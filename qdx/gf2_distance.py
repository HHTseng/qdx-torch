"""Exact stabilizer-distance verification over GF(2) (PyTorch port).

Faithful conversion of the JAX module: identical algorithms, identical
returned quantities, identical semantics. The JAX kernels use fixed-shape
``lax.fori_loop``/``lax.cond`` so they stay ``jit``/``vmap``-compatible;
PyTorch runs eagerly, so the same control flow is written as ordinary Python
loops and ``if`` statements over torch tensors. Every intermediate (RREF
basis, pivots, masks) is therefore directly inspectable in a debugger.

This module is intentionally independent of :class:`qdx.utils.Utils`, whose
``softness``-based KL check remains available unchanged.

Naming map versus the JAX original:

    jax_gf2_rref            -> torch_gf2_rref
    jax_gf2_row_space_mask  -> torch_gf2_row_space_mask
    jax_exact_gf2_kl        -> torch_exact_gf2_kl
    jax_tableau_kl          -> torch_tableau_kl
    jax_softness_kl         -> torch_softness_kl
    JaxKLResult             -> KLResult

The host-side NumPy verifier (``gf2_rref``, ``gf2_row_space_mask``,
``symplectic_commutation_mask``, ``verify_stabilizer_distance_gf2``, ...)
keeps its original names because it was already backend independent.

Parity note: GF(2) arithmetic is done with ``uint8`` matmuls, which wrap
modulo 256 in both JAX and torch. Because 256 is even, wrapping preserves
parity, so the subsequent ``& 1`` yields the exact GF(2) result on both
backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import comb
import re
import time
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch

from qdx.runtime_cache import (
    build_exact_weight_error_operators,
    build_s_structure,
    load_or_build_array_bundle,
)
from qdx.simulators import TableauSimulator


_GATE_PATTERN = re.compile(r"^\.([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)$")


class KLResult(NamedTuple):
    """Exact KL/reward values (torch tensors).

    error_cost has the same positive-cost meaning as CodeDiscovery.check_KL.
    reward is the value used by CodeDiscovery.step_env: -error_cost.
    """

    logical_error_mask: torch.Tensor
    commutes_mask: torch.Tensor
    in_stabilizer_mask: torch.Tensor
    error_count: torch.Tensor
    logical_error_probability: torch.Tensor
    error_cost: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor
    error_count_by_weight: torch.Tensor
    total_count_by_weight: torch.Tensor
    error_rate_by_weight: torch.Tensor


class KLStepScalars(NamedTuple):
    """Compact scalar output used by the repeated-kernel benchmark."""

    error_count: torch.Tensor
    logical_error_probability: torch.Tensor
    error_cost: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor


def _gf2_matmul(left, right) -> torch.Tensor:
    """Binary matrix product reduced mod 2, valid on every torch device.

    torch implements uint8 matmul on CPU only ("addmm_cuda not implemented
    for Byte"), so on accelerators the product is taken in float32. That is
    still exact here: the operands are 0/1 and every accumulated entry is at
    most the matrix width (a few dozen), far below float32's 2**24 exact-
    integer range, so the mod-2 fold is unaffected. (TF32 likewise cannot
    perturb it: 0/1 inputs are exactly representable and accumulation is
    float32.)
    """

    if left.device.type == "cpu":
        return (left.to(torch.uint8) @ right.to(torch.uint8)) & 1
    product = left.to(torch.float32) @ right.to(torch.float32)
    return (product.to(torch.int64) & 1).to(torch.uint8)


def torch_gf2_rref(matrix) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GF(2) RREF, mirroring ``jax_gf2_rref``.

    The pivot array has one entry per input row; unused entries are -1. The
    third return value is the rank (the JAX version's ``pivot_row`` carry).
    Unlike the NumPy :func:`gf2_rref` below, the returned basis keeps all
    ``row_count`` rows so shapes stay static, exactly like the JAX kernel.
    """

    reduced = (torch.as_tensor(matrix).to(torch.uint8) & 1).clone()
    device = reduced.device
    row_count, column_count = reduced.shape
    pivots = torch.full((row_count,), -1, dtype=torch.int32, device=device)
    pivot_row = 0
    if row_count == 0:
        return reduced, pivots, torch.tensor(0, dtype=torch.int32, device=device)

    row_indices = torch.arange(row_count, dtype=torch.int32, device=device)

    for column in range(column_count):
        candidates = (row_indices >= pivot_row) & reduced[:, column].bool()
        has_pivot = bool(torch.any(candidates))
        if has_pivot:
            selected_row = int(torch.argmax(candidates.to(torch.int32)))
            safe_pivot_row = min(pivot_row, row_count - 1)

            selected = reduced[selected_row].clone()
            displaced = reduced[safe_pivot_row].clone()
            reduced[safe_pivot_row] = selected
            reduced[selected_row] = displaced

            pivot_vector = reduced[safe_pivot_row].clone()
            eliminate = reduced[:, column] & (
                row_indices != safe_pivot_row
            ).to(torch.uint8)
            reduced ^= eliminate[:, None] * pivot_vector[None, :]
            pivots[safe_pivot_row] = column
            pivot_row += 1

    return reduced, pivots, torch.tensor(pivot_row, dtype=torch.int32, device=device)


def torch_gf2_row_space_mask(
    vectors,
    reduced_basis,
    pivot_columns,
) -> torch.Tensor:
    """Test a batch of vectors against a torch GF(2) RREF basis."""

    remainders = (torch.as_tensor(vectors).to(torch.uint8) & 1).clone()
    device = remainders.device
    basis = (torch.as_tensor(reduced_basis).to(torch.uint8) & 1).to(device)
    pivots = torch.as_tensor(pivot_columns).to(torch.int32).to(device)
    row_count = basis.shape[0]
    if row_count == 0:
        return ~torch.any(remainders.bool(), dim=1)

    for row in range(row_count):
        pivot = int(pivots[row])
        if pivot < 0:
            continue
        eliminate = remainders[:, pivot]
        remainders = remainders ^ eliminate[:, None] * basis[row][None, :]
    return ~torch.any(remainders.bool(), dim=1)


def _torch_commutes_with_stabilizers(error_operators, check_matrix) -> torch.Tensor:
    errors = torch.as_tensor(error_operators).to(torch.uint8) & 1
    stabilizers = torch.as_tensor(check_matrix).to(torch.uint8) & 1
    n_qubits = stabilizers.shape[1] // 2
    symplectic_products = _gf2_matmul(
        errors[:, :n_qubits], stabilizers[:, n_qubits:].T
    )
    symplectic_products = symplectic_products ^ _gf2_matmul(
        errors[:, n_qubits:], stabilizers[:, :n_qubits].T
    )
    return ~torch.any((symplectic_products & 1).bool(), dim=1)


def _kl_result_from_masks(
    logical_error_mask,
    commutes_mask,
    in_stabilizer_mask,
    error_probabilities,
    lbda,
    error_weights=None,
    weight_values=None,
) -> KLResult:
    probabilities = torch.as_tensor(error_probabilities)
    device = probabilities.device
    logical = torch.as_tensor(logical_error_mask).to(torch.bool).to(device)
    error_count = torch.sum(logical.to(torch.int32))
    logical_probability = torch.sum(
        torch.where(logical, probabilities, torch.zeros_like(probabilities))
    )
    error_cost = torch.as_tensor(
        lbda, dtype=probabilities.dtype, device=device
    ) * logical_probability
    if error_weights is None:
        error_count_by_weight = torch.zeros((0,), dtype=torch.int32, device=device)
        total_count_by_weight = torch.zeros((0,), dtype=torch.int32, device=device)
        error_rate_by_weight = torch.zeros(
            (0,), dtype=probabilities.dtype, device=device)
    else:
        if weight_values is None:
            raise ValueError(
                "weight_values is required when error_weights is supplied"
            )
        weights = torch.as_tensor(error_weights).to(torch.int32).to(device)
        values = torch.as_tensor(weight_values).to(torch.int32).to(device)
        weight_mask = weights[:, None] == values[None, :]
        error_count_by_weight = torch.sum(
            (logical[:, None] & weight_mask).to(torch.int32), dim=0
        )
        total_count_by_weight = torch.sum(weight_mask.to(torch.int32), dim=0)
        error_rate_by_weight = torch.where(
            total_count_by_weight > 0,
            error_count_by_weight.to(probabilities.dtype)
            / total_count_by_weight.to(probabilities.dtype),
            torch.zeros_like(total_count_by_weight, dtype=probabilities.dtype),
        )
    return KLResult(
        commutes_mask=torch.as_tensor(commutes_mask).to(torch.bool).to(device),
        logical_error_mask=logical,
        in_stabilizer_mask=torch.as_tensor(in_stabilizer_mask).to(torch.bool).to(device),
        error_count=error_count,
        logical_error_probability=logical_probability,
        error_cost=error_cost,
        reward=-error_cost,
        terminal=error_count == 0,
        error_count_by_weight=error_count_by_weight,
        total_count_by_weight=total_count_by_weight,
        error_rate_by_weight=error_rate_by_weight,
    )


def torch_exact_gf2_kl(
    check_matrix,
    error_operators,
    error_probabilities,
    lbda,
    *,
    error_weights=None,
    weight_values=None,
) -> KLResult:
    """Return exact KL error count, weighted cost, reward, and terminal.

    An operator is a logical error exactly when it commutes with every row of
    the check matrix but is not in that matrix's complete GF(2) row space.
    error_count is the number of such operators, logical_error_probability is
    their supplied probability mass, error_cost is lambda times that mass, and
    reward is its negative.

    terminal means the encoding condition error_count == 0. The environment's
    full done condition must still OR this with state.time >= max_steps.
    """

    stabilizers = torch.as_tensor(check_matrix).to(torch.uint8) & 1
    errors = torch.as_tensor(error_operators).to(torch.uint8) & 1
    reduced, pivots, _rank = torch_gf2_rref(stabilizers)
    in_stabilizer = torch_gf2_row_space_mask(errors, reduced, pivots)
    commutes = _torch_commutes_with_stabilizers(errors, stabilizers)
    logical = commutes & ~in_stabilizer
    return _kl_result_from_masks(
        logical,
        commutes,
        in_stabilizer,
        error_probabilities,
        lbda,
        error_weights,
        weight_values,
    )


def torch_tableau_kl(
    tableau,
    n_logical: int,
    error_operators,
    error_probabilities,
    lbda,
    *,
    error_weights=None,
    weight_values=None,
) -> KLResult:
    """Check KL conditions directly from a complete Clifford tableau.

    The QDX row convention stores the n-k stabilizer generators in the last
    rows. For a symplectic tableau T, T-inverse = Omega @ T.T @ Omega.
    Transforming each error into reference coordinates avoids the per-step
    RREF and row-space elimination used by torch_exact_gf2_kl.
    """

    full_tableau = torch.as_tensor(tableau).to(torch.uint8) & 1
    errors = torch.as_tensor(error_operators).to(torch.uint8) & 1
    if full_tableau.ndim != 2 or full_tableau.shape[0] != full_tableau.shape[1]:
        raise ValueError("tableau must have shape [2n, 2n]")
    if full_tableau.shape[1] % 2:
        raise ValueError("tableau must have an even width")
    n_qubits = full_tableau.shape[1] // 2
    n_logical = int(n_logical)
    if not 0 <= n_logical <= n_qubits:
        raise ValueError("n_logical must satisfy 0 <= n_logical <= n_qubits")
    if errors.ndim != 2 or errors.shape[1] != 2 * n_qubits:
        raise ValueError("error_operators must have shape [errors, 2n]")

    # E @ Omega swaps the X and Z halves. Multiplication by T.T then gives
    # E @ Omega @ T.T; the final Omega only swaps the coordinate halves, so
    # the two coordinate slices needed below can be selected directly.
    transformed = _gf2_matmul(
        torch.cat((errors[:, n_qubits:], errors[:, :n_qubits]), dim=1),
        full_tableau.T,
    )
    commutes = ~torch.any(transformed[:, n_qubits + n_logical :].bool(), dim=1)
    outside_stabilizer = torch.cat(
        (transformed[:, n_qubits:], transformed[:, :n_logical]),
        dim=1,
    )
    in_stabilizer = ~torch.any(outside_stabilizer.bool(), dim=1)
    logical = commutes & ~in_stabilizer
    return _kl_result_from_masks(
        logical,
        commutes,
        in_stabilizer,
        error_probabilities,
        lbda,
        error_weights,
        weight_values,
    )


def torch_softness_kl(
    check_matrix,
    error_operators,
    error_probabilities,
    s_structure,
    lbda,
) -> KLResult:
    """Torch form of the current softness KL, for benchmarking only."""

    stabilizers = torch.as_tensor(check_matrix).to(torch.uint8) & 1
    errors = torch.as_tensor(error_operators).to(torch.uint8) & 1
    enumerated_stabilizers = _gf2_matmul(
        torch.as_tensor(s_structure).to(torch.uint8).to(stabilizers.device),
        stabilizers,
    )
    in_stabilizer = torch.any(
        torch.all(
            errors[:, None, :] == enumerated_stabilizers[None, :, :],
            dim=-1,
        ),
        dim=1,
    )
    commutes = _torch_commutes_with_stabilizers(errors, stabilizers)
    logical = commutes & ~in_stabilizer
    return _kl_result_from_masks(
        logical,
        commutes,
        in_stabilizer,
        error_probabilities,
        lbda,
    )


def _compact_kl_result(result: KLResult) -> KLStepScalars:
    return KLStepScalars(
        error_count=result.error_count,
        logical_error_probability=result.logical_error_probability,
        error_cost=result.error_cost,
        reward=result.reward,
        terminal=result.terminal,
    )


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_kl_reward_calculation(
    check_matrices,
    error_operators,
    error_probabilities,
    lbda: float,
    *,
    num_steps: int = 100,
    repetitions: int = 5,
    print_summary: bool = True,
    device=None,
) -> dict:
    """Benchmark only repeated per-step KL/reward calculation.

    Model inference, tableau construction, and error/cache construction are
    excluded. Every method is warmed and synchronized first.
    """

    if int(num_steps) < 1 or int(repetitions) < 1:
        raise ValueError("num_steps and repetitions must be positive")
    device = torch.device("cpu") if device is None else torch.device(device)
    matrices = torch.as_tensor(check_matrices).to(torch.uint8).to(device)
    if matrices.ndim == 2:
        matrices = matrices.expand((int(num_steps),) + tuple(matrices.shape))
    elif matrices.ndim == 3:
        num_steps = int(matrices.shape[0])
    else:
        raise ValueError("check_matrices must have shape [m, 2n] or [steps, m, 2n]")

    errors = torch.as_tensor(error_operators).to(torch.uint8).to(device)
    probabilities = torch.as_tensor(error_probabilities).to(device)
    num_stabilizers = int(matrices.shape[1])
    method_kernels = {"gf2": torch_exact_gf2_kl}
    for softness in (1, 2, 3):
        structure = torch.from_numpy(
            np.ascontiguousarray(build_s_structure(num_stabilizers, softness))
        ).to(device)
        method_kernels[f"softness_{softness}"] = (
            lambda matrix, operators, probs, scale, structure=structure: (
                torch_softness_kl(matrix, operators, probs, structure, scale)
            )
        )

    def run_steps(matrix_sequence, step_kernel):
        outputs = []
        for index in range(matrix_sequence.shape[0]):
            outputs.append(
                _compact_kl_result(
                    step_kernel(
                        matrix_sequence[index], errors, probabilities, lbda
                    )
                )
            )
        return outputs

    warmup_seconds = {}
    for name, kernel in method_kernels.items():
        started = time.perf_counter()
        run_steps(matrices, kernel)
        _synchronize(device)
        warmup_seconds[name] = time.perf_counter() - started

    outputs = {}
    timings = {}
    calculations = int(num_steps) * int(repetitions)
    for name, kernel in method_kernels.items():
        started = time.perf_counter()
        output = None
        for _ in range(int(repetitions)):
            output = run_steps(matrices, kernel)
            _synchronize(device)
        total_seconds = time.perf_counter() - started
        outputs[name] = output
        timings[name] = {
            "total_seconds": total_seconds,
            "mean_seconds_per_step": total_seconds / calculations,
            "calculations": calculations,
            "warmup_seconds_excluded": warmup_seconds[name],
        }

    gf2_output = outputs["gf2"]
    comparisons = {}
    gf2_seconds = timings["gf2"]["total_seconds"]
    for softness in (1, 2, 3):
        name = f"softness_{softness}"
        output = outputs[name]
        comparisons[name] = {
            "speedup": timings[name]["total_seconds"] / gf2_seconds,
            "reward_allclose": bool(
                all(
                    torch.allclose(a.reward, b.reward)
                    for a, b in zip(output, gf2_output)
                )
            ),
            "error_count_equal": bool(
                all(
                    torch.all(a.error_count == b.error_count)
                    for a, b in zip(output, gf2_output)
                )
            ),
            "terminal_equal": bool(
                all(
                    torch.all(a.terminal == b.terminal)
                    for a, b in zip(output, gf2_output)
                )
            ),
        }

    result = {
        "num_steps": int(num_steps),
        "repetitions": int(repetitions),
        "calculations_per_method": calculations,
        "timings": timings,
        "speedup_definition": "softness total seconds / GF(2) total seconds",
        "gf2_comparisons": comparisons,
    }
    if print_summary:
        print("Torch KL/reward kernel benchmark")
        print(f"  steps={num_steps} repetitions={repetitions}")
        for name, values in timings.items():
            print(
                f"  {name:10s} mean={values['mean_seconds_per_step']:.9f}s "
                f"total={values['total_seconds']:.6f}s"
            )
        for name, values in comparisons.items():
            print(
                f"  gf2 vs {name}: speedup={values['speedup']:.3f}x "
                f"reward_equal={values['reward_allclose']} "
                f"error_equal={values['error_count_equal']} "
                f"terminal_equal={values['terminal_equal']}"
            )
    return result


@dataclass(frozen=True)
class WeightVerificationStats:
    """Exact counts for one physical Pauli weight."""

    weight: int
    violation_count: int
    total_count: int
    commuting_count: int
    stabilizer_count: int

    @property
    def violation_rate(self) -> float:
        return self.violation_count / self.total_count if self.total_count else 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            {
                "d": self.weight,
                "error_count": self.violation_count,
                "error_count_over_total": (
                    f"{self.violation_count}/{self.total_count}"
                ),
                "error_rate": self.violation_rate,
            }
        )
        return result


@dataclass(frozen=True)
class GF2DistanceResult:
    """Result of an exact search over the requested weight range."""

    target_distance: int
    max_weight_checked: int
    target_met: bool
    estimated_distance: int
    distance_is_exact: bool
    first_logical_weight: int | None
    weight_stats: tuple
    logical_error: np.ndarray | None = None

    @property
    def estimated_distance_label(self) -> str:
        prefix = "" if self.distance_is_exact else ">="
        return f"{prefix}{self.estimated_distance}"

    def to_dict(self) -> dict:
        return {
            "target_distance": self.target_distance,
            "max_weight_checked": self.max_weight_checked,
            "target_met": self.target_met,
            "estimated_distance": self.estimated_distance,
            "estimated_distance_label": self.estimated_distance_label,
            "distance_is_exact": self.distance_is_exact,
            "first_logical_weight": self.first_logical_weight,
            "weight_stats": [item.to_dict() for item in self.weight_stats],
            "logical_error": (
                None if self.logical_error is None else self.logical_error.tolist()
            ),
        }


def _as_binary_matrix(matrix, *, name: str) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only binary values")
    return np.ascontiguousarray(array, dtype=np.uint8)


def gf2_rref(matrix) -> tuple[np.ndarray, tuple]:
    """Return the reduced row echelon form and pivot columns over GF(2)."""

    reduced = _as_binary_matrix(matrix, name="matrix").copy()
    row = 0
    pivots: list = []
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[row:, column])
        if not candidates.size:
            continue
        pivot_row = row + int(candidates[0])
        if pivot_row != row:
            reduced[[row, pivot_row]] = reduced[[pivot_row, row]]

        eliminate = reduced[:, column].astype(bool)
        eliminate[row] = False
        reduced[eliminate] ^= reduced[row]
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break

    return reduced[:row], tuple(pivots)


def gf2_row_space_mask(
    vectors,
    reduced_basis,
    pivot_columns: Sequence[int],
) -> np.ndarray:
    """Vectorized membership test against a GF(2) RREF row basis."""

    candidates = _as_binary_matrix(vectors, name="vectors").copy()
    basis = _as_binary_matrix(reduced_basis, name="reduced_basis")
    pivots = tuple(int(column) for column in pivot_columns)
    if candidates.shape[1] != basis.shape[1]:
        raise ValueError("vectors and reduced_basis must have the same width")
    if len(pivots) != basis.shape[0]:
        raise ValueError("one pivot column is required for every basis row")

    for row, column in enumerate(pivots):
        eliminate = candidates[:, column].astype(bool)
        candidates[eliminate] ^= basis[row]
    return ~np.any(candidates, axis=1)


def symplectic_commutation_mask(errors, check_matrix) -> np.ndarray:
    """Return which Pauli errors commute with every stabilizer generator."""

    errors = _as_binary_matrix(errors, name="errors")
    check_matrix = _validate_check_matrix(check_matrix, validate_commutation=False)
    if errors.shape[1] != check_matrix.shape[1]:
        raise ValueError("errors and check_matrix must have the same width")

    n = check_matrix.shape[1] // 2
    products = errors[:, :n] @ check_matrix[:, n:].T
    products ^= errors[:, n:] @ check_matrix[:, :n].T
    return ~np.any(products & np.uint8(1), axis=1)


def _validate_check_matrix(check_matrix, *, validate_commutation: bool) -> np.ndarray:
    check_matrix = _as_binary_matrix(check_matrix, name="check_matrix")
    if not check_matrix.shape[1] or check_matrix.shape[1] % 2:
        raise ValueError("check_matrix must have nonzero even width [H_X | H_Z]")
    if validate_commutation and check_matrix.shape[0]:
        n = check_matrix.shape[1] // 2
        products = check_matrix[:, :n] @ check_matrix[:, n:].T
        products ^= check_matrix[:, n:] @ check_matrix[:, :n].T
        if np.any(products & np.uint8(1)):
            raise ValueError("check_matrix contains anticommuting stabilizer rows")
    return check_matrix


@lru_cache(maxsize=None)
def cached_exact_weight_pauli_errors(n_qubits: int, weight: int) -> np.ndarray:
    """Load or build an immutable, shared array of exact-weight Pauli errors."""

    n_qubits = int(n_qubits)
    weight = int(weight)
    arrays = load_or_build_array_bundle(
        "utils_exact_weight_error_operators",
        {
            "n_qubits_physical": n_qubits,
            "eval_weight": weight,
        },
        lambda: {
            "error_ops": build_exact_weight_error_operators(n_qubits, weight),
        },
    )
    errors = np.ascontiguousarray(arrays["error_ops"], dtype=np.uint8)
    errors.setflags(write=False)
    return errors


@lru_cache(maxsize=None)
def error_weight_indices_upto(n_qubits: int, code_distance: int) -> np.ndarray:
    """Return the physical weight for each environment error operator.

    The ordering matches runtime_cache.build_error_operators_upto: all
    weight-1 operators, then weight-2 operators, and so on through d-1.
    """
    n_qubits = int(n_qubits)
    max_weight = min(int(code_distance) - 1, n_qubits)
    if max_weight < 1:
        result = np.zeros((0,), dtype=np.int32)
    else:
        result = np.concatenate(
            [
                np.full(
                    comb(n_qubits, weight) * (3 ** weight),
                    weight,
                    dtype=np.int32,
                )
                for weight in range(1, max_weight + 1)
            ]
        )
    result.setflags(write=False)
    return result


def precache_pauli_errors(task_ranges: Iterable) -> None:
    """Preload ``(n_qubits, max_weight)`` error ranges before timed work."""

    unique_ranges = {(int(n), int(max_weight)) for n, max_weight in task_ranges}
    for n_qubits, max_weight in sorted(unique_ranges):
        for weight in range(1, min(n_qubits, max_weight) + 1):
            cached_exact_weight_pauli_errors(n_qubits, weight)


def verify_stabilizer_distance_gf2(
    check_matrix,
    target_distance: int,
    *,
    max_weight: int | None = None,
    chunk_size: int | None = None,
    stop_at_first_logical_weight: bool = True,
    validate_stabilizers: bool = True,
) -> GF2DistanceResult:
    """Exactly verify distance using symplectic and GF(2) row-space tests.

    By default only weights ``1..target_distance-1`` are searched, which is
    sufficient to decide whether the target is met. Set ``max_weight`` to the
    target itself when an exact count at that weight is also desired.
    """

    target_distance = int(target_distance)
    if target_distance < 1:
        raise ValueError("target_distance must be positive")
    required_max_weight = target_distance - 1
    resolved_max_weight = (
        required_max_weight if max_weight is None else int(max_weight)
    )
    if resolved_max_weight < required_max_weight:
        raise ValueError("max_weight must be at least target_distance - 1")
    if chunk_size is not None and int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    check_matrix = _validate_check_matrix(
        check_matrix,
        validate_commutation=validate_stabilizers,
    )
    n_qubits = check_matrix.shape[1] // 2
    resolved_max_weight = min(resolved_max_weight, n_qubits)
    reduced_basis, pivots = gf2_rref(check_matrix)

    stats: list = []
    first_logical_weight: int | None = None
    first_logical_error: np.ndarray | None = None
    for weight in range(1, resolved_max_weight + 1):
        errors = cached_exact_weight_pauli_errors(n_qubits, weight)
        batch_size = len(errors) if chunk_size is None else int(chunk_size)
        violation_count = 0
        commuting_count = 0
        stabilizer_count = 0

        for start in range(0, len(errors), batch_size):
            batch = errors[start : start + batch_size]
            commutes = symplectic_commutation_mask(batch, check_matrix)
            normalizer_errors = batch[commutes]
            commuting_count += int(commutes.sum())
            if not len(normalizer_errors):
                continue
            in_row_space = gf2_row_space_mask(
                normalizer_errors,
                reduced_basis,
                pivots,
            )
            stabilizer_count += int(in_row_space.sum())
            logical_count = int((~in_row_space).sum())
            violation_count += logical_count
            if logical_count and first_logical_error is None:
                first_logical_error = normalizer_errors[~in_row_space][0].copy()

        stats.append(
            WeightVerificationStats(
                weight=weight,
                violation_count=violation_count,
                total_count=len(errors),
                commuting_count=commuting_count,
                stabilizer_count=stabilizer_count,
            )
        )
        if violation_count and first_logical_weight is None:
            first_logical_weight = weight
            if stop_at_first_logical_weight:
                break

    checked_through = stats[-1].weight if stats else 0
    distance_is_exact = first_logical_weight is not None
    estimated_distance = (
        first_logical_weight if distance_is_exact else checked_through + 1
    )
    target_met = (
        first_logical_weight is None or first_logical_weight >= target_distance
    )
    return GF2DistanceResult(
        target_distance=target_distance,
        max_weight_checked=checked_through,
        target_met=target_met,
        estimated_distance=estimated_distance,
        distance_is_exact=distance_is_exact,
        first_logical_weight=first_logical_weight,
        weight_stats=tuple(stats),
        logical_error=first_logical_error,
    )


def stabilizer_check_matrix_from_tableau(tableau, n_logical: int) -> np.ndarray:
    """Extract stabilizer generators from a QDX Clifford tableau."""

    tableau = _as_binary_matrix(tableau, name="tableau")
    if tableau.shape[0] != tableau.shape[1] or tableau.shape[1] % 2:
        raise ValueError("tableau must be a square 2n by 2n binary matrix")
    n_qubits = tableau.shape[0] // 2
    n_logical = int(n_logical)
    if not 0 <= n_logical < n_qubits:
        raise ValueError("n_logical must satisfy 0 <= n_logical < n_qubits")
    return np.ascontiguousarray(tableau[n_qubits + n_logical :])


def stabilizer_check_matrix_from_gates(
    n_qubits: int,
    n_logical: int,
    gates: Sequence[str],
) -> np.ndarray:
    """Apply QDX ``.gate(args)`` strings and return final stabilizers."""

    simulator = TableauSimulator(int(n_qubits))
    for gate in gates:
        match = _GATE_PATTERN.fullmatch(str(gate).strip())
        if match is None:
            raise ValueError(f"invalid QDX gate string: {gate!r}")
        gate_name, raw_arguments = match.groups()
        gate_method = getattr(simulator, gate_name, None)
        if gate_method is None or gate_name.startswith("_"):
            raise ValueError(f"unsupported tableau gate: {gate_name!r}")
        arguments = (
            tuple(int(item.strip()) for item in raw_arguments.split(","))
            if raw_arguments.strip()
            else ()
        )
        gate_method(*arguments)
    tableau = np.asarray(
        simulator.current_tableau[0].detach().cpu().numpy(), dtype=np.uint8
    )
    return stabilizer_check_matrix_from_tableau(tableau, n_logical)
