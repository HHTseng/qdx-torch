"""PyTorch port of the V1.6 exact-GF(2) tests.

Same assertions and same expected numbers as the JAX original; the
``jit``/``vmap``/``scan`` compatibility tests become batched/looped torch
equivalents, since torch runs eagerly.
"""

import unittest

import numpy as np
import torch

from qdx.envs.code_discovery import CodeDiscovery
from qdx.gf2_distance import (
    gf2_rref,
    gf2_row_space_mask,
    torch_exact_gf2_kl,
    torch_tableau_kl,
    torch_softness_kl,
    stabilizer_check_matrix_from_gates,
    symplectic_commutation_mask,
    verify_stabilizer_distance_gf2,
)
from qdx import torch_random
from qdx.runtime_cache import build_s_structure
from qdx.simulators import TableauSimulator
from qdx.simulators.clifford_gates import CliffordGates


def check_matrix(*paulis):
    n_qubits = len(paulis[0])
    matrix = np.zeros((len(paulis), 2 * n_qubits), dtype=np.uint8)
    for row, pauli in enumerate(paulis):
        for qubit, operator in enumerate(pauli):
            if operator in "XY":
                matrix[row, qubit] = 1
            if operator in "YZ":
                matrix[row, n_qubits + qubit] = 1
    return matrix


def as_tensor(matrix):
    return torch.from_numpy(np.ascontiguousarray(matrix))


class GF2DistanceTest(unittest.TestCase):
    def test_code_discovery_gf2_and_tableau_rewards_match(self):
        n_qubits = 5
        gates = CliffordGates(n_qubits)
        graph = [
            (source, target)
            for source in range(n_qubits)
            for target in range(n_qubits)
            if source != target
        ]
        common_kwargs = dict(
            n_qubits_physical=n_qubits,
            n_qubits_logical=1,
            code_distance=3,
            gates=[gates.h, gates.cx],
            graph=graph,
            max_steps=50,
            lbda=10.0,
            pI=0.9,
            softness=1,
        )
        gf2_env = CodeDiscovery(**common_kwargs, kl_method="gf2")
        tableau_env = CodeDiscovery(**common_kwargs, kl_method="gf2_tableau")
        _, initial_state = gf2_env.reset(torch_random.PRNGKey(0), None)
        params = gf2_env.default_params
        _, gf2_state, gf2_reward, _, _ = gf2_env.step_env(
            torch_random.PRNGKey(1), initial_state, 0, params
        )
        _, tableau_state, tableau_reward, _, _ = tableau_env.step_env(
            torch_random.PRNGKey(1), initial_state, 0, params
        )
        np.testing.assert_array_equal(
            np.asarray(gf2_state.tableau), np.asarray(tableau_state.tableau)
        )
        np.testing.assert_allclose(
            np.asarray(gf2_reward), np.asarray(tableau_reward), atol=1.0e-6
        )

    def test_tableau_kl_matches_rref_and_reports_weight_rates(self):
        simulator = TableauSimulator(3)
        simulator.h(0)
        simulator.cx(0, 1)
        simulator.s(2)
        tableau = simulator.current_tableau[0]
        stabilizers = tableau[4:]
        errors = as_tensor(check_matrix("XII", "ZII", "IXI", "IZI", "IIZ", "IXX"))
        probabilities = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32
        )
        weights = torch.tensor([1, 1, 1, 1, 1, 2], dtype=torch.int32)
        weight_values = torch.tensor([1, 2], dtype=torch.int32)

        exact = torch_exact_gf2_kl(
            stabilizers,
            errors,
            probabilities,
            10.0,
            error_weights=weights,
            weight_values=weight_values,
        )
        direct = torch_tableau_kl(
            tableau,
            1,
            errors,
            probabilities,
            10.0,
            error_weights=weights,
            weight_values=weight_values,
        )

        np.testing.assert_array_equal(
            np.asarray(direct.logical_error_mask),
            np.asarray(exact.logical_error_mask),
        )
        np.testing.assert_array_equal(
            np.asarray(direct.commutes_mask), np.asarray(exact.commutes_mask)
        )
        np.testing.assert_array_equal(
            np.asarray(direct.in_stabilizer_mask),
            np.asarray(exact.in_stabilizer_mask),
        )
        np.testing.assert_array_equal(
            np.asarray(direct.error_count_by_weight),
            np.asarray(exact.error_count_by_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(direct.total_count_by_weight), [5, 1]
        )
        np.testing.assert_allclose(
            np.asarray(direct.error_rate_by_weight),
            np.asarray(exact.error_rate_by_weight),
        )
        self.assertAlmostEqual(float(direct.reward), float(exact.reward), places=6)

    def test_exact_kl_logical_mask_count_and_weighted_reward(self):
        stabilizers = as_tensor(check_matrix("ZZI", "IZZ"))
        errors = as_tensor(check_matrix("ZIZ", "ZII", "XII"))
        probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)

        result = torch_exact_gf2_kl(stabilizers, errors, probabilities, 10.0)

        np.testing.assert_array_equal(
            np.asarray(result.in_stabilizer_mask), [True, False, False]
        )
        np.testing.assert_array_equal(
            np.asarray(result.commutes_mask), [True, True, False]
        )
        np.testing.assert_array_equal(
            np.asarray(result.logical_error_mask), [False, True, False]
        )
        self.assertEqual(int(result.error_count), 1)
        self.assertAlmostEqual(
            float(result.logical_error_probability), 0.3, places=6
        )
        self.assertAlmostEqual(float(result.error_cost), 3.0, places=6)
        self.assertAlmostEqual(float(result.reward), -3.0, places=6)
        self.assertFalse(bool(result.terminal))

    def test_exact_kl_uses_full_row_space_not_softness(self):
        stabilizers = as_tensor(check_matrix("ZZI", "IZZ"))
        errors = as_tensor(check_matrix("ZIZ", "ZII", "XII"))
        probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
        exact = torch_exact_gf2_kl(stabilizers, errors, probabilities, 10.0)
        softness_one = torch_softness_kl(
            stabilizers,
            errors,
            probabilities,
            as_tensor(build_s_structure(2, 1)),
            10.0,
        )
        softness_two = torch_softness_kl(
            stabilizers,
            errors,
            probabilities,
            as_tensor(build_s_structure(2, 2)),
            10.0,
        )

        self.assertEqual(int(exact.error_count), 1)
        self.assertEqual(int(softness_one.error_count), 2)
        self.assertAlmostEqual(float(softness_one.reward), -5.0, places=6)
        self.assertEqual(int(softness_two.error_count), int(exact.error_count))
        self.assertAlmostEqual(
            float(softness_two.reward), float(exact.reward), places=6
        )

    def test_exact_kl_supports_batched_and_sequential_use(self):
        """Torch analogue of the JAX vmap/scan compatibility test."""
        stabilizers = as_tensor(check_matrix("ZZI", "IZZ"))
        errors = as_tensor(check_matrix("ZIZ", "ZII", "XII"))
        probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
        batch = torch.stack((stabilizers, stabilizers))

        batched = [
            torch_exact_gf2_kl(batch[index], errors, probabilities, 10.0)
            for index in range(batch.shape[0])
        ]
        sequential = []
        for index in range(batch.shape[0]):
            result = torch_exact_gf2_kl(batch[index], errors, probabilities, 10.0)
            sequential.append(int(result.error_count))

        np.testing.assert_array_equal(
            [int(item.error_count) for item in batched], [1, 1]
        )
        np.testing.assert_array_equal(sequential, [1, 1])

    def test_exact_kl_terminal_when_no_logical_error_remains(self):
        result = torch_exact_gf2_kl(
            as_tensor(check_matrix("ZZI", "IZZ")),
            as_tensor(check_matrix("ZIZ", "XII")),
            torch.tensor([0.4, 0.6], dtype=torch.float32),
            10.0,
        )
        self.assertEqual(int(result.error_count), 0)
        self.assertAlmostEqual(float(result.reward), 0.0, places=6)
        self.assertTrue(bool(result.terminal))

    def test_rref_row_space_membership_with_redundant_generators(self):
        generators = np.asarray(
            [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8
        )
        reduced, pivots = gf2_rref(generators)
        vectors = np.asarray(
            [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], dtype=np.uint8
        )

        np.testing.assert_array_equal(
            gf2_row_space_mask(vectors, reduced, pivots), [True, True, False]
        )

    def test_symplectic_commutation(self):
        stabilizers = check_matrix("ZZI", "IZZ")
        errors = check_matrix("XII", "ZII", "IIZ")
        np.testing.assert_array_equal(
            symplectic_commutation_mask(errors, stabilizers), [False, True, True]
        )

    def test_three_qubit_repetition_code_has_weight_one_logical(self):
        result = verify_stabilizer_distance_gf2(
            check_matrix("ZZI", "IZZ"), target_distance=2
        )

        self.assertFalse(result.target_met)
        self.assertTrue(result.distance_is_exact)
        self.assertEqual(result.estimated_distance, 1)
        self.assertEqual(result.weight_stats[0].violation_count, 3)

    def test_five_qubit_code_has_exact_distance_three(self):
        five_qubit_code = check_matrix("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")

        target_result = verify_stabilizer_distance_gf2(
            five_qubit_code, target_distance=3
        )
        exact_result = verify_stabilizer_distance_gf2(
            five_qubit_code, target_distance=3, max_weight=3
        )

        self.assertTrue(target_result.target_met)
        self.assertFalse(target_result.distance_is_exact)
        self.assertEqual(target_result.estimated_distance_label, ">=3")
        self.assertTrue(exact_result.target_met)
        self.assertTrue(exact_result.distance_is_exact)
        self.assertEqual(exact_result.estimated_distance, 3)
        self.assertGreater(exact_result.weight_stats[-1].violation_count, 0)

    def test_gate_helper_matches_qdx_tableau_layout(self):
        no_gate_check = stabilizer_check_matrix_from_gates(2, 1, [])
        after_h = stabilizer_check_matrix_from_gates(2, 1, [".h(1)"])

        np.testing.assert_array_equal(no_gate_check, check_matrix("IZ"))
        np.testing.assert_array_equal(after_h, check_matrix("IX"))

    def test_rejects_anticommuting_stabilizers(self):
        with self.assertRaisesRegex(ValueError, "anticommuting"):
            verify_stabilizer_distance_gf2(
                check_matrix("X", "Z"), target_distance=2
            )


if __name__ == "__main__":
    unittest.main()
