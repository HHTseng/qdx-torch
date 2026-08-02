# V1.6 "Size-Aware GNN / New Reward": JAX → PyTorch translation record

Branch: `size-aware-gnn-v16-torch` (in `HHTseng/qdx-torch`)
Parent branch: `gnn-multitask-torch`
JAX source: `qdx-JAX-TCC0731`, branch `qdx-Size-Aware-GNN-V1-6-New-Reward`
Reference analysis: `main_vs_qdx_size_aware_gnn_v1_6.md`

---

## 1. Scope and starting point

The PyTorch line already contained a verified conversion of the JAX `main`
branch:

| Layer | Torch branch | JAX counterpart |
|---|---|---|
| Original Olle et al. QDX (MLP policy) | `main` | `jolle-ag/qdx` |
| GNN multitask fork | `gnn-multitask-torch` | `qdx-JAX-TCC0731@main` |
| **V1.6 size-aware GNN + new reward** | **`size-aware-gnn-v16-torch`** | **`qdx-JAX-TCC0731@qdx-Size-Aware-GNN-V1-6-New-Reward`** |

Before porting, the JAX `main` tree was compared file-by-file against the
`qdx_TCC0731_Jul18` snapshot the torch branch was built from. All ported
source files (`envs/*`, `gnn/*`, `utils.py`, `make_train.py`, `main.py`,
`validation.py`, `runtime_cache.py`) hash-matched, confirming
`gnn-multitask-torch` is an exact torch counterpart of JAX `main` and that
the V1.6 delta could be applied cleanly on top.

Per the request, branch topology/commit history in the JAX repo was ignored;
the two branch tips were treated as independent snapshots.

---

## 2. What was translated

### 2.1 New module: `qdx/gf2_distance.py`

Exact GF(2) stabilizer verification. The JAX original keeps everything
`jit`/`vmap`/`scan`-compatible via fixed-shape `lax.fori_loop`/`lax.cond`;
PyTorch runs eagerly, so the same control flow is written as ordinary Python
loops and `if` statements over torch tensors — every intermediate (RREF
basis, pivots, masks) stays inspectable in a debugger.

| JAX symbol | Torch symbol | Notes |
|---|---|---|
| `jax_gf2_rref` | `torch_gf2_rref` | same pivot convention (`-1` for unused rows, full-height basis, rank as third return) |
| `jax_gf2_row_space_mask` | `torch_gf2_row_space_mask` | |
| `jax_exact_gf2_kl` | `torch_exact_gf2_kl` | RREF row-space membership |
| `jax_tableau_kl` | `torch_tableau_kl` | direct symplectic-coordinate test |
| `jax_softness_kl` | `torch_softness_kl` | legacy softness kernel, benchmarking only |
| `JaxKLResult` | `KLResult` | identical 11 fields |
| `benchmark_jax_kl_reward_calculation` | `benchmark_kl_reward_calculation` | |
| `gf2_rref`, `gf2_row_space_mask`, `symplectic_commutation_mask`, `verify_stabilizer_distance_gf2`, `cached_exact_weight_pauli_errors`, `error_weight_indices_upto`, `precache_pauli_errors`, `stabilizer_check_matrix_from_tableau`, `stabilizer_check_matrix_from_gates` | *(unchanged names)* | already backend-independent NumPy; ported verbatim apart from torch→NumPy input coercion |

**GF(2) arithmetic on accelerators.** torch implements `uint8` matmul on CPU
only (`addmm_cuda not implemented for 'Byte'`). `_gf2_matmul` therefore uses
the `uint8` path on CPU and a float32 product on other devices. This is exact,
not an approximation: operands are 0/1 and every accumulated entry is at most
the matrix width (a few dozen), far below float32's 2^24 exact-integer range,
so the mod-2 fold is unaffected — and TF32 cannot perturb it either, since
0/1 inputs are exactly representable and accumulation is float32.

### 2.2 New module: `qdx/action_space.py`

Copied essentially verbatim (pure Python, no array backend):
`ActionSpec`, `canonical_gate_name`, `gate_arity`, `build_action_specs` with
v1.4 symmetric-gate canonicalization (`CZ`, `SQRT_XX` collapse to one
canonical unordered pair; `CX` keeps both control/target orders).

### 2.3 `qdx/envs/code_discovery.py` — the V1.6 MDP

* `kl_method` ∈ {`existing`, `gf2`, `gf2_tableau`} with the same aliases
  (`legacy`/`softness` → `existing`) and the same dispatch in `check_KL`.
* `EnvState` gains `pending_action_mask`, `progress_score`, `success`.
* Action matrices are built from shared `build_action_specs`, and the disk
  cache key gains `action_space_version` so v1.3 and v1.4 caches cannot
  collide.
* `_build_action_relation_tables` computes the commutation table
  `C_ij = 1[M_i M_j = M_j M_i]` and cancellation table `R_ij = 1[M_i M_j = I]`
  over F2 (kept in NumPy `uint16`, exactly as in JAX).
* `update_pending_action_mask` implements the O(A) parallel update
  `p' = where(C[:,x], p XOR R[:,x], 0)`.
* `_distance_progress` implements the V1.6 frontier score
  `P = 1` on success, else `(d_t - 1 + A_t)/(D-1)` with
  `A_t = 1 - log1p(c_{t,d_t})/log1p(M_{d_t})`.
* `step_env` returns
  `r_t = physical_reward + 0.1 (P_{t+1} - P_t) + 1[success_{t+1} ∧ ¬success_t]`
  and the same seven info fields.
* `is_terminal` now uses the exact `state.success` instead of the mutable
  `self.num_KL`.
* `reset_env` computes `P_0` from the identity tableau.

**float32 fidelity.** JAX computes the progress score in float32 with
weak-typed Python scalars. The torch port mirrors that with `np.float32`
arithmetic (`np.float32(0.1) * (P' - P)`), so the shaping term matches
bit-for-bit rather than being computed in float64.

### 2.4 GNN changes

* `qdx/gnn/observation.py` — `action_is_symmetric` added, `action_edge_indices`
  removed, `action_mask` becomes dynamic (`static_mask & ~pending`), action
  ordering shared with the environment through `build_action_specs`. Field
  order of the `GraphObservation` NamedTuple matches the JAX
  `flax.struct.dataclass` field order (asserted by a test).
* `qdx/gnn/model.py` — symmetric two-qubit gates are scored in both qubit
  orders with the **same** MLP parameters and averaged; directional `CX`
  keeps the ordered score.
* `qdx/envs/graph_code_discovery.py` — relation tables reconfigured to the
  padded action size; pending mask forwarded into the observation builder.

### 2.5 Plumbing

`KL_METHOD` in `BASE_CONFIG` → `make_env` → `CodeDiscovery`; selectable
verifier in `distance_error_stats_up_to_target`; `--kl-method` CLI override
in `main.py`; `validation.py` passes the configured method through. SPEC
V1.4/V1.6 docs and the five updated configs plus `configs/1215_1315.yaml`
were copied from the JAX branch. (`configs/main.yaml` additionally keeps the
torch-only `device:` key.)

### 2.6 Deliberate, documented differences

| Item | JAX | PyTorch port | Why |
|---|---|---|---|
| Checkpoint format | `flax.serialization` msgpack (`params.msgpack`) | NumPy `.npz` (`params.npz`) | inherited from the parent branch; no flax dependency |
| RREF control flow | `lax.fori_loop` + `lax.cond` | Python `for`/`if` | eager torch; same algorithm, same outputs |
| `vmap`/`scan` compat tests | `jax.vmap`, `lax.scan` | explicit batched/sequential loops | eager backend has no tracing to be compatible with |
| GF(2) matmul on GPU | `uint8` throughout | `uint8` on CPU, exact float32 elsewhere | torch has no CUDA `uint8` matmul (see 2.1) |

---

## 3. Verification

Comparison baseline: the V1.6 JAX branch snapshot extracted to
`qdx-JAX-V16`. Runner: `python tests/run_all.py --jax-repo <path>`.

### 3.1 Test inventory

| File | What it checks | Result |
|---|---|---|
| `test_prng.py` | threefry PRNG primitives, flax param-key folding | 47/47 |
| `test_simulators.py` | tableau simulators, Clifford gate matrices | 9/9 |
| `test_envs.py` | all four legacy envs + `Utils` | 80/80 |
| `test_network.py` | MLP ActorCritic, PPO loss/grads, optax-equivalent optimizer, GAE | 24/24 |
| `test_gnn_compare.py` | graph observations, GNN init/forward, PPO autograd grads, greedy validation | 27/27 |
| **`test_v16_compare.py`** | **V1.6-specific: exact-GF(2) kernels, reward components, action space, symmetric logits** | **63/63** |
| `test_css_env.py` | icml2024 CSS environment / UtilsCSS | 16/16 |
| `test_end_to_end.py` | CodeFinder training + evaluation (STANDARD/NOISE-AWARE/DELTA/MAX/demo1) | 26/26 |
| `test_end_to_end_css.py` | CSS CodeFinder training + evaluation | 5/5 |
| `test_end_to_end_gnn.py` | joint multitask GNN training + validation, all three KL modes | 21/21 |
| `tests.test_gf2_distance` | port of the V1.6 unit tests | 12 tests OK |
| `tests.test_gnn_qdx` | port of the expanded V1.4 unit tests | 8 tests OK |

Totals: **12/12 files pass; 318 comparison checks + 20 unit tests**, on both
macOS arm64 and Linux x86-64 (tara).

The 20 ported unit tests reproduce the JAX suite's own result
(`Ran 20 tests ... OK`) with the same assertions and the same expected
numbers — including the softness-1 vs exact disagreement (`-5` vs `-3`
reward on the `ZZI`/`IZZ` example), the five-qubit code's exact distance 3,
and the three-qubit repetition code's weight-1 logical.

### 3.2 What `test_v16_compare.py` establishes (63 checks)

* **Exact KL kernels** — all 11 result fields (logical/commutes/in-stabilizer
  masks, counts, probability mass, cost, reward, terminal) compared field by
  field over 10 cases: hand-written algebraic examples plus randomized
  stabilizer groups generated by real Clifford circuits (n = 3…5), through
  both the RREF and direct-tableau paths. **All exact.**
* **Per-weight statistics** — `error_count_by_weight`, `total_count_by_weight`
  exact; `error_rate_by_weight` ≤ 1e-7.
* **Host verifier** — `verify_stabilizer_distance_gf2(...).to_dict()` compared
  as a whole dict (target met, exactness flag, estimated distance, per-weight
  stats): identical.
* **Action space** — action count, stim strings, action matrices, and both
  relation tables `C_ij` / `R_ij` bit-identical (55 actions for the six-gate
  n=5 all-to-all case).
* **V1.6 reward, all three KL modes** — over a 10-step scripted rollout:
  total reward ≤ 3.6e-7, physical component ≤ 3.6e-7, progress score and
  progress delta **exactly 0 difference**, frontier distance / violations /
  success bonus / done flags / pending-mask popcount / tableau checksum all
  exact.
* **Graph observation** — all integer and mask fields (including the dynamic
  action mask and `action_is_symmetric`) exact across a rollout; float
  features ≤ 1.2e-7; NamedTuple field order matches the flax dataclass.
* **GNN** — flax-default init reproduced to 6e-8; forward logits (with
  symmetric averaging) ≤ 4.8e-7; values ≤ 1.5e-7; entropy ≤ 9.5e-7; symmetric
  action logits verified order-invariant (difference exactly 0).
* **Validation verifier** — `distance_error_stats_up_to_target` returns
  identical tuples for `existing`, `gf2`, and `gf2_tableau`.

### 3.3 End-to-end training (`test_end_to_end_gnn.py`, 21 checks)

Joint multitask PPO (3 tasks over 2 hardware graphs, 4 updates) run on both
implementations from identical seeds, for **each** KL mode:

| KL mode | episode/success/timeout counts | per-update reward & loss | final params | validation gates & distances |
|---|---|---|---|---|
| `gf2_tableau` | exact | ≤ 4.8e-6 | ≤ 7.5e-8 | identical |
| `gf2` | exact | ≤ 5.4e-7 | ≤ 3.4e-7 | identical |
| `existing` | exact | ≤ 5.4e-7 | ≤ 3.4e-7 | identical |

Exact episode/success counts certify that every sampled action, environment
transition, exact-success decision and reset matched.

### 3.4 GPU verification (tara, 1× NVIDIA H100 PCIe)

Run with `CUDA_VISIBLE_DEVICES=0` (single GPU, per shared-server etiquette).
See §4 for the measured numbers.

Two real bugs were found **by** the GPU run and fixed:

1. `torch_gf2_rref` allocated `row_indices`/`pivots` on the CPU regardless of
   the input device, raising a device-mismatch error. All GF(2) helpers now
   allocate on the input's device.
2. `uint8` matmul is CPU-only in torch; `_gf2_matmul` now selects an exact
   float32 path on accelerators (§2.1).

Both were invisible on CPU, which is why the GPU pass matters.

### 3.5 Known, expected divergence (unchanged in kind, earlier in onset)

Long training runs eventually diverge between backends: float32 matmuls use
different BLAS implementations (XLA vs torch/ATen) and the RL feedback loop
chaotically amplifies last-bit differences the first time a sampling decision
lands within round-off of a tie. This is inherent to any cross-backend float32
comparison and is not a porting error.

V1.6 moves the onset **earlier** than on the pre-V1.6 branches, because the
new reward adds a progress-delta term whose log-ratio makes trajectories more
sensitive. Measured for the demo1-style seed (N=7, MLP path):

| epochs | 2 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| param max diff | 4.8e-7 | 4.8e-7 | 4.8e-7 | 4.8e-7 | 1.7e-2 | 6.2e-3 |
| return max diff | 2.7e-5 | 2.7e-5 | 2.9e-5 | 2.9e-5 | 6.5e-1 | 5.8e-1 |

Bit-tight through 20 epochs, then divergence between 20 and 30 — and
**non-monotonic** afterwards (30 epochs worse than 40), which is the signature
of chaotic amplification rather than a systematic error. The same test passes
at 40 epochs on Linux, confirming platform dependence. `test_end_to_end.py`
therefore uses a 20-epoch budget (below the onset on every platform tested)
for its tight numerical check, and keeps the 60-epoch run as a qualitative
check that both sides still discover the same code distances.

---

## 4. GPU acceleration (tara, 1× H100 PCIe)

Environment: `QDX_torch` (torch 2.13.0+cu130, Python 3.14), one H100 PCIe
selected with `CUDA_VISIBLE_DEVICES=0`.

### 4.1 Correctness (CPU vs CUDA)

| Check | Result |
|---|---|
| exact KL kernels, n=6 d=3 (2/153 logicals) | CPU == device, and RREF == direct-tableau |
| exact KL kernels, n=8 d=3 (5/276 logicals) | CPU == device, and RREF == direct-tableau |
| exact KL kernels, n=10 d=3 (0/435 logicals) | CPU == device, and RREF == direct-tableau |
| GNN forward logits / values | ≤ 9.5e-7 / ≤ 5.8e-7 |
| PPO loss / autograd gradients | ≤ 6.0e-8 / ≤ 1.8e-7 |
| env `gf2` vs `gf2_tableau` rewards & termination | agree |

The logical-error masks are **bit-identical** between CPU and GPU (not merely
close), which is the point of the exact-arithmetic path: the mod-2 reduction
removes any float rounding sensitivity.

### 4.2 Exact GF(2) KL kernels

| n | d | errors | RREF cpu | RREF cuda | speedup | tableau cpu | tableau cuda | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 276 | 0.66 ms | 2.05 ms | 0.32× | 0.12 ms | 0.28 ms | 0.44× |
| 10 | 3 | 435 | 0.77 ms | 2.47 ms | 0.31× | 0.16 ms | 0.28 ms | 0.59× |
| 12 | 3 | 630 | 0.96 ms | 3.05 ms | 0.31× | 0.25 ms | 0.28 ms | 0.89× |
| 10 | 4 | 3675 | 2.23 ms | 3.04 ms | 0.73× | 1.09 ms | 0.35 ms | **3.17×** |

**Interpretation.** These kernels are small and launch-bound, so the GPU only
wins once the error set is large (n=10, d=4: 3675 operators → 3.2× on the
direct-tableau path). The RREF path stays slower on GPU at every size tested
because its elimination loop issues ~2n sequential small kernels, each
dominated by launch overhead. This is a property of problem size, not of the
translation — and it is exactly why the environment defaults to CPU: **the
recommended configuration is `KL_METHOD: gf2_tableau` with the environment on
CPU**, moving only the network to the GPU.

### 4.3 V1.6 GNN PPO step (forward + backward + optimizer)

| n | hidden | layers | batch | CPU | CUDA | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 64 | 3 | 128 | 62.2 ms | 20.8 ms | 2.99× |
| 9 | 128 | 3 | 512 | 1842.6 ms | 26.7 ms | **69.1×** |
| 9 | 256 | 3 | 2048 | 12118.2 ms | 169.4 ms | **71.5×** |

### 4.4 Joint multitask training (3 tasks, 5 PPO updates, GNN hidden=128)

| device | total | per update |
|---|---:|---:|
| cpu | 32.70 s | ~6.5 s |
| cuda | 7.69 s | ~1.5 s |
| **speedup** | **4.25×** | |

End-to-end speedup is lower than the isolated PPO-step speedup because the
environments (tableau updates and the exact GF(2) verifier) run on CPU and
dominate the remaining time — the same structural property as the earlier
branches.

Note the two devices' training curves diverge slightly after the first update
(`reward -0.360` vs `-0.262` at update 5). That is the expected cross-backend
float32 chaos of §3.5 acting between CPU and GPU BLAS, not a correctness
issue; the deterministic per-step checks in §4.1 are what certify the GPU
path.

---

## 5. How to reproduce

```bash
# comparison suite (needs both stacks in one env; tara: QDX_torch)
cd qdx_jolle_ag_torch/tests
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python run_all.py \
    --jax-repo /path/to/qdx-JAX-V16 --torch-repo /path/to/qdx_jolle_ag_torch

# ported unit tests only
cd qdx_jolle_ag_torch
python -m unittest tests.test_gf2_distance tests.test_gnn_qdx

# GPU correctness + acceleration (single GPU)
CUDA_VISIBLE_DEVICES=0 python tests/bench_gpu_v16.py --device cuda

# training with the exact verifier
python main.py --config configs/main.yaml --kl-method gf2_tableau
```

The JAX baseline is a plain extraction of the V1.6 branch:

```bash
cd qdx-JAX-TCC0731
git archive qdx-Size-Aware-GNN-V1-6-New-Reward | tar -x -C ../qdx-JAX-V16
```

---

## 6. Caveats carried over from the JAX branch

These are properties of the V1.6 design, faithfully reproduced rather than
fixed by the port (see the reference analysis for full discussion):

1. **"Redundant" means phase-free equivalence.** The cancellation table is
   computed from symplectic matrices, so `S² = Z` counts as identity. Valid
   for the distance/commutation objective, not for signed-Clifford circuit
   identity.
2. **Exponential Pauli enumeration remains.** Exact membership is polynomial
   once the batch exists, but the batch is still `Σ_w C(n,w) 3^w`.
3. **Flat observations hide the pending mask.** `CodeDiscovery.get_obs`
   returns only `vec(G_t)`; only the GNN path sees `p_t` via `action_mask`.
4. **`state_space()` metadata is incomplete** — it declares `tableau`, `time`,
   `pending_action_mask` but not `progress_score`/`success`, and keeps the
   `obs_shape`-based tableau box.
5. **`existing` mode is a hybrid.** It restores the softness-based *physical*
   reward, while progress, success and the bonus still come from the exact
   tableau verifier.
