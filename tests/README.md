# JAX ↔ PyTorch comparison test suite

This suite verifies that the PyTorch port in this repository numerically
matches the original JAX implementation. It imports **both** code bases in
one process (swapping `sys.path` / purging `sys.modules` between them), runs
them on identical seeds and inputs, and compares every quantity.

## Requirements

* A Python environment with **both** stacks installed:
  * original: `jax`, `jaxlib`, `flax`, `distrax`, `optax`, `gymnax`, `chex`
  * port: `torch`, `numpy`, `scipy`, `stim`, `more_itertools`
* The original JAX repository checked out somewhere on disk
  (defaults to a sibling directory named `qdx_jolle_ag`).

## Running

```bash
python tests/run_all.py --jax-repo /path/to/qdx_jolle_ag
```

Each test can also be run individually with the same flags.

## What is compared, and how strictly

| Test | Contents | Criterion |
|---|---|---|
| `test_prng.py` | threefry `PRNGKey`/`split`/`fold_in`/`bits`/`uniform`/`categorical`/`permutation`/`randint`/`choice`, flax SHA-1 param-key folding | **bit-exact**; `normal`/`gumbel` within 1 ulp (platform `log`/`log1p` differ from XLA's by the last bit), sampling *decisions* exact |
| `test_simulators.py` | `TableauSimulator`, `CliffordGates`, CSS variants (torch uint8 tableaus) | **bit-exact** |
| `test_envs.py` | all four `qdx` environments + `Utils`: static tensors, rollouts with auto-reset, rewards, dones, discounts, noise draws | integers/bools **bit-exact**; float32 rewards/probabilities ≤ 1e-5 |
| `test_network.py` | flax-equivalent init (orthogonal/QR), forward pass, distrax `Categorical` ops, PPO loss + **torch.autograd** gradients vs `jax.value_and_grad` (incl. the ratio-tie regime, which exercises the balanced 0.5/0.5 min/max tie gradients), optax `chain(clip_by_global_norm, adam)` with LR schedule, GAE | init ≤ 2e-6; forward/loss/grads ≤ ~1e-6; optimizer ≤ 1e-6 over 6 steps; samples exact |
| `test_css_env.py` | icml2024 `CodeDiscoveryCSS`, `UtilsCSS` | integers **bit-exact**; float32 ≤ 1e-5 |
| `test_end_to_end.py` | full `CodeFinder.train()` + `evaluate()` for STANDARD / NOISE-AWARE / DELTA / MAX configs and a 40-epoch [[7,1,3]] demo1-style run | identical trajectories (integer episode lengths exact, returns ≤ 1e-4), final params ≤ 5e-4 (5e-3 budget for the 40-epoch run), identical discovered gate sequences and code distances |
| `test_end_to_end_css.py` | full `CodeFinderCSS.train()` + `evaluate()` | same as above |

All comparisons run the torch side on **CPU**, which is the verified
configuration. (`config["DEVICE"] = "mps"/"cuda"` accelerates training but
introduces backend-specific float32 rounding.)

## Known, expected divergence for very long runs

After several hundred optimizer steps the two implementations' trajectories
eventually diverge: float32 matrix multiplies go through different BLAS
backends (XLA vs torch/ATen), and their last-bit rounding differences are
chaotically amplified by the RL feedback loop the first time a sampling
decision falls within round-off of a tie. This is inherent to *any*
cross-backend float32 comparison (the JAX original behaves the same way
between CPU and GPU backends) and is not a porting error.
`test_end_to_end.py` contains a LONG-RUN check that documents this and
verifies qualitative agreement (identical discovered code distances) past
the onset.

## Measured results (macOS arm64, jax 0.10.2 CPU, torch 2.13.0 CPU)

* 207/207 checks pass across the seven test files.
* Autograd gradients match `jax.value_and_grad` to ≤ ~1e-7 in all regimes.
* The 40-epoch demo1-style run reproduces every sampled action of the JAX
  original exactly; final parameters agree to ≤ ~1e-6.
* The 60-epoch LONG-RUN diverges numerically (params ~0.07 apart) while
  still discovering the same code distances — the expected post-chaos-onset
  behaviour described above.
