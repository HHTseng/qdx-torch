"""Runs the full JAX-vs-PyTorch comparison suite.

On the size-aware-gnn-v16-torch branch, point --jax-repo at the
qdx-Size-Aware-GNN-V1-6-New-Reward snapshot (default sibling directory
``qdx-JAX-V16``): the environment adopts its exact GF(2) verifier, V1.6
reward, and v1.4 action space.

Usage:
    python tests/run_all.py [--jax-repo /path/to/qdx_jolle_ag]
                            [--torch-repo /path/to/qdx_jolle_ag_torch]

Requires an environment with BOTH the original JAX dependencies (jax, flax,
distrax, optax, gymnax, chex) and this repository's requirements installed.
"""

import os
import subprocess
import sys

from compare_utils import parse_args

TESTS = [
    "test_prng.py",
    "test_simulators.py",
    "test_envs.py",
    "test_network.py",
    "test_gnn_compare.py",
    "test_v16_compare.py",
    "test_css_env.py",
    "test_end_to_end.py",
    "test_end_to_end_css.py",
    "test_end_to_end_gnn.py",
]

# Unit-test modules (run with `python -m unittest`), not JAX comparisons.
UNITTEST_MODULES = [
    "tests.test_gf2_distance",
    "tests.test_gnn_qdx",
]


def main():
    args = parse_args(__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for test in TESTS:
        print(f"\n######## {test} ########", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.join(here, test),
             "--jax-repo", args.jax_repo, "--torch-repo", args.torch_repo],
            cwd=here)
        results[test] = proc.returncode == 0

    for module in UNITTEST_MODULES:
        print(f"\n######## {module} ########", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", module],
            cwd=os.path.dirname(here))
        results[module] = proc.returncode == 0

    print("\n================ SUMMARY ================")
    for test, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {test}")
    if not all(results.values()):
        sys.exit(1)
    print("All comparison suites passed.")


if __name__ == "__main__":
    main()
