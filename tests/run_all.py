"""Runs the full JAX-vs-PyTorch comparison suite.

On the gnn-multitask-torch branch, point --jax-repo at the qdx_TCC0731_Jul18
snapshot (the default): the standard environment and Utils adopt its direct
error-operator ordering, and the GNN tests compare against its flax model.

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
    "test_css_env.py",
    "test_end_to_end.py",
    "test_end_to_end_css.py",
    "test_end_to_end_gnn.py",
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

    print("\n================ SUMMARY ================")
    for test, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {test}")
    if not all(results.values()):
        sys.exit(1)
    print("All comparison suites passed.")


if __name__ == "__main__":
    main()
