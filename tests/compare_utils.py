"""Shared helpers for the JAX-vs-PyTorch comparison test suite.

Both repositories define a top-level package named ``qdx`` (and the
icml2024-AI4science folder defines top-level modules ``simulators``, ``envs``,
``make_train``, ``code_finder``, ``utils_css``). To compare them inside one
process we import one side, grab the objects we need, then purge the modules
from ``sys.modules`` and import the other side.

All comparison helpers accept torch tensors, JAX arrays, NumPy arrays, and
plain Python scalars/lists interchangeably (torch tensors are detached and
moved to CPU before conversion).
"""

import argparse
import contextlib
import importlib
import os
import sys

import numpy as np

# Module names that must be purged between repo switches
_QDX_PREFIXES = ("qdx",)
_ICML_PREFIXES = ("simulators", "envs", "make_train", "code_finder",
                  "utils_css", "torch_random", "torch_nn", "torch_env_base")


def asnp(x):
    """Convert torch tensors / JAX arrays / lists to a NumPy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def default_repo_paths():
    """(torch_repo, jax_repo) defaults relative to this file's location."""
    here = os.path.dirname(os.path.abspath(__file__))
    torch_repo = os.path.dirname(here)
    jax_repo = os.path.join(os.path.dirname(torch_repo), "qdx_jolle_ag")
    return torch_repo, jax_repo


def parse_args(description):
    torch_repo, jax_repo = default_repo_paths()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--jax-repo", default=jax_repo,
                   help="Path to the original JAX repository")
    p.add_argument("--torch-repo", default=torch_repo,
                   help="Path to the converted PyTorch repository")
    return p.parse_args()


def _purge(prefixes):
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


@contextlib.contextmanager
def repo_on_path(repo_root, subdir=None, icml=False):
    """Context manager: put ``repo_root[/subdir]`` at the front of sys.path,
    with the relevant module namespaces purged before and after."""
    prefixes = _ICML_PREFIXES if icml else _QDX_PREFIXES
    path = os.path.join(repo_root, subdir) if subdir else repo_root
    _purge(prefixes)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)
        _purge(prefixes)


class Reporter:
    """Collects named checks; exits nonzero if any failed."""

    def __init__(self, title):
        self.title = title
        self.failures = []
        self.count = 0
        print(f"===== {title} =====")

    def check(self, name, ok, detail=""):
        self.count += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok

    def check_exact(self, name, a, b):
        a = asnp(a)
        b = asnp(b)
        ok = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        detail = "" if ok else f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
        if a.shape == b.shape and not np.array_equal(a, b):
            detail = f"{np.sum(a != b)} of {a.size} elements differ"
        return self.check(name, ok, detail)

    def check_value_equal(self, name, a, b):
        """Exact value equality, ignoring dtype differences."""
        a = asnp(a)
        b = asnp(b)
        ok = a.shape == b.shape and np.array_equal(a, b)
        detail = "" if ok else (
            f"shapes {a.shape} vs {b.shape}" if a.shape != b.shape
            else f"{np.sum(a != b)} of {a.size} elements differ")
        return self.check(name, ok, detail)

    def check_close(self, name, a, b, rtol=1e-6, atol=1e-6):
        a = asnp(a).astype(np.float64)
        b = asnp(b).astype(np.float64)
        if a.shape != b.shape:
            return self.check(name, False, f"shapes {a.shape} vs {b.shape}")
        ok = np.allclose(a, b, rtol=rtol, atol=atol)
        diff = np.max(np.abs(a - b)) if a.size else 0.0
        return self.check(name, ok, f"max abs diff {diff:.3g}")

    def finish(self):
        print(f"----- {self.title}: {self.count - len(self.failures)}/{self.count} passed -----")
        if self.failures:
            print("Failed checks:", ", ".join(self.failures))
            sys.exit(1)
        sys.exit(0)


def tree_max_diff(tree_a, tree_b):
    """Max abs difference across two nested dicts of arrays/tensors."""
    diffs = []

    def rec(a, b):
        if isinstance(a, dict):
            for k in a:
                rec(a[k], b[k])
        else:
            diffs.append(np.max(np.abs(asnp(a).astype(np.float64)
                                       - asnp(b).astype(np.float64))))
    rec(tree_a, tree_b)
    return max(diffs) if diffs else 0.0
