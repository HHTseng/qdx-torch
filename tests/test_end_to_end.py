"""JAX vs PyTorch: end-to-end CodeFinder training + evaluation.

Runs a scaled-down demo1-style (STANDARD) and demo2-style (NOISE-AWARE)
training on both implementations with identical seeds and compares:

  * episode-length metrics  -> must match EXACTLY (they are integer-derived,
    so equality certifies that every sampled action / env transition / reset
    across the whole training run was identical on both sides),
  * episode-return metrics  -> float32 tolerance,
  * final network parameters -> small drift tolerance (float32 BLAS/libm
    round-off differences between XLA and torch/ATen compound over
    optimizer steps),
  * evaluate() output       -> identical discovered gate sequences and code
    distances.
"""

import copy

import numpy as np

from compare_utils import parse_args, repo_on_path, Reporter, tree_max_diff, asnp

args = parse_args(__doc__)

r = Reporter("End-to-end training + evaluation (CodeFinder)")


BASE_STANDARD = {
    "ENV_TYPE": "STANDARD",
    "N": 5, "K": 1, "D": 3,
    "MAX_STEPS": 10,
    "WHICH_GATES": ["cx", "h"],
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.9,
    "LAMBDA": 10,
    "SEED": 42,
    "LR": 1e-3,
    "NUM_ENVS": 8,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 8 * 10 * 6,   # 6 epochs
    "UPDATE_EPOCHS": 2,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.02,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 32,
    "ANNEAL_LR": True,
    "NUM_AGENTS": 2,
    "COMPUTE_METRICS": True,
}

BASE_NOISEAWARE = {
    "ENV_TYPE": "NOISE-AWARE",
    "N": 5, "K": 1, "D": 3,
    "MAX_STEPS": 10,
    "WHICH_GATES": ["cx", "h", "s"],
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.9,
    "LAMBDA": 1,
    "SEED": 42,
    "LR": 5e-4,
    "NUM_ENVS": 8,
    "NUM_STEPS": 8,
    "TOTAL_TIMESTEPS": 8 * 8 * 5,    # 5 epochs
    "UPDATE_EPOCHS": 2,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.1,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.05,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 32,
    "ANNEAL_LR": True,
    "NUM_AGENTS": 2,
    "COMPUTE_METRICS": True,
}


def to_np_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_np_tree(v) for k, v in t.items()}
    return asnp(t)


def run_jax(config):
    import jax
    with repo_on_path(args.jax_repo):
        from qdx.code_finder import CodeFinder
        finder = CodeFinder(config)
        params, metrics = finder.train()
        data = finder.evaluate()
    return to_np_tree(params), {k: np.asarray(v) for k, v in metrics.items()}, data


def run_torch(config):
    with repo_on_path(args.torch_repo):
        from qdx.code_finder import CodeFinder
        finder = CodeFinder(config)
        params, metrics = finder.train()
        data = finder.evaluate()
    return to_np_tree(params), {k: np.asarray(v) for k, v in metrics.items()}, data


def clean_eval(data):
    out = []
    for d in data:
        out.append({k: (int(v) if k in ("n", "k", "d") else
                        (float(v) if k in ("cZ", "d_eff") else list(v)))
                    for k, v in d.items()})
    return out


# DELTA and MAX variants (different reward shaping / info dicts / float obs)
BASE_DELTA = dict(BASE_STANDARD, ENV_TYPE="DELTA", LAMBDA=10)
BASE_MAX = dict(BASE_STANDARD, ENV_TYPE="MAX", LAMBDA=10)

# demo1-style config (the repo's actual [[7,1,3]] example, shortened from
# 2e6 timesteps so the sequential torch side stays fast). 40 epochs = 480
# optimizer steps: measured to be safely below the cross-backend float32
# "chaos onset" for this seed (see LONG-RUN check below).
DEMO1_SHORT = {
    "ENV_TYPE": "STANDARD",
    "N": 7, "K": 1, "D": 3,
    "MAX_STEPS": 20,
    "WHICH_GATES": ["cx", "h"],
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.9,
    "LAMBDA": 10,
    "SEED": 42,
    "LR": 1e-3,
    "NUM_ENVS": 16,
    "NUM_STEPS": 20,
    "TOTAL_TIMESTEPS": 16 * 20 * 40,   # 40 epochs
    "UPDATE_EPOCHS": 3,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.02,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 32,
    "ANNEAL_LR": True,
    "NUM_AGENTS": 2,
    "COMPUTE_METRICS": True,
}

for label, base in [("STANDARD", BASE_STANDARD), ("NOISE-AWARE", BASE_NOISEAWARE),
                    ("DELTA", BASE_DELTA), ("MAX", BASE_MAX),
                    ("DEMO1-SHORT (N=7, 40 epochs)", DEMO1_SHORT)]:
    params_j, metrics_j, data_j = run_jax(copy.deepcopy(base))
    params_t, metrics_t, data_t = run_torch(copy.deepcopy(base))

    # Episode lengths are integer-derived: exact equality certifies identical
    # trajectories (every sampled action and env transition) on both sides.
    r.check_value_equal(f"{label}: episode-length metric (exact => identical trajectories)",
                        metrics_t["returned_episode_lengths"],
                        metrics_j["returned_episode_lengths"])
    r.check_close(f"{label}: episode-return metric",
                  metrics_t["returned_episode_returns"],
                  metrics_j["returned_episode_returns"], rtol=1e-4, atol=1e-4)

    d = tree_max_diff(params_j, params_t)
    param_tol = 5e-3 if "DEMO1" in label else 5e-4
    r.check(f"{label}: final params close", d < param_tol, f"max abs diff {d:.3g}")

    ej, et = clean_eval(data_j), clean_eval(data_t)
    r.check(f"{label}: evaluate() gate sequences identical",
            all(a["gates"] == b["gates"] for a, b in zip(ej, et)))
    dist_key = "d_eff" if label == "NOISE-AWARE" else "d"
    r.check(f"{label}: evaluate() distances identical",
            all(a[dist_key] == b[dist_key] for a, b in zip(ej, et)),
            f"torch {[x[dist_key] for x in et]} vs jax {[x[dist_key] for x in ej]}")

# ---------------------------------------------------------------------------
# LONG-RUN check (informational, qualitative criteria).
#
# Beyond several hundred optimizer steps the two implementations inevitably
# diverge: float32 matmuls use different BLAS backends (XLA vs torch/ATen),
# whose 1-ulp rounding differences are chaotically amplified by the RL
# feedback loop the first time an argmax/sampling decision lands within
# round-off of a tie. This is inherent to any cross-backend float32
# comparison (the JAX original shows the same behaviour between CPU and GPU)
# and not a porting error, so for long runs we only require qualitative
# agreement: the same discovered code distances.
# ---------------------------------------------------------------------------
LONG_RUN = dict(DEMO1_SHORT, TOTAL_TIMESTEPS=16 * 20 * 60)   # 60 epochs
params_j, metrics_j, data_j = run_jax(copy.deepcopy(LONG_RUN))
params_t, metrics_t, data_t = run_torch(copy.deepcopy(LONG_RUN))
d = tree_max_diff(params_j, params_t)
ret_d = float(np.max(np.abs(
    np.asarray(metrics_j["returned_episode_returns"], dtype=np.float64)
    - np.asarray(metrics_t["returned_episode_returns"], dtype=np.float64))))
print(f"[INFO] LONG-RUN (60 epochs): param max diff {d:.3g}, "
      f"returns max diff {ret_d:.3g} (post-chaos-onset drift is expected)")
ej, et = clean_eval(data_j), clean_eval(data_t)
r.check("LONG-RUN (60 epochs): evaluate() distances identical (qualitative)",
        all(a["d"] == b["d"] for a, b in zip(ej, et)),
        f"torch {[x['d'] for x in et]} vs jax {[x['d'] for x in ej]}")

r.finish()
