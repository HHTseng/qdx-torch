"""JAX vs PyTorch: end-to-end icml2024-AI4science CSS training + evaluation."""

import copy

import numpy as np

from compare_utils import parse_args, repo_on_path, Reporter, tree_max_diff

args = parse_args(__doc__)

r = Reporter("End-to-end CSS training + evaluation (CodeFinderCSS)")

CONFIG = {
    "N": 6,
    "K": 2,
    "D": 3,
    "INIT_H": [2, 4],
    "BELL": False,
    "MAX_STEPS": 10,
    "WHICH_GATES": ["cx"],
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.7,
    "LAMBDA": 0.1,
    "SEED": 1234,
    "LR": 7e-4,
    "NUM_ENVS": 8,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 8 * 10 * 5,   # 5 epochs
    "UPDATE_EPOCHS": 2,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.05,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 32,
    "ANNEAL_LR": True,
    "NUM_AGENTS": 2,
    "COMPUTE_METRICS": True,
}


def to_np_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_np_tree(v) for k, v in t.items()}
    from compare_utils import asnp as _asnp
    return _asnp(t)


def run_side(repo, is_jax):
    with repo_on_path(repo, "icml2024-AI4science", icml=True):
        from code_finder import CodeFinderCSS
        finder = CodeFinderCSS(copy.deepcopy(CONFIG))
        params, metrics = finder.train(params=None)
        data = finder.evaluate()
    return (to_np_tree(params), {k: np.asarray(v) for k, v in metrics.items()},
            data)


params_j, metrics_j, data_j = run_side(args.jax_repo, True)
params_t, metrics_t, data_t = run_side(args.torch_repo, False)

r.check_value_equal("episode-length metric (exact => identical trajectories)",
                    metrics_t["returned_episode_lengths"],
                    metrics_j["returned_episode_lengths"])
r.check_close("episode-return metric", metrics_t["returned_episode_returns"],
              metrics_j["returned_episode_returns"], rtol=1e-4, atol=1e-4)

d = tree_max_diff(params_j, params_t)
r.check("final params close", d < 5e-4, f"max abs diff {d:.3g}")

for i, (a, b) in enumerate(zip(data_j, data_t)):
    same = (a["gates"] == b["gates"] and int(a["d"]) == int(b["d"])
            and a["G_X"] == b["G_X"] and a["G_Z"] == b["G_Z"])
    r.check(f"evaluate() agent {i}: gates/distance/generators identical", same)

r.finish()
