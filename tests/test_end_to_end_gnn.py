"""JAX(V1.6) vs PyTorch: end-to-end joint multitask GNN training + validation.

Runs the main.py training path (train_joint_multitask + run_validation) on
both implementations with identical seeds, three (n,k,d) tasks over two
hardware graphs, and compares:

  * per-update episode counts / success counts   -> exact (integer-derived,
    certifies identical sampled actions and env transitions),
  * per-update reward means and losses           -> float32 tolerance,
  * final network parameters                     -> small drift tolerance,
  * validation gate sequences and distances      -> exact.
"""

import copy

import numpy as np

from compare_utils import parse_args, repo_on_path, Reporter, tree_max_diff, asnp

args = parse_args(__doc__)

r = Reporter("End-to-end joint multitask GNN training (JAX vs torch)")

CONFIG = {
    "MODEL": "GNN",
    "ENV_TYPE": "STANDARD",
    "D": 3,
    "MAX_STEPS": 10,
    "WHICH_GATES": ("cx", "h"),
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "KL_METHOD": "gf2_tableau",
    "VALIDATION_SOFTNESS": None,
    "P_I": 0.9,
    "LAMBDA": 10,
    "SEED": 42,
    "LR": 1.0e-3,
    "NUM_ENVS_PER_TASK": 4,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 4 * 10 * 3 * 4,   # 4 joint updates over 3 tasks
    "UPDATE_EPOCHS": 2,
    "NUM_MINIBATCHES": 2,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.02,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 16,
    "ANNEAL_LR": True,
    "COMPUTE_METRICS": True,
    "GNN_HIDDEN_DIM": 24,
    "GNN_RELATION_DIM": 8,
    "GNN_GATE_DIM": 8,
    "GNN_NUM_LAYERS": 2,
}

TRAIN_TASKS = [
    {"n": 4, "k": 1, "d": 3, "graph": "All-to-All"},
    {"n": 5, "k": 1, "d": 3, "graph": "All-to-All"},
    {"n": 5, "k": 1, "d": 3, "graph": "NN-1"},
]
VALIDATION_TASKS = [
    {"n": 4, "k": 1, "d": 3, "graph": "All-to-All"},
    {"n": 5, "k": 1, "d": 3, "graph": "NN-1"},
    {"n": 6, "k": 1, "d": 3, "graph": "All-to-All"},
]


def to_np_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_np_tree(v) for k, v in t.items()}
    return asnp(t)


def history_ints(history):
    return [
        (rec["episode_count"], rec["success_count"], rec["timeout_count"])
        for rec in history
    ]


def history_floats(history):
    return np.asarray(
        [
            [rec["reward_mean"], rec["done_rate"], rec["loss"]["total_loss"],
             rec["loss"]["value_loss"], rec["loss"]["actor_loss"],
             rec["loss"]["entropy"]]
            for rec in history
        ],
        dtype=np.float64,
    )


def run_side(repo, is_jax, kl_method="gf2_tableau"):
    with repo_on_path(repo):
        import main as main_mod
        from validation import run_validation
        from qdx.utils import build_graph_padding

        config = copy.deepcopy(CONFIG)
        config["KL_METHOD"] = kl_method
        train_padding = build_graph_padding(TRAIN_TASKS)
        validation_padding = build_graph_padding(VALIDATION_TASKS)
        import time
        params, history, layout = main_mod.train_joint_multitask(
            config,
            total_timesteps=config["TOTAL_TIMESTEPS"],
            train_tasks=copy.deepcopy(TRAIN_TASKS),
            train_graph_padding=train_padding,
            run_started=time.perf_counter(),
        )
        validation = run_validation(
            params,
            config,
            copy.deepcopy(VALIDATION_TASKS),
            validation_graph_padding=validation_padding,
            compute_distance=True,
        )
        if is_jax:
            params_np = to_np_tree(params["params"])
        else:
            params_np = to_np_tree(params["params"])
        return params_np, history, layout, validation


KL_METHOD = "gf2_tableau"
params_j, history_j, layout_j, validation_j = run_side(args.jax_repo, True, KL_METHOD)
params_t, history_t, layout_t, validation_t = run_side(args.torch_repo, False, KL_METHOD)

r.check("training layout identical", layout_t == layout_j,
        f"torch {layout_t} vs jax {layout_j}")
r.check("history length equal", len(history_t) == len(history_j))

r.check("episode/success/timeout counts exact (identical trajectories)",
        history_ints(history_t) == history_ints(history_j),
        f"torch {history_ints(history_t)} vs jax {history_ints(history_j)}")

hf_t, hf_j = history_floats(history_t), history_floats(history_j)
d = float(np.max(np.abs(hf_t - hf_j)))
r.check("per-update reward/done/loss metrics close", d < 1e-4,
        f"max abs diff {d:.3g}")

d = tree_max_diff(params_j, params_t)
r.check("final params close", d < 1e-4, f"max abs diff {d:.3g}")

tasks_j = validation_j["tasks"]
tasks_t = validation_t["tasks"]
r.check("validation: same number of tasks", len(tasks_t) == len(tasks_j))
ok_gates = all(a["gates"] == b["gates"] for a, b in zip(tasks_t, tasks_j))
r.check("validation: identical greedy gate sequences", ok_gates)
ok_dist = all(a["distance"] == b["distance"] for a, b in zip(tasks_t, tasks_j))
r.check("validation: identical distances", ok_dist,
        f"torch {[x['distance'] for x in tasks_t]} vs jax {[x['distance'] for x in tasks_j]}")
ok_stats = all(
    a["distance_stats"] == b["distance_stats"] for a, b in zip(tasks_t, tasks_j)
)
r.check("validation: identical per-weight error stats", ok_stats)
r.check_close(
    "validation: total rewards",
    [x["total_reward"] for x in tasks_t],
    [x["total_reward"] for x in tasks_j],
    rtol=1e-4, atol=1e-4,
)
r.check("validation: distance summary identical",
        validation_t["distance_summary"] == validation_j["distance_summary"])

# ---------------------------------------------------------------------------
# The same joint training under the other two V1.6 KL modes: the physical
# reward kernel differs but progress/success/termination must stay exact.
for method in ("existing", "gf2"):
    pj, hj, lj, vj = run_side(args.jax_repo, True, method)
    pt, ht, lt, vt = run_side(args.torch_repo, False, method)
    r.check(f"[{method}] episode/success/timeout counts exact",
            history_ints(ht) == history_ints(hj),
            f"torch {history_ints(ht)} vs jax {history_ints(hj)}")
    d = float(np.max(np.abs(history_floats(ht) - history_floats(hj))))
    r.check(f"[{method}] per-update reward/loss metrics close", d < 1e-4,
            f"max abs diff {d:.3g}")
    d = tree_max_diff(pj, pt)
    r.check(f"[{method}] final params close", d < 1e-4, f"max abs diff {d:.3g}")
    r.check(f"[{method}] validation gates identical",
            all(a["gates"] == b["gates"] for a, b in zip(vt["tasks"], vj["tasks"])))
    r.check(f"[{method}] validation distances identical",
            all(a["distance"] == b["distance"] for a, b in zip(vt["tasks"], vj["tasks"])),
            f"torch {[x['distance'] for x in vt['tasks']]} vs "
            f"jax {[x['distance'] for x in vj['tasks']]}")

r.finish()
