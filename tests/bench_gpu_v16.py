"""GPU correctness + acceleration benchmark for the V1.6 additions.

Torch-vs-torch (CPU vs accelerator), not a JAX comparison — the JAX-vs-torch
verification lives in test_v16_compare.py / test_end_to_end_gnn.py and runs on
CPU.

  1. Correctness: the exact GF(2) kernels (RREF row-space and direct-tableau),
     the V1.6 GNN forward/PPO gradients, and the environment reward must agree
     between CPU and the accelerator.
  2. Acceleration: timing of the exact KL kernels across (n, d) sizes, of the
     GNN PPO step, and of a short joint multitask training run.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/bench_gpu_v16.py [--device cuda]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdx import torch_nn, torch_random
from qdx.gf2_distance import torch_exact_gf2_kl, torch_tableau_kl
from qdx.gnn.model import GNNQDXActorCritic
from qdx.gnn.observation import GraphPadding, obs_stack, obs_take, obs_to_device
from qdx.envs.graph_code_discovery import GraphCodeDiscovery
from qdx.runtime_cache import build_error_operators_upto
from qdx.simulators import TableauSimulator
from qdx.simulators.clifford_gates import CliffordGates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def tree_max_diff(a, b):
    if isinstance(a, dict):
        return max(tree_max_diff(a[k], b[k]) for k in a)
    return float(torch.max(torch.abs(a.detach().cpu().float() - b.detach().cpu().float())))


def to_device_tree(tree, device):
    return torch_nn._tree_map(
        lambda t: t.detach().to(device).requires_grad_(True), tree)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def random_tableau(n, seed=0):
    rng = np.random.default_rng(seed)
    sim = TableauSimulator(n)
    for _ in range(4 * n):
        if rng.random() < 0.4:
            getattr(sim, rng.choice(["h", "s", "sqrt_x"]))(int(rng.integers(n)))
        else:
            a, b = rng.choice(n, size=2, replace=False)
            getattr(sim, rng.choice(["cx", "cz", "sqrt_xx"]))(int(a), int(b))
    return sim.current_tableau[0]


def make_env(n, k, d, padding, gate_names=("cx", "h", "s"), kl_method="gf2_tableau"):
    gates = CliffordGates(n)
    gate_set = [getattr(gates, name) for name in gate_names]
    return GraphCodeDiscovery(
        n, k, d, gate_set, max_steps=20, lbda=10, pI=0.9, softness=1,
        kl_method=kl_method, graph_padding=padding,
    )


def build_batch(env, batch, seed=0):
    rng = np.random.default_rng(seed)
    obs, state = env.reset(torch_random.PRNGKey(seed), None)
    observations = []
    while len(observations) < batch:
        observations.append(obs)
        action = int(rng.integers(env.num_actions))
        obs, state, _, done, _ = env.step(
            torch_random.PRNGKey(seed + len(observations)), state, action, None)
    return obs_stack(observations)


# ---------------------------------------------------------------- correctness
def correctness_checks(device):
    print(f"\n===== V1.6 correctness: CPU vs {device} =====")
    failures = []

    # (1) exact GF(2) kernels
    for n, d in [(6, 3), (8, 3), (10, 3)]:
        tableau = random_tableau(n, seed=n)
        errors, probs = build_error_operators_upto(n, d, 0.9)
        errors_t = torch.from_numpy(np.ascontiguousarray(errors))
        probs_t = torch.from_numpy(np.ascontiguousarray(probs))
        stabs = tableau[n + 1 :]

        cpu_rref = torch_exact_gf2_kl(stabs, errors_t, probs_t, 10.0)
        dev_rref = torch_exact_gf2_kl(
            stabs.to(device), errors_t.to(device), probs_t.to(device), 10.0)
        cpu_tab = torch_tableau_kl(tableau, 1, errors_t, probs_t, 10.0)
        dev_tab = torch_tableau_kl(
            tableau.to(device), 1, errors_t.to(device), probs_t.to(device), 10.0)

        same = (
            torch.equal(cpu_rref.logical_error_mask, dev_rref.logical_error_mask.cpu())
            and torch.equal(cpu_tab.logical_error_mask, dev_tab.logical_error_mask.cpu())
            and int(cpu_rref.error_count) == int(dev_rref.error_count)
            and int(cpu_tab.error_count) == int(dev_tab.error_count)
            and abs(float(cpu_rref.error_cost) - float(dev_rref.error_cost)) < 1e-5
            and abs(float(cpu_tab.error_cost) - float(dev_tab.error_cost)) < 1e-5
        )
        # RREF and direct-tableau must also agree with each other (exactness)
        agree = torch.equal(
            cpu_rref.logical_error_mask, cpu_tab.logical_error_mask)
        if not check(f"exact KL kernels n={n} d={d}: CPU==device and rref==tableau",
                     same and agree,
                     f"count={int(cpu_rref.error_count)}/{errors.shape[0]}"):
            failures.append(f"kernels-n{n}")

    # (2) GNN forward + PPO gradients under the V1.6 observation schema
    padding = GraphPadding(n_max=7, stabilizers_max=6, hardware_edges_max=42)
    env = make_env(7, 1, 3, padding)
    net = GNNQDXActorCritic(num_gate_types=3, hidden_dim=64, num_gnn_layers=3,
                            activation="relu")
    params_cpu = net.init(torch_random.PRNGKey(5), env.graph_observation_template())
    params_dev = to_device_tree(params_cpu, device)
    obs_cpu = build_batch(env, 64)
    obs_dev = obs_to_device(obs_cpu, device)

    with torch.no_grad():
        pi_c, v_c = net.apply(params_cpu, obs_cpu)
        pi_d, v_d = net.apply(params_dev, obs_dev)
    mask = obs_cpu.action_mask
    d_logits = float(torch.max(torch.abs(pi_c.logits[mask] - pi_d.logits.cpu()[mask])))
    if not check("GNN forward: valid logits CPU vs device", d_logits < 1e-3,
                 f"max abs diff {d_logits:.3g}"):
        failures.append("logits")
    d_v = float(torch.max(torch.abs(v_c - v_d.cpu())))
    if not check("GNN forward: values CPU vs device", d_v < 1e-3,
                 f"max abs diff {d_v:.3g}"):
        failures.append("values")

    B = 64
    config = {"ACTIVATION": "relu", "CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.02}
    action = torch.from_numpy(np.random.default_rng(1).integers(0, env.num_actions, B))
    old_value = v_c.detach() + 0.02
    old_log_prob = pi_c.log_prob(action).detach()
    gae = torch.from_numpy(np.random.default_rng(2).standard_normal(B).astype(np.float32))
    targets = torch.from_numpy(np.random.default_rng(3).standard_normal(B).astype(np.float32))

    (loss_c, _), grads_c = torch_nn.ppo_loss_and_grad_generic(
        net.apply, params_cpu, obs_cpu, action, old_value, old_log_prob, gae,
        targets, config)
    (loss_d, _), grads_d = torch_nn.ppo_loss_and_grad_generic(
        net.apply, params_dev, obs_dev, action.to(device), old_value.to(device),
        old_log_prob.to(device), gae.to(device), targets.to(device), config)
    dl = abs(float(loss_c) - float(loss_d))
    if not check("PPO loss CPU vs device", dl < 1e-4, f"abs diff {dl:.3g}"):
        failures.append("loss")
    dg = tree_max_diff(grads_c["params"], grads_d["params"])
    if not check("PPO autograd gradients CPU vs device", dg < 1e-3,
                 f"max abs diff {dg:.3g}"):
        failures.append("grads")

    # (3) V1.6 environment reward components are identical across KL modes
    #     for the exact methods (gf2 vs gf2_tableau).
    env_gf2 = make_env(6, 1, 3, GraphPadding(n_max=6, stabilizers_max=5,
                                             hardware_edges_max=30),
                       kl_method="gf2")
    env_tab = make_env(6, 1, 3, GraphPadding(n_max=6, stabilizers_max=5,
                                             hardware_edges_max=30),
                       kl_method="gf2_tableau")
    _, s_gf2 = env_gf2.reset(torch_random.PRNGKey(0), None)
    _, s_tab = env_tab.reset(torch_random.PRNGKey(0), None)
    ok_env = True
    for step in range(8):
        a = step % env_gf2.num_actions
        _, s_gf2, r1, d1, i1 = env_gf2.step_env(
            torch_random.PRNGKey(step), s_gf2, a, env_gf2.default_params)
        _, s_tab, r2, d2, i2 = env_tab.step_env(
            torch_random.PRNGKey(step), s_tab, a, env_tab.default_params)
        if (abs(float(r1) - float(r2)) > 1e-5 or d1 != d2
                or i1["distance"] != i2["distance"]):
            ok_env = False
            break
    if not check("env: gf2 and gf2_tableau rewards/termination agree", ok_env):
        failures.append("env-modes")

    return failures


# --------------------------------------------------------------- benchmarks
def kernel_benchmark(device):
    print(f"\n===== Exact GF(2) KL kernel benchmark (CPU vs {device}) =====")
    for n, d, reps in [(8, 3, 60), (10, 3, 40), (12, 3, 20), (10, 4, 8)]:
        tableau = random_tableau(n, seed=n + d)
        errors, probs = build_error_operators_upto(n, d, 0.9)
        errors_t = torch.from_numpy(np.ascontiguousarray(errors))
        probs_t = torch.from_numpy(np.ascontiguousarray(probs))
        row = f"n={n:2d} d={d} errors={errors.shape[0]:7d}"
        for label, fn in (
            ("rref", lambda t, e, p: torch_exact_gf2_kl(t[n + 1 :], e, p, 10.0)),
            ("tableau", lambda t, e, p: torch_tableau_kl(t, 1, e, p, 10.0)),
        ):
            times = {}
            for dev in ("cpu", device):
                tab = tableau.to(dev)
                err = errors_t.to(dev)
                prob = probs_t.to(dev)
                for _ in range(3):
                    fn(tab, err, prob)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    fn(tab, err, prob)
                if dev == "cuda":
                    torch.cuda.synchronize()
                times[dev] = (time.perf_counter() - t0) / reps
            row += (f" | {label}: cpu {times['cpu']*1e3:7.2f}ms "
                    f"{device} {times[device]*1e3:7.2f}ms "
                    f"({times['cpu']/times[device]:5.2f}x)")
        print(row)


def gnn_benchmark(device):
    print(f"\n===== V1.6 GNN PPO step benchmark (CPU vs {device}) =====")
    config = {"ACTIVATION": "relu", "CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.02}
    for n, hidden, layers, batch, reps in [
        (7, 64, 3, 128, 20),
        (9, 128, 3, 512, 10),
        (9, 256, 3, 2048, 5),
    ]:
        padding = GraphPadding(n_max=n, stabilizers_max=n - 1,
                               hardware_edges_max=n * (n - 1))
        env = make_env(n, 1, 3, padding)
        net = GNNQDXActorCritic(num_gate_types=3, hidden_dim=hidden,
                                num_gnn_layers=layers, activation="relu")
        params0 = net.init(torch_random.PRNGKey(7), env.graph_observation_template())
        base_obs = build_batch(env, min(batch, 64))
        idx = torch.arange(batch) % min(batch, 64)
        obs_full = obs_take(base_obs, idx)
        action = torch.from_numpy(
            np.random.default_rng(1).integers(0, env.num_actions, batch))
        gae = torch.from_numpy(
            np.random.default_rng(2).standard_normal(batch).astype(np.float32))
        targets = torch.from_numpy(
            np.random.default_rng(3).standard_normal(batch).astype(np.float32))

        times = {}
        for dev in ("cpu", device):
            params = to_device_tree(params0, dev)
            obs = obs_to_device(obs_full, dev)
            action_d = action.to(dev)
            old_value = torch.zeros(batch, device=dev)
            old_log_prob = torch.zeros(batch, device=dev)
            gae_d, targets_d = gae.to(dev), targets.to(dev)
            opt_state = torch_nn.OptimizerState(params)

            def sync():
                if dev == "cuda":
                    torch.cuda.synchronize()

            for _ in range(3):
                (_, _), grads = torch_nn.ppo_loss_and_grad_generic(
                    net.apply, params, obs, action_d, old_value, old_log_prob,
                    gae_d, targets_d, config)
                upd, opt_state = torch_nn.optimizer_update(grads, opt_state, 0.25, 1e-3)
                params = torch_nn.apply_updates(params, upd)
            sync()
            t0 = time.perf_counter()
            for _ in range(reps):
                (_, _), grads = torch_nn.ppo_loss_and_grad_generic(
                    net.apply, params, obs, action_d, old_value, old_log_prob,
                    gae_d, targets_d, config)
                upd, opt_state = torch_nn.optimizer_update(grads, opt_state, 0.25, 1e-3)
                params = torch_nn.apply_updates(params, upd)
            sync()
            times[dev] = (time.perf_counter() - t0) / reps
        print(f"n={n} hidden={hidden:4d} layers={layers} batch={batch:5d}: "
              f"cpu {times['cpu']*1e3:9.2f} ms/step, "
              f"{device} {times[device]*1e3:9.2f} ms/step, "
              f"speedup {times['cpu']/times[device]:5.2f}x")


def end_to_end_timing(device):
    print(f"\n===== V1.6 joint multitask training (CPU vs {device}) =====")
    import main as main_mod
    from qdx.utils import build_graph_padding

    base_config = {
        "MODEL": "GNN", "ENV_TYPE": "STANDARD", "D": 3, "MAX_STEPS": 20,
        "WHICH_GATES": ("cx", "h", "s", "sqrt_x", "cz", "sqrt_xx"),
        "GRAPH": "All-to-All", "SOFTNESS": 1, "KL_METHOD": "gf2_tableau",
        "VALIDATION_SOFTNESS": None, "P_I": 0.9, "LAMBDA": 10, "SEED": 42,
        "LR": 1.0e-3, "NUM_ENVS_PER_TASK": 16, "NUM_STEPS": 20,
        "TOTAL_TIMESTEPS": 16 * 20 * 3 * 5, "UPDATE_EPOCHS": 3,
        "NUM_MINIBATCHES": 4, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2, "ENT_COEF": 0.02, "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25, "ACTIVATION": "relu", "HIDDEN_DIM": 32,
        "ANNEAL_LR": True, "COMPUTE_METRICS": True, "GNN_HIDDEN_DIM": 128,
        "GNN_RELATION_DIM": 8, "GNN_GATE_DIM": 8, "GNN_NUM_LAYERS": 3,
    }
    train_tasks = [
        {"n": 6, "k": 1, "d": 3, "graph": "All-to-All"},
        {"n": 7, "k": 1, "d": 3, "graph": "All-to-All"},
        {"n": 8, "k": 1, "d": 3, "graph": "All-to-All"},
    ]
    padding = build_graph_padding(train_tasks)
    results = {}
    for dev in ("cpu", device):
        config = dict(copy.deepcopy(base_config), DEVICE=dev)
        started = time.perf_counter()
        params, history, layout = main_mod.train_joint_multitask(
            config, total_timesteps=config["TOTAL_TIMESTEPS"],
            train_tasks=copy.deepcopy(train_tasks),
            train_graph_padding=padding, run_started=started)
        elapsed = time.perf_counter() - started
        results[dev] = elapsed
        print(f"{dev:>6s}: {elapsed:8.3f} s total "
              f"(final reward_mean {history[-1]['reward_mean']:.3f}, "
              f"success {history[-1]['success_count']}/{history[-1]['episode_count']})")
    print(f"speedup ({device} vs cpu): {results['cpu'] / results[device]:.2f}x")


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping GPU checks.")
        sys.exit(0)
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, skipping GPU checks.")
        sys.exit(0)
    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)} "
              f"(visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")

    failures = correctness_checks(device)
    kernel_benchmark(device)
    gnn_benchmark(device)
    end_to_end_timing(device)

    print("\n================ SUMMARY ================")
    if failures:
        print(f"FAILED checks: {failures}")
        sys.exit(1)
    print("All correctness checks passed.")
