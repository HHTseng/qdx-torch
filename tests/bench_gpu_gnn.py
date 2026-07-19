"""GPU (CUDA/MPS) correctness and acceleration benchmark for the GNN path.

Like tests/bench_gpu.py this is a torch-vs-torch check (CPU vs accelerator),
not a JAX comparison — the JAX-vs-torch verification lives in
test_gnn_compare.py / test_end_to_end_gnn.py and runs on CPU.

  1. Correctness: GNN forward pass, PPO loss, and autograd gradients must
     agree between CPU and the accelerator within float32 tolerance.
  2. Acceleration: wall-clock timing of the GNN forward+backward+optimizer
     step across batch sizes / model widths, plus a short
     train_joint_multitask timing comparison (device set via config).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/bench_gpu_gnn.py [--device cuda]

Restrict to a single GPU with CUDA_VISIBLE_DEVICES before launching.
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
from qdx.gnn.model import GNNQDXActorCritic
from qdx.gnn.observation import GraphPadding, obs_stack, obs_take, obs_to_device
from qdx.envs.graph_code_discovery import GraphCodeDiscovery
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


def make_env(n, k, d, padding, gate_names=("cx", "h", "s")):
    gates = CliffordGates(n)
    gate_set = [getattr(gates, name) for name in gate_names]
    return GraphCodeDiscovery(
        n, k, d, gate_set, max_steps=20, lbda=10, pI=0.9, softness=1,
        graph_padding=padding,
    )


def build_batch(env, batch, seed=0):
    """Collect a batch of observations by stepping randomly."""
    rng = np.random.default_rng(seed)
    obs, state = env.reset(torch_random.PRNGKey(seed), None)
    observations = []
    while len(observations) < batch:
        observations.append(obs)
        action = int(rng.integers(env.num_actions))
        obs, state, _, done, _ = env.step(
            torch_random.PRNGKey(seed + len(observations)), state, action, None)
    return obs_stack(observations)


def correctness_checks(device):
    print(f"\n===== GNN correctness: CPU vs {device} =====")
    failures = []
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
    d = float(torch.max(torch.abs(pi_c.logits[mask] - pi_d.logits.cpu()[mask])))
    if not check("forward: valid logits CPU vs device", d < 1e-3, f"max abs diff {d:.3g}"):
        failures.append("logits")
    d = float(torch.max(torch.abs(v_c - v_d.cpu())))
    if not check("forward: values CPU vs device", d < 1e-3, f"max abs diff {d:.3g}"):
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

    d = abs(float(loss_c) - float(loss_d))
    if not check("PPO loss CPU vs device", d < 1e-4, f"abs diff {d:.3g}"):
        failures.append("loss")
    d = tree_max_diff(grads_c["params"], grads_d["params"])
    if not check("PPO autograd gradients CPU vs device", d < 1e-3, f"max abs diff {d:.3g}"):
        failures.append("grads")
    return failures


def microbenchmark(device):
    print(f"\n===== GNN microbenchmark: fwd+bwd+opt (CPU vs {device}) =====")
    config = {"ACTIVATION": "relu", "CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.02}

    cases = [
        # (n, hidden, layers, batch, reps)
        (7, 64, 3, 128, 30),
        (9, 128, 3, 512, 15),
        (9, 256, 3, 2048, 8),
    ]
    for n, hidden, layers, batch, reps in cases:
        padding = GraphPadding(n_max=n, stabilizers_max=n - 1,
                               hardware_edges_max=n * (n - 1))
        env = make_env(n, 1, 3, padding)
        net = GNNQDXActorCritic(num_gate_types=3, hidden_dim=hidden,
                                num_gnn_layers=layers, activation="relu")
        params0 = net.init(torch_random.PRNGKey(7), env.graph_observation_template())
        # collect 64 distinct observations, then index-repeat up to `batch`
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
            gae_d = gae.to(dev)
            targets_d = targets.to(dev)
            opt_state = torch_nn.OptimizerState(params)

            def sync():
                if dev == "cuda":
                    torch.cuda.synchronize()
                elif dev == "mps":
                    torch.mps.synchronize()

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

        speedup = times["cpu"] / times[device]
        print(f"n={n} hidden={hidden:4d} layers={layers} batch={batch:5d}: "
              f"cpu {times['cpu']*1e3:9.3f} ms/step, "
              f"{device} {times[device]*1e3:9.3f} ms/step, "
              f"speedup {speedup:5.2f}x")


def end_to_end_timing(device):
    print(f"\n===== GNN train_joint_multitask timing (CPU vs {device}) =====")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import main as main_mod
    from qdx.utils import build_graph_padding

    base_config = {
        "MODEL": "GNN",
        "ENV_TYPE": "STANDARD",
        "D": 3,
        "MAX_STEPS": 20,
        "WHICH_GATES": ("cx", "h"),
        "GRAPH": "All-to-All",
        "SOFTNESS": 1,
        "VALIDATION_SOFTNESS": None,
        "P_I": 0.9,
        "LAMBDA": 10,
        "SEED": 42,
        "LR": 1.0e-3,
        "NUM_ENVS_PER_TASK": 16,
        "NUM_STEPS": 20,
        "TOTAL_TIMESTEPS": 16 * 20 * 3 * 5,   # 5 joint updates over 3 tasks
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
        "COMPUTE_METRICS": True,
        "GNN_HIDDEN_DIM": 128,
        "GNN_RELATION_DIM": 8,
        "GNN_GATE_DIM": 8,
        "GNN_NUM_LAYERS": 3,
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
            config,
            total_timesteps=config["TOTAL_TIMESTEPS"],
            train_tasks=copy.deepcopy(train_tasks),
            train_graph_padding=padding,
            run_started=started,
        )
        elapsed = time.perf_counter() - started
        results[dev] = elapsed
        print(f"{dev:>6s}: {elapsed:8.3f} s total "
              f"(final reward_mean {history[-1]['reward_mean']:.3f})")

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
    microbenchmark(device)
    end_to_end_timing(device)

    print("\n================ SUMMARY ================")
    if failures:
        print(f"FAILED checks: {failures}")
        sys.exit(1)
    print("All correctness checks passed.")
