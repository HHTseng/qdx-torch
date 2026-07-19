"""GPU (CUDA/MPS) conversion-correctness and acceleration benchmark.

This is NOT a JAX comparison (see test_network.py / test_end_to_end.py for
that, which run entirely on CPU for bit-reproducibility). This script only
exercises the torch port against itself on two devices:

  1. Correctness: CPU and GPU must produce numerically close results for
     parameter init, forward pass, PPO loss + autograd gradients, and a
     short end-to-end training run (float32 tolerance, since GPU cuBLAS
     rounds slightly differently than CPU BLAS -- same caveat as any
     cross-backend float32 comparison).
  2. Acceleration: wall-clock timing of the network forward+backward+
     optimizer step (where DEVICE actually matters -- the environments
     themselves always run on CPU, since their tableau updates are tiny
     integer ops) across a range of batch/hidden sizes, plus a full
     CodeFinder.train() timing comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/bench_gpu.py [--device cuda]

Restrict to a single GPU with CUDA_VISIBLE_DEVICES before launching --
this repo intentionally never requests more than one device itself.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdx import torch_nn, torch_random
from qdx.make_train import ActorCritic, make_train
from qdx.envs.code_discovery import CodeDiscovery
from qdx.simulators.clifford_gates import CliffordGates
from qdx.code_finder import CodeFinder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda",
                   help="Accelerator device to compare against CPU (cuda, mps, ...)")
    return p.parse_args()


def tree_max_diff(a, b):
    if isinstance(a, dict):
        return max(tree_max_diff(a[k], b[k]) for k in a)
    return float(torch.max(torch.abs(a.detach().cpu().float() - b.detach().cpu().float())))


def to_device(tree, device):
    return torch_nn._tree_map(lambda t: t.detach().to(device).requires_grad_(True), tree)


def section(title):
    print(f"\n===== {title} =====")


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


# --------------------------------------------------------------------------
# 1. Correctness: init / forward / PPO loss+grad on CPU vs accelerator
# --------------------------------------------------------------------------

def correctness_checks(device):
    section(f"Correctness: CPU vs {device}")
    failures = []

    OBS_DIM, ACTION_DIM, HIDDEN, B = 98, 60, 64, 256
    net = ActorCritic(ACTION_DIM, activation="relu", hidden_dim=HIDDEN)

    rng = torch_random.PRNGKey(123)
    params_cpu = net.init(rng, torch.zeros(OBS_DIM))
    params_dev = to_device(params_cpu, device)

    d = tree_max_diff(params_cpu["params"], params_dev["params"])
    if not check("init params identical after device move", d < 1e-9, f"max abs diff {d:.3g}"):
        failures.append("init")

    obs_np = (np.random.default_rng(0).random((B, OBS_DIM)) > 0.5).astype(np.float32)
    obs_cpu = torch.from_numpy(obs_np)
    obs_dev = obs_cpu.to(device)

    pi_cpu, v_cpu = net.apply(params_cpu, obs_cpu)
    pi_dev, v_dev = net.apply(params_dev, obs_dev)

    d = float(torch.max(torch.abs(v_cpu - v_dev.cpu())))
    if not check("forward: value CPU vs device", d < 1e-4, f"max abs diff {d:.3g}"):
        failures.append("forward-value")
    d = float(torch.max(torch.abs(pi_cpu.logits - pi_dev.logits.cpu())))
    if not check("forward: logits CPU vs device", d < 1e-4, f"max abs diff {d:.3g}"):
        failures.append("forward-logits")

    action = torch.from_numpy(np.arange(B, dtype=np.int64) % ACTION_DIM)
    old_value = v_cpu.detach() + 0.01
    old_log_prob = pi_cpu.log_prob(action).detach()
    gae = torch.from_numpy(np.random.default_rng(1).standard_normal(B).astype(np.float32)) * 3.0
    targets = torch.from_numpy(np.random.default_rng(2).standard_normal(B).astype(np.float32)) * 2.0
    config = {"ACTIVATION": "relu", "CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.02}

    (loss_cpu, aux_cpu), grads_cpu = torch_nn.ppo_loss_and_grad(
        params_cpu, obs_cpu, action, old_value, old_log_prob, gae, targets, config)
    (loss_dev, aux_dev), grads_dev = torch_nn.ppo_loss_and_grad(
        params_dev, obs_dev, action.to(device), old_value.to(device),
        old_log_prob.to(device), gae.to(device), targets.to(device), config)

    d = abs(float(loss_cpu) - float(loss_dev))
    if not check("PPO loss CPU vs device", d < 1e-4, f"abs diff {d:.3g}"):
        failures.append("ppo-loss")
    d = tree_max_diff(grads_cpu["params"], grads_dev["params"])
    if not check("PPO autograd gradients CPU vs device", d < 1e-3, f"max abs diff {d:.3g}"):
        failures.append("ppo-grad")

    # a few optimizer steps
    opt_cpu = torch_nn.OptimizerState(params_cpu)
    opt_dev = torch_nn.OptimizerState(params_dev)
    p_cpu, p_dev = params_cpu, params_dev
    for step in range(5):
        upd_cpu, opt_cpu = torch_nn.optimizer_update(grads_cpu, opt_cpu, 0.25, 1e-3)
        p_cpu = torch_nn.apply_updates(p_cpu, upd_cpu)
        upd_dev, opt_dev = torch_nn.optimizer_update(grads_dev, opt_dev, 0.25, 1e-3)
        p_dev = torch_nn.apply_updates(p_dev, upd_dev)
    d = tree_max_diff(p_cpu["params"], p_dev["params"])
    if not check("optimizer chain(clip, adam) after 5 steps", d < 1e-3, f"max abs diff {d:.3g}"):
        failures.append("optimizer")

    return failures


# --------------------------------------------------------------------------
# 2a. Microbenchmark: forward+backward+optimizer step only
# --------------------------------------------------------------------------

def microbenchmark(device):
    section(f"Acceleration microbenchmark: network fwd+bwd+opt (CPU vs {device})")
    config = {"ACTIVATION": "relu", "CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.02}

    for hidden, batch, reps in [(32, 512, 200), (256, 4096, 100), (512, 16384, 50)]:
        net = ActorCritic(60, activation="relu", hidden_dim=hidden)
        rng = torch_random.PRNGKey(7)
        params0 = net.init(rng, torch.zeros(98))
        obs_np = (np.random.default_rng(3).random((batch, 98)) > 0.5).astype(np.float32)
        action_np = np.random.default_rng(4).integers(0, 60, size=batch)
        gae_np = np.random.default_rng(5).standard_normal(batch).astype(np.float32)
        targets_np = np.random.default_rng(6).standard_normal(batch).astype(np.float32)

        times = {}
        for tag, dev in [("cpu", "cpu"), (device, device)]:
            params = to_device(params0, dev)
            obs = torch.from_numpy(obs_np).to(dev)
            action = torch.from_numpy(action_np).to(dev)
            old_value = torch.zeros(batch, device=dev)
            old_log_prob = torch.zeros(batch, device=dev)
            gae = torch.from_numpy(gae_np).to(dev)
            targets = torch.from_numpy(targets_np).to(dev)
            opt_state = torch_nn.OptimizerState(params)

            def sync():
                if dev == "cuda":
                    torch.cuda.synchronize()
                elif dev == "mps":
                    torch.mps.synchronize()

            # warmup
            for _ in range(5):
                (loss, aux), grads = torch_nn.ppo_loss_and_grad(
                    params, obs, action, old_value, old_log_prob, gae, targets, config)
                upd, opt_state = torch_nn.optimizer_update(grads, opt_state, 0.25, 1e-3)
                params = torch_nn.apply_updates(params, upd)
            sync()

            t0 = time.perf_counter()
            for _ in range(reps):
                (loss, aux), grads = torch_nn.ppo_loss_and_grad(
                    params, obs, action, old_value, old_log_prob, gae, targets, config)
                upd, opt_state = torch_nn.optimizer_update(grads, opt_state, 0.25, 1e-3)
                params = torch_nn.apply_updates(params, upd)
            sync()
            t1 = time.perf_counter()
            times[tag] = (t1 - t0) / reps

        speedup = times["cpu"] / times[device]
        print(f"hidden_dim={hidden:4d} batch={batch:5d}: "
              f"cpu {times['cpu']*1e3:8.3f} ms/step, "
              f"{device} {times[device]*1e3:8.3f} ms/step, "
              f"speedup {speedup:5.2f}x")


# --------------------------------------------------------------------------
# 2b. End-to-end CodeFinder.train() timing, CPU vs GPU
# --------------------------------------------------------------------------

def end_to_end_timing(device):
    section(f"Acceleration: full CodeFinder.train() (CPU vs {device})")

    base_config = {
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
        "TOTAL_TIMESTEPS": 16 * 20 * 15,   # 15 epochs
        "UPDATE_EPOCHS": 3,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.02,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",
        "HIDDEN_DIM": 256,   # bigger than the paper default, to exercise the GPU
        "ANNEAL_LR": True,
        "NUM_AGENTS": 1,
        "COMPUTE_METRICS": True,
    }

    results = {}
    for tag, dev in [("cpu", "cpu"), (device, device)]:
        config = dict(base_config, DEVICE=dev)
        finder = CodeFinder(config)
        t0 = time.perf_counter()
        params, metrics = finder.train()
        t1 = time.perf_counter()
        results[tag] = (t1 - t0, metrics)
        print(f"{tag:>6s}: {t1 - t0:8.3f} s total "
              f"(final return {float(np.mean(metrics['returned_episode_returns'][0][-3:])):.3f})")

    speedup = results["cpu"][0] / results[device][0]
    print(f"speedup ({device} vs cpu): {speedup:.2f}x")


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
