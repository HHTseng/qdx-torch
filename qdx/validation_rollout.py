"""Shared helpers for greedy validation rollouts (PyTorch port).

The JAX original JIT-compiles a ``lax.scan`` over a masked step; the torch
port runs the same greedy episode as a plain Python loop that stops when the
environment reports ``done``, producing identical rollouts.
"""

from __future__ import annotations

import numpy as np
import torch

from qdx import torch_random


def build_validation_episode_runner(env, network, max_steps):
    """Return a greedy validation episode runner."""

    max_steps = int(max_steps)

    def validate_episode(params, observation, state, rng):
        action_ids = []
        rewards = []
        dones = []
        done = False
        total_reward = np.float32(0.0)
        final_reward = np.float32(0.0)
        final_value = np.float32(0.0)
        steps = 0

        for _ in range(max_steps):
            if done:
                break
            with torch.no_grad():
                policy, value = network.apply(params, observation)
            # greedy: mode of the categorical policy
            action = int(torch.argmax(policy.logits, dim=-1))
            keys = torch_random.split(rng)
            rng, step_rng = keys[0], keys[1]
            observation, state, reward, done, _ = env.step(
                step_rng, state, action, None
            )
            action_ids.append(action)
            rewards.append(np.float32(reward))
            dones.append(bool(done))
            total_reward = np.float32(total_reward + np.float32(reward))
            final_reward = np.float32(reward)
            final_value = np.float32(value.item())
            steps += 1

        return {
            "action_ids": np.asarray(action_ids, dtype=np.int32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "done": bool(done),
            "total_reward": total_reward,
            "final_reward": final_reward,
            "final_value": final_value,
            "steps": np.int32(steps),
        }

    return validate_episode


def summarize_validation_episode(rollout, action_strings):
    """Decode a rollout result into host-side validation metadata."""

    step_count = int(np.asarray(rollout["steps"]))
    action_ids = np.asarray(rollout["action_ids"], dtype=np.int32)[:step_count]
    gates = [action_strings[int(action)] for action in action_ids]
    if step_count:
        final_reward = float(np.asarray(rollout["final_reward"]))
        final_value = float(np.asarray(rollout["final_value"]))
    else:
        final_reward = float("nan")
        final_value = float("nan")
    return {
        "done": bool(np.asarray(rollout["done"])),
        "steps": step_count,
        "total_reward": float(np.asarray(rollout["total_reward"])),
        "final_reward": final_reward,
        "final_value": final_value,
        "gates": gates,
    }
