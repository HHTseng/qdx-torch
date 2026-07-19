import numpy as np
import torch
import time
import random
import os
import itertools
import json
from typing import Tuple, Optional, Sequence, NamedTuple, Any

from qdx import torch_random
from qdx import torch_nn
from qdx.torch_nn import Categorical, apply_actor_critic
from qdx.torch_env_base import FlattenObservationWrapper, LogWrapper
from qdx.gnn.model import GNNQDXActorCritic
from qdx.gnn.observation import (
    GraphObservation,
    obs_index,
    obs_map,
    obs_reshape_lead,
    obs_stack,
    obs_take,
    obs_to_device,
)


"""
This code is a faithful PyTorch port of the JAX/PureJaxRL implementation
(https://github.com/luchris429/purejaxrl), extended with the GNN-QDX policy
factory. The `jax.lax.scan`/`jax.vmap` constructs are replaced by explicit
Python loops, `flax`/`distrax` by the torch modules in qdx.torch_nn and
qdx.gnn, `jax.value_and_grad` by torch autograd, `optax` by the equivalent
optimizer in qdx.torch_nn, and `jax.random` by the bit-exact threefry
implementation in qdx.torch_random.
"""


class ActorCritic:
    """PyTorch stand-in for the flax ActorCritic module.

    Provides ``init`` and ``apply`` with the same call signatures used by the
    rest of the code base. Parameters live in a flax-style nested dict of
    torch tensors.
    """

    def __init__(self, action_dim, activation="tanh", hidden_dim=16):
        self.action_dim = action_dim
        self.activation = activation
        self.hidden_dim = hidden_dim

    def init(self, rng, init_x):
        obs_dim = int(np.shape(init_x)[-1])
        return torch_nn.init_actor_critic_params(
            rng, obs_dim, self.action_dim, self.hidden_dim
        )

    def apply(self, params, x):
        return apply_actor_critic(params, x, self.activation)


class Transition(NamedTuple):
    done: torch.Tensor
    action: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    log_prob: torch.Tensor
    obs: Any
    info: Any


def make_actor_critic(config, env):
    """Construct the selected policy while keeping one PPO implementation."""

    if config.get("MODEL", "MLP").upper() == "GNN":
        if not hasattr(env, "graph_builder"):
            raise ValueError("MODEL='GNN' requires GraphCodeDiscovery")
        return GNNQDXActorCritic(
            num_gate_types=env.graph_builder.num_gate_types,
            hidden_dim=config.get("GNN_HIDDEN_DIM", config["HIDDEN_DIM"]),
            relation_dim=config.get("GNN_RELATION_DIM", 8),
            gate_dim=config.get("GNN_GATE_DIM", 8),
            num_gnn_layers=config.get("GNN_NUM_LAYERS", 3),
            activation=config["ACTIVATION"],
        )
    return ActorCritic(
        env.action_space().n,
        activation=config["ACTIVATION"],
        hidden_dim=config["HIDDEN_DIM"],
    )


def make_train(config, env, network_params_init=None, env_params=None):
    config["NUM_EPOCHS"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    use_gnn = config.get("MODEL", "MLP").upper() == "GNN"
    base_env = env
    if use_gnn:
        init_x = base_env.graph_observation_template()
    else:
        env = FlattenObservationWrapper(env)
        init_x = torch.zeros(env.observation_space(env_params).shape, dtype=torch.float32)
    env = LogWrapper(env)
    network = make_actor_critic(config, base_env)

    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_epochs = int(config["NUM_EPOCHS"])
    num_minibatches = int(config["NUM_MINIBATCHES"])
    update_epochs = int(config["UPDATE_EPOCHS"])
    device = torch.device(config.get("DEVICE", "cpu"))

    def linear_schedule(count):
        # float32 arithmetic to match the jitted original
        frac = np.float32(1.0) - np.float32(
            count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        ) / np.float32(config["NUM_EPOCHS"])

        return np.float32(config["LR"]) * frac

    def _stack_obs(obs_list):
        if use_gnn:
            return obs_to_device(obs_stack(obs_list), device)
        return torch.stack(obs_list).to(device)

    def train(rng, network_params_init=None):

        # INIT NETWORK
        keys = torch_random.split(rng)
        rng, _rng = keys[0], keys[1]

        if network_params_init is None:
            network_params = network.init(_rng, init_x)

        else:
            network_params = network_params_init

        # Move parameters to the requested device
        network_params = torch_nn._tree_map(
            lambda p: p.detach().to(device).requires_grad_(True), network_params)

        if config["ANNEAL_LR"]:
            learning_rate = linear_schedule
        else:
            learning_rate = config["LR"]

        train_params = network_params
        opt_state = torch_nn.OptimizerState(train_params)

        # INIT ENV
        keys = torch_random.split(rng)
        rng, _rng = keys[0], keys[1]
        reset_rng = torch_random.split(_rng, config["NUM_ENVS"])
        obsv_list = []
        env_state = []
        for i in range(num_envs):
            o, s = env.reset(reset_rng[i], env_params)
            obsv_list.append(o)
            env_state.append(s)
        obsv = _stack_obs(obsv_list)

        all_metrics = [] if config["COMPUTE_METRICS"] else None

        # TRAIN LOOP
        keys = torch_random.split(rng)
        rng, _rng = keys[0], keys[1]
        runner_rng = _rng
        last_obs = obsv

        for _update in range(num_epochs):
            # COLLECT TRAJECTORIES
            traj_done = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
            traj_action = torch.zeros((num_steps, num_envs), dtype=torch.int32, device=device)
            traj_value = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
            traj_reward = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
            traj_log_prob = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
            traj_obs_steps = []
            traj_info = {}

            for t in range(num_steps):
                # SELECT ACTION
                keys = torch_random.split(runner_rng)
                runner_rng, _rng = keys[0], keys[1]
                with torch.no_grad():
                    pi, value = network.apply(train_params, last_obs)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                # STEP ENV
                keys = torch_random.split(runner_rng)
                runner_rng, _rng = keys[0], keys[1]
                rng_step = torch_random.split(_rng, config["NUM_ENVS"])
                obs_next = []
                state_next = []
                reward_b = np.zeros(num_envs, dtype=np.float32)
                done_b = np.zeros(num_envs, dtype=bool)
                info_b = []
                for i in range(num_envs):
                    o, s, r, d, info = env.step(
                        rng_step[i], env_state[i], int(action[i]), env_params
                    )
                    obs_next.append(o)
                    state_next.append(s)
                    reward_b[i] = r
                    done_b[i] = d
                    info_b.append(info)

                traj_done[t] = torch.from_numpy(done_b).to(device)
                traj_action[t] = action
                traj_value[t] = value
                traj_reward[t] = torch.from_numpy(reward_b).to(device)
                traj_log_prob[t] = log_prob
                traj_obs_steps.append(last_obs)
                for key in info_b[0].keys():
                    if key not in traj_info:
                        traj_info[key] = np.zeros(
                            (num_steps, num_envs),
                            dtype=np.asarray(info_b[0][key]).dtype)
                    traj_info[key][t] = np.array([inf[key] for inf in info_b])

                env_state = state_next
                last_obs = _stack_obs(obs_next)

            if use_gnn:
                traj_obs = obs_map(lambda *leaves: torch.stack(leaves), *traj_obs_steps)
            else:
                traj_obs = torch.stack(traj_obs_steps)

            # CALCULATE ADVANTAGE
            with torch.no_grad():
                _, last_val = network.apply(train_params, last_obs)

            def _calculate_gae(last_val):
                advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
                gae = torch.zeros_like(last_val)
                next_value = last_val
                for t in reversed(range(num_steps)):
                    done, value, reward = (
                        traj_done[t],
                        traj_value[t],
                        traj_reward[t],
                    )
                    not_done = 1.0 - done.to(torch.float32)
                    delta = reward + float(np.float32(config["GAMMA"])) * next_value * not_done - value
                    gae = (
                        delta
                        + float(np.float32(config["GAMMA"])) * float(np.float32(config["GAE_LAMBDA"])) * not_done * gae
                    )
                    advantages[t] = gae
                    next_value = value
                return advantages, advantages + traj_value

            with torch.no_grad():
                advantages, targets = _calculate_gae(last_val)

            # UPDATE NETWORK
            batch_size = int(config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"])
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"

            if use_gnn:
                flat_obs = obs_reshape_lead(traj_obs, (batch_size,))
            else:
                flat_obs = traj_obs.reshape((batch_size,) + tuple(traj_obs.shape[2:]))
            flat_action = traj_action.reshape(batch_size)
            flat_value = traj_value.reshape(batch_size)
            flat_log_prob = traj_log_prob.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)
            flat_targets = targets.reshape(batch_size)

            for _epoch in range(update_epochs):
                keys = torch_random.split(runner_rng)
                runner_rng, _rng = keys[0], keys[1]
                permutation = torch.from_numpy(
                    np.asarray(torch_random.permutation(_rng, batch_size), dtype=np.int64)).to(device)

                mb = config["MINIBATCH_SIZE"]
                for m in range(num_minibatches):
                    idx = permutation[m * mb:(m + 1) * mb]
                    if use_gnn:
                        obs_mb = obs_take(flat_obs, idx)
                    else:
                        obs_mb = flat_obs[idx]
                    (total_loss, aux), grads = torch_nn.ppo_loss_and_grad_generic(
                        network.apply,
                        train_params,
                        obs_mb,
                        flat_action[idx],
                        flat_value[idx],
                        flat_log_prob[idx],
                        flat_advantages[idx],
                        flat_targets[idx],
                        config,
                    )
                    updates, opt_state = torch_nn.optimizer_update(
                        grads, opt_state,
                        config["MAX_GRAD_NORM"], learning_rate,
                    )
                    train_params = torch_nn.apply_updates(train_params, updates)

            if config["COMPUTE_METRICS"]:
                all_metrics.append(traj_info)

        if config["COMPUTE_METRICS"]:
            # metric has shape (num_updates, num_steps, num_envs)
            metric = {
                k: np.stack([m[k] for m in all_metrics])
                for k in all_metrics[0].keys()
            }
            done = metric["returned_episode"].astype(bool)
            episode_lengths = metric["returned_episode_lengths"]
            success = done & (episode_lengths < config["MAX_STEPS"])
            episode_count = np.sum(done, axis=(1, 2))
            success_count = np.sum(success, axis=(1, 2))
            timeout_count = episode_count - success_count
            with np.errstate(invalid="ignore"):
                success_rate = np.where(
                    episode_count > 0,
                    success_count / np.maximum(episode_count, 1),
                    np.nan,
                )

            metric["episode_count"] = episode_count
            metric["success_count"] = success_count
            metric["timeout_count"] = timeout_count
            metric["success_rate"] = success_rate

            # Average over environments, reshape and return sampled tails.
            metric["returned_episode_returns"] = np.mean(
                metric["returned_episode_returns"], axis=-1
            ).reshape(-1)[config["MAX_STEPS"]::config["MAX_STEPS"]]
            metric["returned_episode_lengths"] = np.mean(
                metric["returned_episode_lengths"], axis=-1
            ).reshape(-1)[config["MAX_STEPS"]::config["MAX_STEPS"]]

            return {"params": train_params, "metrics": metric}
        else:
            return {"params": train_params, "metrics": None}

    return train
