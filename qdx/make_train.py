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


"""
This code is a faithful PyTorch port of the JAX/PureJaxRL implementation
(https://github.com/luchris429/purejaxrl). The `jax.lax.scan`/`jax.vmap`
constructs are replaced by explicit Python loops, `flax`/`distrax` by the
torch modules in qdx.torch_nn, `jax.value_and_grad` by torch autograd,
`optax` by the equivalent optimizer in qdx.torch_nn, and `jax.random` by
the bit-exact threefry implementation in qdx.torch_random.

Every tensor (observations, trajectories, advantages, parameters,
gradients) is an eagerly-evaluated torch.Tensor that can be inspected in a
debugger at any point. Set config["DEVICE"] = "mps"/"cuda" to run the
network and PPO update on an accelerator (default: "cpu", which matches
the JAX CPU results numerically).
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
    obs: torch.Tensor
    info: Any


def make_train(config, env, network_params_init = None, env_params = None):
    config["NUM_EPOCHS"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

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

    def train(rng, network_params_init=None):

        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"], hidden_dim = config["HIDDEN_DIM"])
        keys = torch_random.split(rng)
        rng, _rng = keys[0], keys[1]
        init_x = torch.zeros(env.observation_space(env_params).shape, dtype=torch.float32)

        if network_params_init is None:
            network_params = network.init(_rng, init_x)

        else:
            network_params = network_params_init # typically, network_params_init = train_state.params
            print("IN")

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
        obsv = torch.stack(obsv_list).to(device)

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
            traj_obs = torch.zeros((num_steps,) + tuple(obsv.shape), dtype=obsv.dtype, device=device)
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
                traj_obs[t] = last_obs
                for k in info_b[0].keys():
                    if k not in traj_info:
                        traj_info[k] = np.zeros(
                            (num_steps, num_envs),
                            dtype=np.asarray(info_b[0][k]).dtype)
                    traj_info[k][t] = np.array([inf[k] for inf in info_b])

                env_state = state_next
                last_obs = torch.stack(obs_next).to(device)

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
                    (total_loss, aux), grads = torch_nn.ppo_loss_and_grad(
                        train_params,
                        flat_obs[idx],
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
            # Average over environments, reshape and return samples
            metric["returned_episode_returns"] = np.mean(metric["returned_episode_returns"], axis=-1).reshape(-1)[config["MAX_STEPS"]::config["MAX_STEPS"]]

            metric["returned_episode_lengths"] = np.mean(metric["returned_episode_lengths"], axis=-1).reshape(-1)[config["MAX_STEPS"]::config["MAX_STEPS"]]

            return {"params": train_params, "metrics": metric}
        else:
            return {"params": train_params, "metrics": None}

    return train
