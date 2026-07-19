"""Train and validate one shared GNN-QDX policy on custom (N, K, D) tasks.

PyTorch port of the JAX multitask trainer: same YAML configuration, task
expansion, rollout collection, joint PPO update, artifact layout, and RNG
streams (bit-exact threefry). ``jax.lax.scan``/``jax.jit`` are replaced by
explicit loops on eagerly evaluated torch tensors; ``optax`` by the
equivalent optimizer in ``qdx.torch_nn``; ``flax.serialization`` by a
NumPy ``.npz`` checkpoint (``params.npz``).

Set ``device: cuda`` (or ``mps``) in the YAML to run the network and PPO
update on an accelerator; environments always step on CPU.

Run:
    python main.py
    python main.py --config configs/main.yaml
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from qdx import torch_nn, torch_random
from qdx.gnn.observation import (
    GraphObservation,
    obs_concat,
    obs_map,
    obs_reshape_lead,
    obs_stack,
    obs_take,
    obs_to_device,
)
from qdx.make_train import make_actor_critic
from qdx.torch_env_base import LogWrapper
from qdx.utils import (
    DEFAULT_CONFIG_PATH,
    build_graph_padding,
    format_task,
    graph_padding_to_dict,
    load_run_settings,
    make_task_env,
    params_to_numpy_tree,
    save_params,
)
from validation import run_validation


class Transition(NamedTuple):
    done: torch.Tensor
    action: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    log_prob: torch.Tensor
    obs: Any
    info: Any


class PPOBatch(NamedTuple):
    obs: Any
    action: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor
    advantages: torch.Tensor
    targets: torch.Tensor


def format_metric(value):
    if value is None:
        return "nan"
    return f"{value:.2f}"


def format_duration(seconds):
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def completed_episode_values(info, name):
    if info is None or name not in info:
        return None
    values = np.asarray(info[name]).reshape(-1)
    returned_episode = info.get("returned_episode")
    if returned_episode is not None:
        mask = np.asarray(returned_episode).astype(bool).reshape(-1)
        values = values[mask]
    return values


def completed_episode_mean(info, name):
    values = completed_episode_values(info, name)
    if values is None or values.size == 0:
        return None
    return float(np.mean(values))


def rollout_success_stats(traj_batch, max_steps):
    done = np.asarray(traj_batch.done).astype(bool).reshape(-1)
    episode_count = int(np.sum(done))
    episode_lengths = completed_episode_values(
        traj_batch.info, "returned_episode_lengths"
    )
    if episode_lengths is None:
        success_count = 0
    else:
        episode_count = int(episode_lengths.size)
        success_count = int(np.sum(episode_lengths < max_steps))
    timeout_count = episode_count - success_count
    success_rate = success_count / episode_count if episode_count else None
    return {
        "episode_count": episode_count,
        "success_count": success_count,
        "timeout_count": timeout_count,
        "success_rate": success_rate,
    }


class TrainState:
    """Torch analogue of flax's TrainState with the optax-equivalent chain."""

    def __init__(self, config, params, num_updates):
        def linear_schedule(count):
            updates_completed = count // (
                config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
            )
            frac = np.float32(1.0) - np.float32(updates_completed) / np.float32(num_updates)
            return np.float32(config["LR"]) * frac

        self.params = params
        self.opt_state = torch_nn.OptimizerState(params)
        self.max_grad_norm = config["MAX_GRAD_NORM"]
        self.learning_rate = linear_schedule if config["ANNEAL_LR"] else config["LR"]

    def apply_gradients(self, grads):
        updates, self.opt_state = torch_nn.optimizer_update(
            grads, self.opt_state, self.max_grad_norm, self.learning_rate
        )
        self.params = torch_nn.apply_updates(self.params, updates)


def build_rollout_collector(config, env, network, device):
    num_envs = config["NUM_ENVS_PER_TASK"]

    def _calculate_gae(traj_batch, last_val):
        num_steps = traj_batch.reward.shape[0]
        advantages = torch.zeros_like(traj_batch.reward)
        gae = torch.zeros_like(last_val)
        next_value = last_val
        gamma = float(np.float32(config["GAMMA"]))
        lam = float(np.float32(config["GAE_LAMBDA"]))
        for t in reversed(range(num_steps)):
            not_done = 1.0 - traj_batch.done[t].to(torch.float32)
            delta = (
                traj_batch.reward[t]
                + gamma * next_value * not_done
                - traj_batch.value[t]
            )
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae
            next_value = traj_batch.value[t]
        return advantages, advantages + traj_batch.value

    def collect_rollout(params, env_state, last_obs, rng):
        num_steps = int(config["NUM_STEPS"])
        traj_done = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        traj_action = torch.zeros((num_steps, num_envs), dtype=torch.int32, device=device)
        traj_value = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        traj_reward = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        traj_log_prob = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        traj_obs_steps = []
        traj_info = {}

        for t in range(num_steps):
            keys = torch_random.split(rng)
            rng, policy_rng = keys[0], keys[1]
            with torch.no_grad():
                pi, value = network.apply(params, last_obs)
                action = pi.sample(seed=policy_rng)
                log_prob = pi.log_prob(action)

            keys = torch_random.split(rng)
            rng, step_rng = keys[0], keys[1]
            rng_step = torch_random.split(step_rng, num_envs)
            obs_next = []
            state_next = []
            reward_b = np.zeros(num_envs, dtype=np.float32)
            done_b = np.zeros(num_envs, dtype=bool)
            info_b = []
            for i in range(num_envs):
                o, s, r, d, info = env.step(
                    rng_step[i], env_state[i], int(action[i]), None
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
            last_obs = obs_to_device(obs_stack(obs_next), device)

        traj_obs = obs_map(lambda *leaves: torch.stack(leaves), *traj_obs_steps)
        traj_batch = Transition(
            done=traj_done,
            action=traj_action,
            value=traj_value,
            reward=traj_reward,
            log_prob=traj_log_prob,
            obs=traj_obs,
            info=traj_info,
        )
        with torch.no_grad():
            _, last_val = network.apply(params, last_obs)
            advantages, targets = _calculate_gae(traj_batch, last_val)
        return (env_state, last_obs, rng), (traj_batch, advantages, targets)

    return collect_rollout


def build_joint_update_fn(config, network, batch_size):
    def _update_minibatch(train_state, batch):
        (total_loss, aux), grads = torch_nn.ppo_loss_and_grad_generic(
            network.apply,
            train_state.params,
            batch.obs,
            batch.action,
            batch.value,
            batch.log_prob,
            batch.advantages,
            batch.targets,
            config,
        )
        train_state.apply_gradients(grads)
        value_loss, actor_loss, entropy = aux
        return {
            "total_loss": float(total_loss),
            "value_loss": float(value_loss),
            "actor_loss": float(actor_loss),
            "entropy": float(entropy),
        }

    def update(train_state, batch, rng):
        flat_batch = PPOBatch(
            obs=obs_reshape_lead(batch.obs, (batch_size,)),
            action=batch.action.reshape(batch_size),
            value=batch.value.reshape(batch_size),
            log_prob=batch.log_prob.reshape(batch_size),
            advantages=batch.advantages.reshape(batch_size),
            targets=batch.targets.reshape(batch_size),
        )

        metrics = {"total_loss": [], "value_loss": [], "actor_loss": [], "entropy": []}
        for _epoch in range(int(config["UPDATE_EPOCHS"])):
            keys = torch_random.split(rng)
            rng, permutation_rng = keys[0], keys[1]
            permutation = torch.from_numpy(
                np.asarray(
                    torch_random.permutation(permutation_rng, batch_size),
                    dtype=np.int64,
                )
            )
            minibatch_size = batch_size // int(config["NUM_MINIBATCHES"])
            for m in range(int(config["NUM_MINIBATCHES"])):
                idx = permutation[m * minibatch_size:(m + 1) * minibatch_size]
                minibatch = PPOBatch(
                    obs=obs_take(flat_batch.obs, idx),
                    action=flat_batch.action[idx],
                    value=flat_batch.value[idx],
                    log_prob=flat_batch.log_prob[idx],
                    advantages=flat_batch.advantages[idx],
                    targets=flat_batch.targets[idx],
                )
                minibatch_metrics = _update_minibatch(train_state, minibatch)
                for name, value in minibatch_metrics.items():
                    metrics[name].append(value)
        return train_state, metrics, rng

    return update


def initialize_task_state(env, num_envs_per_task, rng, device):
    keys = torch_random.split(rng)
    rng, reset_rng = keys[0], keys[1]
    reset_rng = torch_random.split(reset_rng, num_envs_per_task)
    obs_list = []
    env_state = []
    for i in range(num_envs_per_task):
        o, s = env.reset(reset_rng[i], None)
        obs_list.append(o)
        env_state.append(s)
    return {
        "env_state": env_state,
        "last_obs": obs_to_device(obs_stack(obs_list), device),
        "rng": rng,
    }


def build_task_context(task, config, graph_padding, network, rng, device):
    env = LogWrapper(make_task_env(task, config, graph_padding))
    task_state = initialize_task_state(env, config["NUM_ENVS_PER_TASK"], rng, device)
    return {
        "task": task,
        "env": env,
        "collector": build_rollout_collector(config, env, network, device),
        **task_state,
    }


def merge_task_batches(task_batches):
    merged_obs = obs_concat([batch.obs for batch in task_batches], dim=1)
    return PPOBatch(
        obs=merged_obs,
        action=torch.cat([b.action for b in task_batches], dim=1),
        value=torch.cat([b.value for b in task_batches], dim=1),
        log_prob=torch.cat([b.log_prob for b in task_batches], dim=1),
        advantages=torch.cat([b.advantages for b in task_batches], dim=1),
        targets=torch.cat([b.targets for b in task_batches], dim=1),
    )


def summarize_loss_metrics(loss_metrics):
    return {
        name: float(np.mean(np.asarray(values)))
        for name, values in loss_metrics.items()
    }


def summarize_task_rollout(task, traj_batch, max_steps):
    info = traj_batch.info
    reward_mean = float(np.mean(np.asarray(traj_batch.reward)))
    done_rate = float(np.mean(np.asarray(traj_batch.done, dtype=np.float32)))
    stats = rollout_success_stats(traj_batch, max_steps=max_steps)
    return {
        "graph": task["graph"],
        "n": task["n"],
        "k": task["k"],
        "d": task["d"],
        "target_distance": task["d"],
        "reward_mean": reward_mean,
        "done_rate": done_rate,
        **stats,
        "episode_return_mean": completed_episode_mean(
            info, "returned_episode_returns"
        ),
        "episode_length_mean": completed_episode_mean(
            info, "returned_episode_lengths"
        ),
    }


def compute_training_layout(base_config, total_timesteps, train_tasks):
    if not train_tasks:
        raise ValueError("at least one training task is required")

    rollout_per_task = (
        base_config["NUM_ENVS_PER_TASK"] * base_config["NUM_STEPS"]
    )
    rollout_per_update = rollout_per_task * len(train_tasks)
    if total_timesteps < rollout_per_update:
        raise ValueError(
            "TOTAL_TIMESTEPS/total_timesteps in the YAML config must cover at "
            f"least one full joint PPO update ({rollout_per_update:,} timesteps)."
        )
    num_updates = total_timesteps // rollout_per_update
    actual_total_timesteps = num_updates * rollout_per_update
    if rollout_per_update % base_config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "NUM_MINIBATCHES must divide the joint rollout size per update "
            f"({rollout_per_update})."
        )
    return {
        "num_updates": int(num_updates),
        "rollout_per_task": int(rollout_per_task),
        "rollout_per_update": int(rollout_per_update),
        "actual_total_timesteps": int(actual_total_timesteps),
        "minibatch_size": int(
            rollout_per_update // base_config["NUM_MINIBATCHES"]
        ),
    }


def train_joint_multitask(
    base_config, total_timesteps, train_tasks, train_graph_padding, run_started
):
    layout = compute_training_layout(base_config, total_timesteps, train_tasks)
    device = torch.device(base_config.get("DEVICE", "cpu"))
    first_task = train_tasks[0]
    first_env = make_task_env(first_task, base_config, train_graph_padding)
    network = make_actor_critic(base_config, first_env)

    rng = torch_random.PRNGKey(base_config["SEED"])
    keys = torch_random.split(rng)
    rng, init_rng = keys[0], keys[1]
    params = network.init(init_rng, first_env.graph_observation_template())
    params = torch_nn._tree_map(
        lambda p: p.detach().to(device).requires_grad_(True), params)
    train_state = TrainState(base_config, params, layout["num_updates"])
    update_rng = torch_random.fold_in(rng, 10_000)

    task_contexts = []
    for task_index, task in enumerate(train_tasks):
        task_contexts.append(
            build_task_context(
                task,
                base_config,
                train_graph_padding,
                network,
                torch_random.fold_in(rng, task_index),
                device,
            )
        )

    joint_update = build_joint_update_fn(
        base_config,
        network,
        batch_size=layout["rollout_per_update"],
    )

    history = []
    training_started = time.perf_counter()
    startup_elapsed = time.perf_counter() - run_started
    print(
        f"Training {len(train_tasks)} tasks jointly for "
        f"{layout['num_updates']} PPO updates; "
        f"{layout['rollout_per_update']:,} timesteps/update "
        f"({layout['rollout_per_task']:,} per task). "
        f"elapsed={format_duration(startup_elapsed)}"
    )

    for update_index in range(layout["num_updates"]):
        started = time.perf_counter()
        task_batches = []
        task_records = []

        for context in task_contexts:
            runner_state, rollout = context["collector"](
                train_state.params,
                context["env_state"],
                context["last_obs"],
                context["rng"],
            )
            context["env_state"], context["last_obs"], context["rng"] = runner_state
            traj_batch, advantages, targets = rollout
            task_batches.append(
                PPOBatch(
                    obs=traj_batch.obs,
                    action=traj_batch.action,
                    value=traj_batch.value,
                    log_prob=traj_batch.log_prob,
                    advantages=advantages,
                    targets=targets,
                )
            )
            task_records.append(
                summarize_task_rollout(
                    context["task"],
                    traj_batch,
                    max_steps=base_config["MAX_STEPS"],
                )
            )

        combined_batch = merge_task_batches(task_batches)
        keys = torch_random.split(update_rng)
        update_rng, step_rng = keys[0], keys[1]
        train_state, loss_metrics, update_rng = joint_update(
            train_state, combined_batch, step_rng
        )
        loss_summary = summarize_loss_metrics(loss_metrics)

        reward_mean = float(
            np.mean([record["reward_mean"] for record in task_records])
        )
        done_rate = float(np.mean([record["done_rate"] for record in task_records]))
        episode_count = int(sum(record["episode_count"] for record in task_records))
        success_count = int(sum(record["success_count"] for record in task_records))
        timeout_count = int(sum(record["timeout_count"] for record in task_records))
        success_rate = success_count / episode_count if episode_count else None
        episode_return_mean = mean_or_none(
            [record["episode_return_mean"] for record in task_records]
        )
        episode_length_mean = mean_or_none(
            [record["episode_length_mean"] for record in task_records]
        )

        finished = time.perf_counter()
        record = {
            "update": update_index + 1,
            "timesteps": (update_index + 1) * layout["rollout_per_update"],
            "seconds": finished - started,
            "elapsed_seconds": finished - training_started,
            "reward_mean": reward_mean,
            "done_rate": done_rate,
            "episode_count": episode_count,
            "success_count": success_count,
            "timeout_count": timeout_count,
            "success_rate": success_rate,
            "episode_return_mean": episode_return_mean,
            "episode_length_mean": episode_length_mean,
            "loss": loss_summary,
            "tasks": task_records,
        }
        history.append(record)
        print(
            f"update={record['update']} "
            f"reward={record['reward_mean']:.2f} "
            f"success={format_metric(record['success_rate'])} "
            f"episodes={record['episode_count']} "
            f"return={format_metric(record['episode_return_mean'])} "
            f"length={format_metric(record['episode_length_mean'])} "
            f"loss={record['loss']['total_loss']:.4f} "
            f"time={record['seconds']:.1f}s "
            f"elapsed={format_duration(record['elapsed_seconds'])}"
        )

    return train_state.params, history, layout


def dry_run(
    base_config,
    train_tasks,
    validation_tasks,
    train_graph_padding,
    validation_graph_padding,
):
    """Check split-specific shapes and train-parameter reuse across paddings."""

    if not train_tasks:
        raise ValueError("at least one training task is required for a dry run")

    first_task = train_tasks[0]
    first_env = make_task_env(first_task, base_config, train_graph_padding)
    network = make_actor_critic(base_config, first_env)
    params = network.init(
        torch_random.PRNGKey(1), first_env.graph_observation_template()
    )

    def check_tasks(label, tasks, graph_padding):
        if not tasks:
            return

        reference_task = tasks[0]
        reference_env = make_task_env(reference_task, base_config, graph_padding)
        expected_shapes = tuple(
            tuple(leaf.shape)
            for leaf in reference_env.graph_observation_template()
        )

        for task_index, task in enumerate(tasks):
            env = make_task_env(task, base_config, graph_padding)
            observation, _ = env.reset(torch_random.PRNGKey(task_index + 2), None)
            shapes = tuple(tuple(leaf.shape) for leaf in observation)
            if shapes != expected_shapes:
                raise ValueError(
                    f"{label} task {task} has incompatible graph shapes"
                )
            with torch.no_grad():
                policy, value = network.apply(params, observation)
            print(
                f"{label} {format_task(task)}: "
                f"nodes={int(observation.node_mask.sum())}, "
                f"actions={int(observation.action_mask.sum())}, "
                f"logits={tuple(policy.logits.shape)}, value_shape={tuple(value.shape)}"
            )

    check_tasks("train", train_tasks, train_graph_padding)
    check_tasks("validation", validation_tasks, validation_graph_padding)


def save_training_results(output_dir, params, history, run_config):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_params(output_dir / "params.npz", params)
    for name, value in (
        ("train_history.json", history),
        ("run_config.json", run_config),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
    print(f"Saved training artifacts to {output_dir}")


def save_validation_results(output_dir, validation):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation.json").open("w", encoding="utf-8") as file:
        json.dump(validation, file, indent=2)
    print(f"Saved validation results to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory from the config file.",
    )
    return parser.parse_args()


def main():
    run_started = time.perf_counter()
    args = parse_args()
    run_settings = load_run_settings(args.config)
    if args.output_dir is not None:
        run_settings["output_dir"] = args.output_dir.expanduser()
    config = run_settings["config"]
    graphs = run_settings["graphs"]
    train_tasks = run_settings["train_tasks"]
    configured_validation_tasks = run_settings["validation_tasks"]
    validation_tasks = (
        [] if run_settings["skip_validation"] else configured_validation_tasks
    )
    train_graph_padding = build_graph_padding(train_tasks)
    validation_graph_padding = (
        build_graph_padding(validation_tasks)
        if validation_tasks
        else train_graph_padding
    )

    print(f"Loaded configuration from {run_settings['config_path']}")

    if run_settings["dry_run"]:
        dry_run(
            config,
            train_tasks,
            validation_tasks,
            train_graph_padding,
            validation_graph_padding,
        )
        return

    params, history, layout = train_joint_multitask(
        config,
        total_timesteps=config["TOTAL_TIMESTEPS"],
        train_tasks=train_tasks,
        train_graph_padding=train_graph_padding,
        run_started=run_started,
    )
    run_config = {
        **config,
        "config_path": run_settings["config_path"],
        "output_dir": str(run_settings["output_dir"]),
        "skip_distance": run_settings["skip_distance"],
        "skip_validation": run_settings["skip_validation"],
        "dry_run": run_settings["dry_run"],
        "graphs": list(graphs),
        "WHICH_GATES": list(config["WHICH_GATES"]),
        "train_tasks": train_tasks,
        "validation_tasks": configured_validation_tasks,
        "train_graph_padding": graph_padding_to_dict(train_graph_padding),
        "validation_graph_padding": (
            None
            if not validation_tasks
            else graph_padding_to_dict(validation_graph_padding)
        ),
        "requested_total_timesteps": config["TOTAL_TIMESTEPS"],
        **layout,
    }
    # move params to CPU for saving/validation
    params = torch_nn._tree_map(lambda p: p.detach().cpu().requires_grad_(True), params)
    save_training_results(run_settings["output_dir"], params, history, run_config)
    validation = run_validation(
        params,
        config,
        validation_tasks,
        validation_graph_padding=validation_graph_padding,
        compute_distance=not run_settings["skip_distance"],
    )
    save_validation_results(run_settings["output_dir"], validation)
    total_runtime = time.perf_counter() - run_started
    print(
        f"Total runtime: {format_duration(total_runtime)} "
        f"({total_runtime:.1f}s)"
    )


if __name__ == "__main__":
    main()
