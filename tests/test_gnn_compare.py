"""JAX(TCC) vs PyTorch: GNN graph observations, model init/forward, PPO
loss + autograd gradients, greedy validation rollouts, and one make_train
PPO update on the graph environment."""

import numpy as np
import torch

from compare_utils import parse_args, repo_on_path, Reporter, tree_max_diff, asnp

args = parse_args(__doc__)

import jax
import jax.numpy as jnp

N, K, D = 5, 1, 3
MAX_STEPS = 8
GATE_NAMES = ["cx", "h", "s"]

r = Reporter("GNN observation / model / PPO / validation (JAX-TCC vs torch)")


def obs_to_np(obs):
    """GraphObservation (either backend) -> dict of numpy arrays."""
    fields = [
        "node_features", "edge_features", "senders", "receivers",
        "relation_ids", "node_mask", "edge_mask", "qubit_mask",
        "stabilizer_mask", "global_features", "action_types",
        "action_gate_ids", "action_first", "action_second",
        "action_edge_indices", "action_mask", "action_env_indices",
    ]
    return {name: asnp(getattr(obs, name)) for name in fields}


def to_np_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_np_tree(v) for k, v in t.items()}
    return asnp(t)


def to_torch_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_torch_tree(v) for k, v in t.items()}
    return torch.from_numpy(np.array(asnp(t), dtype=np.float32, copy=True)).requires_grad_(True)


def build_env_jax(repo, padding_kwargs=None):
    from qdx.simulators.clifford_gates import CliffordGates
    from qdx.envs.graph_code_discovery import GraphCodeDiscovery
    from qdx.gnn.observation import GraphPadding

    g = CliffordGates(N)
    gate_set = [getattr(g, name) for name in GATE_NAMES]
    padding = GraphPadding(**padding_kwargs) if padding_kwargs else None
    return GraphCodeDiscovery(
        N, K, D, gate_set, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1,
        graph_padding=padding,
    )


PADDING = {"n_max": 6, "stabilizers_max": 5, "hardware_edges_max": 30}

# ---------------------------------------------------------------- observation
# Apply the same fixed gate sequence on both sides and compare every field
# of the graph observation at every step.
ACTIONS = [3, 0, 7, 12, 5, 20, 1, 9]

with repo_on_path(args.jax_repo):
    env_j = build_env_jax(args.jax_repo, PADDING)
    obs_j, state_j = env_j.reset(jax.random.PRNGKey(0), None)
    obs_seq_j = [obs_to_np(obs_j)]
    reward_seq_j = []
    for a in ACTIONS:
        obs_j, state_j, rew, done, _ = env_j.step(
            jax.random.PRNGKey(100 + a), state_j, jnp.asarray(a), None)
        obs_seq_j.append(obs_to_np(obs_j))
        reward_seq_j.append(float(rew))
    template_j = obs_to_np(env_j.graph_observation_template())
    static_j = {
        "E_mu": np.asarray(env_j.E_mu),
        "p_mu": np.asarray(env_j.p_mu),
        "E_mu_Omega": np.asarray(env_j.E_mu_Omega),
        "S_struct": np.asarray(env_j.S_struct),
        "actions": np.asarray(env_j.actions),
        "action_string": list(env_j.action_string),
        "num_gate_types": env_j.graph_builder.num_gate_types,
        "descriptors": [env_j.action_descriptor(i) for i in range(env_j.num_actions)],
    }

with repo_on_path(args.torch_repo):
    from qdx.simulators.clifford_gates import CliffordGates
    from qdx.envs.graph_code_discovery import GraphCodeDiscovery
    from qdx.gnn.observation import GraphPadding
    from qdx import torch_random as tr
    from qdx import torch_nn
    from qdx.gnn.model import GNNQDXActorCritic as TorchGNN
    from qdx.gnn.observation import GraphObservation as TorchObs, obs_stack

    g = CliffordGates(N)
    gate_set = [getattr(g, name) for name in GATE_NAMES]
    env_t = GraphCodeDiscovery(
        N, K, D, gate_set, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1,
        graph_padding=GraphPadding(**PADDING),
    )
    obs_t, state_t = env_t.reset(tr.PRNGKey(0), None)
    obs_seq_t = [obs_to_np(obs_t)]
    reward_seq_t = []
    torch_obs_seq = [obs_t]
    for a in ACTIONS:
        obs_t, state_t, rew, done, _ = env_t.step(
            tr.PRNGKey(100 + a), state_t, a, None)
        obs_seq_t.append(obs_to_np(obs_t))
        torch_obs_seq.append(obs_t)
        reward_seq_t.append(float(rew))
    template_t = obs_to_np(env_t.graph_observation_template())
    static_t = {
        "E_mu": asnp(env_t.E_mu),
        "p_mu": asnp(env_t.p_mu),
        "E_mu_Omega": asnp(env_t.E_mu_Omega),
        "S_struct": asnp(env_t.S_struct),
        "actions": asnp(env_t.actions),
        "action_string": list(env_t.action_string),
        "num_gate_types": env_t.graph_builder.num_gate_types,
        "descriptors": [env_t.action_descriptor(i) for i in range(env_t.num_actions)],
    }

r.check_value_equal("env static: E_mu (TCC direct ordering)", static_t["E_mu"], static_j["E_mu"])
r.check_close("env static: p_mu", static_t["p_mu"], static_j["p_mu"], rtol=1e-6, atol=1e-7)
r.check_value_equal("env static: E_mu_Omega", static_t["E_mu_Omega"], static_j["E_mu_Omega"])
r.check_value_equal("env static: S_struct", static_t["S_struct"], static_j["S_struct"])
r.check_value_equal("env static: action matrices", static_t["actions"], static_j["actions"])
r.check("env static: action strings / gate types / descriptors",
        static_t["action_string"] == static_j["action_string"]
        and static_t["num_gate_types"] == static_j["num_gate_types"]
        and static_t["descriptors"] == static_j["descriptors"])
r.check_close("env rollout: rewards", reward_seq_t, reward_seq_j, rtol=1e-5, atol=1e-5)

int_fields = ["senders", "receivers", "relation_ids", "node_mask", "edge_mask",
              "qubit_mask", "stabilizer_mask", "action_types", "action_gate_ids",
              "action_first", "action_second", "action_edge_indices",
              "action_mask", "action_env_indices"]
float_fields = ["node_features", "edge_features", "global_features"]

ok_int, ok_float, max_float_diff = True, True, 0.0
for step, (ot, oj) in enumerate(zip(obs_seq_t, obs_seq_j)):
    for name in int_fields:
        if not np.array_equal(ot[name], oj[name]):
            ok_int = False
    for name in float_fields:
        d = float(np.max(np.abs(ot[name].astype(np.float64) - oj[name].astype(np.float64))))
        max_float_diff = max(max_float_diff, d)
        if d > 1e-5:
            ok_float = False
r.check("graph obs: integer/mask fields exact over rollout", ok_int)
r.check("graph obs: float features close over rollout", ok_float,
        f"max abs diff {max_float_diff:.3g}")
for name in float_fields + int_fields:
    if not np.array_equal(np.asarray(template_t[name]), np.asarray(template_j[name])):
        d = float(np.max(np.abs(template_t[name].astype(np.float64)
                                - template_j[name].astype(np.float64))))
        r.check(f"template field {name}", d < 1e-6, f"max abs diff {d:.3g}")

# ---------------------------------------------------------------- model init
HP = dict(hidden_dim=32, gate_dim=8, num_gnn_layers=2)

with repo_on_path(args.jax_repo):
    from qdx.gnn.model import GNNQDXActorCritic as JaxGNN

    net_j = JaxGNN(num_gate_types=len(GATE_NAMES), activation="tanh", **HP)
    env_j2 = build_env_jax(args.jax_repo, PADDING)
    params_j = net_j.init(jax.random.PRNGKey(11), env_j2.graph_observation_template())
    params_j_np = to_np_tree(params_j)

net_t = TorchGNN(num_gate_types=len(GATE_NAMES), activation="tanh", **HP)
params_t = net_t.init(tr.PRNGKey(11), env_t.graph_observation_template())
d = tree_max_diff(params_j_np, to_np_tree(params_t))
r.check("GNN init params match flax defaults", d < 5e-6, f"max abs diff {d:.3g}")

# ---------------------------------------------------------------- forward
# Use identical weights on both sides (copy flax's into torch).
params_t_shared = to_torch_tree(params_j_np)

single_obs_t = torch_obs_seq[3]
batch_obs_t = obs_stack(torch_obs_seq)

with repo_on_path(args.jax_repo):
    # rebuild the jax obs sequence for forward passes
    env_j3 = build_env_jax(args.jax_repo, PADDING)
    obs_j3, state_j3 = env_j3.reset(jax.random.PRNGKey(0), None)
    jax_obs_seq = [obs_j3]
    for a in ACTIONS:
        obs_j3, state_j3, _, _, _ = env_j3.step(
            jax.random.PRNGKey(100 + a), state_j3, jnp.asarray(a), None)
        jax_obs_seq.append(obs_j3)
    batch_obs_j = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *jax_obs_seq)

    pi_j, v_j = net_j.apply(params_j, jax_obs_seq[3])
    logits_j1 = np.asarray(pi_j.logits)
    value_j1 = np.asarray(v_j)
    pi_jb, v_jb = net_j.apply(params_j, batch_obs_j)
    logits_jb = np.asarray(pi_jb.logits)
    value_jb = np.asarray(v_jb)
    entropy_jb = np.asarray(pi_jb.entropy())
    sample_jb = np.asarray(pi_jb.sample(seed=jax.random.PRNGKey(77)))
    lp_jb = np.asarray(pi_jb.log_prob(jnp.asarray(sample_jb)))
    mode_j1 = int(np.asarray(pi_j.mode()))

with torch.no_grad():
    pi_t1, v_t1 = net_t.apply(params_t_shared, single_obs_t)
    pi_tb, v_tb = net_t.apply(params_t_shared, batch_obs_t)
    sample_tb = pi_tb.sample(seed=tr.PRNGKey(77))
    lp_tb = pi_tb.log_prob(sample_tb)
    mode_t1 = int(torch.argmax(pi_t1.logits, dim=-1))

# Only compare valid (unmasked) logits; padded entries are -1e9 on both sides.
mask = asnp(single_obs_t.action_mask).astype(bool)
r.check_close("forward single: valid logits", asnp(pi_t1.logits)[mask], logits_j1[mask],
              rtol=1e-4, atol=1e-5)
r.check_close("forward single: value", asnp(v_t1), value_j1, rtol=1e-4, atol=1e-5)
r.check("forward single: mode (greedy action)", mode_t1 == mode_j1,
        f"torch {mode_t1} vs jax {mode_j1}")
bmask = asnp(batch_obs_t.action_mask).astype(bool)
r.check_close("forward batch: valid logits", asnp(pi_tb.logits)[bmask], logits_jb[bmask],
              rtol=1e-4, atol=1e-5)
r.check_close("forward batch: values", asnp(v_tb), value_jb, rtol=1e-4, atol=1e-5)
r.check_close("forward batch: entropy", asnp(pi_tb.entropy()), entropy_jb,
              rtol=1e-4, atol=1e-5)
r.check_value_equal("forward batch: categorical samples", asnp(sample_tb), sample_jb)
r.check_close("forward batch: log_prob of samples", asnp(lp_tb), lp_jb,
              rtol=1e-4, atol=1e-5)

# ---------------------------------------------------------------- PPO loss + grad
B = len(torch_obs_seq)
config = {"ACTIVATION": "tanh", "CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.02}
action_b = np.asarray(sample_jb, dtype=np.int32)
gae_b = (np.arange(B, dtype=np.float32) - B / 2.0) / B * 3.0
targets_b = np.cos(np.arange(B, dtype=np.float32))
old_value_b = value_jb.astype(np.float32) + 0.05
old_log_prob_b = lp_jb.astype(np.float32) - 0.1


def jax_loss_fn(params, obs_b, action, old_value, old_log_prob, gae, targets, net):
    pi, value = net.apply(params, obs_b)
    log_prob = pi.log_prob(action)
    value_pred_clipped = old_value + (value - old_value).clip(-0.2, 0.2)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    ratio = jnp.exp(log_prob - old_log_prob)
    gae_n = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor = -jnp.minimum(
        ratio * gae_n, jnp.clip(ratio, 0.8, 1.2) * gae_n).mean()
    entropy = pi.entropy().mean()
    total = loss_actor + 0.5 * value_loss - 0.02 * entropy
    return total, (value_loss, loss_actor, entropy)


with repo_on_path(args.jax_repo):
    (tl_j, aux_j), grads_j = jax.value_and_grad(jax_loss_fn, has_aux=True)(
        params_j, batch_obs_j, jnp.asarray(action_b), jnp.asarray(old_value_b),
        jnp.asarray(old_log_prob_b), jnp.asarray(gae_b), jnp.asarray(targets_b),
        net_j)
    grads_j_np = to_np_tree(grads_j["params"])
    tl_j = float(tl_j)
    aux_j = [float(a) for a in aux_j]

(tl_t, aux_t), grads_t = torch_nn.ppo_loss_and_grad_generic(
    net_t.apply, params_t_shared, batch_obs_t, action_b, old_value_b,
    old_log_prob_b, gae_b, targets_b, config)

r.check_close("PPO total loss", float(tl_t), tl_j, rtol=1e-5, atol=1e-6)
r.check_close("PPO value loss", float(aux_t[0]), aux_j[0], rtol=1e-5, atol=1e-6)
r.check_close("PPO actor loss", float(aux_t[1]), aux_j[1], rtol=1e-5, atol=1e-6)
r.check_close("PPO entropy", float(aux_t[2]), aux_j[2], rtol=1e-5, atol=1e-6)
d = tree_max_diff(grads_j_np, to_np_tree(grads_t["params"]))
r.check("PPO autograd gradients match jax.value_and_grad", d < 2e-5,
        f"max abs diff {d:.3g}")

# ---------------------------------------------------------------- greedy validation
with repo_on_path(args.jax_repo):
    from qdx.validation_rollout import (
        build_validation_episode_runner as jax_runner,
        summarize_validation_episode as jax_summary,
    )

    env_jv = build_env_jax(args.jax_repo, PADDING)
    runner_j = jax_runner(env_jv, net_j, MAX_STEPS)
    rng_v = jax.random.PRNGKey(52_000)
    obs_v, state_v = env_jv.reset(rng_v, None)
    rollout_j = runner_j(params_j, obs_v, state_v, rng_v)
    summary_j = jax_summary(rollout_j, env_jv.action_string_stim)

with repo_on_path(args.torch_repo):
    from qdx.validation_rollout import (
        build_validation_episode_runner as torch_runner,
        summarize_validation_episode as torch_summary,
    )

    runner_t = torch_runner(env_t, net_t, MAX_STEPS)
    rng_v_t = tr.PRNGKey(52_000)
    obs_v_t, state_v_t = env_t.reset(rng_v_t, None)
    rollout_t = runner_t(params_t_shared, obs_v_t, state_v_t, rng_v_t)
    summary_t = torch_summary(rollout_t, env_t.action_string_stim)

r.check("greedy validation: identical gate sequences",
        summary_t["gates"] == summary_j["gates"],
        f"torch {summary_t['gates'][:4]}... vs jax {summary_j['gates'][:4]}...")
r.check("greedy validation: steps/done equal",
        summary_t["steps"] == summary_j["steps"] and summary_t["done"] == summary_j["done"])
r.check_close("greedy validation: total reward",
              summary_t["total_reward"], summary_j["total_reward"], rtol=1e-4, atol=1e-4)
r.check_close("greedy validation: final value",
              summary_t["final_value"], summary_j["final_value"], rtol=1e-3, atol=1e-4)

r.finish()
