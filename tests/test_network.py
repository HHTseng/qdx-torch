"""JAX vs PyTorch: ActorCritic init/forward, distribution ops, PPO loss and
gradients (torch.autograd vs jax.value_and_grad), optimizer updates (vs
optax), and GAE."""

import numpy as np
import torch

from compare_utils import parse_args, repo_on_path, Reporter, tree_max_diff, asnp

args = parse_args(__doc__)

import jax
import jax.numpy as jnp
import optax

with repo_on_path(args.jax_repo):
    from qdx.make_train import ActorCritic as AC_jax
with repo_on_path(args.torch_repo):
    from qdx.make_train import ActorCritic as AC_torch
    from qdx import torch_nn
    from qdx import torch_random as nr

r = Reporter("Network / loss / gradients / optimizer / GAE")

OBS_DIM, ACTION_DIM, HIDDEN = 40, 50, 32
B = 64  # minibatch size

config = {
    "ACTIVATION": "relu",
    "CLIP_EPS": 0.2,
    "VF_COEF": 0.5,
    "ENT_COEF": 0.02,
}


def to_np_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_np_tree(v) for k, v in t.items()}
    return asnp(t)


def to_torch_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_torch_tree(v) for k, v in t.items()}
    return torch.from_numpy(np.array(asnp(t), dtype=np.float32, copy=True))


# ---------------------------------------------------------------- init
for act in ["relu", "tanh"]:
    net_j = AC_jax(ACTION_DIM, activation=act, hidden_dim=HIDDEN)
    net_t = AC_torch(ACTION_DIM, activation=act, hidden_dim=HIDDEN)
    key = jax.random.PRNGKey(11)
    pj = net_j.init(key, jnp.zeros(OBS_DIM))
    pt = net_t.init(nr.PRNGKey(11), torch.zeros(OBS_DIM))
    d = tree_max_diff(to_np_tree(pj["params"]), to_np_tree(pt["params"]))
    r.check(f"init params match ({act})", d < 2e-6, f"max abs diff {d:.3g}")

# ---------------------------------------------------------------- forward
net_j = AC_jax(ACTION_DIM, activation="relu", hidden_dim=HIDDEN)
net_t = AC_torch(ACTION_DIM, activation="relu", hidden_dim=HIDDEN)
key = jax.random.PRNGKey(11)
params_j = net_j.init(key, jnp.zeros(OBS_DIM))
params_t = to_torch_tree(params_j)  # identical weights on both sides

obs = np.asarray(jax.random.uniform(jax.random.PRNGKey(5), (B, OBS_DIM))) > 0.5
obs = obs.astype(np.uint8)

pi_j, v_j = net_j.apply(params_j, jnp.asarray(obs))
pi_t, v_t = net_t.apply(params_t, torch.from_numpy(obs))

r.check_close("forward: value", v_t, np.asarray(v_j), rtol=1e-6, atol=1e-6)
r.check_close("forward: normalized logits", pi_t.logits, np.asarray(pi_j.logits),
              rtol=1e-6, atol=1e-6)
acts = np.arange(B) % ACTION_DIM
r.check_close("forward: log_prob", pi_t.log_prob(torch.from_numpy(acts)),
              np.asarray(pi_j.log_prob(jnp.asarray(acts))), rtol=1e-6, atol=1e-6)
r.check_close("forward: entropy", pi_t.entropy(), np.asarray(pi_j.entropy()),
              rtol=1e-6, atol=1e-6)

skey = jax.random.PRNGKey(77)
sample_j = np.asarray(pi_j.sample(seed=skey))
sample_t = pi_t.sample(seed=nr.PRNGKey(77))
r.check_value_equal("forward: categorical sample", sample_t, sample_j)

# 1-D observation path (used by evaluate())
pi_j1, v_j1 = net_j.apply(params_j, jnp.asarray(obs[0]))
pi_t1, v_t1 = net_t.apply(params_t, torch.from_numpy(obs[0]))
r.check_close("forward 1-D obs: logits", pi_t1.logits, np.asarray(pi_j1.logits),
              rtol=1e-6, atol=1e-6)

# ---------------------------------------------------------------- PPO loss + grad
import distrax


def jax_loss_fn(params, obs_b, action_b, old_value_b, old_log_prob_b, gae_b,
                targets_b, cfg, network):
    """Verbatim re-statement of _loss_fn from the original make_train.py."""
    pi, value = network.apply(params, obs_b)
    log_prob = pi.log_prob(action_b)

    value_pred_clipped = old_value_b + (
        value - old_value_b
    ).clip(-cfg["CLIP_EPS"], cfg["CLIP_EPS"])
    value_losses = jnp.square(value - targets_b)
    value_losses_clipped = jnp.square(value_pred_clipped - targets_b)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )

    ratio = jnp.exp(log_prob - old_log_prob_b)
    gae_n = (gae_b - gae_b.mean()) / (gae_b.std() + 1e-8)
    loss_actor1 = ratio * gae_n
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - cfg["CLIP_EPS"],
            1.0 + cfg["CLIP_EPS"],
        )
        * gae_n
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor
        + cfg["VF_COEF"] * value_loss
        - cfg["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)


rngs = jax.random.PRNGKey(31)
action_b = np.asarray(jax.random.randint(jax.random.fold_in(rngs, 0), (B,), 0, ACTION_DIM)).astype(np.int32)
gae_b = np.asarray(jax.random.normal(jax.random.fold_in(rngs, 1), (B,))) * 3.0
targets_b = np.asarray(jax.random.normal(jax.random.fold_in(rngs, 2), (B,))) * 2.0
gae_b = gae_b.astype(np.float32)
targets_b = targets_b.astype(np.float32)

# Case 1: the "first minibatch" regime -- old log-probs and values exactly
# equal to the current network outputs (ratio == 1 ties everywhere). This
# exercises the balanced 0.5/0.5 tie gradients of minimum/maximum.
with torch.no_grad():
    pi_now, v_now = net_t.apply(params_t, torch.from_numpy(obs))
    old_value_tie = asnp(v_now).astype(np.float32)
    old_log_prob_tie = asnp(pi_now.log_prob(torch.from_numpy(action_b))).astype(np.float32)

# Case 2: generic regime -- old quantities from a perturbed network
perturbed = jax.tree.map(lambda x: x + 0.01 * jnp.sin(jnp.arange(x.size, dtype=jnp.float32)).reshape(x.shape), params_j)
pi_old, v_old = net_j.apply(perturbed, jnp.asarray(obs))
old_value_gen = np.asarray(v_old, dtype=np.float32)
old_log_prob_gen = np.asarray(pi_old.log_prob(jnp.asarray(action_b)), dtype=np.float32)

grad_fn = jax.value_and_grad(jax_loss_fn, has_aux=True)

for case, (ov, olp) in [("tie regime", (old_value_tie, old_log_prob_tie)),
                        ("generic regime", (old_value_gen, old_log_prob_gen))]:
    (tl_j, aux_j), grads_j = grad_fn(
        params_j, jnp.asarray(obs), jnp.asarray(action_b), jnp.asarray(ov),
        jnp.asarray(olp), jnp.asarray(gae_b), jnp.asarray(targets_b),
        config, net_j)
    (tl_t, aux_t), grads_t = torch_nn.ppo_loss_and_grad(
        params_t, obs, action_b, ov, olp, gae_b, targets_b, config)

    r.check_close(f"loss total ({case})", tl_t, np.asarray(tl_j), rtol=1e-5, atol=1e-6)
    r.check_close(f"loss value ({case})", aux_t[0], np.asarray(aux_j[0]), rtol=1e-5, atol=1e-6)
    r.check_close(f"loss actor ({case})", aux_t[1], np.asarray(aux_j[1]), rtol=1e-5, atol=1e-6)
    r.check_close(f"loss entropy ({case})", aux_t[2], np.asarray(aux_j[2]), rtol=1e-5, atol=1e-6)
    d = tree_max_diff(to_np_tree(grads_j["params"]), to_np_tree(grads_t["params"]))
    r.check(f"gradients match ({case})", d < 5e-6, f"max abs diff {d:.3g}")

# tanh path as well
config_tanh = dict(config, ACTIVATION="tanh")
net_j_t = AC_jax(ACTION_DIM, activation="tanh", hidden_dim=HIDDEN)
(tl_j, aux_j), grads_j = jax.value_and_grad(jax_loss_fn, has_aux=True)(
    params_j, jnp.asarray(obs), jnp.asarray(action_b), jnp.asarray(old_value_gen),
    jnp.asarray(old_log_prob_gen), jnp.asarray(gae_b), jnp.asarray(targets_b),
    config_tanh, net_j_t)
(tl_t, aux_t), grads_t = torch_nn.ppo_loss_and_grad(
    params_t, obs, action_b, old_value_gen, old_log_prob_gen, gae_b, targets_b,
    config_tanh)
r.check_close("loss total (tanh)", tl_t, np.asarray(tl_j), rtol=1e-5, atol=1e-6)
d = tree_max_diff(to_np_tree(grads_j["params"]), to_np_tree(grads_t["params"]))
r.check("gradients match (tanh)", d < 5e-6, f"max abs diff {d:.3g}")

# ---------------------------------------------------------------- optimizer
MAX_GRAD_NORM = 0.25
LR = 1e-3
NUM_EPOCHS_F = 100.0
NM, UE = 4, 3


def linear_schedule_jax(count):
    frac = 1.0 - (count // (NM * UE)) / NUM_EPOCHS_F
    return LR * frac


def linear_schedule_np(count):
    frac = np.float32(1.0) - np.float32(count // (NM * UE)) / np.float32(NUM_EPOCHS_F)
    return np.float32(LR) * frac


for sched_name, lr_jax, lr_t in [
        ("anneal", linear_schedule_jax, linear_schedule_np),
        ("const", LR, LR)]:
    tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM),
                     optax.adam(learning_rate=lr_jax, eps=1e-5))
    ostate_j = tx.init(params_j)
    p_j = params_j

    p_t = to_torch_tree(params_t)
    ostate_t = torch_nn.OptimizerState(p_t)

    ok = True
    detail = ""
    for step in range(6):
        # alternate small grads (no clipping) and large grads (clipping active)
        scale = 0.001 if step % 2 == 0 else 10.0
        g_j = jax.tree.map(
            lambda x: scale * jnp.cos(0.1 * step + jnp.arange(x.size, dtype=jnp.float32)).reshape(x.shape),
            p_j)
        g_t = to_torch_tree(g_j)

        upd_j, ostate_j = tx.update(g_j, ostate_j, p_j)
        p_j = optax.apply_updates(p_j, upd_j)

        upd_t, ostate_t = torch_nn.optimizer_update(g_t, ostate_t, MAX_GRAD_NORM, lr_t)
        p_t = torch_nn.apply_updates(p_t, upd_t)

        d = tree_max_diff(to_np_tree(p_j["params"]), to_np_tree(p_t["params"]))
        if d > 1e-6:
            ok = False
            detail = f"step {step}: max abs diff {d:.3g}"
            break
        detail = f"6 steps, final max abs diff {d:.3g}"
    r.check(f"optimizer chain(clip, adam) [{sched_name}]", ok, detail)

# ---------------------------------------------------------------- GAE
T, NE = 20, 16
GAMMA, LAM = 0.99, 0.95
kk = jax.random.PRNGKey(9)
rewards = np.asarray(jax.random.normal(jax.random.fold_in(kk, 0), (T, NE))).astype(np.float32)
values = np.asarray(jax.random.normal(jax.random.fold_in(kk, 1), (T, NE))).astype(np.float32)
dones = np.asarray(jax.random.bernoulli(jax.random.fold_in(kk, 2), 0.15, (T, NE)))
last_val = np.asarray(jax.random.normal(jax.random.fold_in(kk, 3), (NE,))).astype(np.float32)


def gae_jax():
    def _get_advantages(gae_and_next_value, xs):
        gae, next_value = gae_and_next_value
        done, value, reward = xs
        delta = reward + GAMMA * next_value * (1 - done) - value
        gae = delta + GAMMA * LAM * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(jnp.asarray(last_val)), jnp.asarray(last_val)),
        (jnp.asarray(dones), jnp.asarray(values), jnp.asarray(rewards)),
        reverse=True, unroll=16)
    return np.asarray(advantages), np.asarray(advantages + values)


def gae_torch():
    """Torch equivalent of the GAE loop in qdx/make_train.py."""
    dones_t = torch.from_numpy(dones)
    values_t = torch.from_numpy(values)
    rewards_t = torch.from_numpy(rewards)
    advantages = torch.zeros((T, NE), dtype=torch.float32)
    gae = torch.zeros(NE, dtype=torch.float32)
    next_value = torch.from_numpy(last_val)
    for t in reversed(range(T)):
        done, value, reward = dones_t[t], values_t[t], rewards_t[t]
        not_done = 1.0 - done.to(torch.float32)
        delta = reward + float(np.float32(GAMMA)) * next_value * not_done - value
        gae = delta + float(np.float32(GAMMA)) * float(np.float32(LAM)) * not_done * gae
        advantages[t] = gae
        next_value = value
    return asnp(advantages), asnp(advantages + values_t)


adv_j, tgt_j = gae_jax()
adv_t, tgt_t = gae_torch()
r.check_close("GAE advantages", adv_t, adv_j, rtol=1e-6, atol=1e-6)
r.check_close("GAE targets", tgt_t, tgt_j, rtol=1e-6, atol=1e-6)

r.finish()
