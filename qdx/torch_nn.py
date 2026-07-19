"""PyTorch ActorCritic network matching the flax/distrax/optax stack.

Implements, with the same numerics as the JAX original (up to float32
rounding of the underlying BLAS/libm):

  * flax parameter initialization (orthogonal kernels via QR + SHA-1
    parameter-RNG folding, constant-zero biases) — computed with the
    bit-exact threefry PRNG in :mod:`qdx.torch_random`,
  * the ActorCritic forward pass (plain torch ops on a flax-style
    parameter dict, so every intermediate tensor is inspectable),
  * a distrax-compatible ``Categorical`` (normalized logits, log_prob,
    entropy, Gumbel-max sampling),
  * the PPO loss whose gradients are obtained with ``torch.autograd``
    (this replaces both ``jax.value_and_grad`` and the hand-derived
    NumPy backward pass; ``torch.minimum``/``torch.maximum`` share JAX's
    balanced 0.5/0.5 tie-breaking, and clipping is expressed through them
    so the gradient semantics match ``jnp.clip``),
  * optax ``chain(clip_by_global_norm, adam)`` with linear-schedule support.

All floating point work is float32. Everything runs on CPU by default and
matches the JAX CPU results to ~1e-6; moving the parameter tree and batches
to another torch device (e.g. "mps"/"cuda") is supported for acceleration
but will produce slightly different float32 rounding.
"""

import numpy as np
import torch

from qdx import torch_random


# ---------------------------------------------------------------------------
# Numerically faithful helpers (jax.nn equivalents)
# ---------------------------------------------------------------------------

def log_softmax(x, axis=-1):
    x_max = torch.max(x, dim=axis, keepdim=True).values
    shifted = x - x_max
    return shifted - torch.log(torch.sum(torch.exp(shifted), dim=axis, keepdim=True))


def softmax(x, axis=-1):
    x_max = torch.max(x, dim=axis, keepdim=True).values
    unnormalized = torch.exp(x - x_max)
    return unnormalized / torch.sum(unnormalized, dim=axis, keepdim=True)


def one_hot(indices, num_classes, dtype=torch.float32):
    indices = torch.as_tensor(indices)
    classes = torch.arange(num_classes, device=indices.device)
    return (indices[..., None] == classes).to(dtype)


def relu(x):
    # torch.relu backpropagates 0 at exactly 0, matching jax.nn.relu
    return torch.relu(x)


def _clip_balanced(x, lo, hi):
    """jnp.clip equivalent: minimum(maximum(x, lo), hi).

    Unlike ``torch.clamp`` this shares JAX's balanced (0.5/0.5) gradient
    at exact boundary ties.
    """
    return torch.minimum(torch.maximum(x, torch.as_tensor(lo, dtype=x.dtype, device=x.device)),
                         torch.as_tensor(hi, dtype=x.dtype, device=x.device))


# ---------------------------------------------------------------------------
# distrax.Categorical equivalent
# ---------------------------------------------------------------------------

class Categorical:
    """PyTorch re-implementation of distrax.Categorical(logits=...)."""

    def __init__(self, logits):
        # distrax normalizes logits with log_softmax at construction time
        self._logits = log_softmax(torch.as_tensor(logits, dtype=torch.float32), axis=-1)

    @property
    def logits(self):
        return self._logits

    @property
    def probs(self):
        return softmax(self._logits, axis=-1)

    @property
    def num_categories(self):
        return self._logits.shape[-1]

    def sample(self, seed):
        """Single sample, matching distrax's ``sample(seed=key)``.

        ``seed`` is a threefry uint32 key pair from :mod:`qdx.torch_random`.
        Gumbel noise is drawn bit-exactly with NumPy and the argmax is taken
        in torch, reproducing ``jax.random.categorical``.
        """
        with torch.no_grad():
            batch_shape = self._logits.shape[:-1]
            logits = self._logits.detach().cpu().numpy()
            draws = torch_random.categorical(
                seed, logits, axis=-1, shape=(1,) + tuple(batch_shape)
            ).astype(np.int32)
            draws = draws.reshape(tuple(batch_shape))
            return torch.from_numpy(np.asarray(draws)).to(self._logits.device)

    def log_prob(self, value):
        value = torch.as_tensor(value, device=self._logits.device)
        value_one_hot = one_hot(value, self.num_categories, dtype=self._logits.dtype)
        mask_outside_domain = torch.logical_or(value < 0, value > self.num_categories - 1)
        # multiply_no_nan(logits, one_hot): zero wherever one_hot == 0
        zero = torch.zeros((), dtype=self._logits.dtype, device=self._logits.device)
        prod = torch.where(value_one_hot == 0, zero, self._logits * value_one_hot)
        return torch.where(mask_outside_domain,
                           torch.tensor(float("-inf"), device=self._logits.device),
                           torch.sum(prod, dim=-1))

    def entropy(self):
        log_probs = log_softmax(self._logits, axis=-1)
        probs = torch.exp(log_probs)
        # mul_exp(log_probs, log_probs)
        zero = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        x = torch.where(probs == 0, zero, log_probs)
        return -torch.sum(x * probs, dim=-1)


# ---------------------------------------------------------------------------
# Parameter initialization (flax-equivalent)
# ---------------------------------------------------------------------------

def orthogonal_init(key, shape, scale):
    """Replicates jax.nn.initializers.orthogonal()(key, shape) in float32.

    Runs in NumPy (bit-exact threefry normals + LAPACK QR, verified against
    flax) and returns a torch tensor.
    """
    n_rows, n_cols = int(np.prod(shape)) // shape[-1], shape[-1]
    z = torch_random.normal(key, (max(n_rows, n_cols), min(n_rows, n_cols)))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    x = q * np.sign(d)[None, :]
    if n_rows < n_cols:
        x = x.T
    x = x.reshape(shape)
    return torch.from_numpy((np.float32(scale) * x).astype(np.float32))


def init_actor_critic_params(rng, obs_dim, action_dim, hidden_dim):
    """Replicates ``ActorCritic(...).init(rng, zeros(obs_dim))['params']``.

    flax derives each layer's parameter RNG by folding the scope path and a
    per-scope counter (kernel -> 1, bias -> 2) into the root key via SHA-1.
    Biases are constant zeros so their keys are unused.

    Kernels are stored like flax: shape (in_features, out_features).
    All leaves are torch float32 tensors with ``requires_grad=True``.
    """
    layer_shapes = [
        ("Dense_0", (obs_dim, hidden_dim), np.sqrt(2)),   # actor trunk
        ("Dense_1", (hidden_dim, hidden_dim), np.sqrt(2)),
        ("Dense_2", (hidden_dim, action_dim), 0.01),      # actor head
        ("Dense_3", (obs_dim, hidden_dim), np.sqrt(2)),   # critic trunk
        ("Dense_4", (hidden_dim, hidden_dim), np.sqrt(2)),
        ("Dense_5", (hidden_dim, 1), 1.0),                # critic head
    ]
    params = {}
    for name, shape, scale in layer_shapes:
        kernel_key = torch_random.flax_fold_in_static(rng, (name, 1))
        params[name] = {
            "kernel": orthogonal_init(kernel_key, shape, scale).requires_grad_(True),
            "bias": torch.zeros(shape[-1], dtype=torch.float32, requires_grad=True),
        }
    return {"params": params}


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _activation_fn(name):
    return relu if name == "relu" else torch.tanh


def apply_actor_critic(params, x, activation="tanh"):
    """Forward pass; returns (pi, value) like the flax module."""
    p = params["params"]
    act = _activation_fn(activation)
    x = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
    x = x.to(dtype=torch.float32, device=p["Dense_0"]["kernel"].device)

    # Actor
    a_h1 = act(x @ p["Dense_0"]["kernel"] + p["Dense_0"]["bias"])
    a_h2 = act(a_h1 @ p["Dense_1"]["kernel"] + p["Dense_1"]["bias"])
    logits = a_h2 @ p["Dense_2"]["kernel"] + p["Dense_2"]["bias"]

    # Critic
    c_h1 = act(x @ p["Dense_3"]["kernel"] + p["Dense_3"]["bias"])
    c_h2 = act(c_h1 @ p["Dense_4"]["kernel"] + p["Dense_4"]["bias"])
    critic = c_h2 @ p["Dense_5"]["kernel"] + p["Dense_5"]["bias"]
    value = torch.squeeze(critic, dim=-1)

    return Categorical(logits), value


# ---------------------------------------------------------------------------
# PPO loss: value via plain torch ops, gradients via torch.autograd
# ---------------------------------------------------------------------------

def ppo_loss(params, obs, action, old_value, old_log_prob, gae, targets, config):
    """The PPO total loss, mirroring `_loss_fn` in make_train.py.

    Pure forward computation on torch tensors; differentiable end-to-end.
    """
    clip_eps = float(config["CLIP_EPS"])
    vf_coef = float(config["VF_COEF"])
    ent_coef = float(config["ENT_COEF"])

    pi, value = apply_actor_critic(params, obs, config["ACTIVATION"])
    log_prob = pi.log_prob(action)

    # CALCULATE VALUE LOSS
    value_pred_clipped = old_value + _clip_balanced(value - old_value, -clip_eps, clip_eps)
    value_losses = torch.square(value - targets)
    value_losses_clipped = torch.square(value_pred_clipped - targets)
    value_loss = 0.5 * torch.maximum(value_losses, value_losses_clipped).mean()

    # CALCULATE ACTOR LOSS
    ratio = torch.exp(log_prob - old_log_prob)
    gae = (gae - gae.mean()) / (gae.std(correction=0) + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = _clip_balanced(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -torch.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor
        + vf_coef * value_loss
        - ent_coef * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)


def ppo_loss_generic(apply_fn, params, obs, action, old_value, old_log_prob,
                     gae, targets, config):
    """PPO total loss for an arbitrary policy given as ``apply_fn(params, obs)``.

    Same computation as :func:`ppo_loss` but network-agnostic, so it works for
    both the MLP ActorCritic and the GNN actor-critic (whose observations are
    GraphObservation pytrees).
    """
    clip_eps = float(config["CLIP_EPS"])
    vf_coef = float(config["VF_COEF"])
    ent_coef = float(config["ENT_COEF"])

    pi, value = apply_fn(params, obs)
    log_prob = pi.log_prob(action)

    # CALCULATE VALUE LOSS
    value_pred_clipped = old_value + _clip_balanced(value - old_value, -clip_eps, clip_eps)
    value_losses = torch.square(value - targets)
    value_losses_clipped = torch.square(value_pred_clipped - targets)
    value_loss = 0.5 * torch.maximum(value_losses, value_losses_clipped).mean()

    # CALCULATE ACTOR LOSS
    ratio = torch.exp(log_prob - old_log_prob)
    gae = (gae - gae.mean()) / (gae.std(correction=0) + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = _clip_balanced(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -torch.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor
        + vf_coef * value_loss
        - ent_coef * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)


def ppo_loss_and_grad_generic(apply_fn, params, obs, action, old_value,
                              old_log_prob, gae, targets, config):
    """Network-agnostic ``jax.value_and_grad(_loss_fn, has_aux=True)`` analogue."""

    leaves = _tree_leaves(params)
    for leaf in leaves:
        if not leaf.requires_grad:
            leaf.requires_grad_(True)

    device = leaves[0].device
    to_f32 = lambda t: (t if isinstance(t, torch.Tensor)
                        else torch.from_numpy(np.array(t))).to(
                            dtype=torch.float32, device=device)
    action_t = (action if isinstance(action, torch.Tensor)
                else torch.from_numpy(np.array(action))).to(device)

    total_loss, aux = ppo_loss_generic(
        apply_fn, params, obs, action_t,
        to_f32(old_value), to_f32(old_log_prob), to_f32(gae), to_f32(targets),
        config)

    grad_leaves = torch.autograd.grad(total_loss, leaves)
    grads = _tree_unflatten(params, list(grad_leaves))

    return (total_loss.detach(), tuple(a.detach() for a in aux)), grads


def ppo_loss_and_grad(params, obs, action, old_value, old_log_prob, gae,
                      targets, config):
    """Computes the PPO total loss, aux losses, and d(total)/d(params).

    Torch-autograd equivalent of ``jax.value_and_grad(_loss_fn, has_aux=True)``.
    Returns ``((total_loss, (value_loss, loss_actor, entropy)), grads)`` where
    ``grads`` has the same nested-dict structure as ``params``.
    """
    apply_fn = lambda p, o: apply_actor_critic(p, o, config["ACTIVATION"])
    return ppo_loss_and_grad_generic(
        apply_fn, params, obs, action, old_value, old_log_prob, gae, targets,
        config)


# ---------------------------------------------------------------------------
# optax chain(clip_by_global_norm, adam) equivalent
# ---------------------------------------------------------------------------

def _tree_leaves(tree):
    """Flatten a nested dict in sorted-key order (jax pytree order)."""
    if isinstance(tree, dict):
        out = []
        for k in sorted(tree.keys()):
            out.extend(_tree_leaves(tree[k]))
        return out
    return [tree]


def _tree_unflatten(structure, leaves):
    """Rebuild a nested dict with ``leaves`` (in sorted-key order)."""
    if isinstance(structure, dict):
        return {k: _tree_unflatten(structure[k], leaves)
                for k in sorted(structure.keys())}
    return leaves.pop(0)


def _tree_map(fn, *trees):
    if isinstance(trees[0], dict):
        return {k: _tree_map(fn, *[t[k] for t in trees]) for k in trees[0]}
    return fn(*trees)


def global_norm(tree):
    leaves = _tree_leaves(tree)
    sq = torch.zeros((), dtype=torch.float32, device=leaves[0].device)
    for leaf in leaves:
        sq = sq + torch.sum(torch.square(leaf))
    return torch.sqrt(sq)


class OptimizerState:
    """State for chain(clip_by_global_norm(max_norm), adam(lr, eps=1e-5))."""

    def __init__(self, params):
        self.count = 0            # scale_by_adam counter
        self.sched_count = 0      # scale_by_schedule counter
        self.mu = _tree_map(lambda p: torch.zeros_like(p, requires_grad=False),
                            params)
        self.nu = _tree_map(lambda p: torch.zeros_like(p, requires_grad=False),
                            params)


@torch.no_grad()
def optimizer_update(grads, opt_state, max_grad_norm, learning_rate,
                     b1=0.9, b2=0.999, eps=1e-5):
    """One optax update: clip_by_global_norm -> scale_by_adam -> -lr scaling.

    ``learning_rate`` may be a float or a schedule callable of the update
    count (matching optax.scale_by_schedule semantics: the count *before*
    increment is used).
    """
    max_norm = float(max_grad_norm)

    # --- clip_by_global_norm
    g_norm = global_norm(grads)
    trigger = bool(g_norm < max_norm)
    if not trigger:
        grads = _tree_map(lambda t: (t / g_norm) * max_norm, grads)

    # --- scale_by_adam
    count_inc = opt_state.count + 1
    mu = _tree_map(lambda g, m: (1 - b1) * g + b1 * m, grads, opt_state.mu)
    nu = _tree_map(lambda g, v: (1 - b2) * g * g + b2 * v, grads, opt_state.nu)
    # bias corrections in float32, matching optax's jnp arithmetic
    bc1 = float(np.float32(1.0) - np.float32(b1) ** np.float32(count_inc))
    bc2 = float(np.float32(1.0) - np.float32(b2) ** np.float32(count_inc))
    updates = _tree_map(
        lambda m, v: (m / bc1) / (torch.sqrt(v / bc2) + eps), mu, nu)

    # --- scale_by_learning_rate (negative sign; schedule uses pre-increment count)
    if callable(learning_rate):
        step_size = -1.0 * float(learning_rate(opt_state.sched_count))
    else:
        step_size = -1.0 * float(learning_rate)
    updates = _tree_map(lambda u: step_size * u, updates)

    opt_state.count = count_inc
    opt_state.sched_count = opt_state.sched_count + 1
    opt_state.mu = mu
    opt_state.nu = nu
    return updates, opt_state


@torch.no_grad()
def apply_updates(params, updates):
    return _tree_map(
        lambda p, u: (p + u).requires_grad_(True), params, updates)
