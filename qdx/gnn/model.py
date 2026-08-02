"""PyTorch implementation of the GNN-QDX v1.4 actor-critic.

Functional port of the flax original: parameters live in a flax-style nested
dict (``params/node_embed_mlp/dense_0/{kernel,bias}`` ...), initialization
reproduces flax's defaults (lecun-normal kernels via the bit-exact threefry
PRNG in :mod:`qdx.torch_random`, zero biases), and the forward pass mirrors
the JAX computation op-for-op so results match within float32 round-off.
Gradients come from ``torch.autograd``.
"""

from typing import Sequence

import numpy as np
import torch

from qdx import torch_random
from qdx.torch_nn import Categorical
from qdx.gnn.observation import (
    CHECK_Q_TO_S,
    CHECK_S_TO_Q,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    GraphObservation,
    HW_Q_TO_Q,
    NODE_FEATURE_DIM,
    NUM_RELATION_TYPES,
    TWO_QUBIT_ACTION,
)


def _mlp_apply(params, x, activation):
    """Apply a flax-style MLP param subtree {dense_0: {kernel, bias}, ...}."""

    act = torch.relu if activation == "relu" else torch.tanh
    layer_names = sorted(params.keys(), key=lambda name: int(name.split("_")[1]))
    for index, name in enumerate(layer_names):
        layer = params[name]
        x = x @ layer["kernel"] + layer["bias"]
        if index + 1 < len(layer_names):
            x = act(x)
    return x


def _mlp_init(rng, path_prefix, in_dim, features):
    """Initialize a flax MLP subtree with lecun-normal kernels, zero biases."""

    params = {}
    fan_in = in_dim
    for index, out_dim in enumerate(features):
        name = f"dense_{index}"
        kernel_key = torch_random.flax_fold_in_static(
            rng, tuple(path_prefix) + (name, 1)
        )
        params[name] = {
            "kernel": torch.from_numpy(
                torch_random.lecun_normal_init(kernel_key, (fan_in, out_dim))
            ).requires_grad_(True),
            "bias": torch.zeros(out_dim, dtype=torch.float32, requires_grad=True),
        }
        fan_in = out_dim
    return params


def one_hot(indices, num_classes, dtype=torch.float32):
    indices = torch.as_tensor(indices)
    classes = torch.arange(num_classes, device=indices.device)
    return (indices[..., None] == classes).to(dtype)


def _masked_mean(values, mask):
    weights = mask.to(values.dtype)[..., :, None]
    return torch.sum(values * weights, dim=-2) / torch.clamp(
        torch.sum(weights, dim=-2), min=1.0
    )


def _gather_nodes(nodes, indices):
    """Gather from the node axis while preserving arbitrary leading batch axes."""

    indices = indices.to(torch.int64)
    indices = torch.broadcast_to(indices, nodes.shape[:-2] + indices.shape[-1:])
    gather_indices = indices[..., :, None].expand(
        indices.shape + (nodes.shape[-1],)
    )
    return torch.gather(nodes, -2, gather_indices)


def _aggregate_messages(messages, receivers, edge_mask, num_nodes):
    """Aggregate edge messages into receiver nodes using scatter-add."""

    batch_shape = messages.shape[:-2]
    hidden = messages.shape[-1]
    receivers = torch.broadcast_to(
        receivers.to(torch.int64), messages.shape[:-1]
    )
    edge_mask = torch.broadcast_to(edge_mask, messages.shape[:-1])

    flat_messages = messages.reshape((-1, messages.shape[-2], hidden))
    flat_receivers = receivers.reshape((-1, receivers.shape[-1]))
    flat_edge_mask = edge_mask.reshape((-1, edge_mask.shape[-1]))

    edge_weights = flat_edge_mask.to(flat_messages.dtype)
    weighted = flat_messages * edge_weights[..., None]
    index = flat_receivers[..., None].expand(flat_receivers.shape + (hidden,))
    summed = torch.zeros(
        (flat_messages.shape[0], num_nodes, hidden), dtype=flat_messages.dtype,
        device=flat_messages.device,
    ).scatter_add_(1, index, weighted)
    count = torch.zeros(
        (flat_messages.shape[0], num_nodes), dtype=flat_messages.dtype,
        device=flat_messages.device,
    ).scatter_add_(1, flat_receivers, edge_weights)
    aggregated = summed / torch.clamp(count[..., None], min=1.0)
    return aggregated.reshape(batch_shape + (num_nodes, hidden))


def _edge_group_masks(relation_ids, edge_mask):
    """Split valid edges into check-edge and hardware-edge groups."""

    check_mask = edge_mask & (
        (relation_ids == CHECK_S_TO_Q) | (relation_ids == CHECK_Q_TO_S)
    )
    hw_mask = edge_mask & (relation_ids == HW_Q_TO_Q)
    return check_mask, hw_mask


class GNNQDXActorCritic:
    """Variable-size candidate-scoring actor and global-state critic.

    Provides flax-style ``init(rng, obs)`` and ``apply(params, obs)``.
    """

    def __init__(
        self,
        num_gate_types,
        hidden_dim: int = 64,
        relation_dim: int = 8,
        gate_dim: int = 8,
        num_gnn_layers: int = 3,
        activation: str = "tanh",
    ):
        self.num_gate_types = int(num_gate_types)
        self.hidden_dim = int(hidden_dim)
        self.relation_dim = int(relation_dim)
        self.gate_dim = int(gate_dim)
        self.num_gnn_layers = int(num_gnn_layers)
        self.activation = activation

    def init(self, rng, graph_obs: GraphObservation):
        h = self.hidden_dim
        params = {}
        params["node_embed_mlp"] = _mlp_init(
            rng, ("node_embed_mlp",), NODE_FEATURE_DIM, (h, h))
        params["global_embed_mlp"] = _mlp_init(
            rng, ("global_embed_mlp",), GLOBAL_FEATURE_DIM, (h, h))
        params["edge_embed_mlp"] = _mlp_init(
            rng, ("edge_embed_mlp",), EDGE_FEATURE_DIM + NUM_RELATION_TYPES, (h, h))

        gate_kernel_key = torch_random.flax_fold_in_static(
            rng, ("gate_embed_linear", 1))
        params["gate_embed_linear"] = {
            "kernel": torch.from_numpy(
                torch_random.lecun_normal_init(
                    gate_kernel_key, (self.num_gate_types, self.gate_dim))
            ).requires_grad_(True),
            "bias": torch.zeros(self.gate_dim, dtype=torch.float32, requires_grad=True),
        }

        for layer in range(self.num_gnn_layers):
            params[f"edge_message_mlp_{layer}"] = _mlp_init(
                rng, (f"edge_message_mlp_{layer}",), 4 * h, (h, h))
            params[f"node_update_mlp_{layer}"] = _mlp_init(
                rng, (f"node_update_mlp_{layer}",), 4 * h, (h, h))
            params[f"global_update_mlp_{layer}"] = _mlp_init(
                rng, (f"global_update_mlp_{layer}",), 5 * h, (h, h))

        params["single_action_mlp"] = _mlp_init(
            rng, ("single_action_mlp",), 2 * h + self.gate_dim, (h, 1))
        params["two_action_mlp"] = _mlp_init(
            rng, ("two_action_mlp",), 3 * h + self.gate_dim, (h, 1))
        params["value_mlp"] = _mlp_init(rng, ("value_mlp",), h, (h, 1))
        return {"params": params}

    def apply(self, params, graph_obs: GraphObservation):
        p = params["params"]
        activation = self.activation

        h = _mlp_apply(p["node_embed_mlp"], graph_obs.node_features, activation)
        g = _mlp_apply(p["global_embed_mlp"], graph_obs.global_features, activation)
        relation_onehot = one_hot(
            graph_obs.relation_ids,
            NUM_RELATION_TYPES,
            dtype=graph_obs.edge_features.dtype,
        )
        edge_h = _mlp_apply(
            p["edge_embed_mlp"],
            torch.cat([graph_obs.edge_features, relation_onehot], dim=-1),
            activation,
        )
        check_edge_mask, hw_edge_mask = _edge_group_masks(
            graph_obs.relation_ids, graph_obs.edge_mask
        )
        check_edge_pool = _masked_mean(edge_h, check_edge_mask)
        hw_edge_pool = _masked_mean(edge_h, hw_edge_mask)

        gate_onehot = one_hot(
            graph_obs.action_gate_ids,
            self.num_gate_types,
            dtype=graph_obs.node_features.dtype,
        )
        gate_h = gate_onehot @ p["gate_embed_linear"]["kernel"] + p["gate_embed_linear"]["bias"]

        node_mask_f = graph_obs.node_mask.to(h.dtype)[..., :, None]
        h = h * node_mask_f
        for layer in range(self.num_gnn_layers):
            sender_h = _gather_nodes(h, graph_obs.senders)
            receiver_h = _gather_nodes(h, graph_obs.receivers)
            g_edges = torch.broadcast_to(
                g[..., None, :], sender_h.shape[:-1] + (g.shape[-1],)
            )
            message_input = torch.cat(
                [
                    sender_h,
                    receiver_h,
                    edge_h,
                    g_edges,
                ],
                dim=-1,
            )
            messages = _mlp_apply(
                p[f"edge_message_mlp_{layer}"], message_input, activation
            )
            agg_check = _aggregate_messages(
                messages, graph_obs.receivers, check_edge_mask, h.shape[-2]
            )
            agg_hw = _aggregate_messages(
                messages, graph_obs.receivers, hw_edge_mask, h.shape[-2]
            )

            g_nodes = torch.broadcast_to(
                g[..., None, :], h.shape[:-1] + (g.shape[-1],)
            )
            node_delta = _mlp_apply(
                p[f"node_update_mlp_{layer}"],
                torch.cat([h, agg_check, agg_hw, g_nodes], dim=-1),
                activation,
            )
            h = (h + node_delta) * node_mask_f

            q_pool = _masked_mean(h, graph_obs.qubit_mask)
            s_pool = _masked_mean(h, graph_obs.stabilizer_mask)
            global_delta = _mlp_apply(
                p[f"global_update_mlp_{layer}"],
                torch.cat(
                    [g, q_pool, s_pool, check_edge_pool, hw_edge_pool], dim=-1
                ),
                activation,
            )
            g = g + global_delta

        first_h = _gather_nodes(h, graph_obs.action_first)
        second_h = _gather_nodes(h, graph_obs.action_second)
        g_actions = torch.broadcast_to(
            g[..., None, :], first_h.shape[:-1] + (g.shape[-1],)
        )
        single_logits = _mlp_apply(
            p["single_action_mlp"],
            torch.cat([first_h, gate_h, g_actions], dim=-1),
            activation,
        )[..., 0]
        # v1.4: score symmetric two-qubit gates in both qubit orders with the
        # same MLP and average, so CZ/SQRT_XX logits are order invariant.
        two_forward_logits = _mlp_apply(
            p["two_action_mlp"],
            torch.cat([first_h, second_h, gate_h, g_actions], dim=-1),
            activation,
        )[..., 0]
        two_reverse_logits = _mlp_apply(
            p["two_action_mlp"],
            torch.cat([second_h, first_h, gate_h, g_actions], dim=-1),
            activation,
        )[..., 0]
        two_logits = torch.where(
            graph_obs.action_is_symmetric,
            0.5 * (two_forward_logits + two_reverse_logits),
            two_forward_logits,
        )
        logits = torch.where(
            graph_obs.action_types == TWO_QUBIT_ACTION, two_logits, single_logits
        )
        logits = torch.where(
            graph_obs.action_mask,
            logits,
            torch.tensor(-1.0e9, dtype=logits.dtype, device=logits.device),
        )

        value = _mlp_apply(p["value_mlp"], g, activation)
        return Categorical(logits=logits), torch.squeeze(value, dim=-1)
