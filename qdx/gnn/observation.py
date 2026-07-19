"""Padded heterogeneous graph observations for GNN-QDX v1.2 (PyTorch port).

Faithful conversion of the JAX/flax original: identical feature definitions,
padding layout, action ordering, and float32 arithmetic — but every array is
an eagerly evaluated ``torch.Tensor`` and ``GraphObservation`` is a
``NamedTuple`` so it can be stacked/indexed/reshaped with the small tree
helpers at the bottom of this module.
"""

from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch


CHECK_S_TO_Q = 0
CHECK_Q_TO_S = 1
HW_Q_TO_Q = 2
NUM_RELATION_TYPES = 3

SINGLE_ACTION = 0
TWO_QUBIT_ACTION = 1

NODE_FEATURE_DIM = 21
CHECK_EDGE_FEATURE_DIM = 5
HW_EDGE_FEATURE_DIM = 8
EDGE_FEATURE_DIM = CHECK_EDGE_FEATURE_DIM + HW_EDGE_FEATURE_DIM
GLOBAL_FEATURE_DIM = 9
EPSILON = 1.0e-6

GATE_NAME_ALIASES = {"CX": "CNOT"}


class GraphObservation(NamedTuple):
    """One fixed-shape graph observation; leading batch axes may be added."""

    node_features: torch.Tensor
    edge_features: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    relation_ids: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    qubit_mask: torch.Tensor
    stabilizer_mask: torch.Tensor
    global_features: torch.Tensor
    action_types: torch.Tensor
    action_gate_ids: torch.Tensor
    action_first: torch.Tensor
    action_second: torch.Tensor
    action_edge_indices: torch.Tensor
    action_mask: torch.Tensor
    action_env_indices: torch.Tensor


def obs_map(fn: Callable, *observations: GraphObservation) -> GraphObservation:
    """Apply ``fn`` leaf-wise across one or more GraphObservations."""

    return GraphObservation(
        *[fn(*leaves) for leaves in zip(*[tuple(o) for o in observations])]
    )


def obs_stack(observations: Sequence[GraphObservation]) -> GraphObservation:
    """Stack observations along a new leading axis (like jax.vmap batching)."""

    return obs_map(lambda *leaves: torch.stack(leaves), *observations)


def obs_index(observation: GraphObservation, index) -> GraphObservation:
    """Index the leading axis of every leaf."""

    return obs_map(lambda leaf: leaf[index], observation)


def obs_reshape_lead(observation: GraphObservation, new_lead: Tuple[int, ...]) -> GraphObservation:
    """Reshape two leading axes (T, B) into ``new_lead`` for every leaf."""

    return obs_map(
        lambda leaf: leaf.reshape(tuple(new_lead) + tuple(leaf.shape[2:])),
        observation,
    )


def obs_take(observation: GraphObservation, indices: torch.Tensor) -> GraphObservation:
    """Take along the leading axis of every leaf."""

    return obs_map(lambda leaf: leaf[indices], observation)


def obs_concat(observations: Sequence[GraphObservation], dim: int) -> GraphObservation:
    """Concatenate observations along an existing axis."""

    return obs_map(lambda *leaves: torch.cat(leaves, dim=dim), *observations)


def obs_to_device(observation: GraphObservation, device) -> GraphObservation:
    return obs_map(lambda leaf: leaf.to(device), observation)


@dataclass(frozen=True)
class GraphPadding:
    """Static bucket sizes used only for array shapes, never for model parameters."""

    n_max: int
    stabilizers_max: Optional[int] = None
    hardware_edges_max: Optional[int] = None
    actions_max: Optional[int] = None

    @property
    def resolved_stabilizers_max(self) -> int:
        return self.n_max if self.stabilizers_max is None else self.stabilizers_max

    @property
    def resolved_hardware_edges_max(self) -> int:
        if self.hardware_edges_max is None:
            return self.n_max * max(self.n_max - 1, 0)
        return self.hardware_edges_max


@dataclass(frozen=True)
class ActionDescriptor:
    """Host-side description of one environment-compatible candidate action."""

    action_type: str
    gate: str
    qubit: Optional[int] = None
    control: Optional[int] = None
    target: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.action_type, "gate": self.gate}
        if self.action_type == "single":
            result["qubit"] = self.qubit
        else:
            result["control"] = self.control
            result["target"] = self.target
        return result


class GraphObservationBuilder:
    """Turn a stabilizer check matrix and hardware graph into a padded graph."""

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        max_steps: int,
        gates: Sequence[Any],
        hardware_edges: Sequence[Tuple[int, int]],
        padding: Optional[GraphPadding] = None,
    ):
        self.n = int(n)
        self.k = int(k)
        self.d = int(d)
        self.num_stabilizers = self.n - self.k
        self.max_steps = int(max_steps)
        self.gates = tuple(gates)
        self.hardware_edges = tuple((int(i), int(j)) for i, j in hardware_edges)
        self.padding = padding or GraphPadding(n_max=self.n)

        self.n_max = self.padding.n_max
        self.stabilizers_max = self.padding.resolved_stabilizers_max
        self.hardware_edges_max = self.padding.resolved_hardware_edges_max
        if self.n > self.n_max:
            raise ValueError(f"n={self.n} exceeds graph bucket n_max={self.n_max}")
        if self.num_stabilizers > self.stabilizers_max:
            raise ValueError("number of stabilizers exceeds graph bucket capacity")
        if len(self.hardware_edges) > self.hardware_edges_max:
            raise ValueError("hardware edge count exceeds graph bucket capacity")

        self.gate_names = tuple(
            GATE_NAME_ALIASES.get(gate.__name__.upper(), gate.__name__.upper())
            for gate in self.gates
        )
        self.gate_arities = tuple(len(signature(gate).parameters) for gate in self.gates)
        if any(arity not in (1, 2) for arity in self.gate_arities):
            raise ValueError("GNN-QDX v1.2 supports only one- and two-qubit gates")
        self.num_gate_types = len(self.gates)

        self.max_nodes = self.n_max + self.stabilizers_max
        self.max_edges = (
            2 * self.n_max * self.stabilizers_max + self.hardware_edges_max
        )
        default_actions_max = sum(
            self.n_max if arity == 1 else self.hardware_edges_max
            for arity in self.gate_arities
        )
        self.max_actions = (
            default_actions_max
            if self.padding.actions_max is None
            else self.padding.actions_max
        )

        self.action_descriptors = self._build_action_descriptors()
        if len(self.action_descriptors) > self.max_actions:
            raise ValueError("candidate action count exceeds graph bucket capacity")
        self._build_static_arrays()

    def _build_action_descriptors(self) -> Tuple[ActionDescriptor, ...]:
        descriptors = []
        for gate_name, arity in zip(self.gate_names, self.gate_arities):
            if arity == 1:
                descriptors.extend(
                    ActionDescriptor("single", gate_name, qubit=i)
                    for i in range(self.n)
                )
            else:
                descriptors.extend(
                    ActionDescriptor("two", gate_name, control=i, target=j)
                    for i, j in self.hardware_edges
                )
        return tuple(descriptors)

    def _build_static_arrays(self) -> None:
        node_mask = np.zeros(self.max_nodes, dtype=bool)
        qubit_mask = np.zeros(self.max_nodes, dtype=bool)
        stabilizer_mask = np.zeros(self.max_nodes, dtype=bool)
        node_mask[: self.n] = True
        qubit_mask[: self.n] = True
        stab_start = self.n_max
        node_mask[stab_start : stab_start + self.num_stabilizers] = True
        stabilizer_mask[stab_start : stab_start + self.num_stabilizers] = True
        self._node_mask = torch.from_numpy(node_mask)
        self._qubit_mask = torch.from_numpy(qubit_mask)
        self._stabilizer_mask = torch.from_numpy(stabilizer_mask)

        check_pairs = self.n * self.num_stabilizers
        q = np.tile(np.arange(self.n, dtype=np.int32), self.num_stabilizers)
        s = np.repeat(
            np.arange(self.num_stabilizers, dtype=np.int32), self.n
        ) + stab_start
        senders = np.zeros(self.max_edges, dtype=np.int32)
        receivers = np.zeros(self.max_edges, dtype=np.int32)
        relations = np.zeros(self.max_edges, dtype=np.int32)
        senders[:check_pairs], receivers[:check_pairs] = s, q
        relations[:check_pairs] = CHECK_S_TO_Q
        senders[check_pairs : 2 * check_pairs] = q
        receivers[check_pairs : 2 * check_pairs] = s
        relations[check_pairs : 2 * check_pairs] = CHECK_Q_TO_S
        hw_offset = 2 * check_pairs
        for offset, (i, j) in enumerate(self.hardware_edges):
            if not (0 <= i < self.n and 0 <= j < self.n):
                raise ValueError(f"invalid hardware edge {(i, j)} for n={self.n}")
            senders[hw_offset + offset] = i
            receivers[hw_offset + offset] = j
            relations[hw_offset + offset] = HW_Q_TO_Q
        self._senders = torch.from_numpy(senders)
        self._receivers = torch.from_numpy(receivers)
        self._relation_ids = torch.from_numpy(relations)
        self._check_pairs = check_pairs
        self._hw_offset = hw_offset
        self._hw_src = torch.tensor(
            [i for i, _ in self.hardware_edges], dtype=torch.int64
        )
        self._hw_dst = torch.tensor(
            [j for _, j in self.hardware_edges], dtype=torch.int64
        )
        self._hw_edge_indices = torch.arange(
            self._hw_offset,
            self._hw_offset + len(self.hardware_edges),
            dtype=torch.int32,
        )

        neighbors = [set() for _ in range(self.n)]
        for i, j in self.hardware_edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
        hardware_degree = np.asarray(
            [len(node_neighbors) for node_neighbors in neighbors], dtype=np.float32
        )
        self._hardware_degree = torch.from_numpy(hardware_degree)
        self._normalized_hw_degree = torch.from_numpy(
            hardware_degree / float(max(self.n - 1, 1))
        )

        action_types = np.zeros(self.max_actions, dtype=np.int32)
        gate_ids = np.zeros(self.max_actions, dtype=np.int32)
        first = np.zeros(self.max_actions, dtype=np.int32)
        second = np.zeros(self.max_actions, dtype=np.int32)
        edge_indices = np.zeros(self.max_actions, dtype=np.int32)
        action_mask = np.zeros(self.max_actions, dtype=bool)
        env_indices = np.full(self.max_actions, -1, dtype=np.int32)
        cursor = 0
        for gate_id, arity in enumerate(self.gate_arities):
            if arity == 1:
                for i in range(self.n):
                    gate_ids[cursor] = gate_id
                    first[cursor] = i
                    cursor += 1
            else:
                for hw_index, (i, j) in enumerate(self.hardware_edges):
                    action_types[cursor] = TWO_QUBIT_ACTION
                    gate_ids[cursor] = gate_id
                    first[cursor] = i
                    second[cursor] = j
                    edge_indices[cursor] = hw_offset + hw_index
                    cursor += 1
        action_mask[:cursor] = True
        env_indices[:cursor] = np.arange(cursor, dtype=np.int32)
        self._action_types = torch.from_numpy(action_types)
        self._action_gate_ids = torch.from_numpy(gate_ids)
        self._action_first = torch.from_numpy(first)
        self._action_second = torch.from_numpy(second)
        self._action_edge_indices = torch.from_numpy(edge_indices)
        self._action_mask = torch.from_numpy(action_mask)
        self._action_env_indices = torch.from_numpy(env_indices)

    def build(self, check_matrix, time) -> GraphObservation:
        """Build the graph observation with torch float32 arithmetic."""

        check_matrix = torch.as_tensor(check_matrix).to(torch.float32).reshape(
            self.num_stabilizers, 2 * self.n
        )
        h_x = check_matrix[:, : self.n]
        h_z = check_matrix[:, self.n :]

        x_only = torch.logical_and(h_x != 0, h_z == 0)
        z_only = torch.logical_and(h_x == 0, h_z != 0)
        y_like = torch.logical_and(h_x != 0, h_z != 0)
        touched = torch.logical_or(torch.logical_or(x_only, z_only), y_like)

        x_only_f = x_only.to(torch.float32)
        z_only_f = z_only.to(torch.float32)
        y_like_f = y_like.to(torch.float32)
        touched_f = touched.to(torch.float32)

        stabilizer_denominator = float(max(self.num_stabilizers, 1))
        qubit_denominator = float(max(self.n, 1))

        x_degree = torch.sum(x_only_f, dim=0)
        z_degree = torch.sum(z_only_f, dim=0)
        y_degree = torch.sum(y_like_f, dim=0)
        total_check_degree = torch.sum(touched_f, dim=0)
        load_frac = total_check_degree / stabilizer_denominator
        xz_balance = (x_degree - z_degree) / (total_check_degree + EPSILON)

        x_weight = torch.sum(x_only_f, dim=1)
        z_weight = torch.sum(z_only_f, dim=1)
        y_weight = torch.sum(y_like_f, dim=1)
        total_weight = torch.sum(touched_f, dim=1)

        if self.n > 0:
            mean_qubit_check_degree = torch.mean(total_check_degree)
            mean_x_degree = torch.mean(x_degree)
            mean_z_degree = torch.mean(z_degree)
            mean_y_degree = torch.mean(y_degree)
            std_qubit_check_degree = torch.std(total_check_degree, correction=0)
            mean_hardware_degree = torch.mean(self._hardware_degree)
        else:
            mean_qubit_check_degree = torch.tensor(0.0)
            mean_x_degree = torch.tensor(0.0)
            mean_z_degree = torch.tensor(0.0)
            mean_y_degree = torch.tensor(0.0)
            std_qubit_check_degree = torch.tensor(0.0)
            mean_hardware_degree = torch.tensor(0.0)

        if self.num_stabilizers > 0:
            mean_stabilizer_weight = torch.mean(total_weight)
            mean_x_weight = torch.mean(x_weight)
            mean_z_weight = torch.mean(z_weight)
            mean_y_weight = torch.mean(y_weight)
            std_stabilizer_weight = torch.std(total_weight, correction=0)
        else:
            mean_stabilizer_weight = torch.tensor(0.0)
            mean_x_weight = torch.tensor(0.0)
            mean_z_weight = torch.tensor(0.0)
            mean_y_weight = torch.tensor(0.0)
            std_stabilizer_weight = torch.tensor(0.0)

        node_features = torch.zeros(
            (self.max_nodes, NODE_FEATURE_DIM), dtype=torch.float32
        )
        qubit_features = torch.stack(
            [
                torch.ones(self.n),
                torch.zeros(self.n),
                total_check_degree / stabilizer_denominator,
                torch.sum(h_x, dim=0) / stabilizer_denominator,
                torch.sum(h_z, dim=0) / stabilizer_denominator,
                x_degree / stabilizer_denominator,
                z_degree / stabilizer_denominator,
                y_degree / stabilizer_denominator,
                torch.log1p(total_check_degree),
                total_check_degree / (mean_qubit_check_degree + EPSILON),
                x_degree / (mean_x_degree + EPSILON),
                z_degree / (mean_z_degree + EPSILON),
                y_degree / (mean_y_degree + EPSILON),
                self._normalized_hw_degree,
                torch.log1p(self._hardware_degree),
                x_degree / (total_check_degree + EPSILON),
                z_degree / (total_check_degree + EPSILON),
                y_degree / (total_check_degree + EPSILON),
                (x_degree - z_degree) / (total_check_degree + EPSILON),
                torch.zeros(self.n),
                torch.zeros(self.n),
            ],
            dim=-1,
        )
        stabilizer_features = torch.stack(
            [
                torch.zeros(self.num_stabilizers),
                torch.ones(self.num_stabilizers),
                total_weight / qubit_denominator,
                torch.sum(h_x, dim=1) / qubit_denominator,
                torch.sum(h_z, dim=1) / qubit_denominator,
                torch.zeros(self.num_stabilizers),
                torch.zeros(self.num_stabilizers),
                torch.zeros(self.num_stabilizers),
                torch.zeros(self.num_stabilizers),
                torch.zeros(self.num_stabilizers),
                x_weight / (mean_x_weight + EPSILON),
                z_weight / (mean_z_weight + EPSILON),
                y_weight / (mean_y_weight + EPSILON),
                torch.zeros(self.num_stabilizers),
                torch.zeros(self.num_stabilizers),
                x_weight / (total_weight + EPSILON),
                z_weight / (total_weight + EPSILON),
                y_weight / (total_weight + EPSILON),
                (x_weight - z_weight) / (total_weight + EPSILON),
                torch.log1p(total_weight),
                total_weight / (mean_stabilizer_weight + EPSILON),
            ],
            dim=-1,
        )
        node_features[: self.n] = qubit_features
        node_features[
            self.n_max : self.n_max + self.num_stabilizers
        ] = stabilizer_features

        edge_features = torch.zeros((self.max_edges, EDGE_FEATURE_DIM), dtype=torch.float32)
        edge_mask = torch.zeros(self.max_edges, dtype=torch.bool)
        check_features = torch.stack(
            [
                h_x.reshape(-1),
                h_z.reshape(-1),
                x_only_f.reshape(-1),
                z_only_f.reshape(-1),
                y_like_f.reshape(-1),
            ],
            dim=-1,
        )
        src_touched = touched_f[:, self._hw_src]
        dst_touched = touched_f[:, self._hw_dst]
        src_x_only = x_only_f[:, self._hw_src]
        dst_x_only = x_only_f[:, self._hw_dst]
        src_y_like = y_like_f[:, self._hw_src]
        dst_y_like = y_like_f[:, self._hw_dst]
        src_z_only = z_only_f[:, self._hw_src]
        dst_z_only = z_only_f[:, self._hw_dst]

        def _jaccard(src_values, dst_values):
            intersection = torch.sum(src_values * dst_values, dim=0)
            union = torch.sum(torch.maximum(src_values, dst_values), dim=0)
            return intersection / (union + EPSILON)

        hardware_features = torch.stack(
            [
                torch.log1p(self._hardware_degree[self._hw_src]),
                torch.log1p(self._hardware_degree[self._hw_dst]),
                _jaccard(src_touched, dst_touched),
                _jaccard(src_x_only, dst_x_only),
                _jaccard(src_y_like, dst_y_like),
                _jaccard(src_z_only, dst_z_only),
                load_frac[self._hw_src] - load_frac[self._hw_dst],
                xz_balance[self._hw_src] - xz_balance[self._hw_dst],
            ],
            dim=-1,
        )
        touched_flat = touched.reshape(-1)
        edge_features[: self._check_pairs, :CHECK_EDGE_FEATURE_DIM] = check_features
        edge_features[
            self._check_pairs : 2 * self._check_pairs, :CHECK_EDGE_FEATURE_DIM
        ] = check_features
        edge_features[
            self._hw_offset : self._hw_offset + len(self.hardware_edges),
            CHECK_EDGE_FEATURE_DIM:,
        ] = hardware_features
        edge_mask[: self._check_pairs] = touched_flat
        edge_mask[self._check_pairs : 2 * self._check_pairs] = touched_flat
        edge_mask[
            self._hw_offset : self._hw_offset + len(self.hardware_edges)
        ] = True

        time_f = float(time)
        max_steps = float(max(self.max_steps, 1))
        global_features = torch.stack(
            [
                torch.tensor(time_f / max_steps, dtype=torch.float32),
                torch.tensor((max_steps - time_f) / max_steps, dtype=torch.float32),
                torch.tensor(float(self.k) / float(max(self.n, 1)), dtype=torch.float32),
                torch.tensor(float(self.d) / float(max(self.n, 1)), dtype=torch.float32),
                torch.log1p(mean_stabilizer_weight),
                std_stabilizer_weight / (mean_stabilizer_weight + EPSILON),
                torch.log1p(mean_qubit_check_degree),
                std_qubit_check_degree / (mean_qubit_check_degree + EPSILON),
                torch.log1p(mean_hardware_degree),
            ]
        ).to(torch.float32)
        return GraphObservation(
            node_features=node_features,
            edge_features=edge_features,
            senders=self._senders,
            receivers=self._receivers,
            relation_ids=self._relation_ids,
            node_mask=self._node_mask,
            edge_mask=edge_mask,
            qubit_mask=self._qubit_mask,
            stabilizer_mask=self._stabilizer_mask,
            global_features=global_features,
            action_types=self._action_types,
            action_gate_ids=self._action_gate_ids,
            action_first=self._action_first,
            action_second=self._action_second,
            action_edge_indices=self._action_edge_indices,
            action_mask=self._action_mask,
            action_env_indices=self._action_env_indices,
        )

    def empty_observation(self) -> GraphObservation:
        check = torch.zeros((self.num_stabilizers, 2 * self.n), dtype=torch.uint8)
        return self.build(check, 0)

    def action_descriptor(self, action_index: int) -> Dict[str, Any]:
        """Map an actor index to the gate descriptor used by the QDX environment."""

        if not 0 <= int(action_index) < len(self.action_descriptors):
            raise IndexError(f"invalid or padded action index: {action_index}")
        return self.action_descriptors[int(action_index)].as_dict()
