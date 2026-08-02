"""Graph-observation interface for GNN-QDX environment dynamics (torch)."""

from typing import Optional

import torch

from qdx import torch_random
from qdx.torch_env_base import spaces
from qdx.envs.code_discovery import CodeDiscovery, EnvParams, EnvState
from qdx.gnn.observation import (
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphObservation,
    GraphObservationBuilder,
    GraphPadding,
)


class GraphCodeDiscovery(CodeDiscovery):
    """CodeDiscovery with padded GraphObservation outputs.

    Reward calculation, terminal logic, and tableau transitions are inherited from
    CodeDiscovery; graph observations expose the v1.4 dynamic action mask.
    """

    def __init__(self, *args, graph_padding: Optional[GraphPadding] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_builder = GraphObservationBuilder(
            n=self.n_qubits_physical,
            k=self.n_qubits_logical,
            d=self.d,
            max_steps=self.max_steps,
            gates=self.gates,
            hardware_edges=self.graph,
            padding=graph_padding,
        )
        if len(self.graph_builder.action_descriptors) != self.num_actions:
            raise ValueError("graph action ordering does not match environment actions")
        self._configure_action_relations(self.graph_builder.max_actions)

    def get_obs(
        self, state: EnvState, params: Optional[EnvParams] = None
    ) -> GraphObservation:
        check_matrix = self.get_observation(state.tableau)
        return self.graph_builder.build(
            check_matrix,
            state.time,
            state.pending_action_mask,
        )

    def step(self, key, state, action, params=None):
        """Gymnax auto-reset with PyTree-aware graph observation selection."""

        if params is None:
            params = self.default_params
        keys = torch_random.split(key)
        key_step, key_reset = keys[0], keys[1]
        obs_step, state_step, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        if done:
            obs, state = self.reset_env(key_reset, params)
        else:
            obs, state = obs_step, state_step
        return obs, state, reward, done, info

    def graph_observation_template(self) -> GraphObservation:
        return self.graph_builder.empty_observation()

    def action_descriptor(self, action_index: int):
        return self.graph_builder.action_descriptor(action_index)

    def gate_matrix_for_action(self, action_index: int) -> torch.Tensor:
        if not 0 <= int(action_index) < self.num_actions:
            raise IndexError(f"invalid action index: {action_index}")
        return self.actions[int(action_index)]

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Dict:
        builder = self.graph_builder
        return spaces.Dict(
            {
                "node_features": spaces.Box(
                    -1.0e9,
                    1.0e9,
                    (builder.max_nodes, NODE_FEATURE_DIM),
                    dtype=torch.float32,
                ),
                "edge_features": spaces.Box(
                    -1.0e9,
                    1.0e9,
                    (builder.max_edges, EDGE_FEATURE_DIM),
                    dtype=torch.float32,
                ),
                "global_features": spaces.Box(
                    -1.0e9,
                    1.0e9,
                    (GLOBAL_FEATURE_DIM,),
                    dtype=torch.float32,
                ),
                "action_mask": spaces.Box(
                    0, 1, (builder.max_actions,), dtype=torch.bool
                ),
            }
        )
