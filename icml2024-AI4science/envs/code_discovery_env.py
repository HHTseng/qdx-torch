import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional
import itertools
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from simulators import TableauSimulatorCSS
from torch_env_base import Environment, spaces
from inspect import signature
from itertools import combinations
import scipy.special as ss


@dataclass(frozen=True)
class EnvState:
    """
    This class will contain the state of the environment:

    H_X: binary array of size n_qubits_physical*num_H (Hadamards)
    H_Z: binary array of size n_qubits_physical*(n_qubits_logical - num_H)
    time: integer from 0 to max_steps
    """
    S_X: torch.Tensor
    S_Z: torch.Tensor
    time: int

# We ignore this class
@dataclass(frozen=True)
class EnvParams:
    n: int = 7
    k: int = 1
    d: int = 3
    max_steps_in_episode: int = 20


class CodeDiscoveryCSS(Environment):
    """
    Environment for the automatic discovery of QEC CSS codes and encodings.

    Args:
        n_qubits_physical (int): Number of physical qubits available
        n_qubits_logical (int): Number of logical qubits
        code_distance (int): Target code distance
        init_H (list(int)): Positions of initial Hadamard gates
        gates (list(CliffordGates)): List of Clifford gates to prepare the encoding circuit
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 30
        lbda (float, optional): Global rescaling factor for the instantaneous reward. Default: 100
        pI (float, optional): Probability of no error in the noise channel. Default: 0.9
        softness (int, optional): Parameter that controls the size of the stabilizer subgroup to be generated. Default: 1
        bell (bool, optional): Whether we initialize with Bell states or not. Default: False
    """

    def __init__(self,
            n_qubits_physical,
            n_qubits_logical,
            code_distance,
            init_H,
            gates,
            graph=None,
            max_steps = 30,
            lbda = 100,
            pI=0.9,
            softness=1,
            bell = False
                ):
        super().__init__()

        self.n_qubits_physical = n_qubits_physical
        self.n_qubits_logical = n_qubits_logical
        self.gates = gates
        self.max_steps = max_steps
        self.d = code_distance
        self.lbda = lbda # Rescales reward for better convergence
        self.pI = pI # Probability of no error
        self.bell = bell


        self.graph = graph
        if self.graph is None:
            self.graph = []
            ## Fully connected by default
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))


        self.obs_shape = (n_qubits_physical * (n_qubits_physical - n_qubits_logical), )

        # Initialize action tensor
        self.actions = self.action_matrix()

        self.init_H = init_H # Qubit labels where I place an initial H gate
        self.num_H = len(init_H)

        # Initialize error operators
        self.E_mu = self.CSS_error_operators()


        # Initialize error probabilities (rescale with sqrt for improved convergence)
        self.p_mu = torch.sqrt(self.CSS_probabilities())

        # Initialize stabilizer group structure
        self.generate_S_structure(softness) # This generates self.S_struct_X and self.S_struct_Z

        # Initialize num_KL
        self.num_KL = 2 * len(self.p_mu)


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters UNUSED
        return EnvParams()


    def S_struct_loop(self, softness, num):
         # Generate the structure of the stabilizer group (S) based on the softness parameter
         # and the num of generators participating (will generate one struct for X and another for Z)

        # Calculate the number of stabilizer elements
        soft_elements = int(sum([ss.binom(num,i) for i in range(1,softness+1)]))

        # Create an array of zeros
        S_struct = np.zeros((soft_elements, num), dtype=int)

        # Book-keeping
        start_idx = 0

        for m in range(1,softness+1):

            # Update the combinations of positions to place ones in the S structure
            comb = list(combinations(range(num),m))
            indices = np.array(comb)

            # Fill the S structure with ones at the specified indices
            for i,idx in enumerate(indices):
                S_struct[start_idx+i, idx] = 1

            # Update the start_idx
            start_idx += i+1

        # Ensure there are no rows with all zeroes in the S structure
        assert np.prod(np.any(S_struct, axis=1)), "There is a row with all zeroes"

        return S_struct

    def generate_S_structure(self, softness):

        # Helper function that gives the structure of the stabilizer group
        # It's only called once at the very beginning
        num_X = self.num_H
        num_Z = self.n_qubits_physical - self.n_qubits_logical - self.num_H

        # Convert the S structure to uint8 torch tensors for efficient computation
        self.S_struct_X = torch.from_numpy(np.array(self.S_struct_loop(softness, num_X), dtype=np.uint8))
        self.S_struct_Z = torch.from_numpy(np.array(self.S_struct_loop(softness, num_Z), dtype=np.uint8))

        return

    def stabilizer_elements(self, tableau, S_struct):
        # Generate the S matrix by multiplying the S structure with the tableau
        return (S_struct @ tableau) % 2


    def action_matrix(self,
                      params: Optional[EnvParams] = EnvParams) -> torch.Tensor:

        action_matrix = []
        self.action_string = []
        self.action_string_stim = []
        z_action_move = []

        for gate in self.gates:
            ## Two qubit gates. There should not be one-qubit gates
            if len(signature(gate).parameters) == 2:
                for edge in self.graph:
                    action_matrix.append(gate(edge[0], edge[1]))
                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))
                    if edge[0] < edge[1]:
                        z_action_move.append(1)
                    else:
                        z_action_move.append(-1)

        self.z_action_move = np.array(z_action_move, dtype=np.int8)

        return torch.stack(action_matrix).to(torch.uint8)

    def get_observation(self, H_X, H_Z):
        '''
        Extract the check matrix for the observation
        '''
        # Only generators without sign
        return torch.cat([H_X, H_Z]).flatten()


    def CSS_error_operators(self) -> torch.Tensor:
        # We assume same X and Z-type errors to be detected

        size = int(sum([ss.binom(self.n_qubits_physical, w) for w in range(1, self.d)]))

        E_mu = np.zeros((size, self.n_qubits_physical), dtype=np.uint8)

        # Book-keeping
        start_idx = 0

        for w in range(1, self.d):

            # Update the combinations of positions to place the ones
            comb = list(combinations(range(self.n_qubits_physical),w))

            # Convert combinations to a NumPy array
            indices = np.array(comb)

            # Use a loop to set ones at the specified positions
            for i,idx in enumerate(indices):
                E_mu[start_idx+i, idx] = 1


            # Update the start_idx
            start_idx += i+1

        return torch.from_numpy(E_mu)

    def CSS_probabilities(self) -> torch.Tensor:

        p = np.array([self.pI]+[(1-self.pI)/3.]*3, dtype=np.float32)

        p_mu = p[1] ** np.sum(self.E_mu.numpy(), axis=1).astype(np.float32)

        return torch.from_numpy(p_mu)

    def check_KL(self, state: EnvState, params: Optional[EnvParams] = EnvParams):
        # Check the Knill-Laflamme conditions for error correction. This is used to reward the agent

        anticomm_X = torch.any((self.E_mu @ state.S_X.T)%2, dim=1)
        anticomm_Z = torch.any((self.E_mu @ state.S_Z.T)%2, dim=1)

        size = int(ss.binom(self.n_qubits_physical, self.d - 1))

        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        # (vectorized equivalent of jax.vmap(jnp.logical_xor, in_axes=(None, 0)))
        inS_X = torch.logical_xor(state.S_X[None, :, :], self.E_mu[-size:][:, None, :])
        inS_X = torch.logical_not(inS_X).all(dim=-1).to(torch.int32)

        inS_Z = torch.logical_xor(state.S_Z[None, :, :], self.E_mu[-size:][:, None, :])
        inS_Z = torch.logical_not(inS_Z).all(dim=-1).to(torch.int32)

        # Calculate the number of Knill-Laflamme conditions that are not satisfied. This is used for stopping criterion
        self.num_KL = int(2*len(self.p_mu) - torch.sum(anticomm_X, dim=0) - torch.sum(anticomm_Z, dim=0) - torch.sum(inS_X) - torch.sum(inS_Z))

        # Reward is based on the weighted KL sum
        rew = 2 * torch.sum(self.p_mu) - torch.sum(self.p_mu * anticomm_X, dim=0) - torch.sum(self.p_mu * anticomm_Z, dim=0)
        # NOTE: this dangling expression is preserved verbatim from the JAX
        # original, where it is also a no-op statement (missing line
        # continuation), so the inS terms do not contribute to the reward.
        -torch.dot(self.p_mu[-size:], torch.sum(inS_X, dim=-1).to(torch.float32)) -torch.dot(self.p_mu[-size:], torch.sum(inS_Z, dim=-1).to(torch.float32))

        # Return the weighted KL sum rescaled by lbda (as a float32 scalar)
        return np.float32(self.lbda) * np.float32(rew.item())


    def step_env(
        self, key, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[torch.Tensor, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""

        # prev_terminal = self.is_terminal(state, params)
        x_label = action
        z_label = action + self.z_action_move[action]

        # Update state
        state = EnvState( (state.S_X @ self.actions[x_label]) % 2, (state.S_Z @ self.actions[z_label]) % 2, state.time + 1)

        # Update KLs
        reward = -self.check_KL(state)

        # Reward extra if encoding was successful
        reward += 1 * (abs(self.num_KL) <= 1e-7)

        # Evaluate termination conditions
        done = self.is_terminal(state, params)

        return (
            self.get_obs(state),
            state,
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key, params: EnvParams
    ) -> Tuple[torch.Tensor, EnvState]:
        """Performs resetting of environment."""

        tableau = TableauSimulatorCSS(self.n_qubits_physical, self.n_qubits_logical, self.init_H, bell=self.bell)
        S_X = self.stabilizer_elements(tableau.H_X, self.S_struct_X)
        S_Z = self.stabilizer_elements(tableau.H_Z, self.S_struct_Z)

        state = EnvState(
            S_X = S_X,
            S_Z = S_Z,
            time = 0
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> torch.Tensor:
        """Applies observation function to state."""
        H_X = state.S_X[:self.num_H]
        H_Z = state.S_Z[:self.n_qubits_physical - self.n_qubits_logical - self.num_H]

        return self.get_observation(H_X, H_Z).flatten()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done_encoding = (abs(self.num_KL) <= 1e-6)

        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps

        done = done_encoding or done_steps
        return bool(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "CodeDiscoveryCSS"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """Number of actions possible in environment."""
        return self.actions.shape[0]

    def action_space(
        self, params: Optional[EnvParams] = EnvParams
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""

        return spaces.Box(0, 1, self.obs_shape, dtype=torch.float16)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        return spaces.Dict(
            {
                "tableau": spaces.Box(0, 1, self.obs_shape, torch.float16),
                "time": spaces.Discrete(params.max_steps),
            }
        )
