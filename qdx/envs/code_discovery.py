from qdx.torch_env_base import Environment, spaces
import numpy as np
import torch
from dataclasses import dataclass
from inspect import signature
from typing import Tuple, Optional
import itertools
from qdx.simulators import TableauSimulator
import stim
from more_itertools import distinct_permutations
from itertools import combinations
import scipy.special as ss


@dataclass(frozen=True)
class EnvState:
    """
    This class will contain the state of the environment:

    tableau: binary array of size 2*n_qubits_physical*(n_qubits_physical-n_qubits_logical)
    time: integer from 0 to max_steps

    Fixed-noise internal MDP state s_t = (T_t, t) of
    RL_QEC_binary_symplectic_notes.tex Eq. (eq:internal-state), Sec. 8.1
    (no eta term here since the noise channel is fixed; see
    meta_code_discovery.EnvState for the noise-aware s_t=(T_t,eta,t)).
    """
    tableau: torch.Tensor
    time: int

# We ignore this class
@dataclass(frozen=True)
class EnvParams:
    n: int = 7
    k: int = 1
    d: int = 3
    max_steps_in_episode: int = 20


class CodeDiscovery(Environment):
    """
    Environment for the automatic discovery of QEC codes and encodings.

    Args:
        n_qubits_physical (int): Number of physical qubits available
        n_qubits_logical (int): Number of logical qubits
        code_distance (int): Target code distance
        gates (list(CliffordGates)): List of Clifford gates to prepare the encoding circuit
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 30
        lbda (float, optional): Global rescaling factor for the instantaneous reward. Default: 100
        pI (float, optional): Probability of no error in the noise channel. Default: 0.9
        softness (int, optional): Parameter that controls the size of the stabilizer subgroup to be generated. Default: 1
    """
    def __init__(self,
            n_qubits_physical,
            n_qubits_logical,
            code_distance,
            gates,
            graph=None,
            max_steps = 30,
            lbda = 100,
            pI=0.9,
            softness=1,
                ):
        super().__init__()

        self.n_qubits_physical = n_qubits_physical
        self.n_qubits_logical = n_qubits_logical
        self.gates = gates
        self.max_steps = max_steps
        self.d = code_distance
        self.lbda = lbda # Rescales reward for better convergence
        self.pI = pI # Probability of no error


        self.graph = graph
        if self.graph is None:
            self.graph = []
            # Fully connected by default
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))



        self.obs_shape = (2 * n_qubits_physical * (n_qubits_physical - n_qubits_logical), )

        # Initialize action tensor
        self.actions = self.action_matrix()

        # Symplectic metric Omega = [[0,I_n],[I_n,0]] (notes Eq. (eq:omega),
        # Sec. 3.2). u.Omega.v^T mod 2 is 0 iff the Paulis labelled by rows
        # u,v commute (Eq. (eq:commutation-test)); this is what makes
        # syndromes computable as a single matrix product below.
        self.Omega = torch.from_numpy(
            np.kron(np.array([[0,1],[1,0]], dtype=np.uint8), np.eye(n_qubits_physical, dtype=np.uint8)))

        # Initialize error operators and probabilities
        self.E_mu, self.p_mu = self.error_operators()

        # Initialize stabilizer group structure
        self.generate_S_structure(softness) # This generates self.S_struct

        # Initialize num_KL
        self.num_KL = len(self.E_mu)


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def generate_S_structure(self, softness):
        # Generate the structure of the stabilizer group (S) based on the softness parameter.

        num = self.n_qubits_physical - self.n_qubits_logical

        # Calculate the number of stabilizer elements
        soft_elements = int(sum([ss.binom(num,i) for i in range(1,softness+1)]))

        # Create an array of zeros
        S_struct = np.zeros((soft_elements, num), dtype=int)

        # Book-keeping
        start_idx = 0

        for m in range(1,softness+1):

            # Generate combinations of indices to place ones in the S structure
            comb = list(combinations(range(num),m))
            indices = np.array(comb)

            # Fill the S structure with ones at the specified indices
            for i,idx in enumerate(indices):
                S_struct[start_idx+i, idx] = 1

            # Update the start_idx
            start_idx += i+1

        # Ensure there are no rows with all zeroes in the S structure
        assert np.prod(np.any(S_struct, axis=1)), "There is a row with all zeroes"

        # Convert the S structure to a uint8 torch tensor for efficient computation
        self.S_struct = torch.from_numpy(S_struct.astype(np.uint8))

        return

    def stabilizer_elements(self, tableau):
        # Generate the S matrix by multiplying the S structure with the tableau
        return (self.S_struct @ tableau) % 2


    def action_matrix(self,
                      params: Optional[EnvParams] = EnvParams) -> torch.Tensor:

        action_matrix = []
        self.action_string = []
        self.action_string_stim = []

        for gate in self.gates:
            ## One qubit gate
            if len(signature(gate).parameters) == 1:
                for n_qubit in range(self.n_qubits_physical):
                    action_matrix.append(gate(n_qubit))
                    self.action_string.append('%s-%d' % (gate.__name__, n_qubit))
                    self.action_string_stim.append('.%s(%d)' % (gate.__name__.lower(), n_qubit))


            ## Two qubit gates
            elif len(signature(gate).parameters) == 2:
                for edge in self.graph:
                    action_matrix.append(gate(edge[0], edge[1]))
                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))


        return torch.stack(action_matrix).to(torch.uint8)

    def get_observation(self, tableau):
        '''
        Extract the check matrix for the observation

        G_t = T_t[n+k : 2n, :] (notes boxed Eq. (eq:extract-G), Sec. 6.4):
        the ancilla-stabilizer rows n_qubits_physical+n_qubits_logical: of
        the full 2n x 2n tableau are exactly the check matrix of the current
        candidate stabilizer generators g_{t,1},...,g_{t,r} (Eq.
        (eq:time-generators)). o_t = vec(G_t) is the observation, Eq.
        (eq:fixed-observation).
        '''
        ## Only generators without sign
        check_mat = tableau[self.n_qubits_physical + self.n_qubits_logical:].to(torch.uint8)

        return check_mat

    def error_operators(self, params: Optional[EnvParams] = EnvParams) -> torch.Tensor:

        error_list = [1,2,3] # Corresponds to X,Y,Z

        results = []

        for weight in range(1, self.d):
            # Generate all combinations of errors with repetition based on the current weight
            prod = list(list(tup) for tup in itertools.combinations_with_replacement(error_list, weight))

            # Pad the error list to match the number of physical qubits
            prod = [pr + [0] * (self.n_qubits_physical - weight) for pr in prod]

            # Generate distinct permutations of each product to form the error structure
            result = list(list(distinct_permutations(pr, self.n_qubits_physical)) for pr in prod)
            results.append(list(itertools.chain(*result)))

        error_structure = list(itertools.chain(*results))

        # Convert error structure to a torch tensor for efficient computation
        E_mu = torch.from_numpy(
            np.array([np.array(stim.PauliString(p).to_numpy()).flatten() for p in error_structure], dtype=np.uint8))

        # p_X = p_Y = p_Z by assumption
        p_channel = np.array([self.pI] + [(1.-self.pI)/3.]*3, dtype=np.float32)

        # Generate p_mu using p_channel and error_structure
        p_mu = torch.from_numpy(
            np.array([np.prod(p_channel[np.array(error_pauli_character)]) for error_pauli_character in error_structure], dtype=np.float32))

        return E_mu, p_mu


    def check_KL(self, state: EnvState, params: Optional[EnvParams] = EnvParams):
        # Check the Knill-Laflamme conditions for error correction. This is used to reward the agent
        #
        # Implements the binary-matrix-form reward of notes Sec. 8.3
        # (Eqs. (eq:detected-indicator), (eq:membership-indicator),
        # (eq:K-binary), (eq:reward)): for each test error E_mu with
        # binary label e_mu = self.E_mu[mu] and current check matrix
        # G = check_matrix,
        #   d_mu(G) = 1{e_mu . Omega . G^T != 0}   (anticommutes, "anticommutes" below)
        #   m_mu(G) = 1{e_mu in Row(G)}            (is a stabilizer, up to
        #                                            products of <= softness
        #                                            generators, self.S_struct)
        #   K_mu(G) = (1 - d_mu(G)) * (1 - m_mu(G))  -- undetected AND not a
        #                                                stabilizer = harmful
        #                                                logical operator
        # and the returned value is lbda * sum_mu p_mu * K_mu(G), so that
        # step_env's `reward = -self.check_KL(state)` realizes the boxed
        # Eq. (eq:reward): r_t = -sum_mu lambda_mu K_mu(S_{t+1}).

        # Extract the stabilizer generators: G = T_t[n+k:] (Eq. (eq:extract-G))
        check_matrix = state.tableau[self.n_qubits_physical + self.n_qubits_logical:]

        # Update the stabilizer group S: row-space products of up to
        # `softness` generators via self.S_struct (approximates Row(G))
        S = self.stabilizer_elements(check_matrix)

        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        # (vectorized equivalent of jax.vmap(jnp.logical_xor, in_axes=(None, 0))(S, E_mu))
        # -> exact-match test e_mu == (row of S); summing over rows below gives m_mu(G), Eq. (eq:membership-indicator)
        inS = torch.logical_xor(S[None, :, :], self.E_mu[:, None, :])
        inS = torch.logical_not(inS).all(dim=-1).to(torch.int32)

        # self.E_mu @ self.Omega @ check_matrix.T = Sigma = E_M Omega G^T
        # (notes boxed Eq. (eq:all-syndromes)), the syndrome sigma_G(E_mu)
        # for every test error against every generator at once; `anticommutes`
        # is d_mu(G), Eq. (eq:detected-indicator).
        #
        # Calculate the number of Knill-Laflamme conditions that are not satisfied. This is used for stopping criterion
        anticommutes = torch.any(((self.E_mu @ self.Omega) @ check_matrix.T)%2, dim=1)
        self.num_KL = int(len(self.E_mu) - torch.sum(anticommutes, dim=0) - torch.sum(inS))

        # Return the weighted KL sum rescaled by lbda (as a float32 scalar)
        rew = ( torch.sum(self.p_mu) - torch.sum(self.p_mu * anticommutes, dim=0)
                - torch.dot(self.p_mu, torch.sum(inS, dim=-1).to(torch.float32)) )
        return np.float32(self.lbda) * np.float32(rew.item())

    def step_env(
        self, key, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[torch.Tensor, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""

        prev_terminal = self.is_terminal(state, params)

        # Update state: T_{t+1} = T_t M_{A_t} (mod 2), the deterministic
        # transition kernel P_{sas'} of notes boxed Eq. (eq:transition-tableau)
        # / (eq:transition-check), Sec. 8.6.
        state = EnvState( (state.tableau @ self.actions[action]) % 2, state.time + 1)

        # Update KLs -- r_t = -check_KL(state), realizing Eq. (eq:reward)
        reward = -self.check_KL(state)

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

        # T_0 = I_2n, giving G_0 = [0 | 0  I_r] (initial stabilizers
        # Z_{k+1},...,Z_n) -- notes Eq. (eq:initial-check) / Sec. 8.7.
        tableau = TableauSimulator(self.n_qubits_physical)
        init_state = tableau.current_tableau[0]

        state = EnvState(
            tableau = init_state,
            time = 0
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> torch.Tensor:
        """Applies observation function to state."""

        return self.get_observation(state.tableau).flatten()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done_encoding = self.num_KL == 0 # self.threshold

        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps

        done = done_encoding or done_steps
        return bool(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "CodeDiscovery"

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

        return spaces.Box(0, 1, self.obs_shape, dtype=torch.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        return spaces.Dict(
            {
                "tableau": spaces.Box(0, 1, self.obs_shape, torch.uint8),
                "time": spaces.Discrete(params.max_steps),
            }
        )
