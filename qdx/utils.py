import numpy as np
import torch
import stim
import itertools
from more_itertools import distinct_permutations

import scipy.special as ss
from itertools import combinations
from qdx.simulators import TableauSimulator

class Utils():

    def __init__(self,
                 n,
                 k,
                 gates,
                 softness,
                ):

        self.n_qubits_physical = n # Number of physical qubits
        self.n_qubits_logical = k # Number of logical qubits

        # Initialize the tableau simulator with the number of physical qubits and apply gates
        tableau_simulator = TableauSimulator(self.n_qubits_physical)
        for g in gates:
            # Dynamically execute the gate commands on the tableau simulator
            eval("tableau_simulator%s" % g)

        # Extract the current tableau state after applying gates
        self.tableau = tableau_simulator.current_tableau[0]

        # Generate the stabilizer group structure for stabilizer calculations
        self.generate_S_structure(softness) # This generates self.S_struct

        self.S = self.generate_S(self.tableau[self.n_qubits_physical + self.n_qubits_logical:])

        # Symplectic metric Omega
        self.Omega = torch.from_numpy(
            np.kron(np.array([[0,1],[1,0]], dtype=np.uint8), np.eye(self.n_qubits_physical, dtype=np.uint8)))


    def error_operators(self, eval_weight) -> torch.Tensor:
        # Generate error operators based on the evaluation weight.

        error_list = [1,2,3] # Corresponds to X,Y,Z

        # Generate all combinations of errors with repetition based on eval_weight
        prod = list(list(tup) for tup in itertools.combinations_with_replacement(error_list, eval_weight))

        # Pad the error list to match the number of physical qubits
        prod = [pr + [0] * (self.n_qubits_physical - eval_weight) for pr in prod]

        # Generate distinct permutations of each product to form the error structure
        result = list(list(distinct_permutations(pr, self.n_qubits_physical)) for pr in prod)
        error_structure = list(itertools.chain(*result))

        # Convert error structure to a torch tensor for efficient computation
        E_mu = torch.from_numpy(
            np.array([np.array(stim.PauliString(p).to_numpy()).flatten() for p in error_structure], dtype=np.uint8))

        return E_mu

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

    def generate_S(self, tableau):
        # Generate the S matrix by multiplying the S structure with the tableau
        return (self.S_struct @ tableau) % 2

    def remove_rows(self, arr, row_indices):
        mask = torch.ones(arr.shape, dtype=torch.int64)
        mask[torch.as_tensor(row_indices)] = 0
        mask = mask.to(torch.bool)

        a = torch.as_tensor(row_indices).shape[0]
        b,c = arr.shape
        # return arr * mask
        return arr[mask].reshape(b-a,c)

    def check_KL(self, E_mu):
        # Check the Knill-Laflamme conditions for error correction.

        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        # (vectorized equivalent of jax.vmap(jnp.logical_xor, in_axes=(None, 0))(S, E_mu))
        inS = torch.logical_xor(self.S[None, :, :], E_mu[:, None, :])
        inS = torch.logical_not(inS).all(dim=-1).to(torch.int32)

        # Calculate the number of Knill-Laflamme conditions that are not satisfied
        num_KL = len(E_mu) - int(torch.sum(torch.any(((E_mu @ self.Omega) @ self.tableau[self.n_qubits_physical + self.n_qubits_logical:].T)%2, dim=1), dim=0)) - int(torch.sum(inS))

        return num_KL # np.min(KLs), np.argmin(KLs)

    def check_KL_cZ(self, E_mu, cZ):
        # Check the Knill-Laflamme conditions specifically considering a biased error channel with bias parameter cZ

        # Separate X,Y,Z part of errors for effective weight computation
        H_X = E_mu[:,:self.n_qubits_physical]
        H_Z = E_mu[:,self.n_qubits_physical:]
        H_Y = H_X * H_Z

        # Calculate the effective weights (float32, matching the JAX original)
        effective_weights = torch.sum(H_X - H_Y, dim=1).to(torch.float32) + torch.sum(H_Y, dim=1).to(torch.float32) + float(np.float32(cZ))*torch.sum(H_Z - H_Y, dim=1).to(torch.float32)

        # Extract the stabilizer generators
        tableau = self.tableau[self.n_qubits_physical + self.n_qubits_logical:]

        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        # (vectorized equivalent of jax.vmap(jnp.logical_xor, in_axes=(None, 0))(S, E_mu))
        inS = torch.logical_xor(self.S[None, :, :], E_mu[:, None, :])
        inS = torch.logical_not(inS).all(dim=-1).to(torch.int32)

        # Check which error operators don't satisfy the Knill-Laflamme conditions
        KL1 = effective_weights * torch.logical_not(torch.any((E_mu @ self.Omega @ tableau.T)%2, dim=1))
        KL2 = effective_weights * torch.sum(inS, dim=1).to(torch.float32)
        KLs = KL1 - KL2

        # Identify non-zero Knill-Laflamme values (the ones that don't satisfy KL)
        non_zero_ids = torch.nonzero(KLs, as_tuple=True)[0]

        # Determine the smallest undetected effective weight
        if len(KLs[non_zero_ids]) > 0:
            smallest_undetected_effective_weight = torch.min(KLs[non_zero_ids])

        # If all errors are detected, return a large number (it will be discarded anyway)
        else:
            smallest_undetected_effective_weight = 100

        return round(float(smallest_undetected_effective_weight),2)
