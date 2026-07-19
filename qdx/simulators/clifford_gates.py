import torch

class CliffordGates():
    """ Clifford gates class that consists of Clifford gates to update tableau.
    Currently only support the following gates: H, S, X, SQRT_X, CX, and CZ.
    Phases are ignored since they are irrelevant for Knill-Laflamme conditions.
    """

    def __init__(self, n):

        self.n = n # Number of qubits

    def h(self, i):
        # Hadamard rule: X -> Z, Z -> X. For qubit i, this means that I swap columns i and i+n
        h_operator = torch.eye(2*self.n, dtype=torch.uint8)

        # Swap columns i and i+n
        temp = h_operator[:,i].clone()
        h_operator[:,i] = h_operator[:,i+self.n]
        h_operator[:,i+self.n] = temp

        # Return the matrix representation
        return h_operator

    def s(self,i):
        # Phase gate rule: X -> Y, Z -> Z, Y -> -X (sign is ignored).
        s_operator = torch.eye(2*self.n, dtype=torch.uint8)

        # Make qubit i into Y
        s_operator[i, self.n + i] = 1

        # Return the matrix representation
        return s_operator

    def cx(self, control, target):
        # CX rule: X(c) -> X(c)X(t), Z(c) -> Z(c), X(t) -> X(t), Z(t) -> Z(c)Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]

        cx_operator = torch.eye(2*self.n, dtype=torch.uint8)

        # Transform X(c) -> X(c)X(t)
        cx_operator[control, target] = 1

        # Transform Z(t) -> Z(c)Z(t)
        cx_operator[target+self.n, control+self.n] = 1

        return cx_operator

    def sqrt_x(self,i):
        # SQRT X gate rule: X -> X, Z -> -Y , Y -> Z .
        sqrt_x_operator = torch.eye(2*self.n, dtype=torch.uint8)

        ## Make qubit i into Y
        sqrt_x_operator[i + self.n, i] = 1

        # Return the matrix representation
        return sqrt_x_operator

    def cz(self, control, target):
        # CZ rule: X(c) -> X(c)Z(t), Z(c) -> Z(c), X(t) -> Z(c)X(t), Z(t) -> Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]

        cz_operator = torch.eye(2*self.n, dtype=torch.uint8)

        # Transform X(c) -> X(c)Z(t)
        cz_operator[control, target+self.n] = 1

        # Transform X(t) -> Z(c)X(t)
        cz_operator[target, control+self.n] = 1

        # Return the matrix representation
        return cz_operator

    def sqrt_xx(self, control, target):
        # MS or SQRT_XX rule: X(c) -> X(c), Z(c) -> -Y(c)X(t), X(t) -> X(t), Z(t) -> -X(c)Y(t)

        ms_operator = torch.eye(2*self.n, dtype=torch.uint8)

        # Transform Z(c) -> Y(c)X(t)
        ms_operator[control + self.n, control] = 1
        ms_operator[control + self.n, target] = 1

        # Transform Z(t) -> X(c)Y(t)
        ms_operator[target + self.n, control] = 1
        ms_operator[target + self.n, target] = 1

        # Return the matrix representation
        return ms_operator
