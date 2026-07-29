import torch

class CliffordGates():
    """ Clifford gates class that consists of Clifford gates to update tableau.
    Currently only support the following gates: H, S, X, SQRT_X, CX, and CZ.
    Phases are ignored since they are irrelevant for Knill-Laflamme conditions.

    Each `gate(...)` below returns the row-vector symplectic matrix M_A in
    Ftwo^{2n x 2n} satisfying v(A P A^dagger) = v(P) M_A for the ordered Pauli
    basis X_1..X_n, Z_1..Z_n (RL_QEC_binary_symplectic_notes.tex, boxed Eq.
    (eq:clifford-matrix-definition) in Sec. 4.1). It is built by conjugating
    only the 2n canonical generators, i.e. row_i(M_A)=v(A X_i A^dagger) and
    row_{n+i}(M_A)=v(A Z_i A^dagger) (Eqs. (eq:rows-x)/(eq:rows-z)).
    """

    def __init__(self, n):

        self.n = n # Number of qubits

    def h(self, i):
        # Hadamard rule: X -> Z, Z -> X. For qubit i, this means that I swap columns i and i+n
        # M_H = [[0,1],[1,0]] on the (X_i,Z_i) block, identity elsewhere
        # (notes Eq. (eq:MH), Sec. 4.4): (x,z) M_H = (z,x).
        h_operator = torch.eye(2*self.n, dtype=torch.uint8)

        # Swap columns i and i+n
        temp = h_operator[:,i].clone()
        h_operator[:,i] = h_operator[:,i+self.n]
        h_operator[:,i+self.n] = temp

        # Return the matrix representation
        return h_operator

    def s(self,i):
        # Phase gate rule: X -> Y, Z -> Z, Y -> -X (sign is ignored).
        # M_{S_i} = I_2n + E_{i,n+i} (notes Eq. (eq:MSi), Sec. 4.5):
        # (x,z) M_S = (x, x+z).
        s_operator = torch.eye(2*self.n, dtype=torch.uint8)

        # Make qubit i into Y
        s_operator[i, self.n + i] = 1

        # Return the matrix representation
        return s_operator

    def cx(self, control, target):
        # CX rule: X(c) -> X(c)X(t), Z(c) -> Z(c), X(t) -> X(t), Z(t) -> Z(c)Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]
        # M_{CX_{c->t}} = I_2n + E_{c,t} + E_{n+t,n+c} (notes boxed
        # Eq. (eq:MCX-general), Sec. 4.6).

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
