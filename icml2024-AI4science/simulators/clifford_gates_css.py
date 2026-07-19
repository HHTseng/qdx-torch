import torch

class CliffordGatesCSS():
    """
    Clifford gates, which are part of the TableauSimulator but
    without keeping track how the tableau changes
    """

    def __init__(self, n):

        self.n = n # Number of qubits


    def cx(self, control, target):
        # CX rule: X(c) -> X(c)X(t), Z(c) -> Z(c), X(t) -> X(t), Z(t) -> Z(c)Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]

        cx_operator = torch.eye(self.n, dtype=torch.uint8)

        # Transform X(c) -> X(c)X(t)
        cx_operator[control, target] = 1

        return cx_operator
