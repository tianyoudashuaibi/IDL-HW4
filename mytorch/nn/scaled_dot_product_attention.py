import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        """
        Initialize the ScaledDotProductAttention class.
        """
        # We'll apply softmax along the last dimension (the "S" dimension).
        self.softmax = Softmax(dim=-1)
        self.eps = 1e10  # We'll use -self.eps for masked positions

        # Saved for backward
        self.Q = None
        self.K = None
        self.V = None
        self.attention_scores = None
        self.mask = None

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: (N, H, L, E)
        :param K: (N, H, S, E)
        :param V: (N, H, S, Ev)
        :param mask: (N, H, L, S) boolean, True => mask out/ignore
        :return:    (N, H, L, Ev)
        """
        # Save for backward
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        # QK^T / sqrt(d_k)
        d_k = Q.shape[-1]  # E
        # K^T => shape (N, H, E, S)
        K_t = np.swapaxes(K, -1, -2)   # last two dims swapped
        # => scaled_dot: (N, H, L, S)
        scaled_dot = np.matmul(Q, K_t) / np.sqrt(d_k)

        # If mask is provided, fill with -eps so softmax -> 0 for those positions
        if mask is not None:
            scaled_dot[mask] = -self.eps

        # Softmax along the last dim (S)
        self.attention_scores = self.softmax.forward(scaled_dot)  # (N, H, L, S)

        # Multiply by V => (N, H, L, Ev)
        output = np.matmul(self.attention_scores, V)

        return output

    def backward(self, d_output):
        """
        :param d_output: (N, H, L, Ev)
        :return: dQ, dK, dV of shapes (N,H,L,E), (N,H,S,E), (N,H,S,Ev)
        """
        Q, K, V = self.Q, self.K, self.V
        A = self.attention_scores        # shape (N, H, L, S)
        d_k = Q.shape[-1]               # E

        # -- 1) Grad wrt V:  out = A @ V
        #    dV = A^T @ d_output
        A_t = np.swapaxes(A, -1, -2)     # (N,H,S,L)
        d_V = np.matmul(A_t, d_output)   # => (N,H,S,Ev)

        # -- 2) Grad wrt A:  out = A @ V => dA = d_output @ V^T
        V_t = np.swapaxes(V, -1, -2)     # (N,H,Ev,S)
        dA = np.matmul(d_output, V_t)    # => (N,H,L,S)

        # -- 3) Backprop through softmax => dScaled
        dScaled = self.softmax.backward(dA)  # (N,H,L,S)

        # PyTorch’s scaled_dot_product_attention effectively zeroes out masked positions again:
        if self.mask is not None:
            dScaled[self.mask] = 0.

        # -- 4) Account for / sqrt(d_k)
        dScaled = dScaled / np.sqrt(d_k)

        # -- 5) Grad wrt Q, K from scaled_dot = Q @ K^T
        #    dQ = dScaledDot @ K
        #    dK = (dScaledDot^T) @ Q  [then reshape or swap axes]
        d_Q = np.matmul(dScaled, K)  # => (N,H,L,E)

        dScaled_t = np.swapaxes(dScaled, -1, -2)  # => (N,H,S,L)
        dK_temp = np.matmul(dScaled_t, Q)         # => (N,H,S,E)
        d_K = dK_temp  # no further swap needed if we matched shapes exactly

        return d_Q, d_K, d_V
