import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # We'll apply softmax along the last dimension (S dimension)
        self.softmax = Softmax(dim=-1)
        self.eps = 1e10  # DO NOT MODIFY

        # For use in backward:
        self.Q = None
        self.K = None
        self.V = None
        self.attention_scores = None  # the softmax(...) output
        self.mask = None

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: (N, ..., H, L, E)
        :param K: (N, ..., H, S, E)
        :param V: (N, ..., H, S, Ev)
        :param mask: (N, ..., H, L, S) boolean, True => ignore
        :return:     (N, ..., H, L, Ev)
        """
        # Save references for backward
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        # QK^T / sqrt(E)
        d_k = Q.shape[-1]  # E
        # K^T: shape (N, ..., H, E, S)
        K_t = np.swapaxes(K, -1, -2)  # last two dims swapped
        # scaled_dot_product: (N, ..., H, L, S)
        scaled_dot_product = np.matmul(Q, K_t) / np.sqrt(d_k)

        # If mask is given, subtract large number from masked positions
        if mask is not None:
            scaled_dot_product[mask] -= self.eps

        # Apply softmax along last dim (S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Multiply by V => (N, ..., H, L, Ev)
        output = np.matmul(self.attention_scores, V)

        return output

    def backward(self, d_output):
        """
        :param d_output: gradient wrt output, shape (N, ..., H, L, Ev)
        :return: (dQ, dK, dV) with shapes matching Q,K,V
        """
        Q, K, V = self.Q, self.K, self.V
        A = self.attention_scores  # shape (N, ..., H, L, S)
        d_k = Q.shape[-1]          # E in Q,K

        # ---- 1) dV ----
        #  output = A @ V
        #  dV = A^T @ d_output
        A_t = np.swapaxes(A, -1, -2)  # shape (..., S, L)
        d_V = np.matmul(A_t, d_output)  # => (N, ..., H, S, Ev)

        # ---- 2) dA ----
        #  A = softmax(...)
        #  output = A @ V  => dA = d_output @ V^T
        V_t = np.swapaxes(V, -1, -2)  # shape (..., Ev, S)
        dA = np.matmul(d_output, V_t)  # => shape (N, ..., H, L, S)

        # ---- 3) backprop through softmax to get dScaledDotProduct
        d_scaled_dot = self.softmax.backward(dA)  # shape (N, ..., H, L, S)

        # ---- 4) account for the 1/sqrt(d_k) factor
        d_scaled_dot = d_scaled_dot / np.sqrt(d_k)

        # ---- 5a) dQ ----
        # scaled_dot = Q @ K^T, so dQ = d_scaled_dot @ K
        K_t = np.swapaxes(K, -1, -2)  # shape (N, ..., H, E, S)
        d_Q = np.matmul(d_scaled_dot, np.swapaxes(K, -1, -2))  # => (N, ..., H, L, E)

        # ---- 5b) dK ----
        # scaled_dot = Q @ K^T => dK = (d_scaled_dot^T) @ Q
        # but we have to be careful with shapes.  We can do:
        dK_temp = np.matmul(np.swapaxes(d_scaled_dot, -1, -2), Q)  # => (N, ..., H, S, E)
        d_K = np.swapaxes(dK_temp, -1, -2)                          # => (N, ..., H, S, E) is good

        return d_Q, d_K, d_V
