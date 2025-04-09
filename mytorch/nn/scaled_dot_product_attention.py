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
        # We'll apply Softmax along the "S" dimension (the last dimension of K)
        # Typically QK^T yields shape (..., L, S), so we softmax along S:
        self.softmax = Softmax(dim=-1)
        self.eps = 1e10  # DO NOT MODIFY

        # For backward
        self.Q = None
        self.K = None
        self.V = None
        self.attention_scores = None  # softmax(...) result
        self.mask = None

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: (N, ..., H, L, E)
        :param K: (N, ..., H, S, E)
        :param V: (N, ..., H, S, Ev)
        :param mask: (N, ..., H, L, S) boolean, True=ignore
        :return: Output of shape (N, ..., H, L, Ev)
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        # 1) scaled dot product: QK^T / sqrt(E)
        d_k = Q.shape[-1]
        # K^T along last two dims: (N,...,H,S,E) -> (N,...,H,E,S)
        K_t = np.swapaxes(K, -1, -2)  # shape (..., E, S)
        scaled_dot_product = np.matmul(Q, K_t) / np.sqrt(d_k)  # (..., L, S)

        # 2) Apply mask if provided
        if mask is not None:
            # Where mask is True, we add -eps to push softmax logits to near zero
            scaled_dot_product[mask] -= self.eps

        # 3) Softmax along the last dimension (S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # 4) Multiply by V: shape (..., L, S) x (..., S, Ev) -> (..., L, Ev)
        output = np.matmul(self.attention_scores, V)

        return output

    def backward(self, d_output):
        """
        :param d_output: Grad of loss wrt output, shape (N, ..., H, L, Ev)
        :return: (dQ, dK, dV) each shaped like Q,K,V respectively
        """
        # Unpack
        Q, K, V = self.Q, self.K, self.V
        A = self.attention_scores
        d_k = Q.shape[-1]  # for scaling factor sqrt(d_k)

        # 1) Grad wrt V:
        # output = A @ V
        # dV = A^T @ d_output
        # A: (..., L,S), d_output: (..., L,Ev)
        A_t = np.swapaxes(A, -1, -2)   # transpose last two dims -> (..., S,L)
        d_V = np.matmul(A_t, d_output) # (..., S,Ev)

        # 2) Grad wrt attention_scores A:
        # dA = d_output @ V^T
        V_t = np.swapaxes(V, -1, -2)   # shape (..., Ev,S)
        dA = np.matmul(d_output, V_t)  # shape (..., L,S)

        # 3) Backprop through softmax
        d_scaled_dot = self.softmax.backward(dA)

        # 4) Account for the 1/sqrt(d_k) factor in scaled_dot_product
        d_scaled_dot = d_scaled_dot / np.sqrt(d_k)

        # 5) Now scaled_dot_product = Q @ K^T => shape (..., L,S)
        #    dQ = d_scaled_dot @ K
        #    dK = d_scaled_dot^T @ Q
        K_t = np.swapaxes(K, -1, -2)  # (..., E,S)
        d_Q = np.matmul(d_scaled_dot, np.swapaxes(K, -1, -2))  # => (..., L,E)
        
        # dK => we need (Q^T) times d_scaled_dot
        # Q^T: (..., E,L), d_scaled_dot: (..., L,S)
        # => shape for dK: (..., E,S), but we want (..., S,E) so we can swap axes at the end.
        dK = np.matmul(np.swapaxes(d_scaled_dot, -1, -2), Q)  # => (..., S,E)
        # Swap it so it matches K’s shape (N,...,H,S,E)
        dK = np.swapaxes(dK, -1, -2)

        # Reshape d_Q so it matches Q's shape: (N,...,H,L,E)
        # np.matmul gave it shape (...,L,E) which is consistent. 
        # So no special transpose needed unless your matmul needs rearranging.

        return (d_Q, dK, d_V)
