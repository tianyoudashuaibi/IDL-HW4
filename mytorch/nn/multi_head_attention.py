from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention()

        # Projections: embed_dim -> embed_dim
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        # Will store shapes and intermediates for backward
        self.N = None
        self.L = None
        self.S = None
        self.E = None
        self.q = None
        self.k = None
        self.v = None
        self.attn_output = None
        self.mask = None

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) boolean
        :param attn_mask: (L, S) boolean
        :return: (N, L, E)
        """
        self.N, self.L, self.E = query.shape
        self.S = key.shape[1]

        # 1) Project Q,K,V
        q = self.q_proj.forward(query)   # (N,L,E)
        k = self.k_proj.forward(key)     # (N,S,E)
        v = self.v_proj.forward(value)   # (N,S,E)

        # 2) Split into heads
        q = self._split_heads(q)  # (N, H, L, E//H)
        k = self._split_heads(k)  # (N, H, S, E//H)
        v = self._split_heads(v)  # (N, H, S, E//H)

        # 3) Merge masks
        mask = self._merge_masks(key_padding_mask, attn_mask)
        self.mask = mask

        # 4) Scaled dot-product attention
        attn_outputs = self.attention.forward(q, k, v, mask)  # (N,H,L,E//H)

        # 5) Concat heads
        attn_output = self._concat_heads(attn_outputs)  # (N,L,E)

        # 6) Final linear projection
        output = self.out_proj.forward(attn_output)

        # Store for backward
        self.q, self.k, self.v = q, k, v
        self.attn_output = attn_output
        return output

    def backward(self, d_output):
        """
        :param d_output: (N, L, E)
        :return: d_q, d_k, d_v of shape (N,L,E), (N,S,E), (N,S,E)
        """
        # 1) Backprop through final projection
        d_attn_output = self.out_proj.backward(d_output)  # (N,L,E)

        # 2) Split heads
        d_attn_outputs = self._split_heads(d_attn_output)  # (N,H,L,E//H)

        # 3) Backprop through scaled dot-product attention
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)  
        # d_q: (N,H,L,E//H), d_k: (N,H,S,E//H), d_v: (N,H,S,E//H)

        # 4) Concat heads
        d_q = self._concat_heads(d_q)  # (N,L,E)
        d_k = self._concat_heads(d_k)  # (N,S,E)
        d_v = self._concat_heads(d_v)  # (N,S,E)

        # 5) Backprop through input projections
        d_q = self.q_proj.backward(d_q)      # (N,L,E)
        d_k = self.k_proj.backward(d_k)      # (N,S,E)
        d_v = self.v_proj.backward(d_v)      # (N,S,E)

        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask (N, S) and attn_mask (L, S) into (N, H, L, S) if present.
        If neither mask is present, return None.
        """
        if key_padding_mask is None and attn_mask is None:
            return None
        
        # Expand them to (N,H,L,S) and combine with logical_or
        N, H, L, S = self.N, self.num_heads, self.L, self.S
        combined_mask = None

        if key_padding_mask is not None:
            # (N,S) -> (N,1,1,S), then broadcast to (N,H,L,S)
            key_mask = key_padding_mask[:, None, None, :]
            key_mask = np.broadcast_to(key_mask, (N, H, L, S))
        else:
            key_mask = None

        if attn_mask is not None:
            # (L,S) -> (1,1,L,S), then broadcast
            attention_mask = attn_mask[None, None, :, :]
            attention_mask = np.broadcast_to(attention_mask, (N, H, L, S))
        else:
            attention_mask = None

        if key_mask is not None and attention_mask is not None:
            combined_mask = np.logical_or(key_mask, attention_mask)
        elif key_mask is not None:
            combined_mask = key_mask
        else:
            combined_mask = attention_mask

        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Then transpose to (N, num_heads, L, d_k).
        """
        N, L, E = x.shape
        d_k = E // self.num_heads
        # (N,L,E) -> (N,L,num_heads,d_k)
        x = x.reshape(N, L, self.num_heads, d_k)
        # (N,L,num_heads,d_k) -> (N,num_heads,L,d_k)
        x = np.transpose(x, (0, 2, 1, 3))
        return x

    def _concat_heads(self, x):
        """
        Inverse of _split_heads: (N, num_heads, L, d_k) -> (N, L, E)
        """
        N, H, L, d_k = x.shape
        # -> (N,L,H,d_k)
        x = np.transpose(x, (0, 2, 1, 3))
        # -> (N,L,H*d_k)
        x = x.reshape(N, L, H*d_k)
        return x
