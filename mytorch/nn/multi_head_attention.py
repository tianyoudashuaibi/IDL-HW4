from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention()
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        q = self.q_proj.forward(query)   
        k = self.k_proj.forward(key)     
        v = self.v_proj.forward(value)   
        q = self._split_heads(q)  
        k = self._split_heads(k)  
        v = self._split_heads(v) 
        mask = self._merge_masks(key_padding_mask, attn_mask)
        self.mask = mask
        attn_outputs = self.attention.forward(q, k, v, mask)  
        attn_output = self._concat_heads(attn_outputs)  
        output = self.out_proj.forward(attn_output)
        return output

    def backward(self, d_output):
        d_attn_output = self.out_proj.backward(d_output)
        d_attn_outputs = self._split_heads(d_attn_output) 
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)  
        d_q = self._concat_heads(d_q) 
        d_k = self._concat_heads(d_k) 
        d_v = self._concat_heads(d_v)  
        d_q = self.q_proj.backward(d_q)      
        d_k = self.k_proj.backward(d_k)     
        d_v = self.v_proj.backward(d_v)      
        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        N, H, L, S = self.N, self.num_heads, self.L, self.S
        key_mask = key_padding_mask[:, None, None, :]
        key_mask = np.broadcast_to(key_mask, (N, H, L, S))
        attention_mask = attn_mask[None, None, :, :]
        attention_mask = np.broadcast_to(attention_mask, (N, H, L, S))
        combined_mask = np.logical_or(key_mask, attention_mask)
        return combined_mask

    def _split_heads(self, x):
        N, L, E = x.shape
        d_k = E // self.num_heads
        x = x.reshape(N, L, self.num_heads, d_k)
        x = np.transpose(x, (0, 2, 1, 3))
        return x

    def _concat_heads(self, x):
        N, H, L, d_k = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        x = x.reshape(N, L, H*d_k)
        return x
