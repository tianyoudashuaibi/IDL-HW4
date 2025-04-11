import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of transformer architectures.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionEncoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # Sublayers: self-attention (no causal mask needed) + feed-forward
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn       = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the EncoderLayer.
        Args:
            x (torch.Tensor): The input tensor (B, seq_len, d_model)  
            key_padding_mask (torch.Tensor): The padding mask (B, seq_len)

        Returns:
            x (torch.Tensor): The output tensor (B, seq_len, d_model)
            mha_attn_weights (torch.Tensor): The attention weights (B, seq_len, seq_len)
        '''
        # 1) Self-attention (no causal mask). 
        #    The sublayer itself is pre-norm, so we just pass x + mask.
        x_attn, attn_weights = self.self_attn(
            x, 
            key_padding_mask=key_padding_mask,
            attn_mask=None  # no causal mask for encoder
        )

        # 2) Feed-forward
        x_out = self.ffn(x_attn)

        return x_out, attn_weights

        # TODO: Return the output tensor and attention weights
        raise NotImplementedError # Remove once implemented

