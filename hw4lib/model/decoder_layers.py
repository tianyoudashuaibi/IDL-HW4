import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

class SelfAttentionDecoderLayer(nn.Module):
    """
    A single decoder layer for a decoder-only Transformer.
    Contains:
      - Masked self-attention sublayer
      - FeedForward sublayer
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn       = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Masked self-attention
        out, attn_weights = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        # 2) Feed forward
        out = self.ffn(out)
        return out, attn_weights


class CrossAttentionDecoderLayer(nn.Module):
    """
    A single decoder layer for an encoder-decoder Transformer.
    Contains:
      - Masked self-attention
      - Cross-attention
      - FeedForward sublayer
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = SelfAttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)
        self.ffn        = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        dec_key_padding_mask: Optional[torch.Tensor] = None,
        enc_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) Masked self-attn
        out, self_attn_weights = self.self_attn(
            x,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask
        )
        # 2) Cross-attn
        out2, cross_attn_weights = self.cross_attn(
            out,
            enc_output,
            key_padding_mask=enc_key_padding_mask,
            attn_mask=None
        )
        # 3) FeedForward
        out3 = self.ffn(out2)
        return out3, self_attn_weights, cross_attn_weights
