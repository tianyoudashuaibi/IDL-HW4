import torch.nn as nn
import torch
from typing import Tuple, Optional

class SelfAttentionLayer(nn.Module):
    """
    Pre-LN Self-Attention sublayer for the decoder (masked).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: shape (B, T, d_model)
        key_padding_mask: shape (B, T)
        attn_mask: shape (T, T)
        """
        residual = x
        # Pre-norm
        x = self.norm(x)
        # Self-attention
        out, attn_weights = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask, 
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        # dropout + residual
        out = self.dropout(out)
        out = out + residual
        return out, attn_weights


class CrossAttentionLayer(nn.Module):
    """
    Pre-LN cross-attention sublayer for the decoder.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T_dec, d_model)  -- query
        y: (B, T_enc, d_model)  -- key/value
        key_padding_mask: shape (B, T_enc)
        attn_mask: shape (T_dec, T_enc), optional
        """
        residual = x
        x_norm = self.norm(x)
        out, attn_weights = self.mha(
            query=x_norm,
            key=y,
            value=y,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        out = self.dropout(out)
        out = out + residual
        return out, attn_weights


class FeedForwardLayer(nn.Module):
    """
    Pre-LN feed-forward sublayer.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # As specified: Linear -> GELU -> Dropout -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        return x

        # TODO: Return the output tensor
        raise NotImplementedError # Remove once implemented
    
