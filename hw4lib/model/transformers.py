import torch.nn as nn
import torch
import random
from typing import Tuple, Optional
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer
# from .sublayers import ... etc. if needed

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        num_classes: int,
        weight_tying: bool = False,
        layer_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_layers = num_layers

        # A stack of decoder layers
        self.dec_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Embedding, positional encoding, dropout, norm, final projection
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.final_linear = nn.Linear(d_model, num_classes)

        # Optional weight tying
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def forward(
        self,
        padded_targets: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass (training). Returns (seq_out, attn_dict).
        """
        if self.training and target_lengths is None:
            raise ValueError("Must provide target_lengths in training mode.")

        # (1) Padding mask
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths)

        # (2) Causal mask
        causal_mask = CausalMask(padded_targets)

        # (3) Embedding + positional encoding
        x = self.target_embedding(padded_targets)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through each decoder layer
        running_att = {}
        for i, layer in enumerate(self.dec_layers):
            # LayerDrop
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x, att_weights = layer(
                x,
                key_padding_mask=pad_mask_dec,
                attn_mask=causal_mask
            )
            running_att[f'layer{i+1}_dec_self'] = att_weights

        # Final norm + linear
        x = self.norm(x)
        seq_out = self.final_linear(x)  # shape (B,T,num_classes)
        return seq_out, running_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        """
        Inference-time forward for next-token prediction on un-padded prompts.
        Returns only last-token logits.
        """
        if self.training:
            raise ValueError("score is not for training mode")

        # Run forward but we do NOT provide lengths or masks
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # Return the final position's logits: shape (B, num_classes)
        return seq_out[:, -1, :]
