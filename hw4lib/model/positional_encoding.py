import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        """
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table of shape (1, max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        # positions: shape (max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # frequencies for the 2i dims
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape to (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # register it as a buffer so it’s not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        x: shape (B, T, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum of {self.pe.size(1)}")
        # Add the positional encodings
        x = x + self.pe[:, :seq_len, :]
        return x
