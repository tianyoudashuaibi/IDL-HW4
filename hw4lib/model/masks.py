import torch

def PadMask(padded_input, input_lengths):
    """
    Create a mask to identify padding positions.
    Returns a boolean mask of shape (N, T) where:
      True  = padding
      False = non-padding
    """
    # We only care about (N, T) dims, even if the input is (N,T,...).
    N, T = padded_input.shape[:2]
    # Create a [0..T-1] range: shape (T,) -> expand to (N,T)
    # Compare each position index vs the sequence length.
    # If position >= length, it's padding => True
    # Otherwise => False
    device = padded_input.device
    positions = torch.arange(T, device=device).expand(N, T)
    lengths_expanded = input_lengths.unsqueeze(1)
    mask = (positions >= lengths_expanded)
    return mask

def CausalMask(padded_input):
    """
    Create a causal (future-blind) mask of shape (T, T),
    where T = sequence length. True = "do not attend."
    """
    # T is the time dimension from the second axis
    T = padded_input.shape[1]
    # We want an upper-triangular matrix (excluding diagonal)
    # that is True above the diagonal => future positions are masked.
    device = padded_input.device
    causal_mask = torch.triu(
        torch.ones((T, T), device=device, dtype=torch.bool),
        diagonal=1
    )
    return causal_mask
