import torch

def PadMask(padded_input, input_lengths):
    N, T = padded_input.shape[:2]
    device = padded_input.device
    positions = torch.arange(T, device=device).expand(N, T)
    lengths_expanded = input_lengths.unsqueeze(1)
    mask = (positions >= lengths_expanded)
    return mask

def CausalMask(padded_input):
    T = padded_input.shape[1]
    device = padded_input.device
    causal_mask = torch.triu(
        torch.ones((T, T), device=device, dtype=torch.bool),
        diagonal=1
    )
    return causal_mask
