import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
        self,
        score_fn: Callable,
        tokenizer: H4Tokenizer,
        max_length: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.

        Args:
            score_fn: Function that returns logits for next-token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value

        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits  # no change if penalty == 1.0

        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0 / penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0 / penalty)
                    )
        return logits

    def _filter_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Apply temperature, top-k, and top-p filtering to logits.
        """
        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Everything below the k-th logit is set to -inf
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus)
        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0]  = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens

        Returns:
            (sequences, scores):
              sequences -> (batch_size, final_length)
              scores    -> (batch_size,)
        """
        # Basic validations
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        x = x.to(self.device)
        batch_size = x.size(0)
        seq_len    = x.size(1)

        # Keep track of cumulative log-probs and whether each sequence is finished
        scores   = torch.zeros(batch_size, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Generate step by step
        for _ in range(self.max_length - seq_len):
            if finished.all():
                break

            # 1) Score function => (B, vocab_size)
            logits = self.score_fn(x)
            # 2) Repeat penalty
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            # 3) Temperature => then log_softmax
            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1)
            # 4) Greedy => argmax
            next_token = torch.argmax(log_probs, dim=-1)  # shape (B,)

            # 5) Gather next token log-probs
            token_scores = log_probs[torch.arange(batch_size, device=self.device), next_token]
            scores = torch.where(finished, scores, scores + token_scores)

            # 6) Append next_token to x
            next_token = next_token.unsqueeze(-1)
            x = torch.cat([x, next_token], dim=1)

            # 7) Mark finished if EOS
            eos_found = (next_token.squeeze(-1) == self.tokenizer.eos_id)
            finished  = finished | eos_found

        return x, scores

    def generate_beam(
        self,
        x: torch.Tensor,
        beam_width: int,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens

        Returns:
            (sequences, scores):
              sequences -> (batch_size, beam_width, final_length)
              scores    -> (batch_size, beam_width)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        x = x.to(self.device)
        batch_size = x.size(0)
        seq_len    = x.size(1)

        # Duplicate each sequence beam_width times
        # shape => (B, beam_width, seq_len)
        sequences = x.unsqueeze(1).repeat(1, beam_width, 1)

        # Track log-prob scores => shape => (B, beam_width)
        beam_scores = torch.zeros(batch_size, beam_width, device=self.device)

        # Track finished => shape (B, beam_width)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=self.device)

        cur_len = seq_len
        vocab_size = None  # We'll define after first pass

        for _step in range(self.max_length - seq_len):
            if finished.all():
                break

            # Flatten so we can call score_fn
            # shape => (B*beam_width, cur_len)
            flat_sequences = sequences.view(batch_size*beam_width, -1)

            # Evaluate next token logits
            logits = self.score_fn(flat_sequences)  # (B*beam_width, vocab_size)
            if vocab_size is None:
                vocab_size = logits.size(-1)

            # Repeat penalty
            logits = self._apply_repeat_penalty(logits, flat_sequences, repeat_penalty)

            # Temperature
            logits = logits / temperature
            # log_probs
            log_probs = torch.log_softmax(logits, dim=-1)  # shape => (B*beam_width, vocab_size)

            # Current scores => shape (B*beam_width, 1)
            old_scores = beam_scores.view(batch_size*beam_width, 1)

            # Next_scores => shape => (B*beam_width, vocab_size)
            next_scores = log_probs + old_scores
            # Reshape => (B, beam_width, vocab_size)
            next_scores = next_scores.view(batch_size, beam_width, vocab_size)

            # Flatten beam_width x vocab => shape => (B, beam_width*vocab_size)
            next_scores_flat = next_scores.view(batch_size, beam_width * vocab_size)

            # topk => shape => (B, beam_width)
            topk_scores, topk_indices = torch.topk(next_scores_flat, k=beam_width, dim=-1)

            # from topk_indices => decode which beam => (beam_i = index // vocab_size) and token => index % vocab_size
            beam_index = topk_indices // vocab_size
            token_id   = topk_indices % vocab_size

            # Update beam_scores => (B, beam_width)
            beam_scores = topk_scores

            # Build new sequences => (B, beam_width, cur_len+1)
            new_sequences = []
            for b in range(batch_size):
                seq_batch = []
                for beam_i in range(beam_width):
                    parent_idx = beam_index[b, beam_i]
                    chosen_tok = token_id[b, beam_i]
                    old_seq = sequences[b, parent_idx].clone()
                    new_seq = torch.cat([old_seq, chosen_tok.view(1)], dim=0)
                    seq_batch.append(new_seq)
                seq_batch = torch.stack(seq_batch, dim=0)  # (beam_width, cur_len+1)
                new_sequences.append(seq_batch)
            sequences = torch.stack(new_sequences, dim=0)  # (B, beam_width, cur_len+1)

            # Check for EOS => mark finished
            new_tokens = token_id  # shape => (B, beam_width)
            eos_found = (new_tokens == self.tokenizer.eos_id)
            finished = finished | eos_found

            cur_len += 1

        return sequences, beam_scores

    def generate_sample(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from

        Returns:
            (sequences, scores):
              sequences -> (batch_size, final_length)
              scores    -> (batch_size,)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")

        x = x.to(self.device)
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            next_scores = self.score_fn(x)  # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)

            # sample from the distribution
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            scores = torch.where(finished, scores, scores + token_scores)

            # append next_tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # check EOS
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions

        Returns:
            If seq is a single sequence, return a tensor of same shape truncated at first EOS
            If seq is a batch of sequences, return a list of such truncated tensors
        """
        # single-sequence
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq

        # batch of sequences
        eos_mask = (seq == tokenizer.eos_id)
        # find first EOS in each row
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
