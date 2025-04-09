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
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn   = score_fn
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.device     = device

    def _apply_repeat_penalty(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape 
              (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape 
              (batch_size, seq_len) or (batch_size, beam_width, seq_len)
            penalty: Repetition penalty value

        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits  # No change if penalty == 1.0

        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy: shape (B, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                # For each unique token in the sequence, scale the logit
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0 / penalty)
                )
        else:
            # Beam: shape (B, beam_width, vocab_size)
            B, beamW, V = logits.shape
            for b in range(B):
                for beam_i in range(beamW):
                    unique_tokens = torch.unique(sequences[b, beam_i])
                    logits[b, beam_i, unique_tokens] = logits[b, beam_i, unique_tokens] / torch.where(
                        logits[b, beam_i, unique_tokens] > 0,
                        torch.full_like(logits[b, beam_i, unique_tokens], penalty),
                        torch.full_like(logits[b, beam_i, unique_tokens], 1.0 / penalty)
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
        # 1) Temperature
        logits = logits / temperature

        # 2) Top-k
        if top_k > 0:
            # top_k along last dimension
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        # 3) Top-p (nucleus)
        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # shift right
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
            (sequences, scores)
              sequences -> (batch_size, final_seq_len)
              scores    -> (batch_size,) sum of log-probs
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        x = x.to(self.device)
        batch_size = x.size(0)
        seq_len    = x.size(1)

        # We'll track cumulative log-prob for each sequence
        scores   = torch.zeros(batch_size, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(self.max_length - seq_len):
            if finished.all():
                break

            # 1) Score function => shape (B, vocab_size)
            logits = self.score_fn(x)
            # 2) Repetition penalty
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            # 3) Temperature => log-softmax
            logits   = logits / temperature
            log_probs= torch.log_softmax(logits, dim=-1)

            # 4) Greedy pick
            next_token = torch.argmax(log_probs, dim=-1)  # (B,)

            # 5) Update scores
            token_scores = log_probs[torch.arange(batch_size, device=self.device), next_token]
            scores = torch.where(finished, scores, scores + token_scores)

            # 6) Append next token
            next_token = next_token.unsqueeze(1)  # shape (B,1)
            x = torch.cat([x, next_token], dim=1)

            # 7) Check if EOS
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
            beam_width: number of beams
            temperature: temperature for logits scaling
            repeat_penalty: repetition penalty

        Returns:
            (sequences, scores):
              sequences -> (batch_size, beam_width, final_seq_len)
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
        B = x.size(0)
        init_len = x.size(1)

        # 1) We'll replicate each sequence for beam_width
        #    shape => (B, beam_width, init_len)
        sequences = x.unsqueeze(1).repeat(1, beam_width, 1)

        # 2) Keep track of beam scores => shape (B, beam_width)
        beam_scores = torch.zeros(B, beam_width, device=self.device)

        # 3) Keep track of finished => shape (B, beam_width)
        finished = torch.zeros(B, beam_width, dtype=torch.bool, device=self.device)

        # We'll do up to (max_length - init_len) decoding steps
        cur_len = init_len
        vocab_size = None

        for _step in range(self.max_length - init_len):
            if finished.all():
                break

            # (B, beam_width, vocab_size) => we call score_fn separately for each beam
            # then combine
            beam_logits = []
            for beam_i in range(beam_width):
                # shape => (B, seq_len)
                partial_seq = sequences[:, beam_i, :]
                logits_i = self.score_fn(partial_seq)  # => (B, vocab_size)
                beam_logits.append(logits_i.unsqueeze(1))  # => (B,1,vocab_size)
            
            # beam_logits => shape (B, beam_width, vocab_size)
            beam_logits = torch.cat(beam_logits, dim=1)
            if vocab_size is None:
                vocab_size = beam_logits.size(-1)

            # 1) repetition penalty
            beam_logits = self._apply_repeat_penalty(beam_logits, sequences, repeat_penalty)
            # 2) temperature
            beam_logits = beam_logits / temperature
            # 3) log_probs => shape (B, beam_width, vocab_size)
            log_probs   = torch.log_softmax(beam_logits, dim=-1)

            # current beam scores => shape => (B, beam_width, 1)
            old_scores = beam_scores.unsqueeze(-1)

            # new potential scores => shape => (B, beam_width, vocab_size)
            next_scores = log_probs + old_scores

            # Flatten out beam_width * vocab_size => shape => (B, beam_width*vocab_size)
            next_scores_flat = next_scores.view(B, beam_width * vocab_size)

            # topk over that dimension => shape => (B, beam_width)
            topk_scores, topk_indices = torch.topk(next_scores_flat, k=beam_width, dim=-1)

            # decode beam + token => shape => (B, beam_width)
            parent_beam = topk_indices // vocab_size
            token_id    = topk_indices %  vocab_size

            # update beam_scores => shape => (B, beam_width)
            beam_scores = topk_scores

            # build new sequences => shape => (B, beam_width, cur_len+1)
            new_seqs = []
            for b_idx in range(B):
                seq_batch = []
                for beam_i in range(beam_width):
                    par_i   = parent_beam[b_idx, beam_i].item()
                    t_id    = token_id[b_idx, beam_i].unsqueeze(0)
                    old_seq = sequences[b_idx, par_i]
                    new_seq = torch.cat([old_seq, t_id], dim=0)
                    seq_batch.append(new_seq)
                seq_batch = torch.stack(seq_batch, dim=0)
                new_seqs.append(seq_batch)
            sequences = torch.stack(new_seqs, dim=0)

            # check eos
            eos_found = (token_id == self.tokenizer.eos_id)
            finished  = finished | eos_found
            cur_len  += 1

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
              sequences -> (batch_size, final_seq_len)
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

            next_scores = self.score_fn(x)  # (B, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)

            # sample
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_scores = log_probs[torch.arange(batch_size, device=x.device), next_tokens]
            scores = torch.where(finished, scores, scores + token_scores)

            # append next_tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # eos
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape 
               (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions

        Returns:
            If seq is a single sequence, return a truncated version at first EOS
            If seq is a batch of sequences, return a list of truncated sequences
        """
        # single-sequence
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq

        # batch-of-sequences
        eos_mask = (seq == tokenizer.eos_id)
        # find first EOS in each row
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
