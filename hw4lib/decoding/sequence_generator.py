import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
2. Beam Search: Maintains top-k most likely sequences at each step
3. Sampling with Filtering: Uses probabilistic sampling with constraints
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies:
      - greedy search
      - beam search
      - sampling with top-k/nucleus
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
            score_fn:    Function that returns logits for next-token prediction
            tokenizer:   Tokenizer instance for handling token conversions
            max_length:  Maximum sequence length to generate
            device:      Device to run generation on
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
        If penalty=1.0, no change is made.
        """
        if penalty == 1.0:
            return logits

        # (B, vocab) => greedy or (B, beam, vocab) => beam
        if logits.dim() == 2:
            # shape (B, vocab)
            B = sequences.size(0)
            for b in range(B):
                unique_tokens = torch.unique(sequences[b])
                logits[b, unique_tokens] = logits[b, unique_tokens] / torch.where(
                    logits[b, unique_tokens] > 0,
                    torch.full_like(logits[b, unique_tokens], penalty),
                    torch.full_like(logits[b, unique_tokens], 1.0/penalty)
                )
        else:
            # shape (B, beamW, vocab)
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
        Apply temperature, top-k, and top-p nucleus filtering to logits.
        """
        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            # shape (..., vocab)
            top_k_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            # everything < top_k_vals[..., -1] => -inf
            threshold = top_k_vals[..., -1, None]
            mask = logits < threshold
            logits[mask] = float('-inf')

        # Top-p
        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True, dim=-1)
            cdf_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            # compute a mask of cdf>top_p
            sorted_remove = cdf_probs > top_p
            # shift right
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = 0

            # map back
            remove_mask = sorted_remove.scatter(dim=-1, index=sorted_indices, src=sorted_remove)
            logits[remove_mask] = float('-inf')

        return logits

    def generate_greedy(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using GREEDY decoding, step-by-step, until max_length or EOS.

        Args:
            x: (B, seq_len) input sequences
            temperature:   scaling
            repeat_penalty: penalize repeated tokens

        Returns:
            sequences: (B, final_len)
            scores:    (B,) sum of log-probs for each sequence
        """
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("x must be shape (B, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= initial seq_len")

        x = x.to(self.device)
        B, init_len = x.shape
        scores = torch.zeros(B, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Step by step
        for _ in range(self.max_length - init_len):
            if finished.all():
                break
            # 1) get logits => shape (B, vocab)
            logits = self.score_fn(x)
            # 2) apply repeat penalty
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            # 3) scale by temperature => log_softmax
            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1)

            # 4) pick next token = argmax
            next_token = torch.argmax(log_probs, dim=-1)  # (B,)
            # gather the token's log-prob
            token_logp = log_probs[torch.arange(B, device=self.device), next_token]
            # add to running scores, only if not finished
            scores = torch.where(finished, scores, scores + token_logp)

            # 5) append next_token
            x = torch.cat([x, next_token.unsqueeze(1)], dim=1)

            # 6) update finished if EOS
            is_eos = (next_token == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    def generate_beam(
        self,
        x: torch.Tensor,
        beam_width: int,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using multi-step BEAM SEARCH decoding, following the pseudocode.

        Args:
            x: (B, seq_len) input
            beam_width: number of beams
            temperature, repeat_penalty: search params

        Returns:
            sequences: (B, beam_width, final_len)
            scores:    (B, beam_width) in descending order
        """
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("x must be shape (B, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= initial seq_len")

        x = x.to(self.device)
        B, init_len = x.shape

        # 1) Initialize beam scores => shape (B, beam_width)
        #    all zero for the initial beam
        beam_scores = torch.zeros(B, beam_width, device=self.device)

        # 2) Keep track of "finished" => shape (B, beam_width)
        finished = torch.zeros(B, beam_width, dtype=torch.bool, device=self.device)

        # 3) First step: expand the single initial input into beam_width
        # get initial logits => shape (B, vocab)
        logits = self.score_fn(x)  # (B, vocab)
        # apply repeat penalty
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        # scale by temperature => log_softmax
        logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)  # shape (B, vocab)

        # pick topK => shape (B, beam_width)
        topk_logp, topk_ids = torch.topk(log_probs, beam_width, dim=-1)

        # update beam_scores
        beam_scores = topk_logp  # shape (B, beam_width)
        # expand x => shape (B, beam_width, init_len)
        seqs = x.unsqueeze(1).repeat(1, beam_width, 1)
        # append topk_ids => shape => (B, beam_width, init_len+1)
        seqs = torch.cat([seqs, topk_ids.unsqueeze(-1)], dim=-1)

        # update finished if EOS
        is_eos = (topk_ids == self.tokenizer.eos_id)
        finished = finished | is_eos

        cur_len = init_len + 1

        # 4) multi-step loop
        for t in range(1, self.max_length - init_len):
            if finished.all():
                break

            # build next_token_scores => (B, beam_width, vocab)
            # we call score_fn for each beam separately
            all_logits = []
            for beam_i in range(beam_width):
                partial = seqs[:, beam_i, :]  # shape (B, cur_len)
                beam_logits = self.score_fn(partial)  # shape (B, vocab)
                all_logits.append(beam_logits.unsqueeze(1))
            # shape => (B, beam_width, vocab)
            all_logits = torch.cat(all_logits, dim=1)

            # apply repeat penalty
            all_logits = self._apply_repeat_penalty(all_logits, seqs, repeat_penalty)
            # scale by temperature => log_softmax
            all_logits = all_logits / temperature
            log_probs  = torch.log_softmax(all_logits, dim=-1)  # (B, beam_width, vocab)

            # cum_scores => shape (B, beam_width, vocab)
            # each beam+token => beam_scores + new log_prob
            cum_scores = beam_scores.unsqueeze(-1) + log_probs

            # flatten => shape => (B, beam_width*vocab)
            Bv = beam_width * log_probs.size(-1)
            cum_scores_flat = cum_scores.view(B, Bv)
            # topk => (B, beam_width)
            new_topk_scores, new_topk_inds = torch.topk(cum_scores_flat, beam_width, dim=-1)

            # decode which parent beam => (B, beam_width)
            parent_beams = new_topk_inds // log_probs.size(-1)
            next_tokens  = new_topk_inds %  log_probs.size(-1)

            # reorder beam_scores
            beam_scores = new_topk_scores  # shape (B, beam_width)

            # build new seqs => shape => (B, beam_width, cur_len+1)
            new_seqs = []
            for b in range(B):
                seq_batch = []
                for beam_i in range(beam_width):
                    parent_i = parent_beams[b, beam_i]
                    token_id = next_tokens[b, beam_i]
                    old_seq  = seqs[b, parent_i].clone()
                    new_seq  = torch.cat([old_seq, token_id.view(1)], dim=0)
                    seq_batch.append(new_seq)
                seq_batch = torch.stack(seq_batch, dim=0)
                new_seqs.append(seq_batch)
            seqs = torch.stack(new_seqs, dim=0)

            # update finished
            eos_found = (next_tokens == self.tokenizer.eos_id)
            # for beams that are already finished => remain finished
            # for new ones that got EOS => set finished
            # We'll do in place
            # build (B, beam_width)
            finished = finished | eos_found

            cur_len += 1

        # 5) Optionally sort each beam by descending score
        # in many beam search setups, we keep them sorted each step anyway,
        # but let's do final sort to ensure the first beam is highest score
        final_seqs = []
        final_scores = []
        for b in range(B):
            # gather beam scores => shape (beam_width,)
            sc = beam_scores[b]
            # indices for descending
            sorted_sc, sorted_idx = torch.sort(sc, descending=True)
            # reorder seq
            s_b = seqs[b, sorted_idx]
            final_seqs.append(s_b)
            final_scores.append(sorted_sc)

        final_seqs   = torch.stack(final_seqs, dim=0)   # (B, beam_width, final_len)
        final_scores = torch.stack(final_scores, dim=0) # (B, beam_width)

        return final_seqs, final_scores

    def generate_sample(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering, step by step.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")

        x = x.to(self.device)
        B = x.size(0)
        scores = torch.zeros(B, device=x.device)
        finished = torch.zeros(B, dtype=torch.bool, device=x.device)

        init_len = x.size(1)

        for _ in range(self.max_length - init_len):
            if finished.all():
                break

            logits = self.score_fn(x)  # (B, vocab)
            filtered_logits = self._filter_logits(logits, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)

            # sample
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_logp  = log_probs[torch.arange(B, device=x.device), next_tokens]
            scores = torch.where(finished, scores, scores + token_logp)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # eos?
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process: truncate at the first EOS. If input is (N, T), returns list of
        truncated sequences. If input is (T,) returns a single truncated sequence.
        """
        # Single sequence
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq

        # Batch of sequences
        eos_mask = (seq == tokenizer.eos_id)
        # find first EOS in each row
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
