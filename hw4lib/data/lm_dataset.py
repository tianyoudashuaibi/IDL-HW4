from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import H4Tokenizer
    
class LMDataset(Dataset):
    def __init__(
            self, 
            partition: str, 
            config: dict, 
            tokenizer: H4Tokenizer
    ):
        self.config    = config
        self.partition = partition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id
        self.text_dir = os.path.join(self.config["root"], partition)
        all_files = sorted(os.listdir(self.text_dir))
        self.text_files = [os.path.join(self.text_dir, f) for f in all_files]
        subset_size = self.config.get("subset_size", None)
        if subset_size is not None and subset_size > 0:
            self.text_files = self.text_files[:subset_size]
        self.transcripts_shifted = []
        self.transcripts_golden  = []
        self.total_chars  = 0
        self.total_tokens = 0
        self.text_max_len = 0
        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            arr = np.load(file, allow_pickle=True)  
            transcript = "".join(arr.tolist())
            self.total_chars += len(transcript)
            tokenized = self.tokenizer.encode(transcript)
            self.total_tokens += len(tokenized)
            self.text_max_len = max(self.text_max_len, len(tokenized)+1)
            self.transcripts_shifted.append([self.sos_token] + tokenized)
            self.transcripts_golden.append(tokenized + [self.eos_token])
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        if not (len(self.transcripts_shifted) == len(self.transcripts_golden)):
            raise ValueError("Shifted and golden transcripts are misaligned")
        self.length = len(self.transcripts_shifted)
        
    def get_avg_chars_per_token(self) -> float:
        return self.avg_chars_per_token
    
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden  = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden
    
    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shifted_transcripts, golden_transcripts = zip(*batch)
        lengths = [len(s) for s in shifted_transcripts]  
        padded_shifted = pad_sequence(
            shifted_transcripts, 
            batch_first=True, 
            padding_value=self.pad_token
        )
        padded_golden = pad_sequence(
            golden_transcripts, 
            batch_first=True, 
            padding_value=self.pad_token
        )
        lengths = torch.LongTensor(lengths)
        return padded_shifted, padded_golden, lengths

    def sample_prompts(self, num_samples: int, prompt_length: int, seed: int = None) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
        """
        Sample random prompts of fixed length from the dataset and return their original sequences.
        DO NOT MODIFY
        
        Args:
            num_samples: Number of prompts to sample
            prompt_length: Exact number of tokens for each prompt
            seed: Random seed for reproducibility. If None, no seed is set.
            
        Returns:
            tuple: (prompts, originals) where:
                - prompts: torch.LongTensor of tokenized prompts
                - originals: List of torch.LongTensor containing complete original sequences
        """
        # Set random seed if provided
        if seed is not None:
            # Save current random state
            np_state = np.random.get_state()
            # Set seed for sampling
            np.random.seed(seed)
            
        prompts = []
        originals = []
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loops
        
        while len(prompts) < num_samples and attempts < max_attempts:
            # Sample random transcript
            idx = np.random.randint(0, len(self))
            tokens = self.transcripts_shifted[idx][1:] # remove sos token
            
            # Skip if transcript is too short
            if len(tokens) < prompt_length:
                attempts += 1
                continue
                
            # Get exactly prompt_length tokens
            prompt_tokens = tokens[:prompt_length]
            
            # Store prompt and original sequence
            prompts.append(torch.LongTensor([self.sos_token] + prompt_tokens))
            originals.append(torch.LongTensor(tokens + [self.eos_token]))
            
            attempts += 1
            
        if len(prompts) < num_samples:
            print(f"Warning: Could only sample {len(prompts)} valid prompts")
        
        # Restore random state if seed was set
        if seed is not None:
            np.random.set_state(np_state)
            
        # No need for another LongTensor conversion since prompts are already tensors
        return torch.stack(prompts), originals
    
