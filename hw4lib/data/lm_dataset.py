from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import H4Tokenizer

class LMDataset(Dataset):
    """
    Dataset for Language Model training/evaluation.
    """
    def __init__(
        self,
        partition: str, 
        config: dict, 
        tokenizer: H4Tokenizer
    ):
        """
        Initializes the Language Model Dataset for training language models on text data.

        Args:
            partition (str): Data partition subdirectory under root (e.g., 'train', 'valid', 'test')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
        """
        # Store configuration and other args
        # DO NOT MODIFY
        self.config = config
        self.partition = partition
        self.tokenizer = tokenizer
        
        # ---- 1) Load special token IDs
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id
        
        # ---- 2) Set up data path: config["root"] + partition subdirectory
        # e.g. if config["root"] = "/path/to/hw4p1_data"
        self.text_dir = os.path.join(self.config["root"], partition)
        
        # ---- 3) Gather all .npy text files in sorted order
        all_files = sorted(os.listdir(self.text_dir))
        # Filter only .npy if necessary
        self.text_files = [os.path.join(self.text_dir, f)
                           for f in all_files
                           if f.endswith(".npy")]
        
        # ---- 4) Possibly take a subset
        subset_size = self.config.get("subset_size", None)
        if subset_size is not None and subset_size > 0:
            self.text_files = self.text_files[:subset_size]
        
        # ---- 5) Prepare to store transcripts
        self.transcripts_shifted = []
        self.transcripts_golden = []
        
        # Tracking variables (DO NOT MODIFY names)
        self.total_chars = 0
        self.total_tokens = 0
        self.text_max_len = 0
        
        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            # 5a) Load the transcript from a .npy file
            # Typically, np.load(..., allow_pickle=True) -> array of characters
            arr = np.load(file, allow_pickle=True)  # e.g. shape (N,) of chars
            # join to string
            transcript = "".join(arr.tolist())
            
            # 5b) Count characters (before tokenization)
            self.total_chars += len(transcript)
            
            # 5c) Tokenize
            tokenized = self.tokenizer.encode(transcript)
            
            # Track token count (excluding special tokens)
            self.total_tokens += len(tokenized)
            
            # Keep track of max length (include +1 for the added EOS/SOS)
            self.text_max_len = max(self.text_max_len, len(tokenized) + 1)
            
            # 5d) Create shifted & golden
            # shifted => [SOS] + original
            # golden  => original + [EOS]
            shifted = [self.sos_token] + tokenized
            golden  = tokenized + [self.eos_token]
            
            self.transcripts_shifted.append(shifted)
            self.transcripts_golden.append(golden)
        
        # 6) Calculate average chars per token (used for perplexity)
        if self.total_tokens > 0:
            self.avg_chars_per_token = self.total_chars / self.total_tokens
        else:
            self.avg_chars_per_token = 0
        
        # 7) Verify alignment
        if not (len(self.transcripts_shifted) == len(self.transcripts_golden)):
            raise ValueError("Shifted and golden transcripts are misaligned")
        
        # 8) Store the length (#samples)
        self.length = len(self.transcripts_shifted)
    
    def get_avg_chars_per_token(self) -> float:
        """
        Get the average number of characters per token. 
        Used to calculate character-level perplexity.
        DO NOT MODIFY
        """
        return self.avg_chars_per_token
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.length
    
    def __getitem__(self, idx: int):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (shifted_transcript, golden_transcript) 
              where each is a LongTensor.
        """
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden  = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden
    
    def collate_fn(self, batch):
        """
        Collate and pad a batch of samples to create 
        a batch of fixed-length padded shifted and golden transcripts.

        Args:
            batch (list): List of (shifted, golden) transcript pairs

        Returns:
            tuple: (padded_shifted, padded_golden, lengths)
              - padded_shifted: (B, T) 
              - padded_golden: (B, T)
              - lengths: original (un‐padded) sequence lengths as a 1D tensor
        """
        # Unzip the batch
        shifted_transcripts, golden_transcripts = zip(*batch)  # two tuples
        # lengths of each unpadded sample
        lengths = [len(s) for s in shifted_transcripts]  # or golden; same length
        
        # Pad
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
        
        # Return results
        # lengths is typically a LongTensor
        lengths = torch.LongTensor(lengths)
        return padded_shifted, padded_golden, lengths
