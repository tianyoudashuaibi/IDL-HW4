from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

class ASRDataset(Dataset):
    def __init__(
            self,
            partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config: dict,
            tokenizer: H4Tokenizer,
            isTrainPartition: bool,
            global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        Initialize the ASRDataset for training/validation/testing.
        Storing only the base filenames in self.fbank_files and self.text_files
        so that test alignment checks pass.
        """
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # --- 1) Special tokens
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # --- 2) Path setup
        self.fbank_dir = os.path.join(self.config["root"], partition, "fbank")
        all_fbank_files = sorted(os.listdir(self.fbank_dir))
        # Store only base filenames for alignment checks
        self.fbank_files = [f for f in all_fbank_files if f.endswith(".npy")]

        subset_size = self.config.get("subset_size", None)
        if subset_size is not None and subset_size > 0:
            self.fbank_files = self.fbank_files[:subset_size]

        self.length = len(self.fbank_files)

        # If not test, also gather text filenames
        if self.partition != "test-clean":
            self.text_dir = os.path.join(self.config["root"], partition, "text")
            all_text_files = sorted(os.listdir(self.text_dir))
            self.text_files = [f for f in all_text_files if f.endswith(".npy")]
            if subset_size is not None and subset_size > 0:
                self.text_files = self.text_files[:subset_size]

            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature files and text files must match")

        # We'll store the raw data in memory
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden  = []

        self.total_chars  = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        # For Welford's stats if needed
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions using global_mvn")
            self._count = 0
            self._mean  = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            self._M2    = torch.zeros(self.config['num_feats'], dtype=torch.float64)
        else:
            self._count = None
            self._mean  = None
            self._M2    = None

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # 1) Load feature from (fbank_dir + base_filename)
            fbank_filename = self.fbank_files[i]
            feat_path = os.path.join(self.fbank_dir, fbank_filename)
            feat_np = np.load(feat_path)  # shape => (F, T)

            # truncate to config['num_feats']
            F_needed = self.config['num_feats']
            feat_np = feat_np[:F_needed, :]  
            self.feats.append(feat_np)
            self.feat_max_len = max(self.feat_max_len, feat_np.shape[1])

            # update welford if needed
            if (self.config['norm'] == 'global_mvn') and (global_stats is None):
                feat_tensor = torch.from_numpy(feat_np).float()
                time_len = feat_tensor.shape[1]
                self._count += time_len

                delta = feat_tensor - self._mean.unsqueeze(1)
                self._mean += delta.sum(dim=1) / self._count
                delta2 = feat_tensor - self._mean.unsqueeze(1)
                self._M2 += (delta * delta2).sum(dim=1)

            # 2) If not test, load transcript
            if self.partition != "test-clean":
                text_filename = self.text_files[i]
                text_path = os.path.join(self.text_dir, text_filename)
                text_arr = np.load(text_path, allow_pickle=True)
                transcript_str = "".join(text_arr.tolist())

                self.total_chars += len(transcript_str)
                tokenized = self.tokenizer.encode(transcript_str)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                shifted = [self.sos_token] + tokenized
                golden  = tokenized + [self.eos_token]
                self.transcripts_shifted.append(shifted)
                self.transcripts_golden.append(golden)

        # average chars per token
        self.avg_chars_per_token = (self.total_chars / self.total_tokens) if self.total_tokens > 0 else 0

        # alignment check 
        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # finalize global stats
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                # just use the provided stats
                self.global_mean, self.global_std = global_stats
            else:
                # compute from welford accumulators
                variance = self._M2 / (self._count - 1)
                self.global_std  = torch.sqrt(variance + 1e-8).float()
                self.global_mean = self._mean.float()
        else:
            self.global_mean = None
            self.global_std  = None

        # SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=self.config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=self.config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        # load features from memory
        feat_np = self.feats[idx]
        feat = torch.from_numpy(feat_np).float()  # shape => (F, T)

        # apply normalization
        if self.config['norm'] == 'global_mvn':
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            mu = feat.mean(dim=1, keepdim=True)
            sigma = feat.std(dim=1, keepdim=True) + 1e-8
            feat = (feat - mu) / sigma
        else:
            pass

        if self.partition == "test-clean":
            return feat, None, None
        else:
            shifted = self.transcripts_shifted[idx]
            golden  = self.transcripts_golden[idx]
            return feat, torch.LongTensor(shifted), torch.LongTensor(golden)

    def collate_fn(self, batch):
        """
        Return (padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths).
        padded_feats => (B, T, F), etc.
        """
        feats_list    = []
        feat_lengths  = []
        shifted_list  = []
        golden_list   = []
        transcript_lens = []

        for (feat, shifted, golden) in batch:
            # feat => shape (F, T)
            T = feat.shape[1]
            feat_lengths.append(T)
            feats_list.append(feat.transpose(0,1))  # => shape (T, F)

            if self.partition != "test-clean":
                # store transcripts
                shifted_list.append(shifted)
                golden_list.append(golden)
                transcript_lens.append(len(shifted))

        # pad feats => shape (B, max_time, F)
        padded_feats = pad_sequence(feats_list,
                                    batch_first=True,
                                    padding_value=0.0)
        feat_lengths = torch.LongTensor(feat_lengths)

        # transcript handling
        padded_shifted = None
        padded_golden  = None
        transcript_lengths = None

        if self.partition != "test-clean":
            transcript_lengths = torch.LongTensor(transcript_lens)
            padded_shifted = pad_sequence(shifted_list,
                                          batch_first=True,
                                          padding_value=self.pad_token)
            padded_golden  = pad_sequence(golden_list,
                                          batch_first=True,
                                          padding_value=self.pad_token)

        # specaugment if training
        if self.config["specaug"] and self.isTrainPartition:
            # permute to (B, F, T)
            feats_bft = padded_feats.transpose(1,2)  # (B,F,T)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    feats_bft = self.freq_mask(feats_bft)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    feats_bft = self.time_mask(feats_bft)
            # permute back => (B, T, F)
            padded_feats = feats_bft.transpose(1,2)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
