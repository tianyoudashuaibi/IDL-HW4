from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

'''
The ASRDataset class provides data loading and processing for automatic speech recognition. 
It supports:
1) Reading log Mel filterbank features from .npy files
2) Loading optional text transcripts (all partitions except test-clean)
3) Normalizing features with global or per-utterance statistics
4) Optionally applying SpecAugment transformations
5) Converting transcripts to token IDs, constructing "shifted" and "golden" transcripts
6) Collating into batches for DataLoader usage
'''

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

        Args:
            partition (str): 'train-clean-100', 'dev-clean', or 'test-clean'
            config (dict): Configuration dictionary
            tokenizer (H4Tokenizer): for text encoding/decoding
            isTrainPartition (bool): True if we intend to apply SpecAugment, etc.
            global_stats (tuple, optional): (mean, std) for global normalization
        """
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # 1) Special tokens
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # 2) Setup paths
        #    Root + partition => e.g. "hw4_data/hw4p2_data/train-clean-100/fbank"
        self.fbank_dir = os.path.join(self.config["root"], partition, "fbank")
        # Gather sorted list of .npy feature files
        all_fbank_files = sorted(os.listdir(self.fbank_dir))
        self.fbank_files = [os.path.join(self.fbank_dir, f)
                            for f in all_fbank_files
                            if f.endswith(".npy")]

        # Possibly take subset
        subset_size = self.config.get("subset_size", None)
        if subset_size is not None and subset_size > 0:
            self.fbank_files = self.fbank_files[:subset_size]

        self.length = len(self.fbank_files)

        # If not test-clean, we'll also load transcript info
        if self.partition != "test-clean":
            # text dir
            self.text_dir = os.path.join(self.config["root"], partition, "text")
            all_text_files = sorted(os.listdir(self.text_dir))
            self.text_files = [os.path.join(self.text_dir, f)
                               for f in all_text_files
                               if f.endswith(".npy")]
            if subset_size is not None and subset_size > 0:
                self.text_files = self.text_files[:subset_size]

            # Check alignment
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature files and text files must match!")

        # We'll store everything for convenience
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden  = []

        self.total_chars  = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        # For global MVN if needed
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn.")
            # We'll run Welford's algorithm
            self._count = 0
            self._mean  = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            self._M2    = torch.zeros(self.config['num_feats'], dtype=torch.float64)
        else:
            self._count = None
            self._mean  = None
            self._M2    = None

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # 1) Load feature, shape => (num_feats, time)
            feat_path = self.fbank_files[i]
            feat_np   = np.load(feat_path)  # shape (F, T)
            # 2) Truncate if needed
            F_needed = self.config['num_feats']
            feat_np  = feat_np[:F_needed, :]  # just in case it has more rows

            # store in memory as float32
            self.feats.append(feat_np)
            self.feat_max_len = max(self.feat_max_len, feat_np.shape[1])

            # Update global stats if needed
            if (self.config['norm'] == 'global_mvn' and global_stats is None):
                feat_tensor = torch.from_numpy(feat_np).float()  # (F,T)
                batch_count = feat_tensor.shape[1]
                self._count += batch_count

                # Welford's
                # delta => shape (F, T)
                delta = feat_tensor - self._mean.unsqueeze(1)
                # update mean => (F,)
                self._mean += delta.sum(dim=1) / self._count
                delta2 = feat_tensor - self._mean.unsqueeze(1)
                self._M2 += (delta * delta2).sum(dim=1)

            # 3) If not test partition, load transcript
            if self.partition != "test-clean":
                txt_path = self.text_files[i]
                transcript_arr = np.load(txt_path, allow_pickle=True)
                # e.g. array of characters => join
                transcript_str = "".join(transcript_arr.tolist())

                # track chars
                self.total_chars += len(transcript_str)
                # tokenize
                tokenized = self.tokenizer.encode(transcript_str)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                # build shifted/golden
                shifted = [self.sos_token] + tokenized
                golden  = tokenized + [self.eos_token]
                self.transcripts_shifted.append(shifted)
                self.transcripts_golden.append(golden)

        # average chars per token
        self.avg_chars_per_token = (self.total_chars / self.total_tokens) if self.total_tokens > 0 else 0

        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # finalize global stats
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                # compute final stats
                variance = self._M2 / (self._count - 1)
                std      = torch.sqrt(variance + 1e-8).float()
                mean     = self._mean.float()
                self.global_mean = mean
                self.global_std  = std

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

    def get_avg_chars_per_token(self) -> float:
        '''
        Returns average number of characters per token. 
        Used for perplexity at character level.
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """
        Return number of samples in dataset
        """
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (feat, shifted, golden):
          feat: shape (F, T), float
          shifted: LongTensor or None (for test)
          golden:  LongTensor or None
        """
        feat_np = self.feats[idx]  # (F, T)
        feat = torch.from_numpy(feat_np).float()  # (F, T)

        # We'll do normalization in collate_fn or here? 
        # The instructions say "in __getitem__" or "collate"? 
        # Typically we do it in __getitem__ so we have consistent shape. We'll do it here:

        # We'll wait to do it exactly as the skeleton says:
        # "TODO: Apply normalization" => So let's do it here:

        if self.config['norm'] == 'global_mvn':
            # (F, T) => subtract mean => divide by std
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            # per-utterance mean/var
            # shape => (F, T)
            mu = feat.mean(dim=1, keepdim=True)
            sigma = feat.std(dim=1, keepdim=True) + 1e-8
            feat = (feat - mu) / sigma
        else:
            pass  # no norm

        if self.partition == "test-clean":
            return feat, None, None
        else:
            shifted = self.transcripts_shifted[idx]
            golden  = self.transcripts_golden[idx]
            shifted = torch.LongTensor(shifted)
            golden  = torch.LongTensor(golden)
            return feat, shifted, golden

    def collate_fn(self, batch):
        """
        Collate & pad a batch:
          Returns (padded_features, padded_shifted, padded_golden, feat_lengths, transcript_lengths)
        """
        # "batch" is a list of (feat, shifted, golden)
        # feat => shape (F, T) for each sample
        # we want to produce shape => (B, T, F) eventually (batch_first)...

        # 1) collect feats: we'll transpose each (F, T) -> (T, F) for pad_sequence
        feats_list = []
        feat_lengths = []
        for (feat, shifted, golden) in batch:
            # shape => (F, T)
            T = feat.shape[1]
            feat_lengths.append(T)
            # transpose => (T, F)
            transposed = feat.transpose(0, 1)  # now (time, freq)
            feats_list.append(transposed)

        # Now pad them along time dimension
        # we use pad_sequence with batch_first=True => final shape => (B, max_time, F)
        padded_feats = pad_sequence(feats_list,
                                    batch_first=True,
                                    padding_value=0.0)  # => (B, max_time, F)
        feat_lengths = torch.LongTensor(feat_lengths)     # (B,)

        # next transcripts
        padded_shifted = None
        padded_golden  = None
        transcript_lengths = None

        if self.partition != "test-clean":
            # gather
            shifted_list  = []
            golden_list   = []
            transcript_lens= []
            for (_, shifted, golden) in batch:
                transcript_lens.append(len(shifted))
                shifted_list.append(shifted)
                golden_list.append(golden)

            transcript_lengths = torch.LongTensor(transcript_lens)
            # pad them
            padded_shifted = pad_sequence(shifted_list,
                                          batch_first=True,
                                          padding_value=self.pad_token)  # (B, max_len)
            padded_golden = pad_sequence(golden_list,
                                         batch_first=True,
                                         padding_value=self.pad_token)   # (B, max_len)

        # 2) SpecAugment if training
        if self.config["specaug"] and self.isTrainPartition:
            # we want shape => (B, F, T) => so permute
            # padded_feats => (B, T, F)
            feats_BFT = padded_feats.transpose(1, 2)  # => (B, F, T)

            # freq masking
            if self.config["specaug_conf"]["apply_freq_mask"]:
                n_mask = self.config["specaug_conf"]["num_freq_mask"]
                for _ in range(n_mask):
                    feats_BFT = self.freq_mask(feats_BFT)

            # time masking
            if self.config["specaug_conf"]["apply_time_mask"]:
                n_mask = self.config["specaug_conf"]["num_time_mask"]
                for _ in range(n_mask):
                    feats_BFT = self.time_mask(feats_BFT)

            # permute back => (B, T, F)
            padded_feats = feats_BFT.transpose(1, 2)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
