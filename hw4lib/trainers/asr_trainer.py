from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler, create_optimizer
from ..model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
import torchmetrics.text as tmt
from torch.utils.data import Subset
import pandas as pd

class ASRTrainer(BaseTrainer):
    """
    ASR (Automatic Speech Recognition) Trainer class that handles training, validation, and recognition loops.
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        # 1) Initialize CE loss
        #    Use label_smoothing from self.config['loss']['label_smoothing']
        #    and ignore_index with the tokenizer.pad_id
        label_smoothing = self.config['loss'].get('label_smoothing', 0.0)
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=label_smoothing
        )

        # 2) Initialize CTC loss if ctc_weight > 0
        self.ctc_criterion = None
        self.ctc_weight = self.config['loss'].get('ctc_weight', 0.0)
        if self.ctc_weight > 0:
            # Use pad token as the blank index
            self.ctc_criterion = nn.CTCLoss(
                blank=self.tokenizer.pad_id,
                zero_infinity=True
            )

        # Done with initialization
        # Remove the NotImplementedError
        # raise NotImplementedError  # (Removed!)

    def _train_epoch(self, dataloader):
        """
        Train for one epoch.
        """
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training ASR]")

        running_ce_loss   = 0.0
        running_ctc_loss  = 0.0
        running_joint_loss= 0.0
        total_tokens      = 0
        running_att       = {}

        # We zero grad once at start
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Unpack batch & move to device
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch
            feats             = feats.to(self.device)
            feat_lengths      = feat_lengths.to(self.device)
            # targets can be None for test set, but we assume training => not test
            targets_shifted   = targets_shifted.to(self.device)
            targets_golden    = targets_golden.to(self.device)
            transcript_lengths= transcript_lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                # forward pass
                # Expect model to return (seq_out, attn_dict, ctc_inputs)
                seq_out, curr_att, ctc_inputs = self.model(
                    padded_sources=feats,
                    padded_targets=targets_shifted,
                    source_lengths=feat_lengths,
                    target_lengths=transcript_lengths
                )

                # track attn
                running_att = curr_att

                # Compute CE loss
                B, T, V = seq_out.shape
                seq_out_2d   = seq_out.reshape(B*T, V)
                golden_1d    = targets_golden.reshape(B*T)
                ce_loss = self.ce_criterion(seq_out_2d, golden_1d)

                # Possibly compute ctc loss
                if self.ctc_weight > 0.0 and self.ctc_criterion is not None and ctc_inputs is not None:
                    # ctc_inputs['log_probs'] => shape (T_enc, B, vocab)
                    # ctc_inputs['lengths']   => shape (B,)
                    ctc_log_probs = ctc_inputs['log_probs']
                    input_lengths = ctc_inputs['lengths']   # (B,)
                    # target_lengths => transcript_lengths (B,)
                    # targets_golden => shape (B,T) -> flatten?
                    # ctc expects shape (B * T?)
                    # Actually ctc wants the targets as 1D, with no extra special tokens
                    # We'll just flatten except for pads
                    # But we do have to remove the final EOS? Or do we pass them as is, ignoring pad token?
                    # In a typical pipeline, we pass the "golden" minus the final EOS, ignoring pad?
                    # We'll keep it simple & pass golden but remove SOS/EOS/pad or just keep them ignoring them?
                    # We'll do it typical: raw ctc expects (T,B,vocab) so it's good. 
                    # We just flatten 'targets_golden' and remove pad. 
                    # But let's be consistent with how ctc is typically used in the code. 
                    # We'll do the standard approach:
                    ctc_targets = []
                    ctc_tgt_lengths = []
                    for b_i in range(B):
                        # Get the unpadded tokens from 'targets_golden[b_i]'
                        arr = targets_golden[b_i]
                        # remove trailing pads
                        # also remove final EOS if desired
                        arr_valid = arr[arr != self.tokenizer.pad_id]
                        # If we want to remove the last EOS:
                        if len(arr_valid) > 0 and arr_valid[-1].item() == self.tokenizer.eos_id:
                            arr_valid = arr_valid[:-1]
                        ctc_targets.append(arr_valid)
                        ctc_tgt_lengths.append(len(arr_valid))

                    ctc_concat = torch.cat(ctc_targets).to(self.device)
                    ctc_tgt_lengths = torch.LongTensor(ctc_tgt_lengths).to(self.device)

                    ctc_loss = self.ctc_criterion(
                        ctc_log_probs,   # (T_enc, B, vocab)
                        ctc_concat,      # 1D
                        input_lengths,   # (B,)
                        ctc_tgt_lengths  # (B,)
                    )
                    loss = ce_loss + self.ctc_weight * ctc_loss
                else:
                    ctc_loss = torch.tensor(0.0, device=self.device)
                    loss = ce_loss

            # metrics
            batch_tokens = transcript_lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += ce_loss.item() * batch_tokens
            if self.ctc_weight > 0:
                running_ctc_loss += ctc_loss.item() * batch_tokens
            running_joint_loss += loss.item() * batch_tokens

            # scale by grad accum
            loss = loss / self.config['training']['gradient_accumulation_steps']
            self.scaler.scale(loss).backward()

            # update after enough steps
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # progress bar
            avg_ce_loss = running_ce_loss / total_tokens
            avg_ctc_loss= running_ctc_loss / total_tokens
            avg_joint_loss = running_joint_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_ce_loss))

            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ctc_loss=f"{avg_ctc_loss:.4f}",
                joint_loss=f"{avg_joint_loss:.4f}",
                perplexity=f"{perplexity:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps'])+1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # cleanup
            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del seq_out, curr_att, ctc_inputs, loss
            torch.cuda.empty_cache()

        # leftover partial accum
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        # final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ctc_loss= running_ctc_loss / total_tokens
        avg_joint_loss = running_joint_loss / total_tokens
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char  = torch.exp(torch.tensor(avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()))
        batch_bar.close()

        return {
            'ce_loss':       avg_ce_loss,
            'ctc_loss':      avg_ctc_loss,
            'joint_loss':    avg_joint_loss,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char':  avg_perplexity_char.item()
        }, running_att

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        We'll use self.recognize(...) to get results, then compute WER/CER.
        """
        self.model.eval()

        # get recognition results
        # Typically we'd do a small config with beam_width=1 or something
        # but we can just do greedy. Let's do that by default:
        results = self.recognize(
            dataloader=dataloader,
            recognition_config={
                'beam_width': 10,
                'num_batches': None
            },
            config_name='beam',
            max_length=getattr(self, 'text_max_len', 200)
        )

        # references & hypotheses
        references = []
        hypotheses = []
        for r in results:
            if 'target' in r: 
                references.append(r['target'])
                hypotheses.append(r['generated'])

        # if references is empty => it's test set => no metrics
        if len(references) == 0:
            return {
                'word_dist': 0.0,
                'wer': 0.0,
                'cer': 0.0
            }, results

        # compute metrics
        metrics = self._calculate_asr_metrics(references, hypotheses)
        return metrics, results

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for ASR
        """
        if self.scheduler is None:
            raise ValueError("Scheduler not initialized!")
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized!")

        # max transcript length
        self.text_max_len = max(val_dataloader.dataset.text_max_len, train_dataloader.dataset.text_max_len)

        best_val_cer = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            # 1) train
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            # 2) validate
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            # step scheduler if needed
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])

            # log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Save an attention plot
            # If we have attn, let's just pick any key
            if train_attn:
                # for instance, pick the first item in dict
                some_key = next(iter(train_attn.keys()))
                layer_attn = train_attn[some_key]  # shape (B, T, T)
                self._save_attention_plot(layer_attn[0], epoch, "train_attn")

            # Save generated text from val
            self._save_generated_text(val_results, f'val_epoch_{epoch}')

            # Save checkpoint
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            # check if best
            if val_metrics['cer'] < best_val_cer:
                best_val_cer = val_metrics['cer']
                self.best_metric = best_val_cer
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def evaluate(self, dataloader, max_length: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on test set. 
        We'll do a few recognition configs, store results in DataFrame, etc.
        """
        # We'll do the set of beam widths or just do a single?
        recognition_configs = self._get_evaluation_recognition_configs()
        eval_results = {}

        for config_name, config in recognition_configs.items():
            try:
                print(f"Evaluating with {config_name} config")
                results = self.recognize(dataloader, config, config_name, max_length)
                # compile DataFrame
                generated = [r['generated'] for r in results]
                results_df = pd.DataFrame({
                    'id': range(len(generated)),
                    'transcription': generated
                })
                eval_results[config_name] = results_df
                self._save_generated_text(results, f'test_{config_name}_results')
            except Exception as e:
                print(f"Error evaluating with {config_name} config: {e}")
                continue

        return eval_results

    def recognize(self, dataloader, recognition_config: Optional[Dict[str, Any]] = None, config_name: Optional[str] = None, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating transcriptions from audio features.
        If beam_width>1, do beam search (if implemented). If beam=1 => greedy. 
        If we have LM => do shallow fusion by adding LM logits scaled by lm_weight.
        """
        if max_length is None and not hasattr(self, 'text_max_len'):
            raise ValueError("text_max_len is not set. Please run training loop first or provide max_length")

        if recognition_config is None:
            recognition_config = {
                'num_batches': None,
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }
            config_name = 'greedy'

        if recognition_config.get('lm_model'):
            recognition_config['lm_model'].eval()
            recognition_config['lm_model'].to(self.device)

        generator = SequenceGenerator(
            score_fn=None,
            tokenizer=self.tokenizer,
            max_length=max_length if max_length else self.text_max_len,
            device=self.device
        )

        self.model.eval()
        results = []
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Recognizing ASR]:{config_name}")

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                feats, _, targets_golden, feat_lengths, transcript_lengths = batch
                feats          = feats.to(self.device)
                feat_lengths   = feat_lengths.to(self.device)
                if targets_golden is not None:
                    targets_golden = targets_golden.to(self.device)

                # forward to get encoder output
                encoder_output, pad_mask_src, _, _ = self.model.encode(feats, feat_lengths)

                # define a scoring function for the generator
                def get_score(prompt_tokens):
                    # shape => (B, seq_len)
                    asr_logits = self.model.score(prompt_tokens, encoder_output, pad_mask_src)
                    if recognition_config.get('lm_model') is not None:
                        # shallow fusion
                        lm_logits = recognition_config['lm_model'].score(prompt_tokens)
                        return asr_logits + recognition_config['lm_weight']*lm_logits
                    return asr_logits

                generator.score_fn = get_score

                # create a batch of prompts => shape (B,1) all = sos
                B = feats.size(0)
                prompts = torch.full((B,1), self.tokenizer.sos_id, dtype=torch.long, device=self.device)

                # decode
                if recognition_config['beam_width'] > 1:
                    seqs, scores = generator.generate_beam(
                        prompts,
                        beam_width=recognition_config['beam_width'],
                        temperature=recognition_config.get('temperature', 1.0),
                        repeat_penalty=recognition_config.get('repeat_penalty', 1.0)
                    )
                    # pick best beam
                    seqs = seqs[:,0,:]
                    scores = scores[:,0]
                else:
                    seqs, scores = generator.generate_greedy(
                        prompts,
                        temperature=recognition_config.get('temperature', 1.0),
                        repeat_penalty=recognition_config.get('repeat_penalty', 1.0)
                    )

                # post-process
                post_processed_preds = generator.post_process_sequence(seqs, self.tokenizer)

                # store
                if targets_golden is not None:
                    # post-process targets
                    post_processed_targets = generator.post_process_sequence(targets_golden, self.tokenizer)
                    for j, (pred, tgt) in enumerate(zip(post_processed_preds, post_processed_targets)):
                        results.append({
                            'target':    self.tokenizer.decode(tgt.tolist(), skip_special_tokens=True),
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })
                else:
                    for j, pred in enumerate(post_processed_preds):
                        results.append({
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })

                batch_bar.update()
                if recognition_config['num_batches'] is not None:
                    if i >= recognition_config['num_batches'] - 1:
                        break

        batch_bar.close()
        return results

    def _get_evaluation_recognition_configs(self, lm_model: Optional[DecoderOnlyTransformer] = None, lm_weight: float = 0.0) -> Dict[str, Dict[str, Any]]:
        """
        Return some standard decoding configs
        """
        common_config = {
            'num_batches': None,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': lm_weight,
            'lm_model': lm_model
        }
        greedy_config = common_config.copy()
        greedy_config.update({'beam_width': 1})

        beam_10_config = common_config.copy()
        beam_10_config.update({'beam_width': 10})

        beam_20_config = common_config.copy()
        beam_20_config.update({'beam_width': 20})

        return {
            'greedy': greedy_config,
            'beam_10': beam_10_config,
            'beam_20': beam_20_config
        }

    def _calculate_asr_metrics(self, references, hypotheses):
        """
        references: list of strings
        hypotheses: list of strings
        returns { 'word_dist':..., 'wer':..., 'cer':... }
        """
        if len(references) == 0:
            return { 'word_dist':0.0, 'wer':0.0, 'cer':0.0 }

        wer_metric      = tmt.WordErrorRate()
        word_edit_metric= tmt.EditDistance(reduction='mean')
        cer_metric      = tmt.CharErrorRate()

        word_dist = word_edit_metric(hypotheses, references)
        wer       = wer_metric(hypotheses, references)*100.0
        cer       = cer_metric(hypotheses, references)*100.0

        return {
            'word_dist': word_dist.item(),
            'wer': wer.item(),
            'cer': cer.item()
        }
class ProgressiveTrainer(ASRTrainer):
    """
    Progressive Trainer class that implements curriculum learning for ASR training.

    This trainer extends ASRTrainer to implement:
    1. Stage-based training with increasing model complexity
    2. Gradual unfreezing of model layers
    3. Dynamic data subsetting
    4. Smooth transition to full model training

    Implementation Tasks:
    - Store original model layers in __init__
    - Configure model for each stage in configure_stage
    - Implement progressive training loop in progressive_train
    - Handle transition to full training in transition_to_full_training
    - Create data subsets in get_subset_dataloader

    Implementation Notes:
    1. For __init__:
        - Store original encoder and decoder layers
        - Initialize stage counter
        
    2. For configure_stage:
        - Update dropout and label smoothing
        - Activate specified encoder and decoder layers
        - Handle layer freezing based on configuration
        - Print detailed configuration information
        
    3. For progressive_train:
        - Configure model for each stage
        - Create appropriate data subset
        - Train using parent class methods
        
    4. For transition_to_full_training:
        - Restore all model layers
        - Reset loss function parameters
        - Unfreeze all parameters
        - Reset best metrics
        
    5. For get_subset_dataloader:
        - Create subset while preserving dataset attributes
        - Maintain collate function and other dataloader settings

    # -------------------------------------------------------------------------------------------------
    ##### Stage Configuration

    Each stage is defined as a dictionary with the following parameters:
    ```python
    {
        'name': str,                        # Name of the training stage
        'epochs': int,                      # Number of epochs to train in this stage
        'encoder_active_layers': List[int], # Which encoder layers to use
        'decoder_active_layers': List[int], # Which decoder layers to use
        'encoder_freeze': List[bool],       # Whether to freeze each encoder layer
        'decoder_freeze': List[bool],       # Whether to freeze each decoder layer
        'dropout': float,                   # Dropout rate for this stage
        'label_smoothing': float,           # Label smoothing value
        'data_subset': float                # Fraction of training data to use (0.0-1.0)
    }
    ```
    #### Example
    It is best understood by an example. Here is a breakdown of the stages defined below for a model with 6 encoder and 6 decoder layers:

    stages = [
                {
                    # `Initial (1 layers)`:
                    # This stage starts with a model with only 1 encoder and 1 decoder layer.
                    # No freezing or regularization is applied.
                    # It uses 20% of the training data.
                    'name': 'Initial (1 Encoder + 1 Decoder layers)',
                    'epochs': 5,
                    'encoder_active_layers': list(range(1)),
                    'decoder_active_layers': list(range(1)),
                    'encoder_freeze': [False],
                    'decoder_freeze': [False],
                    'dropout': 0.0,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `2 layers`:
                    # This stage increases the number of layers to 2 for both the encoder and decoder.
                    # The previous layer (encoder layer 1 and decoder layer 1) are frozen.
                    # No regularization is applied.
                    # It uses 20% of the training data.
                    'name': '2 Encoder + 2 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(2)),
                    'decoder_active_layers': list(range(2)),
                    'encoder_freeze': [True, False],
                    'decoder_freeze': [True, False],
                    'dropout': 0.0,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `4 layers`:
                    # This stage increases the number of layers to 4 for both the encoder and decoder.
                    # The previous layers (encoder layers 1 and 2 and decoder layers 1 and 2) are frozen.
                    # Dropout is set to 0.05 and label smoothing is set to 0.0.
                    # It uses 20% of the training data.
                    'name': '4 Encoder + 4 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(4)),
                    'decoder_active_layers': list(range(4)),
                    'encoder_freeze': [True, True, False, False],
                    'decoder_freeze': [True, True, False, False],
                    'dropout': 0.05,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `All 6 layers`:
                    # This stage uses all 6 encoder and 6 decoder layers.
                    # The 4 previous layers are frozen and the last 2 layers are trained.
                    # Dropout is set to 0.1 and label smoothing is set to 0.0.
                    # It uses 20% of the training data.
                    'name': '6 Encoder + 6 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(6)),
                    'decoder_active_layers': list(range(6)),
                    'encoder_freeze': [True, True, True, True, False, False],
                    'decoder_freeze': [True, True, True, True, False, False],
                    'dropout': 0.1,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `Final (with label smoothing)`:
                    # This stage uses all 6 encoder and 6 decoder layers.
                    # All layers are unfrozen and trained.
                    # Dropout is set to 0.1 and label smoothing is set to 0.1.
                    # It uses 20% of the training data.
                    'name': 'Final (with label smoothing)',
                    'epochs': 5,
                    'encoder_active_layers': list(range(6)),
                    'decoder_active_layers': list(range(6)),
                    'encoder_freeze': [False, False, False, False, False, False],
                    'decoder_freeze': [False, False, False, False, False, False],
                    'dropout': 0.1,
                    'label_smoothing': 0.1,
                    'data_subset': 0.2
                }
            ]    

    ##### Important Notes
    - Ensure `encoder_freeze` and `decoder_freeze` lists match the length of their respective `active_layers`
    - `data_subset` should be between 0 and 1
    - Stage transitions are handled automatically by the trainer
    - The same optimizer and scheduler are used for all stages so keep that in mind while setting the learning rates and other parameters
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.current_stage = 0
        # Store original layer states
        self.all_encoder_layers = list(self.model.enc_layers)
        self.all_decoder_layers = list(self.model.dec_layers)


    def configure_stage(self, stage_config):
        """Configure model for current training stage"""
        # Create a pretty header
        print("\n" + "="*80)
        print(f"Starting Stage: {stage_config['name']}".center(80))
        print("="*80)
        
        # Print key configuration details
        print(f"\nConfiguration Details:")
        print(f"├── Data Subset: {stage_config['data_subset']*100:.1f}% of training data")
        print(f"├── Training Epochs: {stage_config['epochs']}")
        print(f"├── Dropout: {stage_config['dropout']}")
        print(f"├── Label Smoothing: {stage_config['label_smoothing']}")
        
        # Update dropout and label smoothing
        self.model.dropout.p = stage_config['dropout']
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=stage_config['label_smoothing']
        )
        
        # Get freeze configurations
        encoder_freeze = stage_config.get('encoder_freeze', [])
        decoder_freeze = stage_config.get('decoder_freeze', [])
        
        # Activate and configure encoder layers
        encoder_active_layers = stage_config['encoder_active_layers']
        if encoder_freeze and len(encoder_freeze) != len(encoder_active_layers):
            raise ValueError(f"Encoder freeze list length ({len(encoder_freeze)}) must match number of active encoder layers ({len(encoder_active_layers)})")
        
        # Set the active encoder layers of the model
        self.model.enc_layers = nn.ModuleList([
            self.all_encoder_layers[i] for i in encoder_active_layers
        ])
        self.model.num_encoder_layers = len(encoder_active_layers)
        
        # Activate and configure decoder layers
        decoder_active_layers = stage_config['decoder_active_layers']
        if decoder_freeze and len(decoder_freeze) != len(decoder_active_layers):
            raise ValueError(f"Decoder freeze list length ({len(decoder_freeze)}) must match number of active decoder layers ({len(decoder_active_layers)})")
        
        # Set the active decoder layers of the model
        self.model.dec_layers = nn.ModuleList([
            self.all_decoder_layers[i] for i in decoder_active_layers
        ])
        self.model.num_decoder_layers = len(decoder_active_layers)

        # Handle layer freezing
        frozen_count = 0
        trainable_count = 0
        
        # Configure encoder layers freezing
        print("├── Encoder Layers:")
        for idx, layer in enumerate(self.model.enc_layers):
            should_freeze = encoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {encoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        # Configure decoder layers
        print("├── Decoder Layers:")
        for idx, layer in enumerate(self.model.dec_layers):
            should_freeze = decoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {decoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        print(f"├── Frozen Parameters: {frozen_count:,}")
        print(f"└── Trainable Parameters: {trainable_count:,}")
    

    def progressive_train(self, train_dataloader, val_dataloader, stages: List[Dict[str, Any]]):
        """
        Progressive training through stages
        Each stage configuration is defined as a dictionary with the following parameters:

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            stages: List of dictionaries containing stage configuration
        """
        # Train through stages
        for stage_idx, stage_config in enumerate(stages):
            self.current_stage = stage_idx
            self.configure_stage(stage_config)
            # Get subset of train_dataloader
            subset_train_dataloader = self.get_subset_dataloader(train_dataloader, stage_config['data_subset'])
            super().train(subset_train_dataloader, val_dataloader, epochs=stage_config['epochs'])

    def transition_to_full_training(self):
        """Transition from progressive training to full training"""
        print("\n=== Transitioning to Full Training ===")
        
        # Restore all layers
        self.model.enc_layers = nn.ModuleList(self.all_encoder_layers)
        self.model.dec_layers = nn.ModuleList(self.all_decoder_layers)
        self.model.num_encoder_layers = len(self.all_encoder_layers)
        self.model.num_decoder_layers = len(self.all_decoder_layers)

        # Restore CrossEntropyLoss
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss']['label_smoothing']
        )
        
        # Unfreeze all parameters
        unfrozen_count = 0
        for param in self.model.parameters():
            param.requires_grad = True
            unfrozen_count += param.numel()
        print(f"├── Total Unfrozen Parameters: {unfrozen_count:,}")
        
        # Reset best metrics for new training phase
        self.best_metric = float('inf')

    
    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Run full training phase.
        It is recommended to set the optimizer and scheduler explicitly before calling this function.
        like this:
        cls.optimizer = create_optimizer(self.model, self.config['optimizer'])
        cls.scheduler = create_scheduler(cls.optimizer, cls.config['scheduler'], train_dataloader)
        cls.progressive_train(train_dataloader, val_dataloader, stages)
        """
        self.transition_to_full_training()
        super().train(train_dataloader, val_dataloader, epochs=epochs)


    def get_subset_dataloader(self, dataloader, subset_fraction):
        """
        Creates a new DataLoader with a subset of the original data while preserving dataset attributes.
        
        Args:
            dataloader: Original DataLoader
            subset_fraction: Float between 0 and 1 indicating what fraction of data to keep
        
        Returns:
            New DataLoader containing only the subset of data
        """
        # Calculate how many samples we want to keep
        dataset = dataloader.dataset
        total_samples = len(dataset)
        subset_size = int(total_samples * subset_fraction)
        
        # Create random indices for the subset
        indices = torch.randperm(total_samples)[:subset_size]
        
        # Create a Subset dataset
        subset_dataset = Subset(dataset, indices)
        
        # Add necessary attributes from original dataset to subset
        subset_dataset.text_max_len = dataset.text_max_len
        subset_dataset.feat_max_len = dataset.feat_max_len
        subset_dataset.get_avg_chars_per_token = dataset.get_avg_chars_per_token
        
        # Create new DataLoader with same configuration as original
        subset_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['NUM_WORKERS'],
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
        
        return subset_loader
        
