from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from ..utils import create_scheduler
from ..decoding.sequence_generator import SequenceGenerator

class LMTrainer(BaseTrainer):
    """
    Language Model Trainer class that handles the training, validation, and generation loops.

    This trainer implements:
    1. Training loop with gradient accumulation and mixed precision training
    2. Validation loop for model evaluation
    3. Generation capabilities with different decoding strategies
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        # -- 1) Initialize the criterion
        ignore_index = self.tokenizer.pad_id
        label_smoothing = self.config['training'].get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
        Returns:
            (training_metrics, attn_weights)
        """

        self.model.train()
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc="[Training LM]"
        )

        running_ce_loss = 0.0
        total_tokens = 0
        attn_weights = {}  # We'll store the last batch's attn if you wish

        # Zero grads once at the start
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Unpack batch
            targets_shifted, targets_golden, lengths = batch
            # Move to device
            targets_shifted = targets_shifted.to(self.device)
            targets_golden  = targets_golden.to(self.device)
            lengths         = lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                # Forward pass
                raw_preds, attn_w = self.model.forward(targets_shifted, target_lengths=lengths)
                # raw_preds: (B, T, vocab)
                # targets_golden:  (B, T)
                # Flatten for CrossEntropyLoss
                B, T, V = raw_preds.shape
                raw_preds_2d = raw_preds.reshape(B*T, V)
                golden_1d    = targets_golden.reshape(-1)

                raw_loss = self.criterion(raw_preds_2d, golden_1d)

            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += raw_loss.item() * batch_tokens

            # Normalize by grad_accum_steps if needed
            grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
            loss = raw_loss / grad_accum_steps

            # Backprop
            self.scaler.scale(loss).backward()

            # Update only after enough steps
            if (i + 1) % grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # Just store the last batch's attention if you want
            attn_weights = attn_w

            # Metrics for progress bar
            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
                acc_step=f"{(i % grad_accum_steps)+1}/{grad_accum_steps}"
            )
            batch_bar.update()

            del targets_shifted, targets_golden, lengths, raw_preds_2d, golden_1d, raw_preds, loss
            torch.cuda.empty_cache()

        # Handle leftover partial accum
        if (len(dataloader) % grad_accum_steps) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        # Final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char  = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        metrics = {
            'ce_loss_token':     avg_ce_loss,
            'ce_loss_char':      avg_ce_loss_char,
            'perplexity_token':  avg_perplexity_token.item(),
            'perplexity_char':   avg_perplexity_char.item(),
        }
        return metrics, attn_weights

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        
        Returns:
            (validation_metrics, attn_weights)
        """

        self.model.eval()
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc="[Validating LM]"
        )

        running_ce_loss = 0.0
        total_tokens = 0
        attn_weights = {}

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                targets_shifted, targets_golden, lengths = batch
                targets_shifted = targets_shifted.to(self.device)
                targets_golden  = targets_golden.to(self.device)
                lengths         = lengths.to(self.device)

                raw_preds, attn_w = self.model.forward(targets_shifted, target_lengths=lengths)
                B, T, V = raw_preds.shape
                raw_preds_2d = raw_preds.reshape(B*T, V)
                golden_1d    = targets_golden.reshape(-1)

                loss = self.criterion(raw_preds_2d, golden_1d)

                batch_tokens = lengths.sum().item()
                total_tokens += batch_tokens
                running_ce_loss += loss.item() * batch_tokens

                attn_weights = attn_w  # store last batch's attn

                avg_ce_loss = running_ce_loss / total_tokens
                perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
                batch_bar.set_postfix(
                    ce_loss_token=f"{avg_ce_loss:.4f}",
                    perplexity_token=f"{perplexity_token:.4f}"
                )
                batch_bar.update()

                del targets_shifted, targets_golden, lengths, raw_preds_2d, golden_1d, raw_preds, loss
                torch.cuda.empty_cache()

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char  = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        metrics = {
            'ce_loss_token':     avg_ce_loss,
            'ce_loss_char':      avg_ce_loss_char,
            'perplexity_token':  avg_perplexity_token.item(),
            'perplexity_char':   avg_perplexity_char.item(),
        }
        return metrics, attn_weights

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for LM training.
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized!")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized!")

        best_val_loss = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            # 1) Train
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            # 2) Validate
            val_metrics, val_attn = self._validate_epoch(val_dataloader)
            # 3) Generate (just an example usage)
            gen_results = self.generate(val_dataloader, None)

            # If we use ReduceLROnPlateau, step with val_loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['ce_loss_char'])

            # Logging
            metrics = {
                'train': train_metrics,
                'val':   val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Save some attn plots for debugging
            if train_attn and len(train_attn) > 0:
                any_key = list(train_attn.keys())[0]
                self._save_attention_plot(train_attn[any_key][0], epoch, "train_self")
            if val_attn and len(val_attn) > 0:
                any_key = list(val_attn.keys())[0]
                self._save_attention_plot(val_attn[any_key][0], epoch, "val_self")

            # Save generated text
            self._save_generated_text(gen_results, f'val_epoch_{epoch}')

            # Save model each epoch
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            # Check if best
            val_loss = val_metrics['ce_loss_char']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metric = val_loss
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def evaluate(self, test_dataloader):
        """
        Evaluate the model on the test set.
        """
        test_metrics, test_attn = self._validate_epoch(test_dataloader)
        metrics = { 'test': test_metrics }
        self._log_metrics(metrics, self.current_epoch)

        # Save attention
        if test_attn and len(test_attn) > 0:
            any_key = list(test_attn.keys())[0]
            self._save_attention_plot(test_attn[any_key][0], self.current_epoch, "test_self")

        # Possibly run generate configs
        generation_results = {}
        eval_configs = self._get_evaluation_generation_configs()
        for cname, cset in eval_configs.items():
            try:
                gen_results = self.generate(test_dataloader, generation_config=cset)
                generation_results[cname] = gen_results
                self._save_generated_text(gen_results, f'test_epoch_{self.current_epoch}_{cname}')
            except Exception as e:
                print(f"Could not generate results for {cname}: {e}")
                continue
        return test_metrics, generation_results

    def generate(self, dataloader, generation_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating sequences from prompts.
        """
        if generation_config is None:
            # Default to simple greedy
            generation_config = {
                'num_samples': 10,
                'prompt_length': 20,
                'seed': 11785,
                'max_length': self.model.max_len,
                'temperature': 1.0,
                'beam_width': 1,
                'repeat_penalty': 1.0,
                'top_k': 0,
                'top_p': 0.0
            }

        # Prepare SequenceGenerator
        generator = SequenceGenerator(
            score_fn=lambda x: self.model.score(x),
            tokenizer=self.tokenizer,
            max_length=self.model.max_len,
            device=self.device
        )

        # Sample some random prompts from the dataset
        prompts, originals = dataloader.dataset.sample_prompts(
            num_samples=generation_config.get('num_samples', 10),
            prompt_length=generation_config.get('prompt_length', 10),
            seed=generation_config.get('seed', 11785)
        )
        prompts = prompts.to(self.device)

        # Start generation
        self.model.eval()
        with torch.inference_mode():
            # 1) If top_k or top_p > 0 => sampling
            if generation_config.get('top_k', 0) > 0 or generation_config.get('top_p', 0) > 0:
                # placeholder if you want to use sampling
                seqs, scores = generator.generate_sample(
                    prompts,
                    temperature=generation_config.get('temperature', 1.0),
                    top_k=generation_config.get('top_k', 0),
                    top_p=generation_config.get('top_p', 1.0),
                )
            # 2) If beam_width > 1 => beam
            elif generation_config.get('beam_width', 1) > 1:
                seqs, scores = generator.generate_beam(
                    prompts,
                    beam_width=generation_config.get('beam_width', 5),
                    temperature=generation_config.get('temperature', 1.0),
                    repeat_penalty=generation_config.get('repeat_penalty', 1.0)
                )
                # We take the best beam (the first one)
                seqs = seqs[:, 0, :]  # shape (B, final_len)
                scores = scores[:, 0]
            # 3) Otherwise => greedy
            else:
                seqs, scores = generator.generate_greedy(
                    prompts,
                    temperature=generation_config.get('temperature', 1.0),
                    repeat_penalty=generation_config.get('repeat_penalty', 1.0)
                )

        # post-process
        processed_seqs = generator.post_process_sequence(seqs, self.tokenizer)

        # prepare results
        results = []
        for i, (prompt, seq, score, orig) in enumerate(zip(prompts, processed_seqs, scores, originals)):
            prompt_str    = self.tokenizer.decode(prompt.tolist())
            # Everything after prompt
            generated_str = self.tokenizer.decode(seq[len(prompt):].tolist(), skip_special_tokens=False)
            # Original str after the prompt
            orig_str      = self.tokenizer.decode(orig[len(prompt):].tolist(), skip_special_tokens=False)
            results.append({
                'prompt':    prompt_str,
                'original':  orig_str,
                'generated': generated_str,
                'score':     score.item() if isinstance(score, torch.Tensor) else float(score)
            })
        return results

    def _get_evaluation_generation_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Generation config for evaluating on test set.
        """
        common_config = {
            'num_samples':   50,
            'prompt_length': 10,
            'seed':          11785,
            'max_length':    self.model.max_len,
        }
        
        greedy_config = common_config.copy()
        greedy_config.update({
            'temperature':    1.0,
            'beam_width':     1,
            'repeat_penalty': 1.0,
            'top_k':          0,
            'top_p':          0.0
        })
        
        beam_config = common_config.copy()
        beam_config.update({
            'temperature':    1.0,
            'beam_width':     10,
            'repeat_penalty': 1.2,
            'top_k':          0,
            'top_p':          0.0
        })

        sample_config = common_config.copy()
        sample_config.update({
            'temperature':    1.0,
            'beam_width':     1,
            'repeat_penalty': 1.0,
            'top_k':          10,
            'top_p':          0.95
        })
        
        return {
            'greedy': greedy_config,
            'beam':   beam_config,
            'sample': sample_config
        }
