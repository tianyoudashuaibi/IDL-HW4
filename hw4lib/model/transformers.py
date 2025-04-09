import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary

'''
This file contains two key transformer architectures:

1) DecoderOnlyTransformer: used for LM tasks (GPT-style),
2) EncoderDecoderTransformer: used for ASR tasks (BART-style).

We have removed the "raise NotImplementedError" lines from the constructors so
that the module can be imported without error. You can still fill in the
methods or raise NotImplementedError within them if desired.
'''

## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float, 
        max_len: int, 
        num_classes: int,
        weight_tying: bool = False,
        layer_drop_rate: float = 0.0,
    ):
        """
        Initialize the Decoder-Only Transformer model.
        """
        super().__init__()
        
        # DO NOT MODIFY THESE:
        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers
        
        # Example: define your layer stack
        self.dec_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Embedding + positional encoding + dropout + norm + final linear
        self.target_embedding    = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout             = nn.Dropout(dropout)
        self.norm                = nn.LayerNorm(d_model)
        self.final_linear        = nn.Linear(d_model, num_classes)

        # Optional weight tying
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

        # We do NOT raise an error here to allow import.
        # pass  # If you prefer a no-op

    def forward(
        self, 
        padded_targets: torch.Tensor, 
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for the decoder-only transformer.
        """
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")
        
        # 1) Create padding mask if target_lengths is provided
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths)

        # 2) Causal mask
        causal_mask = CausalMask(padded_targets)

        # 3) Embedding & Positional Encoding & Dropout
        x = self.target_embedding(padded_targets)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # 4) Pass through decoder layers
        running_att = {}
        for i, layer in enumerate(self.dec_layers):
            # Optional layer drop
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                # skip this layer
                continue
            x, att_weights = layer(
                x,
                key_padding_mask=pad_mask_dec,
                attn_mask=causal_mask
            )
            running_att[f'layer{i+1}_dec_self'] = att_weights

        # 5) Final norm + linear
        x = self.norm(x)             # (B,T,d_model)
        seq_out = self.final_linear(x)  # (B,T,num_classes)

        return seq_out, running_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        """
        Inference method for next token: returns last token's logits.
        """
        if self.training:
            raise ValueError("score() is not for training mode.")

        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # last token's logits => shape (B, num_classes)
        logits = seq_out[:, -1, :]
        return logits


## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''
    def __init__(
        self,
        input_dim: int, 
        time_reduction: int, 
        reduction_method: Literal['lstm', 'conv', 'both'], 
        num_encoder_layers: int,
        num_encoder_heads: int,
        d_ff_encoder: int, 
        num_decoder_layers: int,
        num_decoder_heads: int,
        d_ff_decoder: int,
        d_model: int,
        dropout: float, 
        max_len: int, 
        num_classes: int,
        weight_tying: bool = False,
        layer_drop_rate: float = 0.0,
        skip_encoder_pe: bool = False,
        skip_decoder_pe: bool = False,
    ):
        super().__init__()

        # DO NOT MODIFY THESE:
        self.max_len           = max_len
        self.layer_drop_rate   = layer_drop_rate
        self.num_classes       = num_classes
        self.num_encoder_layers= num_encoder_layers
        self.num_decoder_layers= num_decoder_layers
        self.skip_encoder_pe   = skip_encoder_pe
        self.skip_decoder_pe   = skip_decoder_pe

        # 1) Encoder layers (ModuleList)
        self.enc_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(d_model, num_encoder_heads, d_ff_encoder, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 2) Decoder layers (ModuleList)
        self.dec_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(d_model, num_decoder_heads, d_ff_decoder, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 3) Source embedding (speech embedding)
        self.source_embedding = SpeechEmbedding(
            input_dim, 
            d_model, 
            time_reduction, 
            reduction_method, 
            dropout
        )

        # 4) Target embedding
        self.target_embedding = nn.Embedding(num_classes, d_model)

        # 5) Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 6) Final linear layer for decoder
        self.final_linear = nn.Linear(d_model, num_classes)

        # 7) Dropout
        self.dropout = nn.Dropout(dropout)

        # 8) Norm layers for final outputs
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # 9) CTC head: project (d_model -> num_classes) + log_softmax
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # Optional weight tying
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

        # No final NotImplementedError.

    def encode(self, padded_sources: torch.Tensor, source_lengths: torch.Tensor):
        """
        Encode the source features.
        """
        # 1) Speech embedding
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)

        # 2) Optionally apply positional encoding
        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)

        # 3) Dropout
        x_enc = self.dropout(x_enc)

        # 4) Create pad mask
        pad_mask_src = PadMask(x_enc, x_enc_lengths)

        # 5) Pass through encoder layers
        running_att = {}
        for i, layer in enumerate(self.enc_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                # skip the layer
                continue
            x_enc, attn = layer(
                x_enc,
                key_padding_mask=pad_mask_src
            )
            running_att[f'layer{i+1}_enc_self'] = attn

        # 6) Final norm
        x_enc = self.encoder_norm(x_enc)

        # 7) CTC head input
        #   shape for CTC: (T, B, d_model). So we transpose.
        #   but your shape might be (B,T,d_model); be consistent with your data.
        x_enc_for_ctc = x_enc.transpose(0, 1)   # => (src_len, batch_size, d_model)
        ctc_logits = self.ctc_head(x_enc_for_ctc)  # => (src_len, B, num_classes)

        # Return them
        ctc_inputs = {
            'log_probs': ctc_logits,
            'lengths': x_enc_lengths
        }
        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(
        self,
        padded_targets: torch.Tensor,
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None
    ):
        """
        Decode, conditioned on encoder output.
        """
        # 1) Target pad mask
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths)

        if pad_mask_tgt is None and self.training:
            warnings.warn(
                "pad_mask_tgt is None. Typically you pass target_lengths in training."
            )

        # 2) Causal mask
        causal_mask = CausalMask(padded_targets)

        # 3) Embedding
        x_dec = self.target_embedding(padded_targets)
        # Possibly positional encoding
        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)
        # dropout
        x_dec = self.dropout(x_dec)

        # 4) Pass through decoder layers
        running_att = {}
        for i, layer in enumerate(self.dec_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x_dec, self_attn, cross_attn = layer(
                x_dec,
                enc_output=encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask
            )
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        # 5) Final norm + linear projection
        x_dec = self.decoder_norm(x_dec)
        seq_out = self.final_linear(x_dec)   # => (B,T,num_classes)

        return seq_out, running_att

    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for entire encoder-decoder.
        """
        if self.training and target_lengths is None:
            raise ValueError("Need target_lengths in training mode.")
        if self.training and source_lengths is None:
            raise ValueError("Need source_lengths in training mode.")

        # Encode
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = \
            self.encode(padded_sources, source_lengths)

        # Decode
        seq_out, dec_running_att = self.decode(
            padded_targets,
            encoder_output,
            target_lengths=target_lengths,
            pad_mask_src=pad_mask_src
        )

        # Merge attention dicts
        running_att = {**enc_running_att, **dec_running_att}

        # Return final
        return seq_out, running_att, ctc_inputs

    def score(
        self,
        batch_prompts: torch.Tensor,
        encoder_output: torch.Tensor,
        pad_mask_src: torch.Tensor
    ) -> torch.Tensor:
        """
        Score the next token for a given encoder output & partial decode.
        """
        if self.training:
            raise ValueError("score() is not for training mode")

        # decode w/o target_lengths => no pad mask for decode
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        return seq_out[:, -1, :]


    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_checkpoint_path: str,
        config: dict,
    ):
        """
        Helper: load partial weights from a decoder-only checkpoint.
        """
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")
        
        # 1) Create new encoder-decoder
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        # 2) Load decoder checkpoint
        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=True)
        decoder_state_dict = checkpoint['model_state_dict']

        transferred_params = []
        new_params = []

        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            param_count = sum(p.numel() for p in target_module.parameters())
            print(f"  - Transferring {prefix} ({param_count:,} params)")
            target_module.load_state_dict(module_state_dict)
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        # 3) Transfer shared components
        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')

        # 4) Transfer decoder layers
        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        print(f"\nTransferring decoder layers (found {num_layers} layers):")
        for i in range(num_layers):
            print(f"\nLayer {i+1}/{num_layers}:")
            transfer_module_weights(
                model.dec_layers[i].self_attn,
                f'dec_layers.{i}.self_attn.'
            )
            transfer_module_weights(
                model.dec_layers[i].ffn,
                f'dec_layers.{i}.ffn.'
            )

        # 5) Collect new params
        print("\nCollecting new parameters...")
        for name, param in model.named_parameters():
            is_new = True
            for (transferred_name, transferred_param) in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))

        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups: list) -> None:
        """
        Log param group info.
        """
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable  = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable

            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")

## -------------------------------------------------------------------------------------------------
## Test code
## -------------------------------------------------------------------------------------------------
def get_decoder_only_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths

def get_encoder_decoder_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths

def test_decoder_only(num_layers: int = 12, num_heads: int = 8, d_model: int = 512,
                      d_ff: int = 2048, dropout: float = 0.1, max_len: int = 300,
                      num_classes: int = 1000):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])

if __name__ == "__main__":
    test_decoder_only()
