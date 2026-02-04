import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tron_support.flash_attention import FlashAttentionWithTP
import tron_support.process_group_manager as pgm
from tron_support.tensor_parallel_v.tensor_parallel import VocabParallelEmbedding
from rmsnorm import RMSNorm

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.reset_parameters()
    
    def reset_parameters(self):
        def _init_weights(tensor):
            k = 1 / tensor.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(tensor, -bound, bound)

        _init_weights(self.up_proj.weight)
        _init_weights(self.gate_proj.weight)
        _init_weights(self.down_proj.weight)
 
    def forward(self, x):
        gate_output = F.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        return self.down_proj(gate_output * up_output)


class FinalProjection(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.in_features = hidden_size
        self.out_features = vocab_size
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(tensor):
            k = 1 / tensor.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(tensor, -bound, bound)

        _init_weights(self.weight)
        if self.bias is not None:
            _init_weights(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class DecoderLayer(nn.Module):
    # RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
    def __init__(self, 
                 config,
                 layer_idx,
                 device=None,
                 dtype=None):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # RMSNorm layers
        self.input_layernorm = RMSNorm(
            d_model=config.hidden_size,
            eps=config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6,
            device=device,
            dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            d_model=config.hidden_size,
            eps=config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6,
            device=device,
            dtype=dtype
        )
        
        # Attention layer
        self.attention = FlashAttentionWithTP(
            d_model=config.hidden_size,
            n_head=config.num_attention_heads,
            n_kv_head=config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.num_attention_heads,
            max_seq_size=config.max_position_embeddings,
            theta=config.rope_theta if hasattr(config, 'rope_theta') else 10000,
            device=device,
            dtype=dtype,
            use_tp=True,
            async_all_reduce=False
        )
        
        # MLP layer
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, position_ids=None):
        # Attention block with residual connection
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, token_position=position_ids, attention_mask=attention_mask)
        x = residual + x
        
        # MLP block with residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class Llama(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # sanity check 
        assert config.hidden_size % config.num_attention_heads == 0
        num_key_value_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.num_attention_heads
        assert config.num_attention_heads % num_key_value_heads == 0 
        
        # params
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config
        
        # modules
        self.embedding = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i) for i in range(self.num_layers)
        ])
        self.final_proj = FinalProjection(self.hidden_size, self.vocab_size, bias=False)
        self.final_norm = RMSNorm(
            self.hidden_size, 
            eps=config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        
        for layer in self.decoder_layers:
            layer.input_layernorm.reset_parameters()
            layer.attention.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()
            layer.mlp.reset_parameters()

        self.final_norm.reset_parameters()
        self.final_proj.reset_parameters()

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        x = self.embedding(input_ids)
        
        for layer in self.decoder_layers:
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids)
        
        x = self.final_norm(x)
        logits = self.final_proj(x)
        
        return logits  # [batch_size, seq_length, vocab_size]