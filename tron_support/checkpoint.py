import os
import re
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors import safe_open
import contextlib

from tron_support.model import FinalProjection
from tron_support.utils import assert_no_meta_tensors, print
import tron_support.process_group_manager as pgm

@contextlib.contextmanager
def init_model_with_dematerialized_weights(include_buffers: bool = False):
    "在元设备上创建参数，避免初始化时分配完整权重；include_buffers 控制是否同时跳过缓冲区分配。参考 Accelerate： https://github.com/huggingface/accelerate/blob/v0.11.0/src/accelerate/big_modeling.py#L254"
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(torch.device("meta")), **kwargs)

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer

def init_model_with_materialized_weights(model, model_config, save_dir):
    """从单文件或分片权重加载参数，保留预训练输出头及全部权重。"""
    manager = InitializationManager(model, model_config)
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        with safe_open(os.path.join(save_dir, "model.safetensors"), framework="pt", device="cpu") as f:
            weight_map = {key: "model.safetensors" for key in f.keys()}
    state = {}
    for source_name in manager.get_layer_names_in_sft_format():
        if source_name == "lm_head.weight" and source_name not in weight_map:
            if not getattr(model_config, "tie_word_embeddings", False):
                raise ValueError("非共享词嵌入模型缺少 lm_head.weight")
            source_key = "model.embed_tokens.weight"
        else:
            source_key = source_name
        if source_key not in weight_map:
            raise ValueError(f"预训练权重缺少参数：{source_key}")
        with safe_open(os.path.join(save_dir, weight_map[source_key]), framework="pt", device="cpu") as f:
            target_name = manager.convert_safetensors_to_hf_name(source_name)
            state[target_name] = manager.adjust_tensor_size(f.get_tensor(source_key), target_name)
    model.load_state_dict(state, strict=True, assign=True)
    assert_no_meta_tensors(model)
    return model


class InitializationManager:
    """映射标准 LLaMA 权重名称，并按张量并行的进程编号切片。"""
    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config

    def get_layer_names_in_sft_format(self):
        components = ["input_layernorm", "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
                      "post_attention_layernorm", "self_attn.k_proj", "self_attn.o_proj",
                      "self_attn.q_proj", "self_attn.v_proj"]
        return (["model.embed_tokens.weight"] +
                [f"model.layers.{i}.{component}.weight" for i in range(self.model_config.num_hidden_layers)
                 for component in components] + ["model.norm.weight", "lm_head.weight"])

    def adjust_tensor_size(self, tensor, name):
        rank = pgm.process_group_manager.tp_rank
        size = pgm.process_group_manager.tp_world_size
        expected = dict(self.model.named_parameters())[name].shape
        if size > 1 and tensor.shape != expected:
            dim = 1 if ("attention.out_proj" in name or "mlp.down_proj" in name) else 0
            if tensor.shape[dim] != expected[dim] * size:
                raise ValueError(f"预训练参数 {name} 的形状与模型配置不匹配")
            tensor = tensor.narrow(dim, rank * expected[dim], expected[dim]).contiguous()
        if tensor.shape != expected:
            raise ValueError(f"参数 {name}：权重形状 {tuple(tensor.shape)}，模型形状 {tuple(expected)}")
        return tensor

    @staticmethod
    def convert_safetensors_to_hf_name(name):
        if name == "lm_head.weight":
            return "final_proj.weight"
        name = name.removeprefix("model.")
        name = name.replace("layers.", "decoder_layers.").replace("embed_tokens", "embedding")
        name = name.replace("self_attn.", "attention.").replace("o_proj", "out_proj")
        return name.replace("norm.weight", "final_norm.weight") if name == "norm.weight" else name

class CheckpointManager:
    def __init__(self):
        self.tp_rank = pgm.process_group_manager.tp_rank
        self.pp_rank = pgm.process_group_manager.pp_rank
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.pp_world_size = pgm.process_group_manager.pp_world_size
        self.cp_dp_world_size = pgm.process_group_manager.cp_dp_world_size
        self.dp_rank = pgm.process_group_manager.dp_rank
        self.cp_rank = pgm.process_group_manager.cp_rank

    def _get_checkpoint_path(self, out_dir):
        ckpt_name = f"weights_tp_rank_world_size={self.tp_rank}_{self.tp_world_size}_pp_rank_world_size={self.pp_rank}_{self.pp_world_size}.pth"
        return os.path.join(out_dir, ckpt_name)

    def save_checkpoint(self, model, optimizer, trained_steps, trained_tokens, out_dir):
        "保存模型、优化器及训练进度。"
        path = self._get_checkpoint_path(out_dir)

        # 仅数据与上下文并行组的首进程保存相同的模型副本
        if self.dp_rank == 0 and self.cp_rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            raw_model = model.module if self.cp_dp_world_size > 1 else model
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'trained_steps': trained_steps,
                'trained_tokens': trained_tokens
            }
            torch.save(checkpoint, path)

    def load_checkpoint(self, model, optimizer, out_dir):
        "在相同并行拓扑下恢复模型、优化器和训练进度。"
        path = self._get_checkpoint_path(out_dir)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        # 加载模型权重
        raw_model = model.module if self.cp_dp_world_size > 1 else model
        raw_model.load_state_dict(checkpoint['model'])

        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['trained_steps'], checkpoint['trained_tokens']
