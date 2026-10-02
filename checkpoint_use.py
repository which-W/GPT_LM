from pathlib import Path
import warnings
import torch


def model_config(model):
    model = model.module if hasattr(model, 'module') else model
    if isinstance(getattr(model, 'config', None), dict):
        return dict(model.config)
    layer = model.layers[0]
    attention = layer.attention
    rope = getattr(attention, 'rope', None)
    config = dict(d_model=model.embedding.weight.shape[1], vocab_size=model.embedding.weight.shape[0],
                  n_layer=len(model.layers), n_head=attention.n_head,
                  max_seq_len=getattr(rope, 'max_seq_len', 512), theta=getattr(rope, 'theta', None),
                  use_rms_norm=not isinstance(model.ln_final, torch.nn.Identity),
                  norm_model=getattr(layer, 'norm_model', 'pre'), ffn_type='swiglu')
    if hasattr(layer, 'ffn'):
        config['d_ff'] = layer.ffn.w1.out_features
    else:
        config.update(d_ff=layer.moe.experts.experts[0].ffn.w1.out_features,
                      n_experts=layer.moe.n_experts, top_k=layer.moe.top_k)
    if hasattr(model, 'n'):
        config['n'] = model.n
    for key in ('moe_layer_indices', 'engram_layer_indices'):
        if hasattr(model, key):
            config[key] = sorted(getattr(model, key))
    config['model_type'] = model.__class__.__module__ + '.' + model.__class__.__name__
    return config


def get_checkpoint_config(checkpoint, overrides=None):
    """从权重推断维度，旧文件未记录的配置使用文档中的默认值。"""
    state = checkpoint.get('model_state_dict', checkpoint)
    config = dict(checkpoint.get('config', {}))
    if not config:
        warnings.warn('Legacy checkpoint has no config: assuming 8 heads, context 512 and theta 10000; supply config overrides for other settings.', UserWarning)
    inferred = dict(vocab_size=state['embedding.weight'].shape[0],
                    d_model=state['embedding.weight'].shape[1],
                    n_layer=len({int(key.split('.')[1]) for key in state if key.startswith('layers.')}))
    if 'layers.0.ffn.w1.weight' in state:
        inferred['d_ff'] = state['layers.0.ffn.w1.weight'].shape[0]
        inferred['ffn_type'] = 'swiglu' if 'layers.0.ffn.w3.weight' in state else 'silu'
    if 'd_ff' not in inferred and 'd_ff' not in config:
        candidate = next((value for key, value in state.items() if 'ffn.w1.weight' in key), None)
        if candidate is not None:
            inferred['d_ff'] = candidate.shape[0]
    inferred['use_rms_norm'] = 'ln_final.weight' in state or 'layers.0.ln1.weight' in state
    for key, value in inferred.items():
        if key in config and config[key] != value:
            raise ValueError(f'Checkpoint config {key}={config[key]} disagrees with weights ({value})')
        config[key] = value
    for key, value in dict(n_head=8, max_seq_len=512, theta=10000.0, norm_model='pre', ffn_type='swiglu').items():
        config.setdefault(key, value)
    config.update(overrides or {})
    return config


def load_tokenizer(checkpoint, tokenizer_path, vocab_size):
    from tokenizers import Tokenizer
    tokenizer = (Tokenizer.from_str(checkpoint['tokenizer_json']) if checkpoint.get('tokenizer_json')
                 else Tokenizer.from_file(str(tokenizer_path)))
    ids = list(tokenizer.get_vocab().values())
    if not ids or min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError('Tokenizer IDs exceed the checkpoint vocabulary; use the training tokenizer')
    return tokenizer


def mask_tokenizer_logits(logits, tokenizer):
    """采样时排除分词器中不存在的词表位置。"""
    valid = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    valid[list(tokenizer.get_vocab().values())] = True
    return logits.masked_fill(~valid, float('-inf'))

def save_checkpoint(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    iteration:int,
    out,
    config=None,
    tokenizer_path=None,
):
    """
        保存当前训练状态
    """

    model = model.module if hasattr(model, 'module') else model
    #构建一个包含所有必要信息的字典
    checkpoint = {
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'iteration':iteration,
        'config': model_config(model) if config is None else dict(config),
    }
    tokenizer_json = getattr(model, 'tokenizer_json', None)
    if tokenizer_path is not None:
        tokenizer_json = Path(tokenizer_path).read_text(encoding='utf-8')
    if tokenizer_json is not None:
        checkpoint['tokenizer_json'] = tokenizer_json

    #使用torch.save写入
    torch.save(checkpoint,out)

def load_checkpoint(
    src,
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    tokenizer_path=None,
):
    """
        从检查点恢复状态，并返回保存时的迭代次数
    """
    #加载字典
    #使用map_location='cpu'防止出现在没有显卡的机子上报错
    checkpoint = torch.load(src,map_location='cpu',weights_only=True)
    current_config = model_config(model)
    for key, value in checkpoint.get('config', {}).items():
        if key in current_config and current_config[key] != value:
            raise ValueError(f'恢复训练时的模型配置 {key} 与检查点不一致')
    if tokenizer_path is not None and checkpoint.get('tokenizer_json'):
        from tokenizers import Tokenizer
        stored = Tokenizer.from_str(checkpoint['tokenizer_json'])
        current = Tokenizer.from_file(str(tokenizer_path))
        if stored.to_str() != current.to_str():
            raise ValueError('恢复训练必须使用检查点对应的分词器')

    #加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    #恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #返回保存时的迭代次数
    return checkpoint['iteration']
