"""
SFT模型推理脚本

支持:
1. 单轮对话推理
2. 批量推理
3. 交互式对话
"""

import torch
import argparse
import json
from pathlib import Path
from transformer import TransformerLM


def load_tokenizer(tokenizer_path):
    """加载分词器"""
    import sys
    sys.path.insert(0, str(Path(tokenizer_path).parent))
    
    try:
        from tokenizer import encode, decode
        return encode, decode
    except ImportError:
        print("警告: 未找到tokenizer模块,使用简单的字符级编码")
        
        def simple_encode(text):
            return [ord(c) % 30000 for c in text[:512]]
        
        def simple_decode(ids):
            return ''.join([chr(i) for i in ids if i > 0])
        
        return simple_encode, simple_decode


def format_prompt(instruction: str, input_text: str = "", template: str = "default") -> str:
    """格式化提示词"""
    if template == "default":
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    elif template == "alpaca":
        if input_text:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    
    elif template == "chat":
        if input_text:
            return f"User: {instruction}\n{input_text}\n\nAssistant: "
        else:
            return f"User: {instruction}\n\nAssistant: "
    
    else:
        return instruction


@torch.no_grad()
def generate(
    model,
    prompt_ids: list,
    encode_fn,
    decode_fn,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = 'cuda'
):
    """
    生成文本
    
    Args:
        model: TransformerLM模型
        prompt_ids: 提示词的token ids
        encode_fn: 编码函数
        decode_fn: 解码函数
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_k: top-k采样
        top_p: nucleus采样
        device: 设备
    """
    model.eval()
    
    # 清空KV Cache
    model.clear_cache()
    
    # 将prompt转为tensor
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    # Prefill阶段: 处理完整的prompt
    logits = model(input_ids, use_cache=True)
    
    # 获取最后一个token的logits
    next_token_logits = logits[0, -1, :]
    
    # 生成的token列表
    generated_ids = prompt_ids.copy()
    
    # 自回归生成
    for _ in range(max_new_tokens):
        # 应用temperature
        next_token_logits = next_token_logits / temperature
        
        # Top-k过滤
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus)过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # 采样
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 检查是否生成了结束符
        if next_token.item() == 0:
            break
        
        # 添加到生成序列
        generated_ids.append(next_token.item())
        
        # 生成阶段: 只输入新token
        input_ids = next_token.unsqueeze(0)
        logits = model(input_ids, use_cache=True)
        next_token_logits = logits[0, -1, :]
    
    # 解码
    generated_text = decode_fn(generated_ids)
    
    return generated_text


def single_inference(args):
    """单次推理"""
    # 加载分词器
    encode_fn, decode_fn = load_tokenizer(args.tokenizer_path)
    
    # 加载模型
    print("加载模型...")
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    model = TransformerLM(
        d_model=args.d_model,
        n_head=args.n_head,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_ff=args.d_ff,
        theta=args.theta,
        n_layer=args.n_layer,
        device=args.device,
        dtype=dtype,
        use_rms_norm=not args.no_rms_norm,
        norm_model=args.norm_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("模型加载完成!")
    
    # 格式化prompt
    prompt = format_prompt(args.instruction, args.input, args.prompt_template)
    print(f"\n提示词:\n{prompt}")
    
    # 编码
    prompt_ids = encode_fn(prompt)
    
    # 生成
    print("\n生成中...")
    output = generate(
        model=model,
        prompt_ids=prompt_ids,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )
    
    # 提取response部分 (去掉prompt)
    response = output[len(prompt):] if output.startswith(prompt) else output
    
    print(f"\n回答:\n{response}")


def batch_inference(args):
    """批量推理"""
    # 加载分词器
    encode_fn, decode_fn = load_tokenizer(args.tokenizer_path)
    
    # 加载模型
    print("加载模型...")
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    model = TransformerLM(
        d_model=args.d_model,
        n_head=args.n_head,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_ff=args.d_ff,
        theta=args.theta,
        n_layer=args.n_layer,
        device=args.device,
        dtype=dtype,
        use_rms_norm=not args.no_rms_norm,
        norm_model=args.norm_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("模型加载完成!")
    
    # 加载测试数据
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"加载了 {len(test_data)} 条测试数据")
    
    # 批量生成
    results = []
    for i, item in enumerate(test_data):
        print(f"\n处理 {i+1}/{len(test_data)}...")
        
        # 获取instruction和input
        instruction = item.get("instruction", item.get("prompt", ""))
        input_text = item.get("input", "")
        
        # 格式化prompt
        prompt = format_prompt(instruction, input_text, args.prompt_template)
        prompt_ids = encode_fn(prompt)
        
        # 生成
        output = generate(
            model=model,
            prompt_ids=prompt_ids,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        
        response = output[len(prompt):] if output.startswith(prompt) else output
        
        results.append({
            "instruction": instruction,
            "input": input_text,
            "output": response,
            "reference": item.get("output", "")
        })
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n批量推理完成! 结果已保存到 {args.output_file}")


def interactive_chat(args):
    """交互式对话"""
    # 加载分词器
    encode_fn, decode_fn = load_tokenizer(args.tokenizer_path)
    
    # 加载模型
    print("加载模型...")
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    model = TransformerLM(
        d_model=args.d_model,
        n_head=args.n_head,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_ff=args.d_ff,
        theta=args.theta,
        n_layer=args.n_layer,
        device=args.device,
        dtype=dtype,
        use_rms_norm=not args.no_rms_norm,
        norm_model=args.norm_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("模型加载完成!")
    print("\n开始交互式对话 (输入 'quit' 或 'exit' 退出)\n")
    
    while True:
        # 获取用户输入
        user_input = input("User: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("再见!")
            break
        
        if not user_input:
            continue
        
        # 格式化prompt
        prompt = format_prompt(user_input, "", args.prompt_template)
        prompt_ids = encode_fn(prompt)
        
        # 生成
        output = generate(
            model=model,
            prompt_ids=prompt_ids,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        
        response = output[len(prompt):] if output.startswith(prompt) else output
        
        print(f"Assistant: {response}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='SFT模型推理')
    
    # 模式选择
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'batch', 'interactive'],
                       help='推理模式')
    
    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                       help='分词器路径')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--theta', type=float, default=10000.0)
    
    # 实验参数
    parser.add_argument("--no_rms_norm", action="store_true")
    parser.add_argument("--norm_rope", type=str, default="pre")
    parser.add_argument("--ffn_type", type=str, default="swiglu")
    
    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数')
    parser.add_argument('--top_k', type=int, default=50,
                       help='top-k采样')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='nucleus采样')
    
    # 提示词模板
    parser.add_argument('--prompt_template', type=str, default='default',
                       choices=['default', 'alpaca', 'chat'])
    
    # 单次推理参数
    parser.add_argument('--instruction', type=str,
                       help='指令 (single模式)')
    parser.add_argument('--input', type=str, default="",
                       help='输入 (single模式)')
    
    # 批量推理参数
    parser.add_argument('--test_file', type=str,
                       help='测试文件 (batch模式)')
    parser.add_argument('--output_file', type=str,
                       help='输出文件 (batch模式)')
    
    # 设备
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'bfloat16'])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == 'single':
        if not args.instruction:
            print("错误: single模式需要提供 --instruction 参数")
            return
        single_inference(args)
    
    elif args.mode == 'batch':
        if not args.test_file or not args.output_file:
            print("错误: batch模式需要提供 --test_file 和 --output_file 参数")
            return
        batch_inference(args)
    
    elif args.mode == 'interactive':
        interactive_chat(args)


if __name__ == "__main__":
    main()
