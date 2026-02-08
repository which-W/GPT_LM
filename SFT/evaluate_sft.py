"""
SFT模型评估脚本

支持:
1. 自动化指标评估 (BLEU, ROUGE)
2. 人工评估辅助
3. 对比评估 (预训练 vs SFT)
"""

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from transformer import TransformerLM
from SFT.sft_test_inference import load_tokenizer, generate, format_prompt
from checpoint_use import load_checkpoint 


def evaluate_on_dataset(
    model,
    test_data: List[Dict],
    encode_fn,
    decode_fn,
    args
) -> List[Dict]:
    """在数据集上进行评估"""
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for item in tqdm(test_data, desc="评估中"):
            # 获取instruction和input
            instruction = item.get("instruction", item.get("prompt", ""))
            input_text = item.get("input", "")
            reference = item.get("output", item.get("response", ""))
            
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
            
            # 提取response
            response = output[len(prompt):] if output.startswith(prompt) else output
            
            results.append({
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "prediction": response,
            })
    
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """
    计算评估指标
    
    注意: 这里使用简化的指标计算
    实际使用中可以集成 nltk, rouge_score 等库
    """
    
    # 1. 平均长度统计
    pred_lengths = [len(r["prediction"].split()) for r in results]
    ref_lengths = [len(r["reference"].split()) for r in results]
    
    metrics = {
        "num_samples": len(results),
        "avg_pred_length": sum(pred_lengths) / len(pred_lengths),
        "avg_ref_length": sum(ref_lengths) / len(ref_lengths),
    }
    
    # 2. 简单的n-gram重叠度 (简化版BLEU)
    def simple_ngram_overlap(pred: str, ref: str, n: int = 2) -> float:
        """计算n-gram重叠度"""
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        
        if len(pred_words) < n or len(ref_words) < n:
            return 0.0
        
        pred_ngrams = set()
        for i in range(len(pred_words) - n + 1):
            pred_ngrams.add(tuple(pred_words[i:i+n]))
        
        ref_ngrams = set()
        for i in range(len(ref_words) - n + 1):
            ref_ngrams.add(tuple(ref_words[i:i+n]))
        
        if len(pred_ngrams) == 0:
            return 0.0
        
        overlap = len(pred_ngrams & ref_ngrams)
        return overlap / len(pred_ngrams)
    
    unigram_overlaps = []
    bigram_overlaps = []
    
    for r in results:
        unigram_overlaps.append(simple_ngram_overlap(r["prediction"], r["reference"], n=1))
        bigram_overlaps.append(simple_ngram_overlap(r["prediction"], r["reference"], n=2))
    
    metrics["avg_unigram_overlap"] = sum(unigram_overlaps) / len(unigram_overlaps)
    metrics["avg_bigram_overlap"] = sum(bigram_overlaps) / len(bigram_overlaps)
    
    return metrics


def print_sample_outputs(results: List[Dict], num_samples: int = 5):
    """打印样本输出"""
    print("\n" + "="*80)
    print("样本输出:")
    print("="*80)
    
    for i, result in enumerate(results[:num_samples]):
        print(f"\n[样本 {i+1}]")
        print(f"Instruction: {result['instruction']}")
        if result['input']:
            print(f"Input: {result['input']}")
        print(f"\nReference:\n{result['reference']}")
        print(f"\nPrediction:\n{result['prediction']}")
        print("-" * 80)


def human_evaluation_template(results: List[Dict], output_file: str):
    """生成人工评估模板"""
    
    eval_template = []
    
    for i, result in enumerate(results):
        eval_template.append({
            "id": i,
            "instruction": result["instruction"],
            "input": result["input"],
            "reference": result["reference"],
            "prediction": result["prediction"],
            "scores": {
                "relevance": 0,  # 1-5: 相关性
                "fluency": 0,    # 1-5: 流畅性
                "accuracy": 0,   # 1-5: 准确性
                "overall": 0     # 1-5: 总体质量
            },
            "comments": ""
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_template, f, ensure_ascii=False, indent=2)
    
    print(f"\n人工评估模板已保存到: {output_file}")
    print("请填写scores字段 (1-5分) 和comments字段")


def compare_models(
    pretrain_results: List[Dict],
    sft_results: List[Dict]
):
    """对比预训练和SFT模型"""
    
    pretrain_metrics = compute_metrics(pretrain_results)
    sft_metrics = compute_metrics(sft_results)
    
    print("\n预训练模型:")
    for k, v in pretrain_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nSFT模型:")
    for k, v in sft_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n改进:")
    for k in pretrain_metrics.keys():
        if k != "num_samples":
            delta = sft_metrics[k] - pretrain_metrics[k]
            pct = (delta / pretrain_metrics[k] * 100) if pretrain_metrics[k] != 0 else 0
            print(f"  {k}: {delta:+.4f} ({pct:+.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description='SFT模型评估')
    
    # 评估模式
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'compare', 'human_eval'],
                       help='评估模式')
    
    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='SFT模型检查点路径')
    parser.add_argument('--pretrain_checkpoint', type=str,
                       help='预训练模型检查点 (compare模式)')
    parser.add_argument('--tokenizer_path', type=str, required=True)
    
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
    
    # 数据参数
    parser.add_argument('--test_file', type=str, required=True,
                       help='测试数据文件 (JSONL)')
    parser.add_argument('--output_file', type=str, default='eval_results.json',
                       help='评估结果输出文件')
    
    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--prompt_template', type=str, default='default')
    
    # 评估参数
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='评估样本数 (-1表示全部)')
    parser.add_argument('--show_samples', type=int, default=5,
                       help='显示的样本数')
    
    # 设备
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='float32')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载分词器
    print("加载分词器...")
    encode_fn, decode_fn = load_tokenizer(args.tokenizer_path)
    
    # 加载测试数据
    print("加载测试数据...")
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
    
    print(f"测试数据: {len(test_data)} 条")
    
    if args.mode == 'evaluate':
        # 单模型评估
        print("\n加载SFT模型...")
        model = load_checkpoint(args.checkpoint_path, args)
        
        print("\n开始评估...")
        results = evaluate_on_dataset(model, test_data, encode_fn, decode_fn, args)
        
        # 计算指标
        metrics = compute_metrics(results)
        
        print("\n" + "="*80)
        print("评估指标:")
        print("="*80)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # 显示样本
        print_sample_outputs(results, args.show_samples)
        
        # 保存结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果已保存到: {args.output_file}")
    
    elif args.mode == 'compare':
        # 对比评估
        if not args.pretrain_checkpoint:
            print("错误: compare模式需要提供 --pretrain_checkpoint")
            return
        
        print("\n加载预训练模型...")
        pretrain_model = load_checkpoint(args.pretrain_checkpoint, args)
        
        print("加载SFT模型...")
        sft_model = load_checkpoint(args.checkpoint_path, args)
        
        print("\n评估预训练模型...")
        pretrain_results = evaluate_on_dataset(pretrain_model, test_data, encode_fn, decode_fn, args)
        
        print("\n评估SFT模型...")
        sft_results = evaluate_on_dataset(sft_model, test_data, encode_fn, decode_fn, args)
        
        # 对比
        compare_models(pretrain_results, sft_results)
        
        # 保存结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'pretrain_results': pretrain_results,
                'sft_results': sft_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n对比结果已保存到: {args.output_file}")
    
    elif args.mode == 'human_eval':
        # 生成人工评估模板
        print("\n加载SFT模型...")
        model = load_checkpoint(args.checkpoint_path, args)
        
        print("\n生成预测...")
        results = evaluate_on_dataset(model, test_data, encode_fn, decode_fn, args)
        
        # 生成人工评估模板
        human_evaluation_template(results, args.output_file)


if __name__ == "__main__":
    main()
