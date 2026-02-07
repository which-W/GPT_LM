"""
数据准备脚本 - 将原始对话数据转换为SFT训练格式

支持的输入格式:
1. CSV格式: instruction, input, output列
2. JSON格式: 包含conversations的对话数据
3. 纯文本格式: Q&A对

输出格式: JSONL (每行一个JSON对象)
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict


def convert_csv_to_jsonl(input_path: str, output_path: str):
    """
    转换CSV格式到JSONL
    
    CSV格式:
    instruction,input,output
    "如何学习Python?","","从基础语法开始..."
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "instruction": row.get("instruction", "").strip(),
                "input": row.get("input", "").strip(),
                "output": row.get("output", "").strip()
            }
            if item["output"]:  # 只保留有输出的样本
                data.append(item)
    
    # 写入JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(data)} 条数据已保存到 {output_path}")


def convert_conversation_to_jsonl(input_path: str, output_path: str):
    """
    转换对话格式到JSONL
    
    JSON格式:
    {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "你好!有什么可以帮助你的?"}
        ]
    }
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data = []
    for item in raw_data:
        conversations = item.get("conversations", [])
        
        # 提取human-assistant对话对
        for i in range(len(conversations) - 1):
            if conversations[i]["from"] == "human" and conversations[i+1]["from"] == "assistant":
                data.append({
                    "prompt": conversations[i]["value"],
                    "response": conversations[i+1]["value"]
                })
    
    # 写入JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(data)} 条数据已保存到 {output_path}")


def convert_qa_text_to_jsonl(input_path: str, output_path: str):
    """
    转换问答文本格式到JSONL
    
    文本格式:
    Q: 问题1
    A: 回答1
    
    Q: 问题2
    A: 回答2
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_q = None
    current_a = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("A:"):
            current_a = line[2:].strip()
            if current_q and current_a:
                data.append({
                    "prompt": current_q,
                    "response": current_a
                })
                current_q = None
                current_a = None
    
    # 写入JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(data)} 条数据已保存到 {output_path}")


def create_sample_data(output_path: str, num_samples: int = 100):
    """
    创建示例SFT数据 (用于测试)
    """
    samples = []
    
    # 示例类别
    categories = [
        # 问答类
        {
            "instruction": "解释什么是机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个分支,它使计算机系统能够通过经验自动改进。机器学习算法通过分析数据来识别模式,并使用这些模式来做出预测或决策,而无需明确编程。"
        },
        {
            "instruction": "Python中如何创建列表?",
            "input": "",
            "output": "在Python中创建列表有多种方法:\n1. 使用方括号: my_list = [1, 2, 3]\n2. 使用list()函数: my_list = list((1, 2, 3))\n3. 使用列表推导式: my_list = [x for x in range(5)]"
        },
        # 翻译类
        {
            "instruction": "将以下英文翻译成中文",
            "input": "Hello, how are you?",
            "output": "你好,你好吗?"
        },
        # 摘要类
        {
            "instruction": "总结以下文本的主要内容",
            "input": "深度学习是机器学习的一个子领域,它基于人工神经网络。深度学习模型可以自动学习数据的层次化表示,这使得它们在图像识别、语音识别等任务上表现出色。",
            "output": "深度学习是基于神经网络的机器学习方法,能够自动学习数据的层次特征,在视觉和语音任务中表现优异。"
        },
        # 代码生成类
        {
            "instruction": "编写一个Python函数来计算斐波那契数列",
            "input": "",
            "output": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# 或使用迭代方式(更高效)\ndef fibonacci_iterative(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"
        }
    ]
    
    # 复制样本直到达到指定数量
    import random
    for i in range(num_samples):
        sample = random.choice(categories).copy()
        # 添加一些变化
        if i % 10 == 0:
            sample["instruction"] = f"[样本{i}] " + sample["instruction"]
        samples.append(sample)
    
    # 写入JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已创建 {num_samples} 条示例数据: {output_path}")


def split_train_val(input_path: str, train_path: str, val_path: str, val_ratio: float = 0.1):
    """
    将数据集分割为训练集和验证集
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 打乱数据
    import random
    random.shuffle(data)
    
    # 分割
    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]
    
    # 保存
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据分割完成:")
    print(f"  训练集: {len(train_data)} 条 -> {train_path}")
    print(f"  验证集: {len(val_data)} 条 -> {val_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='SFT数据准备工具')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['csv', 'conversation', 'qa_text', 'sample', 'split'],
                       help='转换模式')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--train_output', type=str, help='训练集输出路径 (split模式)')
    parser.add_argument('--val_output', type=str, help='验证集输出路径 (split模式)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--num_samples', type=int, default=100, 
                       help='示例数据数量 (sample模式)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == 'csv':
        convert_csv_to_jsonl(args.input, args.output)
    
    elif args.mode == 'conversation':
        convert_conversation_to_jsonl(args.input, args.output)
    
    elif args.mode == 'qa_text':
        convert_qa_text_to_jsonl(args.input, args.output)
    
    elif args.mode == 'sample':
        create_sample_data(args.output, args.num_samples)
    
    elif args.mode == 'split':
        split_train_val(args.input, args.train_output, args.val_output, args.val_ratio)
    
    print("完成!")


if __name__ == "__main__":
    main()
