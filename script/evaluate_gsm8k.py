import os
import json
import time
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# --- 导入解析工具 ---
# 确保该函数能处理从文本中提取最后一个数字的逻辑
from utils.parse_gsm8k_res import parse_gsm8k_response

# ================= 配置区 =================
VLLM_API_URL = "http://localhost:8010/v1"
MODEL_NAME = "Qwen3-4B-Base"
DATA_PATH = "data/gsm8k/test.jsonl"
OUTPUT_FILE = f"result/gsm8k_{MODEL_NAME}_SFT_results.json"
MAX_WORKERS = 100

# 官方指定的统一 System Prompt
SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer some controversial topics."
)

client = OpenAI(base_url=VLLM_API_URL, api_key="empty")

# ================= 工具函数 =================

def load_gsm8k_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 GSM8K 测试集"""
    items = []
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            # 提取标准答案数字（GSM8K 格式为 "推理过程 #### 数字"）
            gold_str = ex['answer'].split("####")[-1].strip().replace(",", "")
            items.append({
                "question": ex['question'],
                "gold": float(gold_str)
            })
    return items

def call_vllm_api(item: Dict[str, Any]) -> Dict[str, Any]:
    """使用 client.completions.create 进行原始文本补全推理"""
    question = item["question"]
    
    full_raw_prompt =  f"{SYSTEM_PROMPT}\n{question}\nAnswer:"

    try:
        # 使用 completions 接口直接发送原始字符串
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=full_raw_prompt,
            temperature=0.0,  # 官方要求：Greedy Decoding
            max_tokens=512,   # 足够容纳推理过程
            # 停止符：防止 Base 模型在输出答案后复读题目或伪造对话
            stop=["# Query:", "```", "\n\n#", "User:", "Assistant:"] 
        )
        
        # 获取生成内容 (Completions 接口使用 .text)
        gen_text = response.choices[0].text
        
        # 使用解析工具提取数字
        pred_str = parse_gsm8k_response(gen_text)
        
        pred_val = None
        is_correct = False
        
        if pred_str is not None:
            try:
                pred_val = float(pred_str)
                if pred_val == item["gold"]:
                    is_correct = True
            except ValueError:
                pred_val = None

        return {
            "question": question,
            "gold": item["gold"],
            "pred": pred_val,
            "output": gen_text,
            "is_correct": is_correct
        }
    except Exception as e:
        return {"error": str(e), "question": question}

# ================= 主程序 =================

def main():
    # 1. 准备数据
    all_items = load_gsm8k_data(DATA_PATH)
    if not all_items:
        return
    print(f"已加载 {len(all_items)} 个 GSM8K 测试题目。")

    # 2. 并发评估逻辑
    all_results = []
    start_time = time.time()

    print(f"正在调用 {MODEL_NAME} 进行原始补全评测 (并发数: {MAX_WORKERS})...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_item = {executor.submit(call_vllm_api, item): item for item in all_items}
        
        # 收集结果
        for future in tqdm(as_completed(future_to_item), total=len(all_items)):
            res = future.result()
            if res and "error" not in res:
                all_results.append(res)
            elif res:
                print(f"\n警告: 请求失败 -> {res['error']}")

    duration = time.time() - start_time

    # 3. 结果分析与统计
    if not all_results:
        print("未收集到任何有效结果，请检查 vLLM 服务状态。")
        return

    correct_count = sum(1 for r in all_results if r["is_correct"])
    parse_failed_count = sum(1 for r in all_results if r["pred"] is None)
    total_evaluated = len(all_results)
    
    accuracy = correct_count / total_evaluated
    throughput = total_evaluated / duration

    metrics = {
        "model": MODEL_NAME,
        "accuracy": accuracy,
        "throughput_samples_per_sec": throughput,
        "parsing_failure_count": parse_failed_count,
        "total_evaluated": total_evaluated,
        "total_original": len(all_items)
    }

    # 4. 序列化保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "method": "client.completions",
                "model": MODEL_NAME,
                "temperature": 0.0,
                "stop_sequences": ["# Query:", "```"]
            },
            "metrics": metrics, 
            "details": all_results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print(f"评测完成！")
    print(f"模型: {MODEL_NAME}")
    print(f"准确率 (Accuracy): {accuracy:.2%}")
    print(f"吞吐量 (Throughput): {throughput:.2f} samples/s")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()