import numpy as np
from tokenizers import Tokenizer
from pathlib import Path

def preprocess_file(input_file, output_file, tokenizer_path="tokenizer.json", chunk_size=10_000_000):
    """
    将文本文件转换为二进制token文件（分块处理避免内存溢出）
    
    Args:
        input_file: 输入的文本文件路径
        output_file: 输出的二进制文件路径
        tokenizer_path: tokenizer文件路径
        chunk_size: 每次读取的字符数
    """
    print(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"读取文本文件: {input_file}")
    
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 打开输出文件
    output_path = Path(output_file)
    if output_path.exists():
        output_path.unlink()  # 删除旧文件
    
    total_tokens = 0
    chunk_num = 0
    
    # 分块读取和处理
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk_num += 1
            print(f"\r处理块 {chunk_num}...", end="", flush=True)
            
            # 读取一块文本
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # 编码
            encoding = tokenizer.encode(chunk)
            tokens = encoding.ids
            
            # 转换为数组
            tokens_array = np.array(tokens, dtype=np.int64)
            total_tokens += len(tokens)
            
            # 追加到文件
            with open(output_file, "ab") as out:
                tokens_array.tofile(out)
    
    print(f"\n已保存到: {output_file}")
    print(f"总 Token 数: {total_tokens:,}")
    
    # 计算文件大小
    file_size_mb = Path(output_file).stat().st_size / 1024 / 1024
    print(f"文件大小: {file_size_mb:.2f} MB")
    
    return total_tokens


if __name__ == "__main__":
    # 预处理训练集
    print("处理训练集...")
    train_tokens = preprocess_file(
        input_file="data/TinyStories-train.txt",
        output_file="data/TinyStories-train.bin",
        tokenizer_path="tokenizer.json",
        chunk_size=50_000_000  # 每次读取 50MB 文本
    )
    
    print("处理验证集...")

    
    # 预处理验证集
    valid_tokens = preprocess_file(
        input_file="data/TinyStories-valid.txt",
        output_file="data/TinyStories-valid.bin",
        tokenizer_path="tokenizer.json",
        chunk_size=50_000_000
    )
    
    print("预处理完成!")
    print(f"训练集 tokens: {train_tokens:,}")
    print(f"验证集 tokens: {valid_tokens:,}")