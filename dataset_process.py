import numpy as np
from tokenizers import Tokenizer
from pathlib import Path
import argparse
import hashlib
import json
from utils.token_data import TOKEN_DTYPES

def preprocess_file(input_file, output_file, tokenizer_path="tokenizer_tinystories.json", chunk_size=10_000_000, dtype="int64"):
    """
    将文本文件转换为二进制token文件(分块处理避免内存溢出)
    
    参数：
        input_file: 输入的文本文件路径
        output_file: 输出的二进制文件路径
        tokenizer_path: tokenizer文件路径
        chunk_size: 每次读取的字符数
    """
    print(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if dtype not in TOKEN_DTYPES or chunk_size < 1:
        raise ValueError("Invalid token dtype or chunk_size")
    if max(tokenizer.get_vocab().values()) > np.iinfo(np.dtype(dtype)).max:
        raise ValueError(f"Tokenizer IDs do not fit in {dtype}")

    print(f"读取文本文件: {input_file}")

    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_file)
    if output_path.resolve() in (Path(input_file).resolve(), Path(tokenizer_path).resolve()):
        raise ValueError("输出路径不能覆盖输入文本或分词器")
    import tempfile
    total_tokens = 0
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output_path.parent, delete=False) as out:
            temporary_path = Path(out.name)
            with open(input_file, "r", encoding="utf-8") as f:
                chunk_num = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    # 将最后一行补齐，避免读取边界把一个单词拆成两个 token。
                    if not chunk.endswith("\n"):
                        chunk += f.readline()
                    chunk_num += 1
                    print(f"处理块 {chunk_num}...", flush=True)
                    tokens = tokenizer.encode(chunk).ids
                    np.asarray(tokens, dtype=dtype).tofile(out)
                    total_tokens += len(tokens)
        if total_tokens == 0:
            raise ValueError("输入文本未产生任何 token")
        metadata = {"dtype": dtype, "num_tokens": total_tokens,
                    "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()}
        # 全部编码成功后替换输出；失败时保留原有数据。
        temporary_path.replace(output_path)
        Path(str(output_path) + ".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"已保存到: {output_file}；总 Token 数: {total_tokens:,}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return total_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode text as a binary token file")
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--tokenizer_path", default="tokenizer_tinystories.json")
    parser.add_argument("--dtype", choices=TOKEN_DTYPES, default="int64")
    parser.add_argument("--chunk_size", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.input_path or args.output_path:
        if not args.input_path or not args.output_path:
            parser.error("--input_path and --output_path must be supplied together")
        preprocess_file(args.input_path, args.output_path, args.tokenizer_path, args.chunk_size, args.dtype)
        raise SystemExit(0)
    # 预处理训练集
    print("处理训练集...")
    train_tokens = preprocess_file(
        input_file="data/TinyStories-train.txt",
        output_file="data/TinyStories-train.bin",
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
        chunk_size=args.chunk_size  # 每次读取指定数量的字符
    )

    print("处理验证集...")


    # 预处理验证集
    valid_tokens = preprocess_file(
        input_file="data/TinyStories-valid.txt",
        output_file="data/TinyStories-valid.bin",
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
        chunk_size=args.chunk_size
    )

    print("预处理完成!")
    print(f"训练集 tokens: {train_tokens:,}")
    print(f"验证集 tokens: {valid_tokens:,}")
