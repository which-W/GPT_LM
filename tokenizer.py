"""从文本训练项目的 BPE 分词器；导入模块时不会开始训练或覆盖文件。"""
import argparse
from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def train_tokenizer(files, output_path="tokenizer.json", vocab_size=30000):
    """使用空白及标点预分词，并同时注册终止符和未知词符号。"""
    if not files or any(not Path(path).is_file() for path in files):
        raise ValueError("训练文本文件不存在")
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|unk|>"], show_progress=True)
    tokenizer.train([str(path) for path in files], trainer)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="训练 BPE 分词器，之后需重新预处理 token 数据")
    parser.add_argument("--input_paths", nargs="+", default=["data/TinyStories-train.txt"])
    parser.add_argument("--output_path", default="tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=30000)
    args = parser.parse_args()
    tokenizer = train_tokenizer(args.input_paths, args.output_path, args.vocab_size)
    print(f"已保存分词器，实际词表大小：{tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
