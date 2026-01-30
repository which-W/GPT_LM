from tokenizers import Tokenizer, pre_tokenizers, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path

# 初始化 tokenizer
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

# 组合多个预分词器
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    # 先按空格分词
    pre_tokenizers.Whitespace(),
    # 将标点符号独立出来,但使用正则保护缩写词中的撇号
    pre_tokenizers.Split(
        pattern=r"'(?=[a-zA-Z])|(?<=[a-zA-Z])'(?![a-zA-Z])|[.,!?;:\"""''—(){}\-]|[\[\]]",
        behavior='isolated'
    )
])

# 配置训练器
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["<|endoftext|>"],
    show_progress=True,
    min_frequency=2,
)

# 收集训练文件
import os
print(f"当前工作目录: {os.getcwd()}")

data_path = Path("data")
files = [str(data_path / "TinyStories-train.txt")]

if not Path(files[0]).exists():
    print(f"尝试查找的文件: {Path(files[0]).absolute()}")
    print(f"data 目录下的所有文件: {list(data_path.glob('*')) if data_path.exists() else '目录不存在'}")
    raise ValueError("找不到 TinyStories-train.txt 文件")

print(f"使用训练文件: {files[0]}")

print(f"找到 {len(files)} 个训练文件")
print("开始训练 tokenizer...")

# 训练 tokenizer
tokenizer.train(files, trainer)

# 保存 tokenizer
tokenizer.save("tokenizer.json")
print("\nTokenizer 已保存到: tokenizer.json")
print(f"词汇表大小: {tokenizer.get_vocab_size()}")