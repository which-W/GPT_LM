"""按统一、明确的磁盘格式读取 token 文件。"""
import json
import hashlib
from pathlib import Path

import numpy as np


TOKEN_DTYPES = ("int64", "uint16", "uint32")


def load_token_data(path, dtype="int64", tokenizer_path=None):
    path = Path(path)
    metadata_path = Path(str(path) + ".meta.json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_dtype = metadata["dtype"]
        if dtype is not None and dtype != stored_dtype:
            raise ValueError(f"{path}: stored dtype is {stored_dtype}, requested {dtype}")
        dtype = stored_dtype
        if tokenizer_path is not None and "tokenizer_sha256" in metadata:
            fingerprint = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
            if fingerprint != metadata["tokenizer_sha256"]:
                raise ValueError("数据使用的分词器与当前分词器不一致，请重新生成数据或使用原分词器")
    dtype = dtype or "int64"
    if dtype not in TOKEN_DTYPES:
        raise ValueError(f"Unsupported token dtype: {dtype}")
    itemsize = np.dtype(dtype).itemsize
    size = path.stat().st_size
    if not size or size % itemsize:
        raise ValueError(f"{path}: file size is incompatible with {dtype}")
    data = np.memmap(path, dtype=dtype, mode="r")
    if "num_tokens" in metadata and len(data) != metadata["num_tokens"]:
        raise ValueError(f"{path}: token count does not match metadata")
    return data


class TokenWindowDataset:
    """连续语言模型窗口，额外保留一个目标 token。"""

    def __init__(self, path, seq_len, dtype="int64", vocab_size=None, tokenizer_path=None):
        self.data = load_token_data(path, dtype, tokenizer_path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        if seq_len < 1 or len(self.data) <= seq_len:
            raise ValueError("Token data must contain at least seq_len + 1 tokens")

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, index):
        import torch
        start = index * self.seq_len
        tokens = np.array(self.data[start:start + self.seq_len + 1], dtype=np.int64)
        if self.vocab_size is not None and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("Token IDs are outside the configured vocabulary")
        return torch.from_numpy(tokens)
