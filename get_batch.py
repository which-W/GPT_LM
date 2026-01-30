import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset:npt.NDArray,
    batch_size:int,
    max_seq_len:int,
    device:str
):
    """
    Docstring for get_batch
    随机采样批次
    
    返回：
    x:输入张量，[b,s]
    y:输出张量，[b,s]
    """
    n = len(dataset)
    
    #最后一个可用的起点，必须流出max_seq_len 的空间给x,在多留1位给y
    max_idx = n - max_seq_len -1
    
    #随机选择batch_size个起点
    ix = torch.randint(0,max_idx + 1,(batch_size,))
    
    #提取序列并转为np数组，再转为tensor
    x = torch.stack([torch.from_numpy(dataset[i:i+max_seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i+1:i+max_seq_len+1].astype(np.int64)) for i in ix])
    
    #一次性转移到GPU上
    return x.to(device),y.to(device)