import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int,device=None,dtype=None):
        super().__init__()
        # 创建参数矩阵,
        factory_para = {"device" : device,"dtype" : dtype}
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim,**factory_para))
        # 使用截断正态分布初始化
        trunc_normal_(self.weight, mean =0.0 ,std=1 , a = -3 , b = 3 ) #[-3,3]的范围防止产生过大的数
    
    def forward(self, token_id : torch.tensor):
        # x: [batch_size, seq_len] 索引张量
        # 直接索引查表
        return self.weight[token_id]  # [batch_size, seq_len, embedding_dim]