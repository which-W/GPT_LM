import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSoftmax(nn.Module):
    """
    数值稳定的Softmax实现
    
    核心原理:
        softmax(x) = exp(x_i) / sum(exp(x_j))
                   = exp(x_i - c) / sum(exp(x_j - c))  (对任意常数c成立)
                   = exp(x_i - max(x)) / sum(exp(x_j - max(x)))  (选择c=max(x))
    
    优势:
        - 防止上溢: max(x - max(x)) = 0, 所以 exp(0) = 1.0
        - 防止下溢: 减去最大值后，指数值在合理范围内
        - 数学等价: 结果与标准softmax完全相同
    
    Examples:
        >>> softmax = StableSoftmax(dim=-1)
        >>> x = torch.tensor([1000.0, 1001.0, 1002.0])
        >>> result = softmax(x)
        >>> print(result)
        tensor([0.0900, 0.2447, 0.6652])
    """
    
    def __init__(self, dim=-1):
        """
        Args:
            dim (int): 计算softmax的维度，默认为-1(最后一维)
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        数值稳定的softmax前向传播
        
        Args:
            x (torch.Tensor): 输入张量，任意形状 [..., n, ...]
        
        Returns:
            torch.Tensor: softmax结果，形状与输入相同
                         在指定维度上求和为1
        """
        # 步骤1: 找到指定维度的最大值 (keepdim=True用于广播)
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        
        # 步骤2: 减去最大值，防止exp溢出
        x_shifted = x - x_max
        
        # 步骤3: 计算exp
        exp_x = torch.exp(x_shifted)
        
        # 步骤4: 归一化
        sum_exp = torch.sum(exp_x, dim=self.dim, keepdim=True)
        softmax_x = exp_x / sum_exp
        
        return softmax_x