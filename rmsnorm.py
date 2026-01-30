import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    PyTorch 版本的 RMSNorm 实现
    
    RMSNorm 相比 LayerNorm 更简单，只使用 RMS (Root Mean Square) 进行归一化，
    不减去均值，计算效率更高。常用于 LLaMA、GPT-NeoX 等大模型。
    
    Args:
        d_model: 需要归一化的维度大小
        eps: 防止除零的小常数
        elementwise_affine: 是否使用可学习的缩放参数
    """
    def __init__(self, d_model, eps=1e-6, elementwise_affine=True,device=None,dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device
        self.dtype = dtype
        if self.elementwise_affine:
            # 可学习的缩放参数 (gain)
            self.weight = nn.Parameter(torch.ones(d_model,device=self.device,dtype=self.dtype))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x):
        in_dtype = x.dtype
        #转为float32防止计算均值或者方差时溢出
        x_float = x.to(torch.float32)
        # 计算 RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化
        result = x_float / rms
        
        # 应用缩放
        if self.elementwise_affine:
            result = result * self.weight
        
        return result.to(in_dtype)