import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    PyTorch 版本的 LayerNorm 实现
    
    Args:
        d_model: 需要归一化的维度大小
        eps: 防止除零的小常数
        elementwise_affine: 是否使用可学习的仿射变换参数,默认为true
    """
    def __init__(self, d_model, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            # 可学习的缩放参数 gamma 
            self.weight = nn.Parameter(torch.ones(d_model))
            # 可学习的平移参数 beta
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        in_dtype = x.dtype
        #转为float32防止计算均值或者方差时溢出
        x_float = x.to(torch.float32)
        # 计算均值和方差（在最后 d_model 维度上），keepdim保证之后可以广播
        mean = x.mean(dim=-1, keepdim=True)
        #官方写法
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        result = (x_float - mean) / torch.sqrt(var + self.eps)
        
        # 应用仿射变换
        if self.elementwise_affine:
            result = result * self.weight + self.bias
        
        return result.to(in_dtype)