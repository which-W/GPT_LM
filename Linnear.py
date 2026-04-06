import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, constant_

class CustomLinear(nn.Module):
    def __init__(self, in_features: int , out_features: int , bias=True ,device = None ,dtype = None):
        super().__init__()
        
        #配置工厂
        factory_par = {"device" : device , "dtype" : dtype}
        # 权重矩阵 [out_features, in_features]
        self.weight = nn.Parameter(torch.empty(out_features, in_features,**factory_par))
        
        # 偏置项 [out_features]
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features,**factory_par))
        else:
            self.register_parameter('bias', None)
        
        # 初始化参数
        self.reset_parameters(in_features , out_features)
    
    def reset_parameters(self,in_features,out_features):
        # 权重使用截断正态分布初始化,std的设置为了防止方差抖动太大或直接为0
        std = (2.0 / (in_features * out_features)) ** 0.5
        trunc_normal_(self.weight,mean=0.0, std=std , a=std*(-3) , b=std*(3))
        
        # 偏置初始化为0
        if self.bias is not None:
            constant_(self.bias, 0)
    
    def forward(self, x):
        # x: [..., in_features]
        # weight: [out_features, in_features]
        # output: [..., out_features]
        # einsum实现: 最后一个维度做点积
        output = torch.einsum('...i,oi->...o', x, self.weight)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


# 使用示例
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建自定义Linear层
    linear = CustomLinear(in_features=512, out_features=256, bias=True,device = device)
    
    # 输入数据
    batch_size = 32
    seq_len = 128
    x = torch.randn(batch_size, seq_len, 512,device = device)
    
    # 前向传播
    output = linear(x)  # [32, 128, 256]
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证参数
    print(f"Weight shape: {linear.weight.shape}")  # [256, 512]
    print(f"Bias shape: {linear.bias.shape}")      # [256]
    
    # 对比官方nn.Linear
    official_linear = nn.Linear(512, 256).to(device)
    output_official = official_linear(x)
    print(f"Official output shape: {output_official.shape}")