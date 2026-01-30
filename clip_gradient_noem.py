import torch
from collections.abc import Iterator

def Clip_gradient_noem(parameters:Iterator[torch.nn.Parameter],max_norm:float):
    """
    Docstring for Clip_gradient_noem
    全局梯度裁剪 g_new = g_old * M / g_ouleng + eps
    :param parameters: 模型所有参数
    :type parameters: Iterator[torch.nn.Parameter]
    :param max_norm: 允许的最大梯度L2范数(M)
    :type max_norm: float
    """
    #过滤掉没有梯度的参数
    params_with_grads = [p for p in parameters if p.grad is not None]
    if not params_with_grads:
        return
    #计算全局L2范数
    total_norm = 0.0
    for p in params_with_grads:
        #使用.detache()非常重要
        #梯度裁剪是在计算完之后对倒数进行的数值操作，不能将计算范数也计入计算图
        # p=2->计算出当前梯度L2范数L_i
        param_norm = torch.norm(p.grad.detach() , p=2)
        
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    #检查是否触发裁剪
    eps = 1e-6 #防止除以0
    
    if total_norm > max_norm:
        #计算统一的缩放系数
        clip_coef = max_norm / (total_norm + eps)
        
        #原地修改各个参数的梯度
        #使用_mul直接修改内存不创建副本减少内存使用
        for p in params_with_grads:
            p.grad.detach().mul_(clip_coef)