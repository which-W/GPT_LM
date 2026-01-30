import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """
    AdamW优化器
    参数:
        params: 可迭代的参数或定义参数组的字典
        lr: 学习率 (默认: 1e-3)
        betas: 用于计算梯度及其平方的运行平均值的系数 (默认: (0.9, 0.999))
        eps: 为了提高数值稳定性而添加到分母的项 (默认: 1e-8)
        weight_decay: 权重衰减系数 (默认: 0.01)
        amsgrad: 是否使用AMSGrad变体 (默认: False)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, amsgrad=False):
        
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        """
        执行单个优化步骤
        """
        loss = None
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩估计
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 二阶矩估计
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] 
                state['step'] += 1
                
                # 更新一阶矩和二阶矩的指数移动平均
                # m = betal * m + (1 - betal) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = betal * v + (1 - betal) * grand^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正,消除初始值为0带来的偏移计算矫正后的alpha
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                
                #参数更新：theta = theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                #应用解耦的权重衰减
                # theta = theta - alpha * lambda * theta
                if wd:
                    p.add_(p, alpha=-lr*wd)
        return loss


# 使用示例
if __name__ == "__main__":
    # 创建一个简单的模型
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 模拟训练
    print("开始训练...")
    for epoch in range(5):
        # 创建随机数据
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        
        # 前向传播
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print("\n优化器状态示例 (第一个参数):")
    first_param = next(model.parameters())
    state = optimizer.state[first_param]
    print(f"步数: {state['step']}")
    print(f"一阶矩均值: {state['exp_avg'].mean().item():.6f}")
    print(f"二阶矩均值: {state['exp_avg_sq'].mean().item():.6f}")