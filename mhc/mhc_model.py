import torch
from torch import nn
import math
import torch.nn.functional as F

def sinkhorn_knopp(matrix: torch.Tensor, num_iter: int = 20, epsilon: float = 1e-20) -> torch.Tensor:
    """
    Sinkhorn-Knopp 算法: 双随机归一化
    所有行和为1,所有列和为1,且元素非负。
    """
    # 确保元素非负
    K = torch.exp(matrix.float() - matrix.float().amax(dim=(-1, -2), keepdim=True))
    for _ in range(num_iter):
        # 行归一化，使每行和为1
        K = K / (K.sum(dim=-1, keepdim=True) + epsilon)
        # 列归一化，使每列和为1
        K = K / (K.sum(dim=-2, keepdim=True) + epsilon)
    return K

class mHC(nn.Module):
    """
    Multi-branch Hierarchical Connection (mHC)

    实现多分支层次连接机制,用于:
    1. Width connection: 不同分支之间的信息交互
    2. Depth connection: 不同层之间的信息传递 (残差连接)
    """
    def __init__(self, d_model, n, device=None, dtype=None):
        super(mHC, self).__init__()
        self.d_model = d_model
        self.n = n
        self.nc = n * d_model
        self.n2 = n * n

        self.phi = nn.Linear(self.nc, self.n2 + 2 * self.n, bias=False, device=device, dtype=dtype)
        # 也可拆分成3个矩阵




        self.a = nn.Parameter(torch.full((3,), 0.01, device=device, dtype=dtype))
        # 也可拆分成3个



        self.b = nn.Parameter(torch.zeros(self.n2 + 2 * self.n, device=device, dtype=dtype))
        # 也可拆分成3个矩阵




    # 不同分支之间信息交互
    def width_connection(self, hidden_states):
        """
        Width connection: 分支间信息交互
        
        参数：
            hidden_states: [B, L, n, d_model] 前一层的输出
            
        返回：
            h_pre: [B, L, 1, d_model] 压缩后的表示
            h_res: [B, L, n, d_model] 残差分支
            H_post: [B, L, n] 后处理权重
        """
        B, L, n, D = hidden_states.shape  # 张量形状或计算公式：[B, L, n, d_model]
        hidden_states_flatten = hidden_states.flatten(2)  # 张量形状或计算公式：[B, L, n*d_model]

        # 计算归一化因子
        r = (hidden_states_flatten.float().norm(dim=-1, keepdim=True) / math.sqrt(self.nc)).clamp(min=1e-6).to(hidden_states.dtype)  # 形状为 [B, L, 1]

        # 通过线性层计算连接权重
        H = self.phi(hidden_states_flatten)  # 张量形状或计算公式：[B, L, n*n + 2*n]
        H_pre = (1/r) * H[:, :, :self.n] * self.a[0] + self.b[0:self.n]  # 张量形状或计算公式：[B, L, n]
        H_post = (1/r) * H[:, :, self.n:self.n*2] * self.a[1] + self.b[self.n:self.n*2]  # 张量形状或计算公式：[B, L, n]
        H_res = (1/r) * H[:, :, self.n*2:] * self.a[2] + self.b[self.n*2:]  # 张量形状或计算公式：[B, L, n*n]

        # 应用激活函数
        H_pre = F.sigmoid(H_pre)
        H_post = 2 * F.sigmoid(H_post)

        # H_res 使用 Sinkhorn-Knopp 算法进行双随机归一化
        H_res = H_res.reshape(B, L, self.n, self.n)  # 张量形状或计算公式：[B, L, n, n]
        H_res = sinkhorn_knopp(H_res).to(hidden_states.dtype)

        # 计算压缩表示和残差分支
        H_pre = H_pre.unsqueeze(dim=2)  # 张量形状或计算公式：[B, L, 1, n]
        h_pre = torch.matmul(H_pre, hidden_states)  # 张量形状或计算公式：[B, L, 1, n] @ [B, L, n, d_model] = [B, L, 1, d_model]
        h_res = torch.matmul(H_res, hidden_states)  # 张量形状或计算公式：[B, L, n, n] @ [B, L, n, d_model] = [B, L, n, d_model]

        return h_pre, h_res, H_post

    # 不同层之间信息传递，残差连接
    def depth_connection(self, h_res, hidden_states, H_post):
        """
        Depth connection: 层间信息传递
        
        参数：
            h_res: [B, L, n, d_model] 残差分支
            hidden_states: [B, L, d_model] 经过 attention 或 FFN 后的输出
            H_post: [B, L, n] 后处理权重
            
        返回：
            output: [B, L, n, d_model] 输出
        """
        # 张量形状或计算公式：H_post: [B, L, n]
        # hidden_states: [B, L, d_model]，经过 attention 或者 FFN 后的输出
        h_post = torch.matmul(H_post.unsqueeze(-1), hidden_states.unsqueeze(-2))  # 张量形状或计算公式：[B, L, n, 1] * [B, L, 1, d_model] = [B, L, n, d_model]
        output = h_post + h_res

        return output  # 张量形状或计算公式：[B, L, n, d_model]
