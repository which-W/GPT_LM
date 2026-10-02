"""保留单精度参数，使用自动混合精度及梯度缩放执行训练。"""
import torch


class TrainingPrecision:
    """在 CUDA 上按指定精度计算，CPU 默认使用单精度。"""
    def __init__(self, device, dtype):
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.enabled = self.device_type == "cuda" and dtype != torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled and dtype == torch.float16)

    def context(self):
        return torch.autocast(self.device_type, dtype=self.dtype if self.enabled else torch.bfloat16,
                              enabled=self.enabled)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def unscale(self, optimizer):
        self.scaler.unscale_(optimizer)

    def step(self, optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()
