"""统一专家模型的设备选择；空设备列表用于 CPU 验证。"""
import torch


def moe_main_device(device_ids=None, main_device=0):
    if device_ids == [] or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(f'cuda:{main_device}')
