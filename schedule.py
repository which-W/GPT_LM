import numpy as np
import math

class CosineAnnealingWarmupScheduler:
    """
    带预热的余弦退火学习率调节器
    
    参数:
        base_lr: 基础学习率
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: 预热步数
        total_steps: 总训练步数
    """
    def __init__(self,max_lr:float, min_lr:float, 
                 warmup_steps:int, total_steps:int):
        
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine_steps = total_steps - warmup_steps
        
    def get_lr_cosine_shedule(self, step):
        """
        获取当前步数对应的学习率
        
        参数:
            step: 当前训练步数
        返回:
            当前学习率
        """
        if step < self.warmup_steps:
            # 线性预热阶段,从0线性增长到max_lr
            return self.max_lr * step / self.warmup_steps
            #衰减周期后维持最小值 
        if step > self.total_steps:
            return self.min_lr

        #余弦退火核心逻辑
        #计算当前处于退火阶段的进度百分比（0.0到1.0）
        #step - warmup_steps：距离预热结束还要走多少步
        #total_steps - warmup_steps为整个退火阶段总长度
        decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        
        #计算余弦系数,[0.0到1.0]
        # math.cos(math.pi*decay_ratio)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        
        #最终计算，学习率从max降到min
        return self.min_lr + coeff * (self.max_lr - self.min_lr) 