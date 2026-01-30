import torch

def Cross_entropy(logits:torch.Tensor,targets:torch.Tensor):
    """
        logits:形状为(batch,seq,vocal_size)的预测分值
        tragets:(batch,seq)的真实Token ID
    
    """
    
    #计算每一组logits的最大值M，用于数值稳定
    m = torch.max(logits , dim=-1, keepdim= True).values
    
    #提取目标位置对应的原始分值o_y
    targets_logits = torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    
    #计算log_sum_exp项
    shifted_logits = logits - m
    
    #公式:M + log(sum(exp(o-M)))
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits),dim=-1))
    
    #计算每一个token的独立损失值
    loss = log_sum_exp - targets_logits
    
    #对批次求平均并返回
    return torch.mean(loss)