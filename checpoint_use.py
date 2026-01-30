import torch

def save_checkpoint(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    iteration:int,
    out,
):
    """
        保存当前训练状态
    """
    
    #构建一个包含所有必要信息的字典
    checkpoint = {
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'iteration':iteration
    }
    
    #使用torch.save写入
    torch.save(checkpoint,out)

def load_checkpoint(
    src,
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
):
    """
        从检查点恢复状态，并返回保存时的迭代次数
    """
    #加载字典
    #使用map_location='cpu'防止出现在没有显卡的机子上报错
    checkpoint = torch.load(src,map_location='cpu')
    
    #加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #返回保存时的迭代次数
    return checkpoint['iteration']