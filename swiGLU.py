import torch
from torch import nn
def silu_fn(in_featrue):
    
    return in_featrue * torch.sigmoid(in_featrue)
class SwiGLU(nn.Module):
    #d_ff必须为64的倍数，并且为d_model的8/3倍
    def __init__(self,d_model:int,d_ff:int,device = None , dtype = None):
        super().__init__()
        factory_par = {"device":device , "dtype":dtype}
        #w1,w3为升维层，d_model->d_ff
        self.w1 = nn.Linear(d_model ,d_ff , **factory_par)
        self.w3 = nn.Linear(d_model,d_ff, **factory_par)
        
        #w2为降维层d_ff->d_model
        self.w2 = nn.Linear(d_ff,d_model, **factory_par)
    def forward(self,x:torch.Tensor):
        gate = silu_fn(self.w1(x))
        signal = self.w3(x)
        
        return self.w2(gate * signal)