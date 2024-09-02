import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cpu')
class Nbeats(nn.Module):
    def __init__(self,unit,theta_dim,seq_length,pred_length):
        super(Nbeats,self).__init__()
        self.unit=unit
        self.seq_length=seq_length
        self.pred_length=pred_length
        self.theta_dim=theta_dim
        self.fc=nn.Sequential(nn.Linear(seq_length,unit),nn.ReLU(),
                              nn.Linear(unit,unit),nn.ReLU(),
                              nn.Linear(unit,unit),nn.ReLU(),
                              nn.Linear(unit,unit),nn.ReLU())
        self.theta_f_fc = self.theta_b_fc = nn.Linear(unit, theta_dim, bias=False)
    def forward(self,x):
        out=self.fc(x)
        return out
#%%
def linspace(input_len,output_len):
    total=input_len+output_len
    start=0
    end=input_len+output_len-1
    s=np.linspace(start/total,end/total,total,dtype=np.float32)
    back_s=s[:input_len]
    pred_ls=s[input_len:]
    return back_s,pred_ls
#%%
class Trendmodel(Nbeats):
    def __init__(self,unit,seq_length,pred_length,theta_dim):
        super().__init__(
          unit=unit,
          seq_length=seq_length,
          pred_length=pred_length,
          theta_dim=theta_dim,
        )
        seq_lin,pred_lin = linspace(seq_length,pred_length)
        seq=torch.Tensor([seq_lin**i for i in range(theta_dim)]).to(torch.float32).to(device)
        norm=np.sqrt(seq_length/theta_dim)
        self.register_buffer('back',seq*norm)
        pred=torch.Tensor([pred_lin**i for i in range(theta_dim)]).to(torch.float32).to(device)
        self.register_buffer('fore',pred*norm)
    def forward(self,x):
        x = super().forward(x)
        backcast=self.theta_b_fc(x).mm(self.back)
        forecast=self.theta_f_fc(x).mm(self.fore)
        return backcast,forecast
#%%
class Seasonalmodel(Nbeats):
    def __init__(self,unit,seq_length,pred_length,theta_dim,min_period=1):
        super().__init__(
          unit=unit,
          seq_length=seq_length,
          pred_length=pred_length,
          theta_dim=theta_dim,
        )
        self.min_period=min_period
        seq_lin,pred_lin=linspace(seq_length,pred_length)
        p1,p2=(theta_dim//2,theta_dim//2) if theta_dim % 2==0 else (theta_dim//2, theta_dim//2+1)
        s1_b=torch.tensor([np.cos(2*np.pi*i*seq_lin) for i in self.get_frequencies(p1)], dtype=torch.float32)
        s2_b=torch.tensor([np.sin(2*np.pi*i*seq_lin) for i in self.get_frequencies(p2)], dtype=torch.float32)
        self.register_buffer("S_back",torch.cat([s1_b, s2_b]))
        s1_f=torch.tensor([np.cos(2*np.pi*i*pred_lin)for i in self.get_frequencies(p1)], dtype=torch.float32)
        s2_f=torch.tensor([np.cos(2*np.pi*i*pred_lin)for i in self.get_frequencies(p2)], dtype=torch.float32)
        self.register_buffer("S_fore",torch.cat([s1_f, s2_f]))
    def forward(self,x):
        x=super().forward(x)
        backcast=self.theta_b_fc(x).mm(self.S_back)
        forecast=self.theta_f_fc(x).mm(self.S_fore)
        return backcast,forecast
    def get_frequencies(self,n):
        return np.linspace(0,(self.seq_length+ self.pred_length)/self.min_period,n)
#%%
class Generralmodel(Nbeats):
    def __init__(self,unit,seq_length,pred_length,theta_dim):
        super().__init__(
          unit=unit,
          seq_length=seq_length,
          pred_length=pred_length,
          theta_dim=theta_dim,
        )
        self.backcast_fc=nn.Linear(theta_dim, seq_length)
        self.forecast_fc=nn.Linear(theta_dim, pred_length)
    def forward(self, x):
        x = super().forward(x)
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
#%%
class NBEATModel(nn.Module):
    def __init__(self,unit=6,seq_length=10,pred_length=2,theta_dim=8,block_num=3):
        super(NBEATModel,self).__init__()
        self.unit=unit
        self.seq_length=seq_length
        self.pred_length=pred_length
        self.theta_dim=theta_dim
        self.blocks=nn.ModuleList()
        for i in range(block_num):
            block1=Trendmodel(unit=unit,seq_length=seq_length,pred_length=pred_length,theta_dim=theta_dim)
            self.blocks.append(block1)
            block2=Seasonalmodel(unit=unit,seq_length=seq_length,pred_length=pred_length,theta_dim=theta_dim)
            self.blocks.append(block2)
    def forward(self,x):
        x = x.squeeze(-1)
        back_z=torch.zeros((x.size(0),self.seq_length),dtype=torch.float32,device=device)
        fore_z=torch.zeros((x.size(0),self.pred_length),dtype=torch.float32,device=device)
        for block in self.blocks:
            back,fore=block(x)
            back_z=back-back_z
            fore_z=fore+fore_z
        return fore_z