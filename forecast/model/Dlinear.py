import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class moving_average(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_average, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_average(kernel_size, stride=1)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
#%%
class DLinearModel(nn.Module):
    def __init__(self, seq_len,pred_len,channels):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = channels
        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        for i in range(self.channels):
            self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
            self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
        trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
        for i in range(self.channels):
            seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
            trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        x = seasonal_output + trend_output
        return x.permute(0,2,1).squeeze(-1)