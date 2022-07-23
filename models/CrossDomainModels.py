import torch
import torch.nn as nn
import torch_dct
from functools import partial
from scipy.fftpack import dct, idct
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce
from .Models import BasicBlockall

class CrossDomainResCNN(nn.Module):
    def __init__(self, data_num=512):
        super(CrossDomainResCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.linear = nn.Linear(32 * 2 * data_num, data_num)
        
    def forward(self, x):
        f = torch_dct.dct(x, norm='ortho')
        f = self.model_f(f).view(x.shape[0], -1)
        t = self.model_t(x).view(x.shape[0], -1)
        f = torch_dct.idct(f, norm='ortho')
        return self.linear(torch.cat((t, f), dim=-1))

class DualTimeResCNN(nn.Module):
    def __init__(self, data_num=512):
        super(DualTimeResCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.linear = nn.Linear(32 * 2 * data_num, data_num)
        
    def forward(self, x):
        f = self.model_f(x).view(x.shape[0], -1)
        t = self.model_t(x).view(x.shape[0], -1)
        return self.linear(torch.cat((t, f), dim=-1))
    
class CrossDomainSimpleCNN(nn.Module):
    def __init__(self, data_num=512):
        super(CrossDomainSimpleCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.linear = nn.Linear(64 * 2 * data_num, data_num)

    def forward(self, x):
        f = torch_dct.dct(x, norm='ortho')
        f = self.model_f(f).view(x.shape[0], -1)
        t = self.model_t(x).view(x.shape[0], -1)
        f = torch_dct.idct(f, norm='ortho')
        return self.linear(torch.cat((t, f), dim=-1))

class DualTimeSimpleCNN(nn.Module):
    def __init__(self, data_num=512):
        super(DualTimeSimpleCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.linear = nn.Linear(64 * 2 * data_num, data_num)

    def forward(self, x):
        f = self.model_f(x).view(x.shape[0], -1)
        t = self.model_t(x).view(x.shape[0], -1)
        return self.linear(torch.cat((t, f), dim=-1))

class CrossDomainFCNN(nn.Module):
    def __init__(self, data_num=512):
        super(CrossDomainFCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        
        self.model_t = nn.Sequential(
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        self.linear = nn.Linear(2 * data_num, data_num)
    
    def forward(self, x):
        f = torch_dct.dct(x, norm='ortho')
        f = self.model_f(f).view(f.shape[0], -1)
        t = self.model_t(x).view(x.shape[0], -1)
        # f = torch_dct.idct(f, norm='ortho')
        return self.linear(torch.cat((t, f), dim=-1))
    
class DualTimeFCNN(nn.Module):
    def __init__(self, data_num=512):
        super(DualTimeFCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        
        self.model_t = nn.Sequential(
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        self.linear = nn.Linear(2 * data_num, data_num)
    
    def forward(self, x):
        t = self.model_t(x).view(x.shape[0], -1)
        f = self.model_f(x).view(x.shape[0], -1)
        return self.linear(torch.cat([t, f], dim=-1))
        