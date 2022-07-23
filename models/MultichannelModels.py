'''
Author: JinYin
Date: 2022-07-01 12:21:42
LastEditors: JinYin
LastEditTime: 2022-07-16 21:30:56
FilePath: \01_FreqTimeEEG_new\models\MultichannelModels.py
Description: 多通道网络
'''
from matplotlib.pyplot import plot
import torch
import torch.nn as nn
import torch_dct
from .Models import BasicBlockall
import matplotlib.pyplot as plt

class MultichannelFCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelFCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
class MultichannelSimpleCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelSimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel_num, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.conv = nn.Conv1d(64, channel_num, 3, 1, 1)
        

    def forward(self, x):
        t = self.model(x)
        return self.conv(t)

class MultichannelResCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelResCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel_num, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.cov = nn.Conv1d(32, channel_num, 3, 1, 1)

    def forward(self, x):
        t = self.model(x)
        return self.cov(t)

class MultichannelCrossDomainFCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelCrossDomainFCNN, self).__init__()
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
        self.linear = nn.Linear(data_num * 2, data_num)
        
    def forward(self, x):
        t = self.model_t(x)
        f = self.model_f(torch_dct.dct(x, norm='ortho'))
        # f = torch_dct.idct(f, norm='ortho')
        return self.linear(torch.cat([t, f], dim=-1))
    
class MultichannelDualTimeFCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelDualTimeFCNN, self).__init__()
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
        t = self.model_t(x)
        f = self.model_f(x)
        return self.linear(torch.cat([t, f], dim=-1))
       
class MultichannelCrossDomainSimpleCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelCrossDomainSimpleCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(channel_num, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(channel_num, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.conv = nn.Conv1d(64*2, channel_num, 3, 1, 1)
        

    def forward(self, x):
        t = self.model_t(x)
        f = self.model_f(torch_dct.dct(x, norm='ortho'))
        f = torch_dct.idct(f, norm='ortho')
        return self.conv(torch.cat([t, f], dim=1))

class MultichannelDualTimeSimpleCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelDualTimeSimpleCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(channel_num, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(channel_num, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.conv = nn.Conv1d(64 * 2, channel_num, 3, 1, 1)
        

    def forward(self, x):
        t = self.model_t(x)
        f = self.model_f(x)
        return self.conv(torch.cat([t, f], dim=1))

class MultichannelCrossDomainResCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelCrossDomainResCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(channel_num, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(channel_num, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.conv = nn.Conv1d(32 * 2, channel_num, 3, 1, 1)

    def forward(self, x):
        t = self.model_t(x)
        f = self.model_f(torch_dct.dct(x, norm='ortho'))
        f = torch_dct.idct(f, norm='ortho')
        return self.conv(torch.cat([t, f], dim=1))

class MultichannelDualTimeResCNN(nn.Module):
    def __init__(self, data_num=512, channel_num=4):
        super(MultichannelDualTimeResCNN, self).__init__()
        self.model_f = nn.Sequential(
            nn.Conv1d(channel_num, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.model_t = nn.Sequential(
            nn.Conv1d(channel_num, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.conv = nn.Conv1d(32 * 2, channel_num, 3, 1, 1)

    def forward(self, x):
        t = self.model_t(x)
        f = self.model_f(x)
        return self.conv(torch.cat([t, f], dim=1))
    