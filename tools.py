'''
Author: JinYin
Date: 2022-07-01 21:46:06
LastEditors: JinYin
LastEditTime: 2022-07-23 14:30:27
FilePath: \CrossDomainFramework\tools.py
Description: 
'''
from models import *
from opts import get_opts

import librosa
import mne
import os
from scipy.fftpack import fft
from scipy import signal
from matplotlib import pyplot as plt
from audtorch.metrics.functional import pearsonr

def pick_models(opts, data_num=540):
    if opts.denoise_network == 'SimpleCNN':
            if opts.channel_type == "single_channel":
                if opts.mode == "DTD":
                    model = DualTimeSimpleCNN(data_num).to(opts.device)
                elif opts.mode == "CD":
                    model = CrossDomainSimpleCNN(data_num).to(opts.device)
                else:
                    model = SimpleCNN(data_num).to(opts.device)
            else:
                if opts.mode == "DTD":
                    model = MultichannelDualTimeSimpleCNN(data_num).to(opts.device)
                elif opts.mode == "CD":
                    model = MultichannelCrossDomainSimpleCNN(data_num).to(opts.device)
                else:
                    model = MultichannelSimpleCNN(data_num).to(opts.device)
                     
    elif opts.denoise_network == 'FCNN':  
        if opts.channel_type == "single_channel": 
            if opts.mode == "DTD":
                model = DualTimeFCNN(data_num).to(opts.device)
            elif opts.mode == "CD":
                model = CrossDomainFCNN(data_num).to(opts.device)
            else:
                model = FCNN(data_num).to(opts.device)
        else:
            if opts.mode == "DTD":
                model = MultichannelDualTimeFCNN(data_num).to(opts.device)
            elif opts.mode == "CD":
                model = MultichannelCrossDomainFCNN(data_num=540).to(opts.device)
            else:
                model = MultichannelFCNN(data_num).to(opts.device)
                
                
    elif opts.denoise_network == 'ResCNN':
        if opts.channel_type == "single_channel": 
            if opts.mode == "DTD":
                model = DualTimeResCNN(data_num).to(opts.device)
            elif opts.mode == "CD":
                model = CrossDomainResCNN(data_num).to(opts.device)
            else:
                model = ResCNN(data_num).to(opts.device)
        else:
            if opts.mode == "DTD":
                model = MultichannelDualTimeResCNN(data_num).to(opts.device)
            elif opts.mode == "CD":
                model = MultichannelCrossDomainResCNN(data_num).to(opts.device)
            else:
                model = MultichannelResCNN(data_num).to(opts.device)
    else:
        print("model name is error!")
        pass
    return model

if __name__ == "__main__":
    pass
    