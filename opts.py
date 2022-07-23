'''
Author: Yin Jin
Date: 2022-03-08 19:38:55
LastEditTime: 2022-07-05 21:43:02
LastEditors: JinYin
Description: 
'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--noise_type', type=str, default='EOG')
    parser.add_argument('--EEG_path', type=str, default='./data/EEG.npy')
    parser.add_argument('--NOS_path', type=str, default='./data/EOG.npy')
    parser.add_argument('--denoise_network', type=str, default='FCNN')
    parser.add_argument('--channel_type', type=str, default='single_channel')       # single_channel multi_channel
    parser.add_argument('--save_path', type=str, default=r'E:\03_PhD_study\02_personal_work\04_Data\02_Result\02_FreqTime\Denoise\Summary_Result/')
    parser.add_argument('--mode', type=str, default='Src')     # DTD(DualTimeDomain)/CrossDomain(CD)/Src
    
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--depth', type=float, default=6)
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_result', type=bool, default=True)
    
    
    
    opts = parser.parse_args()
    return opts
