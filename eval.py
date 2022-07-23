'''
Author: JinYin
Date: 2022-07-02 17:42:47
LastEditors: JinYin
LastEditTime: 2022-07-23 14:32:24
FilePath: \CrossDomainFramework\eval.py
Description: 
'''
from cProfile import label
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import numpy as np
import torch
import torch.nn as nn
import torch_dct
from opts import get_opts
from models import *
from preprocess.MultichannelPreprocess import *
from audtorch.metrics.functional import pearsonr

from tools import pick_models

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def cal_SNR_multichannel(predict, truth):
    PS = np.sum(np.sum(np.square(truth), axis=-1), axis=-1)  # power of signal
    PN = np.sum(np.sum(np.square((predict - truth)), axis=-1), axis=-1)  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def rrmse_multichannel(predict, truth):
    res = np.sqrt(((predict - truth) ** 2).mean().mean()) / np.sqrt((truth ** 2).mean().mean())
    return res

def rrmse_singlechannel(predict, truth):
    res = np.sqrt(((predict - truth) ** 2).mean()) / np.sqrt((truth ** 2).mean())
    return res

def acc_multichannel(predict, truth):
    acc = []
    for i in range(predict.shape[0]):
        acc.append(np.corrcoef(predict[i], truth[i])[1, 0])
    return np.mean(np.array(acc))

def acc_singlechannel(predict, truth):
    return np.corrcoef(predict, truth)[1, 0]

# Average results for the entire dataset
def multi_channel():
    opts = get_opts()
    opts.epochs = 200
    opts.depth = 6
    opts.noise_type = 'EOG'
    opts.batch_size = 8
    Network = "FCNN"
    
    opts.EEG_path = r"./data/Pure_Data.mat"
    opts.NOS_path = r"./data/Contaminated_Data.mat"
    # FCNN SimpleCNN ResCNN
    all_rrmse, all_acc, all_snr = [], [], []
    acc, rrmse, snrs = [], [], []
    
    for fold in range(10):
        torch_path = r"E:\02_personal_work\02_Result\Summary_Result\{}\multi_channel\{}_EOG_multi_channel_Src_200_{}/best_{}.pth".format(Network, Network, fold, Network)
        model = torch.load(torch_path)
        
        train_list, test_list = DivideDataset(54)       # 54表示总人数
        _, _, _, _, EEG_test_data, NOS_test_data = LoadEEGData(opts.EEG_path, opts.NOS_path, train_list, test_list, fold=fold)
        test_data = GetEEGData(EEG_test_data, NOS_test_data, opts.batch_size)
        
        model.eval()
        losses = []
        
        for batch_id in range(test_data.len()):
            x_t, y_t = test_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device), torch.Tensor(y_t).to(opts.device)
            
            with torch.no_grad():
                p_t = model(x_t)
                loss = (((p_t - y_t) ** 2).mean(dim=-1).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).mean(dim=-1).sqrt()).detach()
                losses.append(loss)
                p_t, y_t = p_t.cpu().numpy(), y_t.cpu().numpy()
                for i in range(p_t.shape[0]):
                    rrmse.append(rrmse_multichannel(p_t[i], y_t[i]))
                    acc.append(acc_multichannel(p_t[i], y_t[i]))
                    snrs.append(cal_SNR_multichannel(p_t[i], y_t[i]))
        
    all_rrmse.append(np.mean(np.array(rrmse)))
    all_acc.append(np.mean(np.array(acc)))
    all_snr.append(np.mean(np.array(snrs)))
    print(np.array(rrmse).shape)
    print('test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(np.mean(np.array(rrmse)), np.mean(np.array(acc)), np.mean(np.array(snrs))))
    
    print('test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(np.mean(np.array(all_rrmse)), np.mean(np.array(all_acc)), np.mean(np.array(all_snr))))  
    
multi_channel()
    
              
    
    
    


