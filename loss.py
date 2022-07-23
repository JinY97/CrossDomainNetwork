'''
Author: Yin Jin
Date: 2022-03-08 20:17:04
LastEditTime: 2022-07-23 13:49:13
LastEditors: JinYin
Description: 定义loss函数
'''

import torch
from torch import nn
import numpy as np
from audtorch.metrics.functional import pearsonr

def denoise_loss_mse(denoise, clean):      
  loss = torch.nn.MSELoss()
  return loss(denoise, clean)

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return torch.from_numpy(10 * np.log10(ratio))

