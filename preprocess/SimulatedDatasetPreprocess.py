'''
Author: Yin Jin
Date: 2022-03-08 20:15:16
LastEditTime: 2022-07-23 14:22:41
LastEditors: JinYin
Description: 数据预处理
'''

import math
import torch
import numpy as np
import scipy.io
from tqdm import trange
from sklearn.model_selection import KFold

def DivideDataset(person_num):
    x = np.arange(person_num)      
    kf = KFold(n_splits=10, shuffle=False)
    train_list, test_list = [], []
    for train_index, test_index in kf.split(x):
        train_list.append(train_index)
        test_list.append(test_index)
    return train_list, test_list
    
def LoadEEGData(EEG_path, NOS_path, train_list, test_list, fold):
    data_signals = scipy.io.loadmat(EEG_path)
    noisy_artifact = scipy.io.loadmat(NOS_path)
    
    train_id, test_id = train_list[fold], test_list[fold]
    len_val = int(len(train_id) * 0.1)
    val_id, train_id = train_id[:len_val], train_id[len_val:]
    
    fix_length = 5400
    train_data, train_noisy = [], []
    for n in train_id:
        n = n + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']
        shapes = data_artifact.shape
        if shapes[1] < fix_length:
            data_artifact = np.concatenate([data_artifact, torch.zeros(size=(shapes[0], fix_length - shapes[1]))], axis=1)
            reference = np.concatenate([reference, torch.zeros(size=(shapes[0], fix_length - shapes[1]))], axis=1)
        else:
            data_artifact = data_artifact[:, :fix_length]
            reference = reference[:, :fix_length]
        train_data.append(reference)
        train_noisy.append(data_artifact)

    EEG_train_data, NOS_train_data = train_data, train_noisy
    
    val_data, val_noisy = [], []
    for n in val_id:
        n = n + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']
        
        reference, data_artifact = reference[:, 0:5400], data_artifact[:, 0:5400]
        reference, data_artifact = reference.reshape(-1, 540), data_artifact.reshape(-1, 540)
        
        val_data.append(reference)
        val_noisy.append(data_artifact)
        
    EEG_val_data = np.array(val_data).reshape(-1, 540)
    NOS_val_data = np.array(val_noisy).reshape(-1, 540)

    test_data, test_noisy = [], []
    for n in test_id:
        n = n + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']
        
        reference, data_artifact = reference[:, 0:5400], data_artifact[:, 0:5400]
        reference, data_artifact = reference.reshape(-1, 540), data_artifact.reshape(-1, 540)
        
        test_data.append(reference)
        test_noisy.append(data_artifact)
        
    EEG_test_data = np.array(test_data).reshape(-1, 540)
    NOS_test_data = np.array(test_noisy).reshape(-1, 540)

    return EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data

class GetEEGData(object):
    def __init__(self, EEG_data, NOS_data, batch_size=128):
        super(GetEEGData, self).__init__()
        self.EEG_data, self.NOS_data = EEG_data, NOS_data
        self.batch_size = batch_size

    def len(self):
        return math.ceil(self.EEG_data.shape[0] / self.batch_size)     # ceil

    def get_item(self, item):
        EEG_data = self.EEG_data[item, :]
        NOS_data = self.NOS_data[item, :]
        return NOS_data, EEG_data

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min((batch_id + 1) * self.batch_size, self.EEG_data.shape[0])
        EEG_NOS_batch, EEG_batch = [], []
        for item in range(start_id, end_id):
            EEG_NOS_data, EEG_data = self.get_item(item)
            EEG_NOS_batch.append(EEG_NOS_data), EEG_batch.append(EEG_data)
        EEG_NOS_batch, EEG_batch = np.array(EEG_NOS_batch), np.array(EEG_batch)
        return EEG_NOS_batch, EEG_batch

    def shuffle(self):
        EEG_id = np.random.permutation(self.EEG_data.shape[0])
        self.EEG_data = self.EEG_data[EEG_id, :]
        self.NOS_data = self.NOS_data[EEG_id, :]

class GetEEGData_train(object):
    def __init__(self, EEG_data, NOS_data, batch_size=128, device='cuda:0'):
        super(GetEEGData_train, self).__init__()
        self.device = device
        self.EEG_list = torch.Tensor(np.concatenate(np.array(EEG_data), axis=0)).to(self.device)
        self.NOS_list = torch.Tensor(np.concatenate(np.array(NOS_data), axis=0)).to(self.device)
        self.batch_size = batch_size
        self.random_shuffle()

    def len(self):
        return math.floor(self.start_point_idxs.shape[0] / self.batch_size)  # ceil

    def get_item(self, item):
        EEG_data = self.EEG_data[item, :]
        NOS_data = self.NOS_data[item, :]
        return NOS_data, EEG_data

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min((batch_id + 1) * self.batch_size, self.start_point_idxs.shape[0])
        start_point_batch = self.start_point_idxs[start_id:end_id]
        sample_batch = self.sample_idxs[start_id:end_id]
        EEG_samples = self.EEG_list[sample_batch, ...]
        NOS_samples = self.NOS_list[sample_batch, ...]
        gather_idx = torch.Tensor(np.array(list(range(540)))).to(self.device).long().unsqueeze(0)
        gather_idx = gather_idx.repeat((EEG_samples.shape[0], 1))
        gather_idx = gather_idx + start_point_batch.unsqueeze(-1) # 这句话是关键 添加偏移量
        EEG_batch = EEG_samples.gather(index=gather_idx, dim=1)
        EEG_NOS_batch = NOS_samples.gather(index=gather_idx, dim=1)
        return EEG_NOS_batch, EEG_batch

    def random_shuffle(self):
        num_per_epoch_sample = 50
        self.start_point_idxs, self.sample_idxs = [], []
        for i in range(self.EEG_list.shape[0]):
            idx = np.random.permutation(self.EEG_list.shape[1] - 540)[:num_per_epoch_sample]
            self.start_point_idxs.append(idx), self.sample_idxs.append(np.zeros(shape=(num_per_epoch_sample,)) + i)
        self.start_point_idxs = np.concatenate(self.start_point_idxs, axis=0)
        self.sample_idxs = np.concatenate(self.sample_idxs, axis=0)
        shuffle = np.random.permutation(self.start_point_idxs.shape[0])
        self.start_point_idxs = torch.Tensor(self.start_point_idxs[shuffle]).to(self.device).long()
        self.sample_idxs = torch.Tensor(self.sample_idxs[shuffle]).to(self.device).long()

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    EEG_path = "./data/Pure_Data.mat"
    NOS_path = "./data/Contaminated_Data.mat"

    train_list, test_list = DivideDataset(54)
    EEG_train_data, NOS_train_data, EEG_test_data, NOS_test_data = LoadEEGData(EEG_path, NOS_path, train_list, test_list, fold=1)
    train_data = GetEEGData_train(EEG_train_data, NOS_train_data)
    for batch_id in trange(train_data.len()):
        x_t, y_t = train_data.get_batch(batch_id)
    train_data.random_shuffle()




