'''
Author: Yin Jin
Date: 2022-06-18 15:08:21
LastEditTime: 2022-07-23 14:22:25
LastEditors: JinYin
Description: 0-18: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz
'''

import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch

from opts import get_opts
from sklearn.model_selection import KFold

def DivideDataset(person_num):
    x = np.arange(person_num)
    kf = KFold(n_splits=10, shuffle=False)
    train_list, test_list = [], []
    for train_index, test_index in kf.split(x):
        train_list.append(train_index)
        test_list.append(test_index)
    return train_list, test_list
    
# Random sampling to recordings
def LoadEEGData(EEG_path, NOS_path, train_list, test_list, fold):
    data_signals = scipy.io.loadmat(EEG_path)
    noisy_artifact = scipy.io.loadmat(NOS_path)
    pick_channel = [0, 1, 2, 3]

    train_id, test_id = train_list[fold], test_list[fold]
    len_val = int(len(train_id) * 0.1)
    val_id, train_id = train_id[::len_val], train_id[:len(train_id)-len_val]

    fix_length = 5400  # Fixed length to 5400, length is adjustable
    train_data, train_noisy = [], []
    for n in train_id:
        n = n + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']
        reference, data_artifact = reference[pick_channel, :], data_artifact[pick_channel, :]
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

    val_data, val_noisy = np.zeros((4, 10 * len(val_id), 540)), np.zeros((4, 10 * len(val_id), 540))
    for i in range(len(val_id)):
        n = val_id[i] + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']
        reference, data_artifact = reference[pick_channel, :], data_artifact[pick_channel, :]

        reference, data_artifact = reference[:, 0:5400], data_artifact[:, 0:5400]
        reference, data_artifact = reference.reshape(4, -1, 540), data_artifact.reshape(4, -1, 540)

        val_data[:, i * 10:(i + 1) * 10, :] = reference
        val_noisy[:, i * 10:(i + 1) * 10, :] = data_artifact

    EEG_val_data, NOS_val_data = val_data, val_noisy
    
    test_data, test_noisy = np.zeros((4, 10 * len(test_id), 540)), np.zeros((4, 10 * len(test_id), 540))
    for i in range(len(test_id)):
        n = test_id[i] + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']
        reference, data_artifact = reference[pick_channel, :], data_artifact[pick_channel, :]

        reference, data_artifact = reference[:, 0:5400], data_artifact[:, 0:5400]
        reference, data_artifact = reference.reshape(4, -1, 540), data_artifact.reshape(4, -1, 540)

        test_data[:, i * 10:(i + 1) * 10, :] = reference
        test_noisy[:, i * 10:(i + 1) * 10, :] = data_artifact

    EEG_test_data, NOS_test_data = test_data, test_noisy

    return EEG_train_data, NOS_train_data,  EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data

# validation and test data partition
class GetEEGData(object):
    def __init__(self, EEG_data, NOS_data, batch_size=128):
        super(GetEEGData, self).__init__()
        self.EEG_data, self.NOS_data = EEG_data, NOS_data
        self.batch_size = batch_size

    def len(self):
        return math.ceil(self.EEG_data.shape[1] / self.batch_size)

    def get_item(self, item):
        EEG_data = self.EEG_data[:, item, :]
        NOS_data = self.NOS_data[:, item, :]
        return NOS_data, EEG_data

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min((batch_id + 1) * self.batch_size, self.EEG_data.shape[1])
        EEG_NOS_batch, EEG_batch = [], []
        for item in range(start_id, end_id):
            EEG_NOS_data, EEG_data = self.get_item(item)
            EEG_NOS_batch.append(EEG_NOS_data), EEG_batch.append(EEG_data)
        EEG_NOS_batch, EEG_batch = np.array(EEG_NOS_batch), np.array(EEG_batch)
        return EEG_NOS_batch, EEG_batch

# Training data partition
class GetEEGData_train(object):
    def __init__(self, EEG_data, NOS_data, batch_size=128, device='cuda:0'):
        super(GetEEGData_train, self).__init__()
        self.device = device
        self.EEG_list = torch.Tensor(np.array(EEG_data)).to(self.device)
        self.NOS_list = torch.Tensor(np.array(NOS_data)).to(self.device)
        self.batch_size = batch_size
        self.random_shuffle()

    def len(self):
        return math.floor(self.start_point_idxs.shape[0] / self.batch_size)  # ceil

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min((batch_id + 1) * self.batch_size, self.start_point_idxs.shape[0])
        start_point_batch = self.start_point_idxs[start_id:end_id]
        sample_batch = self.sample_idxs[start_id:end_id]
        EEG_samples = self.EEG_list[sample_batch, ...]
        NOS_samples = self.NOS_list[sample_batch, ...]
        gather_idx = torch.Tensor(np.array(list(range(540)))).to(self.device).long().unsqueeze(0).unsqueeze(0)
        gather_idx = gather_idx.repeat((EEG_samples.shape[0], EEG_samples.shape[1], 1))
        gather_idx = gather_idx + start_point_batch.unsqueeze(-1).unsqueeze(-1)  # add offset
        EEG_batch = EEG_samples.gather(index=gather_idx, dim=2)
        EEG_NOS_batch = NOS_samples.gather(index=gather_idx, dim=2)
        return EEG_NOS_batch, EEG_batch

    def random_shuffle(self):
        num_per_epoch_sample = 200
        self.start_point_idxs, self.sample_idxs = [], []
        for i in range(self.EEG_list.shape[0]):
            idx = np.random.permutation(self.EEG_list.shape[2] - 540)[:num_per_epoch_sample]
            self.start_point_idxs.append(idx), self.sample_idxs.append(np.zeros(shape=(num_per_epoch_sample,)) + i)
        self.start_point_idxs = np.concatenate(self.start_point_idxs, axis=0)
        self.sample_idxs = np.concatenate(self.sample_idxs, axis=0)
        shuffle = np.random.permutation(self.start_point_idxs.shape[0])
        self.start_point_idxs = torch.Tensor(self.start_point_idxs[shuffle]).to(self.device).long()
        self.sample_idxs = torch.Tensor(self.sample_idxs[shuffle]).to(self.device).long()


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    opts = get_opts()
    opts.batch_size = 8
    opts.EEG_path = "./data/Pure_Data.mat"
    opts.NOS_path = "./data/Contaminated_Data.mat"

    train_list, test_list = DivideDataset(54)       # 54表示总人数

    EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = LoadEEGData(opts.EEG_path, opts.NOS_path, train_list, test_list, fold=0)
    
    EEG_0, NOS_0 = EEG_train_data[1], NOS_train_data[1]
    plt.style.use(['science'])
    y_lim = np.array(list(range(4))) * 150
    x_lim = np.array(list(range(0, 6))) * 1000
    plt.figure(figsize=(5., 2.2))
    for i in range(NOS_0.shape[0]):
        plt.yticks(y_lim, ["FP1", "FP2", "F3", "F4"], fontsize=10)
        plt.xticks(x_lim, [0, 5, 10, 15, 20, 25], fontsize=10)
        plt.plot(np.arange(EEG_0.shape[1]), NOS_0[i, :] + i * 150, linewidth=0.001)     # , color="black"
        plt.xlabel("Time(s)", fontsize=10)
    plt.savefig("./img/few_channel_data.pdf")
            
    for i in range(EEG_0.shape[0]):
        plt.plot(np.arange(EEG_0.shape[1]), EEG_0[i, :] + i * 100)
        plt.plot(np.arange(EEG_0.shape[1]), NOS_0[i, :] + i * 100)
    
    plt.show()
    
    train_data = GetEEGData_train(EEG_train_data, NOS_train_data, opts.batch_size)
    
    val_data = GetEEGData(EEG_val_data, NOS_val_data, opts.batch_size)
    test_data = GetEEGData(EEG_test_data, NOS_test_data, opts.batch_size)

    for batch_id in range(train_data.len()):
        x_t, y_t, batch_id = train_data.get_batch(batch_id)
        train_data.random_shuffle()

    for batch_id in range(val_data.len()):
        x_t, y_t, batch_id = val_data.get_batch(batch_id)
        val_data.shuffle()
