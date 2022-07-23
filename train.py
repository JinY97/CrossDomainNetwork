'''
Author: Yin Jin
Date: 2022-03-08 19:50:50
LastEditTime: 2022-07-23 14:29:25
LastEditors: JinYin
'''
import argparse, torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from opts import get_opts
from audtorch.metrics.functional import pearsonr

import os
from models import *
from loss import *
from torch.utils.tensorboard import SummaryWriter
import torch

from tools import pick_models

opts = get_opts()
if opts.channel_type == "single_channel":
    from preprocess.SimulatedDatasetPreprocess import *
else:
    from preprocess.MultichannelPreprocess import *
        
def train(opts, model, train_log_dir, val_log_dir, data_save_path, fold):
    train_list, test_list = DivideDataset(54)       # 54表示总人数
    EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = LoadEEGData(opts.EEG_path, opts.NOS_path, train_list, test_list, fold=fold)
    train_data = GetEEGData_train(EEG_train_data, NOS_train_data, opts.batch_size)
    val_data = GetEEGData(EEG_val_data, NOS_val_data, opts.batch_size)
    test_data = GetEEGData(EEG_test_data, NOS_test_data, opts.batch_size)

    if opts.denoise_network == 'FCNN':
        learning_rate = 0.0001   
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100], 0.1)
    elif opts.denoise_network == 'SimpleCNN':
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100], 0.1)
    elif opts.denoise_network == 'ResCNN':
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100], 0.1)

    best_val_mse = 10
    if opts.save_result:
        train_summary_writer = SummaryWriter(train_log_dir)
        val_summary_writer = SummaryWriter(val_log_dir)
        f = open(data_save_path + "result.txt", "a+")
    
    for epoch in range(opts.epochs):
        model.train()
        losses = []
        for batch_id in trange(train_data.len()):
            x_t, y_t = train_data.get_batch(batch_id)
            if opts.channel_type == "single_channel":
                x_t = x_t.unsqueeze(dim=1)
                p_t = model(x_t).view(x_t.shape[0], -1)
            else:
                p_t = model(x_t)
                
            loss = denoise_loss_mse(p_t, y_t)
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        train_data.random_shuffle()
        scheduler.step()
        train_loss = torch.stack(losses).mean().item()
        if opts.save_result:
            train_summary_writer.add_scalar("Train loss", train_loss, epoch)

        model.eval()
        losses = []
        for batch_id in range(val_data.len()):
            x_t, y_t = val_data.get_batch(batch_id)
            if opts.channel_type == "single_channel":
                x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            else:
                x_t, y_t = torch.Tensor(x_t).to(opts.device), torch.Tensor(y_t).to(opts.device)
            
            with torch.no_grad():
                if opts.channel_type == "single_channel":
                    p_t = model(x_t).view(x_t.shape[0], -1)
                    loss = (((p_t - y_t) ** 2).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).sqrt()).detach()
                    losses.append(loss)
                else:
                    p_t = model(x_t)
                    loss = (((p_t - y_t) ** 2).mean(dim=-1).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).mean(dim=-1).sqrt()).detach()
                    losses.append(loss)

        val_mse = torch.cat(losses, dim=0).mean().item()
        val_summary_writer.add_scalar("Val loss", val_mse, epoch)
        
        model.eval()
        losses = []
        clean_data, output_data, input_data = [], [], []
        for batch_id in range(test_data.len()):
            x_t, y_t = test_data.get_batch(batch_id)
            if opts.channel_type == "single_channel":
                x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            else:
                x_t, y_t = torch.Tensor(x_t).to(opts.device), torch.Tensor(y_t).to(opts.device)
            with torch.no_grad():
                if opts.channel_type == "single_channel":
                    p_t = model(x_t).view(x_t.shape[0], -1)
                    loss = (((p_t - y_t) ** 2).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).sqrt()).detach()
                    losses.append(loss)
                else:
                    p_t = model(x_t)
                    loss = (((p_t - y_t) ** 2).mean(dim=-1).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).mean(dim=-1).sqrt()).detach()
                    losses.append(loss)
   
            output_data.append(p_t.cpu().numpy()), clean_data.append(y_t.cpu().numpy()), input_data.append(x_t.cpu().numpy())
        test_rrmse = torch.cat(losses, dim=0).mean().item()
        val_summary_writer.add_scalar("test rrmse", test_rrmse, epoch)
        
        # save best results
        if val_mse < best_val_mse:
            best_rrmse = test_rrmse
            best_val_mse = val_mse
            print("Save best result")
            f.write("Save best result \n")
            val_summary_writer.add_scalar("best rrmse", best_rrmse, epoch)
            if opts.save_result:
                np.save(f"{data_save_path}/best_input_data.npy", np.array(input_data))
                np.save(f"{data_save_path}/best_output_data.npy", np.array(output_data))
                np.save(f"{data_save_path}/best_clean_data.npy", np.array(clean_data))
                torch.save(model, f"{data_save_path}/best_{opts.denoise_network}.pth")

        print('epoch: {:3d}, train_loss:{:.4f}, val_mse: {:.4f},  test_rrmse: {:.4f}'.format(epoch, train_loss, val_mse, test_rrmse))
        f.write('epoch: {:3d}, val_mse: {:.4f},  test_rrmse: {:.4f}'.format(epoch, val_mse, test_rrmse) + "\n")

    with open(os.path.join('./json_file/Semisimulated/Semisimulation{}_{}_{}.log'.format(opts.denoise_network, opts.mode, opts.channel_type)), 'a+') as fp:
        fp.write('fold:{}, test_rrmse: {:.4f}'.format(fold, best_rrmse) + "\n")
    
    if opts.save_result:
        np.save(f"{data_save_path}/last_input_data.npy", test_data.EEG_data)
        np.save(f"{data_save_path}/last_output_data.npy", np.array(output_data))
        np.save(f"{data_save_path}/last_clean_data.npy", np.array(clean_data))
        torch.save(model, f"{data_save_path}/last_{opts.denoise_network}.pth")

if __name__ == '__main__':
    opts.epochs = 200
    opts.depth = 6
    opts.noise_type = 'EOG'
    if opts.channel_type == "multi_channel":
        opts.batch_size = 8
    opts.EEG_path = "./data/Pure_Data.mat"
    opts.NOS_path = "./data/Contaminated_Data.mat"
    opts.save_path = "./temp/{}".format(opts.denoise_network)

    print(opts)
    for fold in range(10):
        print(f"fold:{fold}")
        model = pick_models(opts)
        print(opts.denoise_network)
        print(model)
        
        foldername = '{}_{}_{}_{}_{}_{}'.format(opts.denoise_network, opts.noise_type, opts.channel_type, opts.mode, opts.epochs, fold)
        
        train_log_dir = opts.save_path +'/'+foldername +'/'+ '/train'
        val_log_dir = opts.save_path +'/'+foldername +'/'+ '/test'
        data_save_path = opts.save_path +'/'+foldername +'/'
        
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        
        if not os.path.exists(val_log_dir):
            os.makedirs(val_log_dir)
        
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)

        train(opts, model, train_log_dir, val_log_dir, data_save_path, fold)


