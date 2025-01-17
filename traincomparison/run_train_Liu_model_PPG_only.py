"""
Train Liu et al. 2022 JAHA model on Pulsewatch data.
(experiments/try_10_Liu_JAHA_2022/main_02_train_on_Pulsewatch.py)

Dong, 08/20/2024.

I did not add training resume feature to load checkpoint.

Copied from https://github.com/zdzdliu/PPGArrhythmiaDetection
Multiclass Arrhythmia Detection and Classification from Photoplethysmography Signals Using a Deep Convolutional Neural Network

@article{liu2022multiclass,
  title={Multiclass Arrhythmia Detection and Classification From Photoplethysmography Signals Using a Deep Convolutional Neural Network},
  author={Liu, Zengding and Zhou, Bin and Jiang, Zhiming and Chen, Xi and Li, Ye and Tang, Min and Miao, Fen},
  journal={Journal of the American Heart Association},
  volume={11},
  number={7},
  pages={e023555},
  year={2022},
  publisher={Am Heart Assoc}
}
"""

# Multiclass Arrhythmia Classification from Photoplethysmography Signals based on VGGNet
path_your_code = '/content/drive/MyDrive/Public_Datasets/PulsewatchRelease/GitHub/PulsewatchRelease'
import argparse
import os
import operator as op #
import random
import math
from scipy import signal as sig
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--multiLabel', type=bool, default=False, help='enable multiple label')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--recordLength', type=int, default=2500, help='the length of input record')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. ')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

# This path was not used in this code. 
opt = parser.parse_args(['--dataroot','/content/drive/MyDrive/Public_Datasets/Liu_2022_JAHA','--workers','2','--manualSeed','42','--cuda'])

# ------------------------
# --- Dong's argparser ---
flag_linux=True # Default run on Linux system.
flag_HPC=False # Default run not on HPC.
flag_Colab=True # False means on CentOS server.

# --- Training data config ---
fold_name = 'fold_2' # 'fold_1' or 'fold_2'
data_dim='1D_PPG' # '1D_PPG', '1D_PPG_HR', '2D_TFS', '2D_Poin', '2D_TFS_HR', '2D_Poin_HR'.
dataset_name='Pulsewatch' # 'Pulsewatch', 'Simband', 'MIMICIII'.
normalize_type='zero_one_normalize' # 'zero_one_normalize', 'standardize'.
data_format='pt' # 'pt' (1D PPG+HR, 2D TFS, 2D Poin), 'csv' (1D ACC)
n_classes=3 # Three classes: NSR, AF, PAC/PVC.
batch_size = opt.batchSize
# --- Model config ---
str_which_model='RNN-GRU' # 'Luis', 'RNN-GRU'.

# --- Output config ---
filename_output='train_RNN_GRU_20240713_'+fold_name+'.csv' # Default run not on HPC.
output_folder_name='TestRNNGRU_batch32' # 'TestPyTorch'
# -------------------------


print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    os.makedirs(opt.outf)
except OSError:
    pass
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# Get all the paths for the type of data we will use. 1D or 2D.
import sys
# Append the directory to your python path using sys
if flag_linux:
    print('Inside Linux')
    if flag_Colab:
        print('Inside Colab')
        # Add path for func 'my_pathdef', 'untar_files'
        path_for_utils = os.path.join(path_your_code,'utils')
        print('path_for_utils:',path_for_utils)
        sys.path.append(path_for_utils)
        # Add path for func 'my PPGVGGNet_4channels'
        path_for_models = os.path.join(path_your_code,'traincomparison')
        print('path_for_models:',path_for_models)
        sys.path.append(path_for_models)
# Unpack the input output paths.

import my_pathdef # Inside utils
dict_paths = my_pathdef.get_data_paths(
                            data_dim = data_dim,
                            dataset_name = dataset_name,
                            output_folder_name = output_folder_name)
data_path = dict_paths['data_path']
tar_path = dict_paths['tar_path']
labels_path = dict_paths['labels_path']
saving_path = dict_paths['saving_path']
path_train_val_test_split = dict_paths['path_train_val_test_split']
ACC_path = dict_paths['ACC_path']
tar_ACC_path = dict_paths['tar_ACC_path']
aug_path = dict_paths['aug_path']+fold_name # Add the fold name to the path. Only needed for aug path.
tar_aug_path = dict_paths['tar_aug_path']+fold_name # Add the fold name to the path. Only needed for aug path.
dict_paths['aug_path'] = aug_path
dict_paths['tar_aug_path'] = tar_aug_path
print('dict_paths[''aug_path'']',dict_paths['aug_path'])
print('dict_paths[''tar_aug_path'']',dict_paths['tar_aug_path'])

# Untar the data first.
import untar_files
untar_files.my_untar_PPG_filt_30sec_csv(tar_path,data_path) # Returns None.
print('Debug: data_dim',data_dim)
if data_dim == '1D_PPG_HR_ACC' \
    or data_dim == '1D_HR_ACC' \
    or data_dim == '1D_PPG_HR_ACC_rescaleHR' \
    or data_dim == '1D_PPG_ElgendiHR_ACC' \
    or data_dim == '1D_PPG_twoHRs_rescaleHRs_ACC' \
    or data_dim == '1D_PPG_WEPD_rescale_HR_ACC' \
    or data_dim == '1D_PPG_Elgendi_rescale_HR_ACC':
    # Untar the ACC as well.
    print('Debug: ACC_path',ACC_path)
    print('Debug: tar_ACC_path',tar_ACC_path)
    untar_files.my_untar_PPG_filt_30sec_csv(tar_ACC_path,ACC_path) # Returns None.
if data_dim == '1D_PPG_aug_HR_ACC_rescaleHR'or data_dim == '1D_PPG_aug5k_HR_ACC_rescaleHR':
    # Untar the ACC as well.
    print('Debug: ACC_path',ACC_path)
    print('Debug: tar_ACC_path',tar_ACC_path)
    print('Debug: aug_path',aug_path)
    print('Debug: tar_aug_path',tar_aug_path)
    untar_files.my_untar_PPG_filt_30sec_csv(tar_ACC_path,ACC_path) # Returns None.
    untar_files.my_untar_PPG_filt_30sec_csv(tar_aug_path,aug_path) # Returns None.



# The pre-splitted train, validation, and test segment names with labels as dict.
import my_dataloader
df_train, df_valid, df_test = my_dataloader.split_data_Cassey_ROS_ImageNet(path_train_val_test_split,fold_name)

# Preprocess data
# Initialize the train_loader and valid_loader, so finished_seg_names was set to empty.
train_loader, valid_loader, train_dataset, val_dataset = my_dataloader.load_train_valid_dataloader(
                            fold_name, df_train, df_valid,
                            dict_paths, normalize_type, data_format, data_dim,
                            batch_size, finished_seg_names = [])

menu_segment_names = train_loader.dataset.segment_names # All the segments to be run in the training dataset.
menu_segment_labels = train_loader.dataset.segment_labels # All the ground truth labels
print('Debug: len(menu_segment_names)',len(menu_segment_names))
print('Debug: len(menu_segment_labels)',len(menu_segment_labels))

# Initialize result storage
results = {
    'train_metrics':{'epoch':[],'train_loss': [],'train_accuracy':[]},
    'validation_metrics':{'epoch':[],'valid_loss': [],'valid_accuracy': [],
                          'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []}
}

import PPGVGGNet_4channels

ngpu = int(opt.ngpu)
num_types = 1
# Get input dimension.
if data_dim[:3] == '1D_':
    # Input is 1-D, so 30x50.
    print('Debug: dir(train_dataset)',dir(train_dataset))
    input_h = train_dataset.reset_sec # Without passing variable is fetching the private variable.
    input_w = train_dataset.reset_fs
    if data_dim == '1D_PPG_HR' or data_dim == '1D_HR_ACC':
        input_d = 2 # Input PPG and HR at the same time. Or input HR and ACC without PPG.
    elif data_dim == '1D_PPG_HR_ACC' or data_dim == '1D_PPG_ElgendiHR_ACC':
        input_d = 3 # Input PPG, HR, and ACC at the same time.
        print('Debug: input_d',input_d)
    elif data_dim == '1D_PPG_HR_ACC_rescaleHR' or data_dim == '1D_PPG_Elgendi_rescale_HR_ACC' or data_dim == '1D_PPG_aug_HR_ACC_rescaleHR'\
        or data_dim == '1D_PPG_aug5k_HR_ACC_rescaleHR':
        input_d = 4 # Input PPG, HR, rescaled HR, and ACC at the same time.
        print('Debug: input_d',input_d)
    elif data_dim == '1D_PPG_twoHRs_rescaleHRs_ACC':
        input_d = 6 # Input PPG, WEPD HR, rescaled WEPD HR, Elgendi HR, rescaled Elgendi HR, and ACC at the same time.
    else:
        input_d = 1
        print('Debug: input_d',input_d)
    input_size_L = input_h * input_w
    in_channels_d = input_d
else:
    # Input is 2D, so 128x128.
    input_h = train_dataset.reset_img_size # 128.
    input_w = input_h
    if data_dim[-3:] == '_HR':
        input_d = 2
    elif data_dim[-3:] == 'ACC':
        input_d = 3 # Input PPG, HR, and ACC at the same time.
    else:
        input_d = 1
    input_size_L = input_h
    in_channels_d = input_w
print('in_channels_d',in_channels_d)
model = PPGVGGNet_4channels.vgg16_bn(in_channels = in_channels_d,ngpu=ngpu,num_classes=n_classes)

# Print the model size and parameters.
print(model)

ngpu = int(opt.ngpu)
in_x = Variable(torch.randn(20,in_channels_d,1500))
in_y = Variable(torch.randn(20,1))
out_x, out_prob = model(in_x)

print(in_x.shape)
print(out_x.shape)

loss_fun = nn.CrossEntropyLoss()
if opt.cuda:
    model.to(device)
    loss_fun.to(device)
optimizer = optim.Adam(list(model.parameters()), lr = opt.lr, betas=(opt.beta1, 0.999),weight_decay=0.004)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

EpochAcc = np.zeros((2,)) # Dong, 08/20/2024: Initializing EpochAcc.
EpochValidAcc = np.zeros((2,)) # Dong, 08/20/2024: Initializing EpochAcc.
ValidAcc = 0 # Dong, 08/20/2024: Initializing validation accuracy.
for epoch in range(opt.niter):
    # Model training
    losses = []
    predictions = []
    labelsall = []
    model.train()
    for i, train_data in enumerate(train_loader, 0):
        model.zero_grad()
        # inputs, labels = train_data['signal'], train_data['labels']
        inputs, labels = train_data['data'], train_data['label']
        # print('inputs.shape',inputs.shape)
        # print('labels.shape',labels.shape)
        if train_data['data'].dim() == 3:
            # Only 1D.
            inputs = torch.tensor(train_data['data']).to(device)
        else:
            inputs = torch.squeeze(train_data['data'], dim=1).to(device)  # 1D PPG: (none, 1, d, L) -> (none, d, L)
        if opt.cuda:
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

        output, y_pred_prob = model(inputs)
        # labels = labels[:,0].long()
        labels = labels.long()
        loss = loss_fun(output, labels)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu()
        labels = labels.cpu()
        predictions.extend(predicted.numpy())
        labelsall.extend(labels.numpy())

    scheduler.step()
    train_acc = accuracy_score(labelsall, predictions)
    train_loss = np.average(losses)
    print('[%d/%d] Loss: %.4f Training Accuracy: %.4f' %(epoch, opt.niter, train_loss,
                                                         train_acc))
    results['train_metrics']['train_loss'] = train_loss
    results['train_metrics']['train_accuracy'] = train_acc
    results['train_metrics']['epoch'] = epoch
    EpochAcc = np.vstack((EpochAcc,np.array((epoch, np.average(losses)))))
    # Model validation
    model.eval()
    if (epoch+1) % 5 == 0:
        # validate
        predictions = []
        labelsall = []
        losses = []
        for i, valid_data in enumerate(valid_loader, 0):
            # inputs, labels = valid_data['signal'], valid_data['labels']
            inputs, labels = valid_data['data'], valid_data['label']
            if opt.cuda:
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
            if train_data['data'].dim() == 3:
                # Only 1D.
                inputs = torch.tensor(valid_data['data']).to(device)
            else:
                inputs = torch.squeeze(valid_data['data'], dim=1).to(device)  # 1D PPG: (none, 1, d, L) -> (none, d, L)
            output, y_pred_prob = model(inputs)
            # labels = labels[:,0].long()
            labels = labels.long()
            loss = loss_fun(output, labels)
            losses.append(loss.item())
            _,predicted = torch.max(output.data, 1)
            predicted = predicted.cpu()
            labels = labels.cpu()

            predictions.extend(predicted.numpy())
            labelsall.extend(labels.numpy())
        valid_loss = np.average(losses)
        valid_accuracy = accuracy_score(labelsall, predictions)
        print('[%d/%d] Validation Loss: %.4f Validation Accuracy: %.4f' %(epoch, opt.niter, valid_loss,
                                                         valid_accuracy))
        EpochValidAcc = np.vstack((EpochValidAcc,np.array((epoch, np.average(losses)))))
        results['validation_metrics']['valid_loss'] = valid_loss
        results['validation_metrics']['valid_accuracy'] = valid_accuracy
        results['validation_metrics']['epoch'] = epoch
        if accuracy_score(labelsall, predictions) > ValidAcc:
            ValidAcc = accuracy_score(labelsall, predictions)
            labelsAll = labelsall
            predictionsAll = predictions
            # torch.save(model, 'modelPPG.pkl')
            # Since I save the checkpoint at the end of each epoch, I do not need to save the model again.
            str_epoch = '{epoch:{fill}{width}}'.format(epoch=epoch, fill='0', width=4)
            str_val_acc = '{val_acc:.{prec}f}'.format(val_acc=ValidAcc, prec=4)
            best_model_ckpt_name = 'Liu_JAHA_2022_PPG_only_'+fold_name+'_epoch_'+str_epoch+'_ValAcc_'+str_val_acc+'.pt'
            checkpoint_path = saving_path
            save_ckpt_best_model_path = os.path.join(checkpoint_path,best_model_ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'likelihood_state_dict': likelihood.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ValidAcc': ValidAcc,
                # Include other metrics as needed
                'metrics':results,
                'PARAMS_config':opt
            }, save_ckpt_best_model_path)
