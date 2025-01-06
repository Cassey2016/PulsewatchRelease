# # Step 0-0: Map personal Google drive to here.
# from google.colab import drive
# drive.mount('/content/drive')

"""
Test Liu et al. 2022 JAHA model on Pulsewatch data.
(experiments/try_10_Liu_JAHA_2022/main_03_test_on_Pulsewatch_PPG_only.py)

Dong, 08/21/2024.
"""
# Get all the paths for the type of data we will use. 1D or 2D.
# All the parameters:
import sys
import gc
import argparse
import os
import torch
print('torch.cuda.is_available()',torch.cuda.is_available())
print('torch.cuda.device_count()',torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# opt = parser.parse_args(['--dataroot','/mnt/r/ENGR_Chon/Dong/Public_Database/PPGArrhythmiaDetection-main','--cuda'])
opt = parser.parse_args(['--dataroot','/content/drive/MyDrive/Public_Datasets/Liu_2022_JAHA','--workers','2','--manualSeed','42','--cuda'])

flag_linux = True # Default run on Linux system.
flag_HPC = False # Default run not on HPC.
flag_Colab = True # False means on CentOS server.

# --- Training data config ---
fold_name = 'fold_1' # 'fold_1' or 'fold_2'
data_dim = '1D_PPG' # '1D_PPG', '1D_PPG_HR', '2D_TFS', '2D_Poin', '2D_TFS_HR', '2D_Poin_HR'.
dataset_name = 'Pulsewatch' # 'Pulsewatch', 'Simband', 'MIMICIII'.
normalize_type = 'zero_one_normalize' # 'zero_one_normalize', 'standardize'.
data_format = 'pt' # 'pt' (1D PPG+HR, 2D TFS, 2D Poin), 'csv' (1D ACC)
n_classes = 3 # Three classes: NSR, AF, PAC/PVC.

# --- Model config ---
str_which_model='RNN-GRU' # 'Luis', 'RNN-GRU'.
flag_resume_training=True # 'True' means Resume training from checkpoints.
batch_size=32 # 32, 64, 1024.
loss_type='default' # 'default', 'l1', 'l2', 'l1+l2'.
learning_rate=1e-4 # 1e-4.
num_iterations=200 # Epoch.
patience=10 # Num of epoch before validation does not improve.
# --- Output config ---
# parser.add_argument("--filename_output", type=str, default='train_Luis_20240604_'+fold_name+'.csv') # Default run not on HPC.
filename_output='train_RNN_GRU_20240713_'+fold_name+'.csv' # Default run not on HPC.
output_folder_name='TestRNNGRU_batch32' # 'TestPyTorch'

# Append the directory to your python path using sys
if flag_linux:
    print('Inside Linux')
    if flag_Colab:
        print('Inside Colab')
        # For 'my_pathdef'
        sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Github_private_another_prompt/Pulsewatch_labeling/DeepBeat/utils')
        # Add Luis' active learning code 'ss_active_learning'
        sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Github_private_another_prompt/Pulsewatch_labeling/BML_project/active_learning')
        # Add Luis' Gaussian Process model 'ss_gp_model'
        sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Github_private_another_prompt/Pulsewatch_labeling/BML_project/models')
        # Add Luis' 'dataloader' file, the 'update_train_loader_with_uncertain_samples' inside.
        sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Github_private_another_prompt/Pulsewatch_labeling/BML_project/utils_gp')
        # Add my my_RNN_GRU_model
        sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Github_private_another_prompt/Pulsewatch_labeling/DeepBeat/experiments/try_02_RNN_GRU')
        # Add remove_train_seg.py
        sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Github_private_another_prompt/Pulsewatch_labeling/ResNet_classification/utils')
        # Add test_model.py
        sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Github_private_another_prompt/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models')
        path_GT = r'/content/drive/MyDrive/Adjudication_UConn/final_attemp_4_1_Dong_Ohm_2024_02_18_copy'
    else:
        print('Not inside Colab')
        # For 'my_pathdef'
        sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/utils')
        # Add Luis' active learning code 'ss_active_learning'
        sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/BML_project/active_learning')
        # Add Luis' Gaussian Process model 'ss_gp_model'
        sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/BML_project/models')
        # Add Luis' 'dataloader' file, the 'update_train_loader_with_uncertain_samples' inside, visualization.py, 'plot_comparative_results'
        sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/BML_project/utils_gp')
        # Add my my_RNN_GRU_model
        sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_02_RNN_GRU')
        # Add remove_train_seg.py
        sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/ResNet_classification/utils')
        import pretty_errors


import my_pathdef # Inside utils

dict_paths = my_pathdef.get_data_paths(
                            data_dim = data_dim,
                            dataset_name = dataset_name,
                            output_folder_name = output_folder_name)

print('Debug main: dict_paths',dict_paths)
data_path = dict_paths['data_path']
tar_path = dict_paths['tar_path']
labels_path = dict_paths['labels_path']
saving_path = dict_paths['saving_path']
path_train_val_test_split = dict_paths['path_train_val_test_split']
ACC_path = dict_paths['ACC_path']
tar_ACC_path = dict_paths['tar_ACC_path']

# Know the ckpt, fold name
dict_paths_ckpt = my_pathdef.my_ckpt_path_2024_08_21_PPG_only(flag_Colab, fold_name)
path_ckpt_model = dict_paths_ckpt['path_ckpt_model']
filename_ckpt_best_model = dict_paths_ckpt['filename_ckpt_best_model']
filename_ckpt_model = dict_paths_ckpt['filename_ckpt_model']

# Untar the data first. Should work on both 1D PPG and 2D TFS or Poin.
import untar_files
untar_files.my_untar_PPG_filt_30sec_csv(tar_path,data_path) # Returns None.
if data_dim == '1D_PPG_HR_ACC' or data_dim == '1D_HR_ACC' or data_dim == '1D_PPG_HR_ACC_rescaleHR':
    # Untar the ACC as well.
    print('Debug: ACC_path',ACC_path)
    print('Debug: tar_ACC_path',tar_ACC_path)
    untar_files.my_untar_PPG_filt_30sec_csv(tar_ACC_path,ACC_path) # Returns None.
# Unpack the input output paths.

n_classes = 3 # NSR, AF, PACPVC
num_classes = n_classes
if data_dim[:3] == '1D_':
    # Input is 1-D, so 30x50.
    input_h = 30 # Sec. Since I am too lazy to find the data, I will use fixed value here.
    input_w = 50 # Hz
    if data_dim == '1D_PPG_HR' or data_dim == '1D_HR_ACC':
        input_d = 2 # Input PPG and HR at the same time.
    elif data_dim == '1D_PPG_HR_ACC':
        input_d = 3 # Input PPG, HR, and ACC at the same time.
        print('Debug: input_d',input_d)
    elif data_dim == '1D_PPG_HR_ACC_rescaleHR':
        input_d = 4 # Input PPG, HR, ACC, and rescaled HR at the same time.
    else:
        input_d = 1
    input_size_L = input_h * input_w
    in_channels_d = input_d
else:
    input_h = 128 # Since I am too lazy to find the data, I will use fixed value here.
    input_w = input_h
    if data_dim[-3:] == '_HR':
        input_d = 2
    elif data_dim[-3:] == 'ACC':
        input_d = 3 # Input PPG, HR, and ACC at the same time.
    else:
        input_d = 1
    input_size_L = input_h
    in_channels_d = input_w

import PPGVGGNet_4channels
ngpu = int(opt.ngpu)
if data_dim[:2] == '1D':
    # model = my_RNN_GRU_model.myRNNGRUmodel(input_size_L, in_channels_d, num_classes).to(device)
    model = PPGVGGNet_4channels.vgg16_bn(in_channels = in_channels_d,ngpu=ngpu,num_classes=n_classes).to(device)
# elif data_dim[:2] == '2D':
#     model = my_RNN_GRU_model.myRNNGRUmodel_2D(input_size_L, in_channels_d, num_classes).to(device)

# Load the last model here.
# path_temp = os.path.join(path_ckpt_model, filename_ckpt_model)
# Load the best model here.
print('device',device)
print('torch.cuda.device_count()',torch.cuda.device_count())
path_temp = os.path.join(path_ckpt_model, filename_ckpt_best_model)
print(f'Debug: loading ckpt: {path_temp}')
checkpoint = torch.load(path_temp,map_location='cuda:0')

model.load_state_dict(checkpoint['model_state_dict'])
# metrics = checkpoint['metrics'] # Load the metrics.
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(output_folder_name,fold_name,'pytorch_total_params',pytorch_total_params)

# Load the unused data
import my_dataloader
if fold_name == 'fold_2':
    test_fold_name = 'fold_1'
elif fold_name == 'fold_1':
    test_fold_name = 'fold_2'

_, _, df_test = my_dataloader.split_data_Cassey_ROS_ImageNet(path_train_val_test_split,test_fold_name)

test_loader, test_dataset = my_dataloader.load_test_dataloader(fold_name, df_test,
                              dict_paths, normalize_type, data_format, data_dim,
                              batch_size,num_workers=1)
# Evaluate the model
metrics_test = {'segment_names':[],
                'test_accuracy':[],
                'precision':[],
                'recall':[],
                'f1_score':[],
                # 'auc_roc':[],
                'y_true':[],
                'y_pred':[],
                'y_pred_prob':[],
                'y_pred_logit':[]
                }

import test_model
metrics_test = test_model.my_test_model(model, test_loader, n_classes, metrics_test)

# Save the metrics
output_filename = 'test_01_Liu_JAHA_2022_same_subject_test_'+filename_output[:-4]+'.pt'
path_save_metrics = os.path.join(saving_path,output_filename)
torch.save(metrics_test,path_save_metrics)
print('Saved the pt results to',path_save_metrics)
# Clean cache.
torch.cuda.empty_cache()
gc.collect()
# Load the remaining NSR data and save those results as well.

############## The other fold testing #############
import remove_train_seg
df_the_other_fold = remove_train_seg.my_test_the_other_fold(path_GT,fold_name)
df_the_other_fold = df_the_other_fold.rename(columns={'final_AF_GT_20230921': 'values', 'table_file_name': 'keys'})

print('Debug: df_the_other_fold',df_the_other_fold)

the_other_fold_loader, the_other_fold_dataset = my_dataloader.load_test_dataloader(fold_name, df_the_other_fold,
                              dict_paths, normalize_type, data_format, data_dim,
                              batch_size,num_workers=1)

# Evaluate the model
metrics_the_other_fold = {'segment_names':[],
                'test_accuracy':[],
                'precision':[],
                'recall':[],
                'f1_score':[],
                # 'auc_roc':[],
                'y_true':[],
                'y_pred':[],
                'y_pred_prob':[],
                'y_pred_logit':[]
                }

import test_model
metrics_the_other_fold = test_model.my_test_model(model, the_other_fold_loader, n_classes, metrics_the_other_fold)

# Save the metrics
output_filename = 'test_02_Liu_JAHA_2022_the_other_fold_'+filename_output[:-4]+'.pt'
path_save_metrics = os.path.join(saving_path,output_filename)
torch.save(metrics_the_other_fold,path_save_metrics)
print('Saved the pt results to',path_save_metrics)

############## The other fold testing end #############


############## Remaining NSR testing ################

dict_remain_labels = remove_train_seg.my_remove_both_folds(path_GT,fold_name)

import pandas as pd
df_remain_labels = pd.DataFrame({'keys':dict_remain_labels.keys(),'values':dict_remain_labels.values()})

remain_loader, remain_dataset = my_dataloader.load_test_dataloader(fold_name, df_remain_labels,
                              dict_paths, normalize_type, data_format, data_dim,
                              batch_size,num_workers=1)

# Evaluate the model
metrics_remain = {'segment_names':[],
                'test_accuracy':[],
                'precision':[],
                'recall':[],
                'f1_score':[],
                # 'auc_roc':[],
                'y_true':[],
                'y_pred':[],
                'y_pred_prob':[],
                'y_pred_logit':[]
                }

import test_model
metrics_remain = test_model.my_test_model(model, remain_loader, n_classes, metrics_remain)

# Save the metrics
output_filename = 'test_03_Liu_JAHA_2022_remain_'+filename_output[:-4]+'.pt'
path_save_metrics = os.path.join(saving_path,output_filename)
torch.save(metrics_remain,path_save_metrics)
print('Saved the pt results to',path_save_metrics)
############## Remaining NSR testing ends ################
# # Debug batch 219:
# import remove_train_seg
# dict_debug = remove_train_seg.my_debug_test(labels_path)
# print('Debug main: dict_debug',dict_debug)
# import pandas as pd
# df_debug = pd.DataFrame({'keys':dict_debug.keys(),'values':dict_debug.values()})
# print('Debug main: df_debug',df_debug)

# remain_loader, remain_dataset = my_dataloader.load_test_dataloader(fold_name, df_debug,
#                               dict_paths, normalize_type, data_format, data_dim,
#                               batch_size=1,num_workers=1)

# # Evaluate the model
# metrics_remain = {'segment_names':[],
#                 'test_accuracy':[],
#                 'precision':[],
#                 'recall':[],
#                 'f1_score':[],
#                 'auc_roc':[],
#                 'y_true':[],
#                 'y_pred':[],
#                 'y_pred_prob':[],
#                 'y_pred_logit':[]
#                 }

# import test_model
# metrics_remain = test_model.my_test_model(model, remain_loader, n_classes, metrics_remain)

# # Save the metrics
# output_filename = 'test_03_debug_'+filename_output[:-4]+'.pt'
# path_save_metrics = os.path.join(saving_path,output_filename)
# torch.save(metrics_remain,path_save_metrics)