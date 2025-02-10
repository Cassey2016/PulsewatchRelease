"""
Test model with PPG + ACC as input.
(DeepBeat/experiments/try_04_run_several_models/test_run_07_PPG_ACC_only.py)
Dong, 09/19/2024.
"""
# Get all the paths for the type of data we will use. 1D or 2D.
# All the parameters:
import sys
import gc
path_your_code = '/content/drive/MyDrive/Public_Datasets/PulsewatchRelease/GitHub/PulsewatchRelease'
path_GT = r'/content/drive/MyDrive/Public_Datasets/PulsewatchRelease/GitHub/Adjudication_UConn/final_attemp_4_1_Dong_Ohm_2024_02_18_copy'
flag_linux = True # Default run on Linux system.
flag_HPC = False # Default run not on HPC.
flag_Colab = True # False means on CentOS server.

# --- Training data config ---
fold_name = 'fold_2' # 'fold_1' or 'fold_2'
data_dim = '1D_PPG_ACC_only' # '1D_PPG', '1D_PPG_HR', '2D_TFS', '2D_Poin', '2D_TFS_HR', '2D_Poin_HR'.
dataset_name = 'Pulsewatch' # 'Pulsewatch', 'Simband', 'MIMICIII'.
normalize_type = 'zero_one_normalize' # 'zero_one_normalize', 'standardize'.
data_format = 'pt' # 'pt' (1D PPG+HR, 2D TFS, 2D Poin), 'csv' (1D ACC)
n_classes = 3 # Three classes: NSR, AF, PAC/PVC.

# --- Model config ---
str_which_model='RNN-GRU' # 'RNN-GRU'.
flag_resume_training=True # 'True' means Resume training from checkpoints.
batch_size=32 # 32, 64, 1024.
loss_type='default' # 'default', 'l1', 'l2', 'l1+l2'.
learning_rate=1e-4 # 1e-4.
num_iterations=200 # Epoch.
patience=10 # Num of epoch before validation does not improve.
# --- Output config ---
# parser.add_argument("--filename_output", type=str, default='train_Luis_20240604_'+fold_name+'.csv') # Default run not on HPC.
filename_output='train_RNN_GRU_20240917_'+fold_name+'.csv' # Default run not on HPC.
output_folder_name='TestRNNGRU_PPG_ACC_only' # 'TestPyTorch'

# Append the directory to your python path using sys
if flag_linux:
    print('Inside Linux')
    if flag_Colab:
        print('Inside Colab')
        # Add path for func 'my_pathdef', 'untar_files'
        path_for_utils = os.path.join(path_your_code,'utils')
        print('path_for_utils:',path_for_utils)
        sys.path.append(path_for_utils)
        # Add path for func 'my my_RNN_GRU_model'
        path_for_models = os.path.join(path_your_code,'model')
        print('path_for_models:',path_for_models)
        sys.path.append(path_for_models)

        # Add remove_train_seg.py
        path_for_rm_seg = os.path.join(path_your_code,'test')
        print('path_for_rm_seg:',path_for_rm_seg)
        sys.path.append(path_for_rm_seg)

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
dict_paths_ckpt = my_pathdef.my_ckpt_path_2024_09_19_PPG_ACC_only(flag_Colab, fold_name)
path_ckpt_model = dict_paths_ckpt['path_ckpt_model']
filename_ckpt_best_model = dict_paths_ckpt['filename_ckpt_best_model']
filename_ckpt_model = dict_paths_ckpt['filename_ckpt_model']

# Untar the data first. Should work on both 1D PPG and 2D TFS or Poin.
import untar_files
untar_files.my_untar_PPG_filt_30sec_csv(tar_path,data_path) # Returns None.
if data_dim == '1D_PPG_HR_ACC' or data_dim == '1D_HR_ACC' or data_dim == '1D_PPG_HR_ACC_rescaleHR'\
    or data_dim == '1D_PPG_ACC_only':
    # Untar the ACC as well.
    print('Debug: ACC_path',ACC_path)
    print('Debug: tar_ACC_path',tar_ACC_path)
    untar_files.my_untar_PPG_filt_30sec_csv(tar_ACC_path,ACC_path) # Returns None.
# Unpack the input output paths.
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the last model here.
# path_temp = os.path.join(path_ckpt_model, filename_ckpt_model)
# Load the best model here.
path_temp = os.path.join(path_ckpt_model, filename_ckpt_best_model)
print(f'Debug: loading ckpt: {path_temp}')
checkpoint = torch.load(path_temp)

n_classes = 3 # NSR, AF, PACPVC
num_classes = n_classes
if data_dim[:3] == '1D_':
    # Input is 1-D, so 30x50.
    input_h = 30 # Sec. Since I am too lazy to find the data, I will use fixed value here.
    input_w = 50 # Hz
    if data_dim == '1D_PPG_HR' or data_dim == '1D_HR_ACC' or data_dim == '1D_PPG_ACC_only':
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

import my_RNN_GRU_model
if data_dim[:2] == '1D':
    model = my_RNN_GRU_model.myRNNGRUmodel(input_size_L, in_channels_d, num_classes).to(device)
elif data_dim[:2] == '2D':
    model = my_RNN_GRU_model.myRNNGRUmodel_2D(input_size_L, in_channels_d, num_classes).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
# metrics = checkpoint['metrics'] # Load the metrics.

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
output_filename = 'test_01_same_subject_test_'+filename_output[:-4]+'.pt'
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
output_filename = 'test_02_the_other_fold_'+filename_output[:-4]+'.pt'
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
output_filename = 'test_03_remain_'+filename_output[:-4]+'.pt'
path_save_metrics = os.path.join(saving_path,output_filename)
torch.save(metrics_remain,path_save_metrics)
print('Saved the pt results to',path_save_metrics)
############## Remaining NSR testing ends ################