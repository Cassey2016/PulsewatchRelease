"""
Input PPG, HR, and ACC to the 1D-bi-GRU model.
(DeepBeat/experiments/try_04_run_several_models/run_04_ACC.py)

Dong, 06/25/2024.
"""
import my_main
import argparse

parser = argparse.ArgumentParser()
# --- System arguments ---
parser.add_argument("--flag_linux", type=bool, default=True) # Default run on Linux system.
parser.add_argument("--flag_HPC", type=bool, default=False) # Default run not on HPC.
parser.add_argument("--flag_Colab", type=bool, default=True) # False means on CentOS server.

# --- Training data config ---
fold_name = 'fold_2' # 'fold_1' or 'fold_2'
parser.add_argument("--fold_name", type=str, default=fold_name) # 'fold_1', 'fold_2'.
parser.add_argument("--data_dim",type=str, default='1D_PPG_HR_ACC') # '1D_PPG', '1D_PPG_HR', '2D_TFS', '2D_Poin', '2D_TFS_HR', '2D_Poin_HR'.
parser.add_argument("--dataset_name",type=str, default='Pulsewatch') # 'Pulsewatch', 'Simband', 'MIMICIII'.
parser.add_argument("--normalize_type",type=str,default='zero_one_normalize') # 'zero_one_normalize', 'standardize'.
parser.add_argument("--data_format",type=str,default='pt') # 'pt' (1D PPG+HR, 2D TFS, 2D Poin), 'csv' (1D ACC)
parser.add_argument("--n_classes",type=int,default=3) # Three classes: NSR, AF, PAC/PVC.

# --- Model config ---
parser.add_argument("--str_which_model",type=str, default='RNN-GRU') # 'RNN-GRU'.
parser.add_argument("--flag_resume_training",type=bool, default=True) # 'True' means Resume training from checkpoints.
parser.add_argument("--batch_size",type=int,default=32) # 32, 64, 1024.
parser.add_argument("--loss_type",type=str,default='default') # 'default', 'l1', 'l2', 'l1+l2'.
parser.add_argument("--learning_rate",type=float,default=1e-4) # 1e-4.
parser.add_argument("--num_iterations",type=int,default=200) # Epoch.
parser.add_argument("--patience",type=int,default=10) # Num of epoch before validation does not improve.
# --- Output config ---
parser.add_argument("--filename_output", type=str, default='train_RNN_GRU_20240708_'+fold_name+'.csv') # Default run not on HPC.
parser.add_argument("--output_folder_name",type=str, default='TestRNNGRU_HR_ACC') # 'TestPyTorch'

# Hyperparameters for the model
my_main.my_main(parser.parse_args())

# Copied from https://neptune.ai/blog/how-to-manage-track-visualize-hyperparameters
# Hyperparameters for the model



