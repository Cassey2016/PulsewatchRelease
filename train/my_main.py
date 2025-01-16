"""
Modularized main code. The main code is never changed for different custom settings for better version control. 
Remember to change these paths:
- path_your_code:
Output:

Dong, 01/04/2025.
"""
import os
import torch
file_path = os.path.realpath(__file__)
path_your_code = '/content/drive/MyDrive/Public_Datasets/PulsewatchRelease/GitHub/PulsewatchRelease'
from datetime import datetime

now = datetime.now() # Get the time now for model checkpoint saving.

dt_string = now.strftime("%Y_%m_%d_%H_%M_%S") # YYYY_mm_dd_HH_MM_SS, for model saving.
print("The date and time suffix of the model file is", dt_string)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
my_random_seed_num = 42
torch.manual_seed(my_random_seed_num)
import random
random.seed(my_random_seed_num)
import numpy as np
np.random.seed(my_random_seed_num)

def my_main(config):
    # ======== Customize parameter start here =======
    # System config:
    flag_linux = config.flag_linux # Default is True to run my code on a Linux server for debugging.
    flag_HPC = config.flag_HPC # Default is False to not run on UConn HPC.
    flag_Colab = config.flag_Colab # Default is True for running on Google Colab. Both this and flag_linux have to be True.

    # Training data config:
    fold_name = config.fold_name # 'fold_1' or 'fold_2' in Pulsewatch dataset.
    data_dim = config.data_dim # '1D_PPG' for PPG only, '1D_HR_ACC' for HR+ACC, '1D_PPG_HR_ACC_rescaleHR' for four channels.
    dataset_name = config.dataset_name # 'Pulsewatch'
    normalize_type = config.normalize_type # Default 'zero_one_normalize'
    data_format = config.data_format # Default 'pt' for input data format.
    n_classes = config.n_classes # 3, three classes.

    # Model config:
    str_which_model = config.str_which_model # 'RNN-GRU'
    flag_resume_training = config.flag_resume_training # 'True' for resume training from check points.
    batch_size = config.batch_size # Default 32.
    loss_type = config.loss_type # 'default', 'l1', 'l2', 'l1+l2'
    learning_rate = config.learning_rate # Default 1e-4
    num_iterations = config.num_iterations # Default 200
    patience = config.patience # Default 10

    # Output config:
    filename_output = config.filename_output # Output file name, prefix (e.g., 'train_RNN_GRU_20240612_')+fold_name+'.csv'
    output_folder_name = config.output_folder_name # Folder created for this run, e.g. 'TestPyTorch'
    
    # Print the input customize config so you can see it running the code.
    print('---System config---')
    print('flag_linux',flag_linux,'type',type(flag_linux))
    print('flag_HPC',flag_HPC,'type',type(flag_HPC))
    print('flag_Colab',flag_Colab,'type',type(flag_Colab))
    print('---Training data config---')
    print('fold_name',fold_name,'type',type(fold_name))
    print('data_dim',data_dim,'type',type(data_dim))
    print('dataset_name',dataset_name,'type',type(dataset_name))
    print('normalize_type',normalize_type,'type',type(normalize_type))
    print('data_format',data_format,'type',type(data_format))
    print('n_classes',n_classes,'type',type(n_classes))
    print('---Model config---')
    print('str_which_model',str_which_model,'type',type(str_which_model))
    print('flag_resume_training',flag_resume_training,'type',type(flag_resume_training))
    print('batch_size',batch_size,'type',type(batch_size))
    print('loss_type',loss_type,'type',type(loss_type))
    print('learning_rate',learning_rate,'type',type(learning_rate))
    print('num_iterations',num_iterations,'type',type(num_iterations))
    print('patience',patience,'type',type(patience))
    print('---Output config---')
    print('filename_output',filename_output,'type',type(filename_output))
    print('output_folder_name',output_folder_name,'type',type(output_folder_name))

    # Store the config into a dictionary for future check.
    PARAMS_config = {'flag_linux':flag_linux,
                     'flag_HPC':flag_HPC,
                     'flag_Colab':flag_Colab,
                     'fold_name':fold_name,
                     'data_dim':data_dim,
                     'dataset_name':dataset_name,
                     'normalize_type':normalize_type,
                     'data_format':data_format,
                     'n_classes':n_classes,
                     'str_which_model':str_which_model,
                     'flag_resume_training':flag_resume_training,
                     'batch_size':batch_size,
                     'loss_type':loss_type,
                     'learning_rate':learning_rate,
                     'num_iterations':num_iterations,
                     'patience':patience,
                     'filename_output':filename_output,
                     'output_folder_name':output_folder_name} # Save all the model config into one dict.

    # ======== Customize parameter end here =======

    # Path for subfunctions.
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
            # Add path for func 'my my_RNN_GRU_model'
            path_for_models = os.path.join(path_your_code,'model')
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
    print('dict_paths[''data_path'']',dict_paths['data_path'])
    print('dict_paths[''tar_path'']',dict_paths['tar_path'])

    # Untar the data first. Should work on both 1D PPG and 2D TFS or Poin.
    import untar_files
    untar_files.my_untar_PPG_filt_30sec_csv(tar_path,data_path) # Returns None.
    print('Debug: data_dim',data_dim)
    if data_dim == '1D_PPG_HR_ACC' \
        or data_dim == '1D_HR_ACC' \
        or data_dim == '1D_PPG_HR_ACC_rescaleHR' \
        or data_dim == '1D_PPG_ElgendiHR_ACC' \
        or data_dim == '1D_PPG_twoHRs_rescaleHRs_ACC' \
        or data_dim == '1D_PPG_WEPD_rescale_HR_ACC' \
        or data_dim == '1D_PPG_Elgendi_rescale_HR_ACC'\
        or data_dim == '1D_PPG_ACC_only'\
        or data_dim == '1D_HR_rescaleHR_ACC_noPPG':
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
    train_loader, val_loader, train_dataset, val_dataset = my_dataloader.load_train_valid_dataloader(
                                fold_name, df_train, df_valid,
                                dict_paths, normalize_type, data_format, data_dim,
                                batch_size, finished_seg_names = [])

    menu_segment_names = train_loader.dataset.segment_names # All the segments to be run in the training dataset.
    menu_segment_labels = train_loader.dataset.segment_labels # All the ground truth labels 
    print('Debug: len(menu_segment_names)',len(menu_segment_names))
    print('Debug: len(menu_segment_labels)',len(menu_segment_labels))

    # Initialize result storage
    results = {
        'train_loss_Cassey': [],'train_loss_Darren': [],'valid_loss': [],
        'train_accuracy': [], 'valid_accuracy': [], 'train_epoch': [],
        'validation_metrics': {'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []},
        'active_learning_metrics': {'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []},
        'al_loss_Cassey': [],'al_loss_Darren': [], 'al_valid_loss':[],
        'al_accuracy': [], 'al_valid_accuracy': [], 'al_iteration': [],
    }
    
    # Initial model training
    if str_which_model == 'RNN-GRU':
        import my_RNN_GRU_model
        datackpt_name = 'ckpt_dataset'+filename_output.split('.')[0]+'.pt'
        modelckpt_name = 'ckpt_train_model'+filename_output.split('.')[0]+'.pt'
        best_model_ckpt_name = filename_output.split('.')[0]
        # 01/15/2025: If you want to resume training using our saved models,
        # replace the 'modelckpt_name' with the corresponding model name in 
        # /utils/my_pathdef.py.
        # modelckpt_name = filename_output.split('.')[0]+'_epoch_'+'0043'+'_val_loss_'+'0.1470'+'.pt'
        # best_model_ckpt_name = filename_output.split('.')[0]

        PARAMS_all = {'train_loader':train_loader,
                      'val_loader':val_loader,
                      'train_dataset':train_dataset,
                      'val_dataset':val_dataset,
                      'batch_size':batch_size,
                      'fold_name':fold_name,
                      'df_train':df_train,
                      'df_valid':df_valid,
                      'dict_paths':dict_paths,
                      'normalize_type':normalize_type,
                      'data_format':data_format,
                      'data_dim':data_dim,
                      'loss_type':loss_type,
                      'learning_rate':learning_rate,
                      'num_iterations':num_iterations,
                      'n_classes':n_classes,
                      'patience':patience,
                      'flag_resume_training':flag_resume_training,
                      'datackpt_name':datackpt_name,
                      'modelckpt_name':modelckpt_name,
                      'best_model_ckpt_name':best_model_ckpt_name,
                      'PARAMS_config':PARAMS_config          
        }
        model, \
            training_metrics = my_RNN_GRU_model.train_valid_RNN_GRU_model_all(PARAMS_all)

    # Save the training metrics for future visualization
    results['train_loss_Cassey'].extend(training_metrics['train_loss_Cassey'])
    results['train_loss_Darren'].extend(training_metrics['train_loss_Darren'])
    results['valid_loss'].extend(training_metrics['valid_loss'])
    results['train_accuracy'].extend(training_metrics['train_accuracy'])
    results['valid_accuracy'].extend(training_metrics['valid_accuracy'])
    results['train_epoch'].extend(training_metrics['epoch'])

    results['validation_metrics']['precision'].extend(training_metrics['precision'])
    results['validation_metrics']['recall'].extend(training_metrics['recall'])
    results['validation_metrics']['f1_score'].extend(training_metrics['f1_score'])
    # results['validation_metrics']['auc_roc'].extend(training_metrics['auc_roc'])