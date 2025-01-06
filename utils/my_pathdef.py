import socket # For getting the host name.
import os
import pathlib

def get_data_paths(data_dim,dataset_name,output_folder_name) -> dict:
    your_computer_name = socket.gethostname()
    print('Debug: your_computer_name',your_computer_name)
    if not your_computer_name == 'localhost.localdomain' or not your_computer_name == 'HPC_computer_name' or not your_computer_name == 'Darren_computer_name':
        print('Debug: You are in Google Colab.')
        base_path = '/content'
        labels_base_path = '/content/drive/MyDrive/Adjudication_UConn'
        saving_base_path = '/content/drive/MyDrive/Checkpoint_Colab'
        path_train_val_test_split = '/content/drive/MyDrive/Adjudication_UConn/SMOTE_everything_fold_2_2024_03_30'
        # path_1D_PPG_Pulsewatch = '/content/PPG_filt_30sec_csv/' # Never put it in your personal drive. Toss it after this session since too many files.
        # path_tar_file_1D_PPG_Pulsewatch = '/content/drive/MyDrive/Pulsewatch_Data/tar_PPG_filt_30sec_csv'
        path_1D_PPG_Pulsewatch = '/content/PPG_filt_30sec_HR_pt' # Never put it in your personal drive. Toss it after this session since too many files.
        path_tar_file_1D_PPG_Pulsewatch = '/content/drive/MyDrive/Pulsewatch_Data/tar_PT_1D_PPG_HR_Pulsewatch'
        path_1D_PPG_aug_Pulsewatch = '/content/PPG_filt_30sec_aug_pt_rs_' # Remember to add 'fold_name'
        path_tar_file_1D_PPG_aug_Pulsewatch = '/content/drive/MyDrive/Pulsewatch_Data/tar_PPG_filt_30sec_aug_pt_rs_' # Remember to add 'fold_name'
        path_1D_ACC_Pulsewatch = '/content/PPG_Elgendi_30sec_HR_ACC_pt'
        path_tar_file_1D_ACC_Pulsewatch = '/content/drive/MyDrive/Pulsewatch_Data/tar_PPG_Elgendi_30sec_HR_ACC_pt'
        path_2D_tfs_Pulsewatch = '/content/PT_tfs_Pulsewatch'
        path_tar_file_2D_tfs_Pulsewatch = '/content/drive/MyDrive/Pulsewatch_Data/tar_PT_tfs_Pulsewatch'
        path_2D_poin_Pulsewatch = '/content/PT_Poin_Pulsewatch'
        path_tar_file_2D_poin_Pulsewatch = '/content/drive/MyDrive/Pulsewatch_Data/tar_PT_Poin_Pulsewatch'

        path_1D_PPG_Simband = '/content/PPG_filt_30sec_HR_pt' # Same path but UID start with 6XX.
        path_tar_file_1D_PPG_Simband = '/content/drive/MyDrive/pt_files/tar_1DPPG_WEPD_HR_Simband' # PPG is stored in both WEPD and Elgendi files.
        path_1D_ACC_Simband = '/content/PPG_Elgendi_30sec_HR_ACC_pt' # Elgendi's HR also includes the ACC.
        path_tar_file_1D_ACC_Simband = '/content/drive/MyDrive/pt_files/tar_1DPPG_Elgendi_HR_ACC_Simband'
        path_2D_tfs_Simband = '/content/tfrecords_tfs_mm_Simband'
        path_tar_file_2D_tfs_Simband = '/content/drive/MyDrive/tfrecords/tar_tfrecords_tfs_mm_Simband'
        path_2D_poin_Simband = '/content/tfrecords_poin_mm_Simband'
        path_tar_file_2D_poin_Simband = '/content/drive/MyDrive/tfrecords/tar_tfrecords_poin_mm_Simband'

        path_1D_PPG_MIMICIII = '/content/PPG_filt_30sec_HR_pt' # Same path but UID start with 7XX.
        path_tar_file_1D_PPG_MIMICIII = '/content/drive/MyDrive/pt_files/tar_1DPPG_WEPD_HR_MIMICIII_top_peaks' # WEPD should use top peaks (08/09/2024)
        path_1D_ACC_MIMICIII = '/content/PPG_Elgendi_30sec_HR_ACC_pt' # Elgendi's HR also includes the ACC.
        path_tar_file_1D_ACC_MIMICIII = '/content/drive/MyDrive/pt_files/tar_1DPPG_Elgendi_HR_ACC_MIMICIII_bottom_peaks' # Elgendi should use bottom peaks (before 08/09/2024)
        path_2D_tfs_MIMICIII = '/content/tfrecords_tfs_mm_MIMICIII'
        path_tar_file_2D_tfs_MIMICIII = '/content/drive/MyDrive/tfrecords/tar_tfrecords_tfs_mm_Simband'
        path_2D_poin_MIMICIII = '/content/tfrecords_poin_mm_MIMICIII'
        path_tar_file_2D_poin_MIMICIII = '/content/drive/MyDrive/tfrecords/tar_tfrecords_poin_mm_MIMICIII'
        # print('ERROR! YOUR DID NOT GET THE PATH.')
        # raise ValueError
    
    if data_dim[:3] == '1D_':
        if dataset_name == 'Pulsewatch':
            data_path = path_1D_PPG_Pulsewatch
            tar_path = path_tar_file_1D_PPG_Pulsewatch
            ACC_path = path_1D_ACC_Pulsewatch
            tar_ACC_path = path_tar_file_1D_ACC_Pulsewatch
            aug_path = path_1D_PPG_aug_Pulsewatch
            tar_aug_path = path_tar_file_1D_PPG_aug_Pulsewatch
            labels_path = os.path.join(labels_base_path, "final_attemp_4_1_Dong_Ohm")
        elif dataset_name == 'Simband':
            data_path = path_1D_PPG_Simband
            tar_path = path_tar_file_1D_PPG_Simband
            ACC_path = path_1D_ACC_Simband # 08/06/2024.
            tar_ACC_path = path_tar_file_1D_ACC_Simband # 08/06/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.
            labels_path = os.path.join(labels_base_path, "Dong_Simband") # Need to revise this.
            # '/content/drive/MyDrive/Adjudication_UConn/Dong_Simband/simband_segments_labels_2024_04_25.csv'
        elif dataset_name == 'MIMICIII':
            data_path = path_1D_PPG_MIMICIII
            tar_path = path_tar_file_1D_PPG_MIMICIII
            ACC_path = path_1D_ACC_MIMICIII # 08/06/2024.
            tar_ACC_path = path_tar_file_1D_ACC_MIMICIII # 08/06/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.
            labels_path = os.path.join(labels_base_path, "Dong_MIMICIII")
            # For Sensor paper: '/content/drive/MyDrive/Adjudication_UConn/Dong_MIMICIII/2020_Han_Sensors_MIMICIII_Ground_Truth_2024_05_05.csv'
            # For newly adjudicated segments: '/content/drive/MyDrive/Adjudication_UConn/Dong_MIMICIII/MIMICIII_Dong_20240809.csv'
        
        saving_path = os.path.join(saving_base_path, output_folder_name)
    elif data_dim == '2D_TFS':
        if dataset_name == 'Pulsewatch':
            data_path = path_2D_tfs_Pulsewatch
            tar_path = path_tar_file_2D_tfs_Pulsewatch
            ACC_path = '' # Fill it later. 07/08/2024.
            tar_ACC_path = '' # Fill it later. 07/08/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.
        elif dataset_name == 'Simband':
            data_path = path_2D_tfs_Simband
            tar_path = path_tar_file_2D_tfs_Simband
            ACC_path = '' # Fill it later. 07/08/2024.
            tar_ACC_path = '' # Fill it later. 07/08/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.
        elif dataset_name == 'MIMICIII':
            data_path = path_2D_tfs_MIMICIII
            tar_path = path_tar_file_2D_tfs_MIMICIII
            ACC_path = '' # Fill it later. 07/08/2024.
            tar_ACC_path = '' # Fill it later. 07/08/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.

        labels_path = os.path.join(labels_base_path, "final_attemp_4_1_Dong_Ohm")
        saving_path = os.path.join(saving_base_path, output_folder_name)
    elif data_dim == '2D_Poin':
        if dataset_name == 'Pulsewatch':
            data_path = path_2D_poin_Pulsewatch
            tar_path = path_tar_file_2D_poin_Pulsewatch
            ACC_path = '' # Fill it later. 07/08/2024.
            tar_ACC_path = '' # Fill it later. 07/08/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.
        elif dataset_name == 'Simband':
            data_path = path_2D_poin_Simband
            tar_path = path_tar_file_2D_poin_Simband
            ACC_path = '' # Fill it later. 07/08/2024.
            tar_ACC_path = '' # Fill it later. 07/08/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.
        elif dataset_name == 'MIMICIII':
            data_path = path_2D_poin_MIMICIII
            tar_path = path_tar_file_2D_poin_MIMICIII
            ACC_path = '' # Fill it later. 07/08/2024.
            tar_ACC_path = '' # Fill it later. 07/08/2024.
            aug_path = '' # Fill it later, 07/26/2024.
            tar_aug_path = '' # Fill it later, 07/26/2024.
            
        labels_path = os.path.join(labels_base_path, "final_attemp_4_1_Dong_Ohm")
        saving_path = os.path.join(saving_base_path, output_folder_name)
    else:
        raise ValueError("Invalid data format. Choose 'csv' or 'png.")
    
    # Create the parent path for checkpoints.
    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)

    dict_paths = {'data_path': data_path, 
                  'tar_path': tar_path,
                  'ACC_path':ACC_path,
                  'tar_ACC_path':tar_ACC_path,
                  'aug_path':aug_path,
                  'tar_aug_path':tar_aug_path,
                  'labels_path': labels_path, 
                  'saving_path': saving_path,
                  'path_train_val_test_split': path_train_val_test_split}
    return dict_paths

def my_ckpt_path(flag_Colab):
    if flag_Colab:
        path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_2024_06_26_HR'
        filename_ckpt_best_model = 'ckpt_best_modeltrain_RNN_GRU_20240612_fold_2.pt'
        filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240612_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_01(flag_Colab, fold_name):
    # For 1D PPG+HR.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = '/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240629_fold_1_epoch_0075_val_loss_0.1493.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240629_fold_1_epoch_0075_val_loss_0.1493.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240629_fold_2_epoch_0066_val_loss_0.1862.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240629_fold_2_epoch_0066_val_loss_0.1862.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_09(flag_Colab, fold_name):
    # For 1D PPG+HR+ACC.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = '/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC'
            filename_ckpt_best_model = 'train_RNN_GRU_20240708_fold_1_epoch_0035_val_loss_0.1308.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240708_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC'
            filename_ckpt_best_model = 'train_RNN_GRU_20240708_fold_2_epoch_0029_val_loss_0.1690.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240629_fold_2_epoch_0029_val_loss_0.1690.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_12(flag_Colab, fold_name):
    # For 1D HR+ACC without PPG.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = '/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC_noPPG'
            filename_ckpt_best_model = 'train_RNN_GRU_20240710_fold_1_epoch_0034_val_loss_0.1483.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240710_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC_noPPG'
            filename_ckpt_best_model = 'train_RNN_GRU_20240710_fold_2_epoch_0054_val_loss_0.1429.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240710_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_16(flag_Colab, fold_name):
    # For 1D PPG+HR+ACC+rescaled HR.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = '/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240712_fold_1_epoch_0033_val_loss_0.1207.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240712_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240712_fold_2_epoch_0043_val_loss_0.1470.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240712_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_16_batch512(flag_Colab, fold_name):
    # For 1D PPG+HR+ACC+rescaled HR.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = '/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_batch512'
            filename_ckpt_best_model = 'train_RNN_GRU_20240712_fold_1_epoch_1239_val_loss_0.1523.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240712_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_batch512'
            filename_ckpt_best_model = 'train_RNN_GRU_20240712_fold_2_epoch_1264_val_loss_0.2995.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240712_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_16_batch32(flag_Colab, fold_name):
    # For 1D PPG+HR+ACC+rescaled HR.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = '/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_batch32'
            filename_ckpt_best_model = 'train_RNN_GRU_20240713_fold_1_epoch_0092_val_loss_0.2038.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240713_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_batch32'
            filename_ckpt_best_model = 'train_RNN_GRU_20240713_fold_2_epoch_0130_val_loss_0.1885.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240713_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_18(flag_Colab, fold_name):
    # For 1D PPG+ElgendiHR+ACC.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = '/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_ElgendiHR_ACC'
            filename_ckpt_best_model = 'train_RNN_GRU_20240717_fold_1_epoch_0094_val_loss_0.1499.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240717_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_ElgendiHR_ACC'
            filename_ckpt_best_model = 'train_RNN_GRU_20240717_fold_2_epoch_0090_val_loss_0.1599.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240717_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_19(flag_Colab, fold_name):
    # For 1D PPG+ElgendiHR+rescale Elgendi HR+ACC.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_Elgendi_rescaleHR_ACC'
            filename_ckpt_best_model = 'train_RNN_GRU_20240719_fold_1_epoch_0062_val_loss_0.1448.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240719_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_Elgendi_rescaleHR_ACC'
            filename_ckpt_best_model = 'train_RNN_GRU_20240719_fold_2_epoch_0096_val_loss_0.1706.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240719_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_26(flag_Colab, fold_name):
    # For 1D PPG+ElgendiHR+rescale Elgendi HR+ACC.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_aug_PPG_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240726_fold_1_epoch_0065_val_loss_0.1690.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240726_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_aug_PPG_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240726_fold_2_epoch_0103_val_loss_0.2120.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240726_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_07_29(flag_Colab, fold_name):
    # For aug 5k 1D PPG+WEPD HR+rescale HR+ACC.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_aug5k_PPG_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240728_fold_1_epoch_0062_val_loss_0.1336.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240728_fold_1.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_aug5k_PPG_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'train_RNN_GRU_20240728_fold_2_epoch_0094_val_loss_0.1829.pt'
            filename_ckpt_model = 'ckpt_train_modeltrain_RNN_GRU_20240728_fold_2.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths
def my_ckpt_path_2024_08_21_PPG_only(flag_Colab, fold_name):
    # For aug 5k 1D PPG+WEPD HR+rescale HR+ACC.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_batch32'
            filename_ckpt_best_model = 'Liu_JAHA_2022_PPG_only_fold_1_epoch_0019_ValAcc_0.9657.pt'
            filename_ckpt_model = 'Liu_JAHA_2022_PPG_only_fold_1_epoch_0019_ValAcc_0.9657.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_batch32'
            filename_ckpt_best_model = 'Liu_JAHA_2022_PPG_only_fold_2_epoch_0014_ValAcc_0.9531.pt'
            filename_ckpt_model = 'Liu_JAHA_2022_PPG_only_fold_2_epoch_0014_ValAcc_0.9531.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_08_21_four_channels(flag_Colab, fold_name):
    # For aug 5k 1D PPG+WEPD HR+rescale HR+ACC.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'Liu_JAHA_2022_four_channels_fold_1_epoch_0024_ValAcc_0.9639.pt'
            filename_ckpt_model = 'Liu_JAHA_2022_four_channels_fold_1_epoch_0024_ValAcc_0.9639.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_ACC_rescaleHR'
            filename_ckpt_best_model = 'Liu_JAHA_2022_four_channels_fold_2_epoch_0019_ValAcc_0.9496.pt'
            filename_ckpt_model = 'Liu_JAHA_2022_four_channels_fold_2_epoch_0019_ValAcc_0.9496.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_09_19_PPG_ACC_only(flag_Colab, fold_name):
    # For PPG and ACC only.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_PPG_ACC_only'
            filename_ckpt_best_model = 'train_RNN_GRU_20240917_fold_1_epoch_0054_val_loss_0.2112.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240917_fold_1_epoch_0054_val_loss_0.2112.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_PPG_ACC_only'
            filename_ckpt_best_model = 'train_RNN_GRU_20240917_fold_2_epoch_0072_val_loss_0.1829.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240917_fold_2_epoch_0072_val_loss_0.1829.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_09_19_HR_only(flag_Colab, fold_name):
    # For PPG and ACC only.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_only'
            filename_ckpt_best_model = 'train_RNN_GRU_20240917_fold_1_epoch_0043_val_loss_0.1582.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240917_fold_1_epoch_0043_val_loss_0.1582.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_only'
            filename_ckpt_best_model = 'train_RNN_GRU_20240917_fold_2_epoch_0061_val_loss_0.2029.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240917_fold_2_epoch_0061_val_loss_0.2029.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_09_19_HR_rescaleHR_only(flag_Colab, fold_name):
    # For PPG and ACC only.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_rescaleHR_only'
            filename_ckpt_best_model = 'train_RNN_GRU_20240917_fold_1_epoch_0041_val_loss_0.1566.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240917_fold_1_epoch_0041_val_loss_0.1566.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_rescaleHR_only'
            filename_ckpt_best_model = 'train_RNN_GRU_20240917_fold_2_epoch_0029_val_loss_0.2001.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240917_fold_2_epoch_0029_val_loss_0.2001.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths

def my_ckpt_path_2024_09_21_HR_rescaleHR_ACC_noPPG(flag_Colab, fold_name):
    # For PPG and ACC only.
    if flag_Colab:
        if fold_name == 'fold_1':
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_rescaleHR_ACC_noPPG'
            filename_ckpt_best_model = 'train_RNN_GRU_20240921_fold_1_epoch_0045_val_loss_0.1401.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240921_fold_1_epoch_0045_val_loss_0.1401.pt'
        else:
            path_ckpt_model = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR_rescaleHR_ACC_noPPG'
            filename_ckpt_best_model = 'train_RNN_GRU_20240921_fold_2_epoch_0037_val_loss_0.1592.pt'
            filename_ckpt_model = 'train_RNN_GRU_20240921_fold_2_epoch_0037_val_loss_0.1592.pt'

    dict_paths = {'path_ckpt_model': path_ckpt_model,
                 'filename_ckpt_best_model':filename_ckpt_best_model,
                 'filename_ckpt_model':filename_ckpt_model}
    
    return dict_paths