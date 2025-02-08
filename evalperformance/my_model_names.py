path_y_pred_root = '/content/drive/MyDrive/Public_Datasets/PulsewatchRelease/GitHub/y_pred'
import os
def my_model_names(str_model_idx):
    if str_model_idx == '03':
        # Only 1D PPG with batch size 32.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_batch32')
        file_date = '20240713'
        model_name = '1D_PPG_batch32' # 1D_PPG
        # Train code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/run_01_batch_1024.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/test_run_01_batch32.py
    elif str_model_idx == '04':
        # Only 1D PPG with batch size 512.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_batch512')
        file_date = '20240712'
        model_name = '1D_PPG_batch512'
        # Training code path: same as model 03.
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/test_run_01_batch512.py
    elif str_model_idx == '05':
        # 1D PPG + HR.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR')
        file_date = '20240629'
        model_name = '1D_PPG_HR'
        # Training code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/run_02_fold_1.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/test_run_02_fold_1.py
    elif str_model_idx == '06':
        # 1D PPG + HR + ACC.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR_ACC')
        file_date = '20240708'
        model_name = '1D_PPG_HR_ACC'
        # Training code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/run_04_ACC.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/test_run_04_fold_1.py
    elif str_model_idx == '07':
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR_ACC_noPPG')
        file_date = '20240710'
        model_name = '1D_HR_ACC_noPPG'
        # Train code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/run_05_no_PPG.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/test_run_05_fold_1.py
    elif str_model_idx == '08':
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR_ACC_rescaleHR')
        file_date = '20240712'
        model_name = '1D_PPG_WEPD-HR_ACC_rescale-WEPD-HR'
        # Train code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/run_06_rescale_HR.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_04_run_several_models/test_run_06_fold_1.py
    elif str_model_idx == '09':
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_ElgendiHR_ACC')
        file_date = '20240717'
        model_name = '1D_PPG_Elgendi-HR_ACC'
        # Train code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_05_Elgendi_HR/run_01_Elgendi_HR.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_05_Elgendi_HR/test_run_01_Elgendi_HR.py
    elif str_model_idx == '10':
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_Elgendi_rescaleHR_ACC')
        file_date = '20240719'
        model_name = '1D_PPG_Elgendi-HR_ACC_rescale-Elgendi-HR'
        # Train code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_05_Elgendi_HR/run_04_Elgendi_rescale_HR_ACC.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_05_Elgendi_HR/test_run_04_Elgendi_rescale_HR_ACC.py
    elif str_model_idx == '11':
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_aug_PPG_HR_ACC_rescaleHR')
        file_date = '20240726'
        model_name = '1D_PPG_WEPD-HR_ACC_rescale-HR_aug30k'
        # Train code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_06_augment_PACPVC/train/run_01_aug_train_PPG_HR_rescaledHR_ACC.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_06_augment_PACPVC/train/test_01_aug_PPG_HR_ACC_rescaleHR.py
    elif str_model_idx == '12':
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_aug5k_PPG_HR_ACC_rescaleHR')
        file_date = '20240728'
        model_name = '1D_PPG_WEPD-HR_ACC_rescale-HR_aug5k'
        # Train code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_06_augment_PACPVC/train/run_02_aug5k_train_PPG_HR_rescaledHR_ACC.py
        # Test code path: /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_06_augment_PACPVC/train/test_02_aug5k_PPG_HR_ACC_rescaleHR.py
    elif str_model_idx == '18':
        # Liu's retrained model.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_batch32')
        file_date = '20240713'
        model_name = 'Liu_2022_JAHA_retrained_Pulsewatch_PPG_only'
        # Test code path:
    elif str_model_idx == '19':
        # Liu's retrained model.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR_ACC_rescaleHR')
        file_date = '20240712'
        model_name = 'Liu_2022_JAHA_retrained_Pulsewatch_four_channels'
        # Test code path: 
    elif str_model_idx == '20':
        # Darren's model.
        path_ckpt = os.path.join(path_y_pred_root,'y_pred_2024_08_26_Darren')
        file_date = '20240826'
        model_name = 'Darren_2024_EMBC'
    elif str_model_idx == '21':
        # PPG+ACC only.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_PPG_ACC_only')
        file_date = '20240917'
        model_name = '1D_PPG_ACC_only'
    elif str_model_idx == '22':
        # HR only.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR_only')
        file_date = '20240917'
        model_name = '1D_HR_only'
    elif str_model_idx == '23':
        # HR+rescaleHR only.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR_rescaleHR_only')
        file_date = '20240917'
        model_name = '1D_HR_rescaleHR_only'
    elif str_model_idx == '24':
        # HR+rescaleHR+ACC.
        path_ckpt = os.path.join(path_y_pred_root,'TestRNNGRU_HR_rescaleHR_ACC_noPPG')
        file_date = '20240921'
        model_name = '1D_HR_rescaleHR_ACC_noPPG'
    else:
        pass

    return path_ckpt, file_date, model_name