"""
Testing all models for same subject testing.
It should be the results for the 'test.csv' in this folder:
R:\ENGR_Chon\Dong\Python_generated_results\deep_learning_2023\SMOTE_everything_fold_2_2024_03_30
"""
import os
import torch
import pandas as pd
import sys
import itertools
import pathlib
import numpy as np
# === Start of path and model name for all models. ===

# For 1D PPG aug5k:
# path_ckpt = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/y_pred_2024_07_29_aug5k/TestRNNGRU_aug5k_PPG_HR_ACC_rescaleHR'
# file_date = '20240728'
# model_name = '1D_PPG_aug5k_HR_ACC_rescaleHR'
sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/experiments/try_07_eval_models')
import my_model_names
list_str_model_idx = ['01','02','03','04','05','06','07','08','09','10','11','12','18','19','20','21','22','23','24']
# list_str_model_idx = ['20']
for str_model_idx in list_str_model_idx:
    path_ckpt, file_date, model_name = my_model_names.my_model_names(str_model_idx)
    # === End of path and model names. ====

    flag_linux = True
    flag_Colab = False
    # Append the directory to your python path using sys
    if flag_linux:
        print('Inside Linux')
        if flag_Colab:
            print('Inside Colab')
            # For 'my_pathdef'
        else:
            print('Not inside Colab')
            # For 'my_cfmatrix'
            sys.path.append('/mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/DeepBeat/utils')
            import pretty_errors

    import my_cfmatrix

    # ====
    if str_model_idx == '01' or str_model_idx == '02':
        # TensorFlow model, so output saved in csv format.
        if str_model_idx == '01':
            filename_fold_1 = 'test_01_tfs_mm_fold_1_same_subjects.csv'
            filename_fold_2 = 'test_01_tfs_mm_fold_2_same_subjects.csv'
        else:
            filename_fold_1 = 'test_02_poin_mm_fold_1_same_subjects.csv'
            filename_fold_2 = 'test_02_poin_mm_fold_2_same_subjects.csv'

        df_fold_1 = pd.read_csv(os.path.join(path_ckpt,filename_fold_1))
        df_fold_2 = pd.read_csv(os.path.join(path_ckpt,filename_fold_2))

        # Since 'y_pred_prob_0', 'y_pred_prob_1', 'y_pred_prob_2' are not in the table, I will fake one until I have them.
        df_fold_1.insert(3,'y_pred_prob_0',np.zeros((df_fold_1.shape[0],),dtype=float))
        df_fold_1.insert(3,'y_pred_prob_1',np.zeros((df_fold_1.shape[0],),dtype=float))
        df_fold_1.insert(3,'y_pred_prob_2',np.zeros((df_fold_1.shape[0],),dtype=float))
        df_fold_2.insert(3,'y_pred_prob_0',np.zeros((df_fold_2.shape[0],),dtype=float))
        df_fold_2.insert(3,'y_pred_prob_1',np.zeros((df_fold_2.shape[0],),dtype=float))
        df_fold_2.insert(3,'y_pred_prob_2',np.zeros((df_fold_2.shape[0],),dtype=float))
    elif str_model_idx == '18' or str_model_idx == '19':
        fold_name = 'fold_1'
        filename_ckpt = 'test_01_Liu_JAHA_2022_same_subject_test_train_RNN_GRU_'+file_date+'_'+fold_name+'.pt'
        df_fold_1 = my_cfmatrix.my_load_pt_results(path_ckpt, filename_ckpt)

        fold_name = 'fold_2'
        filename_ckpt = 'test_01_Liu_JAHA_2022_same_subject_test_train_RNN_GRU_'+file_date+'_'+fold_name+'.pt'
        df_fold_2 = my_cfmatrix.my_load_pt_results(path_ckpt, filename_ckpt)
    elif str_model_idx == '20':
        y_pred_filename_fold_1 = 'predictions_fold_1_test_set.csv'
        y_pred_filename_fold_2 = 'predictions_fold_2_test_set.csv'

        y_true_filename_fold_1 = 'labels_fold_1_test_set.csv'
        y_true_filename_fold_2 = 'labels_fold_2_test_set.csv'

        pred_prob_filename_fold_1 = 'prediction_proba_fold_1_test_set.csv'
        pred_prob_filename_fold_2 = 'prediction_proba_fold_2_test_set.csv'

        y_pred_fold_1 = pd.read_csv(os.path.join(path_ckpt,y_pred_filename_fold_1))
        y_pred_fold_2 = pd.read_csv(os.path.join(path_ckpt,y_pred_filename_fold_2))
        y_true_fold_1 = pd.read_csv(os.path.join(path_ckpt,y_true_filename_fold_1))
        y_true_fold_2 = pd.read_csv(os.path.join(path_ckpt,y_true_filename_fold_2))
        pred_prob_fold_1 = pd.read_csv(os.path.join(path_ckpt,pred_prob_filename_fold_1))
        pred_prob_fold_2 = pd.read_csv(os.path.join(path_ckpt,pred_prob_filename_fold_2))

        print('y_pred_fold_1.columns',y_pred_fold_1.columns) # 'predictions'
        print('y_true_fold_1.columns',y_true_fold_1.columns) # 'labels'
        print('pred_prob_fold_1.columns',pred_prob_fold_1.columns) # '0', '1', '2'

        df_fold_1 = pd.DataFrame({'y_pred':y_pred_fold_1['predictions'].to_list(),
                                  'y_true':y_true_fold_1['labels'].to_list(),
                                  'y_pred_prob_0':pred_prob_fold_1['0'].to_list(),
                                  'y_pred_prob_1':pred_prob_fold_1['1'].to_list(),
                                  'y_pred_prob_2':pred_prob_fold_1['2'].to_list()})
        
        print('df_fold_1',df_fold_1)

        df_fold_2 = pd.DataFrame({'y_pred':y_pred_fold_2['predictions'].to_list(),
                                  'y_true':y_true_fold_2['labels'].to_list(),
                                  'y_pred_prob_0':pred_prob_fold_2['0'].to_list(),
                                  'y_pred_prob_1':pred_prob_fold_2['1'].to_list(),
                                  'y_pred_prob_2':pred_prob_fold_2['2'].to_list()})
        
        print('df_fold_2',df_fold_2)
    else:
        fold_name = 'fold_1'
        filename_ckpt = 'test_01_same_subject_test_train_RNN_GRU_'+file_date+'_'+fold_name+'.pt'
        df_fold_1 = my_cfmatrix.my_load_pt_results(path_ckpt, filename_ckpt)

        fold_name = 'fold_2'
        filename_ckpt = 'test_01_same_subject_test_train_RNN_GRU_'+file_date+'_'+fold_name+'.pt'
        df_fold_2 = my_cfmatrix.my_load_pt_results(path_ckpt, filename_ckpt)

    path_output = '/'.join(path_ckpt.split('/')[:-1])

    # Calculate Sensitivity (TPR), Specificity (TNR), Precision (PPV), NPV, FPR, FNR, FDR, Accuracy (ACC).
    dict_metrics_fold_1 = my_cfmatrix.my_average_none_metrics(y_true = df_fold_1['y_true'], y_pred = df_fold_1['y_pred'])
    dict_metrics_fold_2 = my_cfmatrix.my_average_none_metrics(y_true = df_fold_2['y_true'], y_pred = df_fold_2['y_pred'])

    import numpy as np
    print('dict_metrics_fold_1[TPR]',dict_metrics_fold_1['TPR'])
    def my_mean_calculate(val_fold_1, val_fold_2):
        a = np.array(val_fold_1)
        b = np.array(val_fold_2)
        c = np.vstack((a,b))
        my_mean = np.mean(c,axis=0)
        return my_mean
    avg_TPR = my_mean_calculate(dict_metrics_fold_1['TPR'],dict_metrics_fold_2['TPR'])
    print('avg_TPR',avg_TPR)
    avg_TNR = my_mean_calculate(dict_metrics_fold_1['TNR'],dict_metrics_fold_2['TNR'])
    avg_PPV = my_mean_calculate(dict_metrics_fold_1['PPV'],dict_metrics_fold_2['PPV'])
    avg_NPV = my_mean_calculate(dict_metrics_fold_1['NPV'],dict_metrics_fold_2['NPV'])
    avg_FPR = my_mean_calculate(dict_metrics_fold_1['FPR'],dict_metrics_fold_2['FPR'])
    avg_FNR = my_mean_calculate(dict_metrics_fold_1['FNR'],dict_metrics_fold_2['FNR'])
    avg_FDR = my_mean_calculate(dict_metrics_fold_1['FDR'],dict_metrics_fold_2['FDR'])
    avg_ACC = my_mean_calculate(dict_metrics_fold_1['ACC'],dict_metrics_fold_2['ACC'])

    # Calculate the AUROC.
    n_classes = 3
    my_class_names = {0: 'NSR', 1: 'AF', 2: 'PAC/PVC'}
    test_name = 'same_subject_test'
    path_output = os.path.join(r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/analysis_2024_07_30',test_name)
    path_output_fig = os.path.join(path_output,'plots')
    pathlib.Path(path_output_fig).mkdir(parents=True, exist_ok=True)
    micro_roc_auc_ovr_1, macro_roc_auc_ovr_1 = my_cfmatrix.my_auroc_cal_plot(df_fold_1, n_classes, my_class_names, path_output_fig, model_name, test_name, str_model_idx, fold_name='fold_1')
    micro_roc_auc_ovr_2, macro_roc_auc_ovr_2 = my_cfmatrix.my_auroc_cal_plot(df_fold_2, n_classes, my_class_names, path_output_fig, model_name, test_name, str_model_idx, fold_name='fold_2')

    avg_micro_auroc = np.mean(np.array([micro_roc_auc_ovr_1,micro_roc_auc_ovr_2]))
    avg_macro_auroc = np.mean(np.array([macro_roc_auc_ovr_1,macro_roc_auc_ovr_2]))

    dict_metrics_avg = {'avg_TPR':avg_TPR,
                        'avg_TNR':avg_TNR,
                        'avg_PPV':avg_PPV,
                        'avg_NPV':avg_NPV,
                        'avg_FPR':avg_FPR,
                        'avg_FNR':avg_FNR,
                        'avg_FDR':avg_FDR,
                        'avg_ACC':avg_ACC,
                        'avg_micro_auroc':avg_micro_auroc,
                        'avg_macro_auroc':avg_macro_auroc}

    # df_metrics = pd.DataFrame(dict_metrics_avg.items(),columns=dict_metrics_avg.keys())
    df_metrics = pd.DataFrame.from_dict(dict_metrics_avg, orient='columns').reset_index()
    df_metrics.to_csv(os.path.join(path_output,str_model_idx+'_'+model_name+'_'+test_name+'.csv'),header=True,index=True)