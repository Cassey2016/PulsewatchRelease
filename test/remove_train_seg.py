"""
When calculating the performance metrics, remove segments used in training.

Dong, 07/09/2024.
"""

import os
import glob
import pandas as pd

def my_remove_both_folds(path_GT,fold_name):
    # Get the segment name:
    filename_GT = glob.glob(os.path.join(path_GT,'*.csv'))

    all_UIDs_GT = [xx.split('/')[-1][:3] for xx in filename_GT]

    print('Number of all adjudicated UIDs:',len(all_UIDs_GT))

    df_all = []
    for new_GT_filename in filename_GT:
    # if True:
        # UID = unique_UIDs_df_1[0]
        # '011_final_attemp_4_1_Dong.csv'
        df_new_GT = pd.read_csv(new_GT_filename)
        # df_new_GT_trim = df_new_GT.loc[(df_new_GT['final_AF_GT_20230921'] == 0) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 1) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 2) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 3)]
        df_new_GT_trim = df_new_GT.loc[(df_new_GT['final_AF_GT_20230921'] == 0) | \
                            (df_new_GT['final_AF_GT_20230921'] == 1) | \
                            (df_new_GT['final_AF_GT_20230921'] == 2)] # Dong, 07/01/2024

        if new_GT_filename.split('/')[-1][:3] == '408':
            print('408 GT is:',df_new_GT_trim)
        if len(df_all) > 0:
            # Have initialized the dataframe before.
            df_all = pd.concat([df_all,df_new_GT_trim])
        else:
            df_all = df_new_GT_trim.copy()

    df_all['final_AF_GT_20230921'] = pd.to_numeric(df_all['final_AF_GT_20230921'], downcast='integer')
    # print('df_all',df_all)
    # print('df_all.columns',df_all.columns)

    # Convert dataframe into dictionary.
    dict_df_all = {}
    for idx,rr in df_all.iterrows():
        # print('rr[''table_file_name'']',rr['table_file_name'])
        # print('rr[''final_AF_GT_20230921'']',rr['final_AF_GT_20230921'])
        dict_df_all[rr['table_file_name']] = rr['final_AF_GT_20230921']
    
    # Load all the training, validation, and testing set.
    # Copied from /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/ResNet_classification/experiments/try_06_swap_AF/gen_AF/main_03_final_UIDs_fold_1_2.ipynb
    # Fold 1 remove fold 1 data.
    # UID info for PACPVC. 03/30/2024.
    UIDs_fold_1_PACPVC = ['110','419','408','053','113','054','039','080','120','064','104','042','069','089','007','090','026','022','093']
    UIDs_fold_2_PACPVC = ['075','100','045','005','112','086','013','073','002','028','327','052','068','021','087','078','070','038','029']

    # UID info for AF. 03/30/2024.
    UIDs_fold_1_AF = ['110','419','408','423','413','416','415','400','409','405','321','305','318','320','322','310','422']
    UIDs_fold_2_AF = ['075','017','410','402','421','406','414','407','420','302','307','311','301','329','319','324','312','306']

    # UID info for NSR. 03/30/2024.
    UIDs_fold_1_NSR = ['024','057','037']
    UIDs_fold_2_NSR = []

    # All UIDs:
    UIDs_fold_1 = list(set(UIDs_fold_1_PACPVC+UIDs_fold_1_AF+UIDs_fold_1_NSR))
    UIDs_fold_2 = list(set(UIDs_fold_2_PACPVC+UIDs_fold_2_AF+UIDs_fold_2_NSR))

    # if fold_name == 'fold_1':
    #     UIDs_fold = UIDs_fold_1
    # else:
    #     UIDs_fold = UIDs_fold_2
    UIDs_fold = UIDs_fold_1 + UIDs_fold_2

    df_used = []
    for UID in UIDs_fold:
    # if True:
        # UID = unique_UIDs_df_1[0]
        # '011_final_attemp_4_1_Dong.csv'
        new_GT_filename = [tt for tt in filename_GT if UID in tt.split('/')[-1][:3]]
        df_new_GT = pd.read_csv(new_GT_filename[0])
        # df_new_GT_trim = df_new_GT.loc[(df_new_GT['final_AF_GT_20230921'] == 0) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 1) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 2) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 3)]
        df_new_GT_trim = df_new_GT.loc[(df_new_GT['final_AF_GT_20230921'] == 0) | \
                            (df_new_GT['final_AF_GT_20230921'] == 1) | \
                            (df_new_GT['final_AF_GT_20230921'] == 2)] # Dong, 07/01/2024.

        if len(df_used) > 0:
            # Have initialized the dataframe before.
            df_used = pd.concat([df_used,df_new_GT_trim])
        else:
            df_used = df_new_GT_trim.copy()

    finished_seg_names = df_used['table_file_name'].to_list()
    # print('len(finished_seg_names)',len(finished_seg_names))

    # From extract_segment_names_and_labels:
    # Input: 
    #   dict_df_all: dictionary, with table_file_name as key and final_AF_GT_20230921 as value.
    #   finished_seg_names: list of string.
    remain_labels = dict_df_all.copy() 
    # print('Debug: type(remain_labels)',type(remain_labels))
    for key in finished_seg_names:
        remain_labels.pop(key)
    # print('Debug: len(labels)',len(dict_df_all))
    # print('Debug: len(remain_labels)',len(remain_labels))

    return remain_labels

def my_test_the_other_fold(path_GT,fold_name):
    # Get the segment name:
    filename_GT = glob.glob(os.path.join(path_GT,'*.csv'))

    # Load all the training, validation, and testing set.
    # Copied from /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/ResNet_classification/experiments/try_06_swap_AF/gen_AF/main_03_final_UIDs_fold_1_2.ipynb
    # Fold 1 remove fold 1 data.
    # UID info for PACPVC. 03/30/2024.
    UIDs_fold_1_PACPVC = ['110','419','408','053','113','054','039','080','120','064','104','042','069','089','007','090','026','022','093']
    UIDs_fold_2_PACPVC = ['075','100','045','005','112','086','013','073','002','028','327','052','068','021','087','078','070','038','029']

    # UID info for AF. 03/30/2024.
    UIDs_fold_1_AF = ['110','419','408','423','413','416','415','400','409','405','321','305','318','320','322','310','422']
    UIDs_fold_2_AF = ['075','017','410','402','421','406','414','407','420','302','307','311','301','329','319','324','312','306']

    # UID info for NSR. 03/30/2024.
    UIDs_fold_1_NSR = ['024','057','037']
    UIDs_fold_2_NSR = []

    # All UIDs:
    UIDs_fold_1 = list(set(UIDs_fold_1_PACPVC+UIDs_fold_1_AF+UIDs_fold_1_NSR))
    UIDs_fold_2 = list(set(UIDs_fold_2_PACPVC+UIDs_fold_2_AF+UIDs_fold_2_NSR))

    if fold_name == 'fold_1':
        # Test on the other fold.
        the_other_UIDs_fold = UIDs_fold_2
    else:
        the_other_UIDs_fold = UIDs_fold_1
    # UIDs_fold = UIDs_fold_1 + UIDs_fold_2

    df_the_other_fold = []
    for UID in the_other_UIDs_fold:
    # if True:
        # UID = unique_UIDs_df_1[0]
        # '011_final_attemp_4_1_Dong.csv'
        new_GT_filename = [tt for tt in filename_GT if UID in tt.split('/')[-1][:3]]
        df_new_GT = pd.read_csv(new_GT_filename[0])
        # df_new_GT_trim = df_new_GT.loc[(df_new_GT['final_AF_GT_20230921'] == 0) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 1) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 2) | \
        #                     (df_new_GT['final_AF_GT_20230921'] == 3)]
        df_new_GT_trim = df_new_GT.loc[(df_new_GT['final_AF_GT_20230921'] == 0) | \
                            (df_new_GT['final_AF_GT_20230921'] == 1) | \
                            (df_new_GT['final_AF_GT_20230921'] == 2)] # Dong, 07/01/2024.

        if len(df_the_other_fold) > 0:
            # Have initialized the dataframe before.
            df_the_other_fold = pd.concat([df_the_other_fold,df_new_GT_trim])
        else:
            df_the_other_fold = df_new_GT_trim.copy()
    return df_the_other_fold

def my_same_subject_test_seg(path_fold_info,fold_name):
    print('path_fold_info',path_fold_info)
    if fold_name == 'fold_1':
        # Fold 1 remove fold 1 data.
        filename_fold_2 = 'fold_1_test_20240330.csv'
    else:
        filename_fold_2 = 'fold_2_test_20240330.csv'
    df_fold_2 = pd.read_csv(os.path.join(path_fold_info,filename_fold_2))
    # Convert dataframe into dictionary.
    remain_labels = {}
    for idx,rr in df_fold_2.iterrows():
        remain_labels[rr['table_file_name']] = rr['final_AF_GT_20230921']

    return remain_labels

def my_test_Simband(flag_Colab):
    if flag_Colab:
        path_Simband_GT = r'/content/drive/MyDrive/Adjudication_UConn/Darren_Simband/simband_segments_labels.csv'
    else:
        path_Simband_GT = r'/mnt/r/ENGR_Chon/Darren/Public_Database/PPG_PeakDet_Simband/Darren_conversion/simband_segments_labels.csv'
    df_Simband = pd.read_csv(path_Simband_GT)
    # Convert dataframe into dictionary.
    remain_labels = {}
    for idx,rr in df_Simband.iterrows():
        # ['segment_names', 'labels']
        remain_labels[rr['segment_names']] = rr['labels']

    return remain_labels

def my_test_Simband_poin(flag_Colab):
    if flag_Colab:
        path_Simband_GT = r'/content/drive/MyDrive/Adjudication_UConn/Dong_Simband/simband_segments_labels_check_all_labels_2024_04_25.csv'
    else:
        path_Simband_GT = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/Adjudication_UConn/Simband/simband_segments_labels_check_all_labels_2024_04_25.csv'
    df_new_GT = pd.read_csv(path_Simband_GT)

    # Only keep the 0, 1, 2 in the testing.
    df_new_GT_trim = df_new_GT.loc[(df_new_GT['labels'] == 0) | \
                            (df_new_GT['labels'] == 1) | \
                            (df_new_GT['labels'] == 2)]
    df_Simband = df_new_GT_trim.copy()

    # Convert dataframe into dictionary.
    remain_labels = {}
    for idx,rr in df_Simband.iterrows():
        # ['segment_names', 'labels']
        remain_labels[rr['segment_names']] = rr['labels']

    return remain_labels

def my_test_MIMICIII(flag_Colab,flag_tfs):
    if flag_Colab:
        if flag_tfs:
            path_MIMICIII_GT = r'/content/drive/MyDrive/Adjudication_UConn/Darren_MIMICIII/2020_Han_Sensors_MIMICIII_Ground_Truth.csv'
        else:
            path_MIMICIII_GT = r'/content/drive/MyDrive/Adjudication_UConn/Dong_MIMICIII/2020_Han_Sensors_MIMICIII_Ground_Truth_2024_05_05.csv'
    else:
        if flag_tfs:
            path_MIMICIII_GT = r'/mnt/r/ENGR_Chon/Darren/Public_Database/PPG_PeakDet_MIMICIII/Darren_conversion/2020_Han_Sensors_MIMICIII_Ground_Truth.csv'
        else:
            path_MIMICIII_GT = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/Adjudication_UConn/MIMICIII/2020_Han_Sensors_MIMICIII_Ground_Truth_2024_05_05.csv'
    df_MIMICIII = pd.read_csv(path_MIMICIII_GT)
    # Convert dataframe into dictionary.
    remain_labels = {}
    for idx,rr in df_MIMICIII.iterrows():
        # ['segment_names', 'labels']
        remain_labels[rr['segment_names']] = rr['labels']

    return remain_labels
