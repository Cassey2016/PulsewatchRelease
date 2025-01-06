import os
# For saving checkpoints
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn
from torchvision.transforms import ToTensor
import socket
# Downsampling image
import cv2
from pyarrow import csv
import random
# import torchvision.transforms as T
# transform for rectangular resize
ppg_size = 1500 # Dong, 05/30/2024: this is for Pulsewatch data length.
flag_debug_plot = False # 06/26/2024, debugging the generated HR.

def my_RandomUpSample_Shuffle(dict_df_train):
    X_fold_1_reshape = list(dict_df_train.keys())
    y_fold_1_parse = list(dict_df_train.values())
    import numpy as np
    import pandas as pd
    X_fold_1_reshape = np.array(X_fold_1_reshape)
    df_y_fold_1_bf_ros = pd.DataFrame({'X_fold_1_reshape':X_fold_1_reshape,'y_fold_1_parse':y_fold_1_parse})
    print('Before upsample:',df_y_fold_1_bf_ros['y_fold_1_parse'].value_counts())
    print('X_fold_1_reshape.shape',X_fold_1_reshape.shape)
    X_fold_1_reshape = np.reshape(X_fold_1_reshape,(-1,1))
    print('X_fold_1_reshape.shape',X_fold_1_reshape.shape)
    
    # === Only here is different from try 06 same name file ===
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42,sampling_strategy='not majority')
    X_fold_1_ros, y_fold_1_ros = ros.fit_resample(X_fold_1_reshape, y_fold_1_parse)

    X_fold_1_ros = np.squeeze(X_fold_1_ros)
    
    df_y_fold_1_ros = pd.DataFrame({'X_fold_1_ros':X_fold_1_ros,'y_fold_1_ros':y_fold_1_ros})
    print(df_y_fold_1_ros['y_fold_1_ros'].value_counts())
    # Let us shuffle the data again. Dong, 03/14/2024, my validation accuracy is always low.
    from sklearn.utils import shuffle
    X_fold_1_ros_shuffle, y_fold_1_ros_shuffle = shuffle(X_fold_1_ros, y_fold_1_ros, random_state=42)

    # I no longer can use dictionary for output format because of the duplication.
    df_train_ros_shuffle = pd.DataFrame({'keys':X_fold_1_ros_shuffle,'values':y_fold_1_ros_shuffle})
    return df_train_ros_shuffle

def split_data_Cassey_ROS_ImageNet(path_train_val_test_split,fold_name):
    # 
    # Splitting the entire dataset into two folds + some pure NSR.
    # Stratified splitting each fold into training, validation, and testing based
    # on subjects (UIDs).
    #
    # Parameter:
    #    - fold_name: fold_1 or fold_2.
    #    - 
    # Output:
    # 
    # /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/ResNet_classification/experiments/try_06_swap_AF/gen_AF/main_04_stratify_split_train_valid_test.ipynb
    
    # The split of UIDs is here:
    # /mnt/r/ENGR_Chon/Dong/Github_private/Pulsewatch_labeling/ResNet_classification/utils/fold_1_2_UIDs_2024_03_30.py

    # # I am still using the subjects and splits from try 06, SMOTE everything.
    # path_train_val_test_split = '/content/drive/MyDrive/Adjudication_UConn/SMOTE_everything_fold_2_2024_03_30'
    filename_train = fold_name+'_train_20240330.csv' # Make sure this fold match with the last cell's output file name.
    # filename_train = fold_name+'_train_aug_2024_07_26.csv' # Merge the origin train and the new aug file name together. Shuffled.
    # filename_train = fold_name+'_train_aug_2024_07_28.csv' # Reduced the aug PACPVC from 30k+ to 5k+. Shuffled.
    print('Training data file name:',filename_train)
    df_train = pd.read_csv(os.path.join(path_train_val_test_split,filename_train))
    filename_valid = fold_name+'_valid_20240330.csv' # Make sure this fold match with the last cell's output file name.
    df_valid = pd.read_csv(os.path.join(path_train_val_test_split,filename_valid))
    filename_test = fold_name+'_test_20240330.csv' # Make sure this fold match with the last cell's output file name.
    df_test = pd.read_csv(os.path.join(path_train_val_test_split,filename_test))

    print('Before my_convert_df_to_dict:',df_train['final_AF_GT_20230921'].value_counts())
    print('Before my_convert_df_to_dict:',df_train.shape)
    def my_convert_df_to_dict(df_train):
        # Convert dataframe into dictionary.
        dict_df_train = {}
        for idx,rr in df_train.iterrows():
            dict_df_train[rr['table_file_name']] = rr['final_AF_GT_20230921']
        return dict_df_train
    
    dict_df_train = my_convert_df_to_dict(df_train)
    print('After my_convert_df_to_dict:',len(dict_df_train))
    dict_df_valid = my_convert_df_to_dict(df_valid)
    dict_df_test = my_convert_df_to_dict(df_test)
    # # For 1D PPG: 004_2019_09_27_10_08_59_ppg_0000_filt.csv
    # filename_tfrecords_fold_1 = data_path+'/'+df_train['table_file_name'].str[:3]+'/'+df_train['table_file_name']+'_filt.csv'
    # # For tfs:
    # filename_tfrecords_fold_1 = data_path+'/'+df_train['table_file_name'].str[:3]+'/'+df_train['table_file_name']+'_tfs_1_mm.tfrecords'
    # # For poin:
    # filename_tfrecords_fold_1 = data_path+'/'+df_train['table_file_name'].str[:3]+'/'+df_train['table_file_name']+'_Poin_0_mm.tfrecords'
    # filename_tfrecords_fold_1 = filename_tfrecords_fold_1.values
    # print(filename_tfrecords_fold_1)
    # y_train_true = df_train['final_AF_GT_20230921'].values

    # # I don't think I need to use tfrecords, so maybe just returning filename will be enough.
    df_train_ros_shuffle = my_RandomUpSample_Shuffle(dict_df_train)
    df_valid_ros_shuffle = my_RandomUpSample_Shuffle(dict_df_valid)
    # I have to unify the use of dataframe instead of dictionary.
    df_test = pd.DataFrame({'keys':list(dict_df_test.keys()),'values':list(dict_df_test.values())})

    return df_train_ros_shuffle, df_valid_ros_shuffle, df_test

def extract_segment_names_and_labels(UIDs,labels_path,read_all_labels=False):
    # 
    # Extract all segment names and labels when starting the main function.
    #
    # Parameters:
    #     - UIDs: all the UIDs that you will run for your training. We will 
    #             use all the segments in this UID.
    #     - labels_path: the ground truth path.
    #     - read_all_labels: True means reading noisy segments as well.
    #
    # Output: 
    #   segment_names: list of string.
    #   labels: dictionary, with segment_names as key and label as value.
    segment_names = []
    labels = {}

    for UID in UIDs:
        label_file = os.path.join(labels_path, UID + "_final_attemp_4_1_Dong.csv")
        if os.path.exists(label_file):
            print('Debug: this file exists',label_file)
            label_data = pd.read_csv(label_file, sep=',', header=0, names=['segment', 'label'])
            label_segment_names = label_data['segment'].apply(lambda x: x.split('.')[0])
            for idx, segment_name in enumerate(label_segment_names):
                label_val = label_data['label'].values[idx]
                if read_all_labels:
                    # Assign -1 if label is not in [0, 1, 2, 3]
                    labels[segment_name] = label_val if label_val in [0, 1, 2, 3] else -1
                    if segment_name not in segment_names:
                        segment_names.append(segment_name)
                else:
                    # Only add segments with labels in [0, 1, 2, 3]
                    if label_val in [0, 1, 2, 3] and segment_name not in segment_names:
                        segment_names.append(segment_name)
                        labels[segment_name] = label_val
    print('>>> Number of segments in this dataloader:',len(segment_names)) # Dong, 01/29/2024: know the number of segments before running training epochs.
    print('>>> Number of labels in this dataloader:',len(labels))
    return segment_names, labels

def remove_finished_segment_names_and_labels(labels,finished_seg_names):
    # From extract_segment_names_and_labels:
    # Input: 
    #   labels: dictionary, with segment_names as key and label as value.
    #   finished_seg_names: list of string.
    remain_labels = labels.copy()
    print('Debug: type(remain_labels)',type(remain_labels))
    for batch in finished_seg_names:
        for key in batch:
            remain_labels.pop(key)
    print('Debug: len(labels)',len(labels))
    print('Debug: len(remain_labels)',len(remain_labels))

    return remain_labels

class CustomDataset(Dataset):
    def __init__(self, data_path, ACC_path, aug_path, labels_path, normalize_type, data_format, segment_names, segment_labels, data_dim):
        self.data_path = data_path
        self.ACC_path = ACC_path
        self.aug_path = aug_path
        self.labels_path = labels_path
        self.normalize_type = normalize_type
        self.data_format = data_format
        self.segment_names = segment_names
        self.segment_labels = segment_labels
        self.data_dim = data_dim
        # Write a set func later if this dimension needs to be changed. 
        self._seg_sec = 30 # The PPG length in second.
        self._seg_fs = 50 # The PPG original sampling frequency.
        
    def __len__(self):
        return len(self.segment_names)

    def __getitem__(self, idx):
        #
        # PyTorch built-in func to load single segment.
        # We are calling the self-made load_data
        segment_name = self.segment_names[idx]
        label = self.segment_labels[idx]

        if hasattr(self, 'all_data') and idx < len(self.all_data):
            # Data is stored in memory
            time_freq_tensor = self.all_data[idx]
        else:
            # Load data on-the-fly based on the segment_name
            time_freq_tensor = self.my_load_data(segment_name)

        return {'data': time_freq_tensor, 'label': label, 'segment_name': segment_name, 'idx': idx}

    def my_load_data(self, segment_name):
        # 
        # Loading single segment using segment name.
        # 
        # Parameters:
        #    - self.data_path: the data path.
        #    - self.data_dim: 1-D or 2-D input. 
        #    - self.data_format: it is csv for 1-D PPG, and pt file for 2-D images.
        #    - self.seg_sec: (for 1-D) the duration in second for 1-D PPG. It should be 30-sec but could be 25-sec.
        #    - self.seg_fs: (for 1-D)the sampling frequency of 1-D PPG.
        #    - self.normalize_type: (for 1-D) standardize or normalize the 1-D PPG.
        # 
        # Output:
        #    - torch_tensor: in 1-D or 2-D (128x128), or a zero value tensor with correct 1-D or 2-D dimension.
        if self.data_dim == '1D_PPG':
            # try:
              # 1D PPG and HR are already stored in the same pt file.
              data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
              seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
              dict_pt = torch.load(seg_path)

              # Load the HR and peaks.
              # print('dict_pt[PPG_filt_csv].shape[1]',dict_pt['PPG_filt_csv'].shape[1])
              if dict_pt['PPG_filt_csv'].shape[1] > 1:
                  # print('Entered shape > 1')
                  PPG_filt_csv = dict_pt['PPG_filt_csv'][:,1]
              else:
                  PPG_filt_csv = dict_pt['PPG_filt_csv']
              
              PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
                  
              if self.normalize_type == 'standardize':
                  torch_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
              elif self.normalize_type == 'zero_one_normalize':
                  torch_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

              # print('Debug dataloader after norm: torch_tensor',torch_tensor)
              # print('Debug dataloader after norm: torch.max(torch_tensor)',torch.max(torch_tensor))
              # print('Debug dataloader after norm: torch.min(torch_tensor)',torch.min(torch_tensor))
            # except Exception as e:
            #     print(f"Error processing segment: {segment_name}. Exception: {str(e)}")
              # return torch.zeros((1, self._seg_sec * self._seg_fs))  # Return zeros in case of an error
              return torch_tensor
        elif self.data_dim == '1D_PPG_HR':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            # print('seg_path',seg_path)
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))
            torch_tensor = torch.stack((PPG_tensor, HR_tensor), dim=1) # (none, 1, 1500) -> (none, 2, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor',torch_tensor)
        elif self.data_dim == '1D_PPG_ACC_only':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']


            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            if PPG_filt_csv.shape[1] > 1:
                PPG_tensor = torch.Tensor(PPG_filt_csv[:,1]).reshape(1,self._seg_sec * self._seg_fs)
            else:
                PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # Loading ACC, 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load ACC.
            if dict_pt['ACC_raw_csv'].shape[1] > 1:
                ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
            else:
                ACC_raw_csv = dict_pt['ACC_raw_csv']

            if segment_name[0] == '6':
                # Simband data
                ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data

            # print('Debug dataloader after norm: torch_tensor',torch_tensor)
            # print('Debug dataloader after norm: torch.max(torch_tensor)',torch.max(torch_tensor))
            # print('Debug dataloader after norm: torch.min(torch_tensor)',torch.min(torch_tensor))

            torch_tensor = torch.stack((PPG_tensor, ACC_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)

        elif self.data_dim == '1D_HR_only':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            # print('seg_path',seg_path)
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.

            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))
            torch_tensor = HR_tensor.clone().detach() # (none, 1, 1500) -> (none, 2, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor',torch_tensor)
        elif self.data_dim == '1D_HR_rescaleHR_only':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_30_220_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            HR_rescale_tensor = self.zero_one_normalization_scaling(HR_tensor) # data own min max rescaling.
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            torch_tensor = torch.stack((HR_30_220_tensor, HR_rescale_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_HR_rescaleHR_ACC_noPPG':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_30_220_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            HR_rescale_tensor = self.zero_one_normalization_scaling(HR_tensor) # data own min max rescaling.
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            # Loading ACC, 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load ACC.
            if dict_pt['ACC_raw_csv'].shape[1] > 1:
                ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
            else:
                ACC_raw_csv = dict_pt['ACC_raw_csv']

            if segment_name[0] == '6':
                # Simband data
                ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data

            # print('Debug dataloader after norm: torch_tensor',torch_tensor)
            # print('Debug dataloader after norm: torch.max(torch_tensor)',torch.max(torch_tensor))
            # print('Debug dataloader after norm: torch.min(torch_tensor)',torch.min(torch_tensor))


            torch_tensor = torch.stack((HR_30_220_tensor, HR_rescale_tensor, ACC_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_PPG_HR_ACC':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            if PPG_filt_csv.shape[1] > 1:
                PPG_tensor = torch.Tensor(PPG_filt_csv[:,1]).reshape(1,self._seg_sec * self._seg_fs)
            else:
                PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            # Loading ACC, 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load ACC.
            if dict_pt['ACC_raw_csv'].shape[1] > 1:
                ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
            else:
                ACC_raw_csv = dict_pt['ACC_raw_csv']

            if segment_name[0] == '6':
                # Simband data
                ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data

            # print('Debug dataloader after norm: torch_tensor',torch_tensor)
            # print('Debug dataloader after norm: torch.max(torch_tensor)',torch.max(torch_tensor))
            # print('Debug dataloader after norm: torch.min(torch_tensor)',torch.min(torch_tensor))

            torch_tensor = torch.stack((PPG_tensor, HR_tensor, ACC_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_HR_ACC':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            # PPG_tensor = torch.Tensor(PPG_filt_csv[:,1]).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            # if self.normalize_type == 'standardize':
            #     PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            # elif self.normalize_type == 'zero_one_normalize':
            #     PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            # Loading ACC, 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load ACC.
            if dict_pt['ACC_raw_csv'].shape[1] > 1:
                ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
            else:
                ACC_raw_csv = dict_pt['ACC_raw_csv']

            if segment_name[0] == '6':
                # Simband data
                ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data
            torch_tensor = torch.stack((HR_tensor, ACC_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_PPG_HR_ACC_rescaleHR':
            # 1D PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            if PPG_filt_csv.shape[1] > 1:
                PPG_tensor = torch.Tensor(PPG_filt_csv[:,1]).reshape(1,self._seg_sec * self._seg_fs)
            else:
                PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_30_220_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            HR_rescale_tensor = self.zero_one_normalization_scaling(HR_tensor) # data own min max rescaling.
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            # Loading ACC, 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load ACC.
            if dict_pt['ACC_raw_csv'].shape[1] > 1:
                ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
            else:
                ACC_raw_csv = dict_pt['ACC_raw_csv']

            if segment_name[0] == '6':
                # Simband data
                ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data
        
            torch_tensor = torch.stack((PPG_tensor, HR_30_220_tensor, ACC_tensor, HR_rescale_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_PPG_aug_HR_ACC_rescaleHR' or self.data_dim == '1D_PPG_aug5k_HR_ACC_rescaleHR':
            # 1D WEPD PPG and HR are already stored in the same pt file.
            # 064_2020_12_08_23_08_21_ppg_0001
            # 113_2021_07_23_05_39_24_ppg_0013_aug_005_010_0...
            if len(segment_name) > 32:
                # The aug pt file.
                data_path_UID = os.path.join(self.aug_path, segment_name.split('_')[0], segment_name[:32]) # Get the UID and the source path name from the segment name.
                seg_path = os.path.join(data_path_UID, segment_name) # Go to the UID folder and append the appendix name to it. 
                dict_pt = torch.load(seg_path)
                
                # Load the HR and peaks.
                PPG_filt_csv = dict_pt['new_PPG_filt_csv']
                new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
                PPG_HR_WEPD = dict_pt['new_PPG_HR']
                ACC_raw_csv = dict_pt['new_ACC_raw_csv']
            else:
                # The original pt file.
                data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
                seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
                dict_pt = torch.load(seg_path)

                # Load the HR and peaks.
                if dict_pt['PPG_filt_csv'].shape[1] > 1:
                    PPG_filt_csv = dict_pt['PPG_filt_csv'][:,1]
                else:
                    PPG_filt_csv = dict_pt['PPG_filt_csv']
                PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
                PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
                PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
                new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
                new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv)
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            HR_WEPD_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_WEPD_30_220_tensor = self.zero_one_normalization_scaling_HR(HR_WEPD_tensor)
            HR_WEPD_rescale_tensor = self.zero_one_normalization_scaling(HR_WEPD_tensor) # data own min max rescaling.
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            if len(segment_name) > 32:
                pass
            else:
                # 1D Elgendi already saved the filt PPG and ACC.
                data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
                seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
                dict_pt = torch.load(seg_path)

                # Load the HR and peaks.
                if dict_pt['ACC_raw_csv'].shape[1] > 1:
                    ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
                else:
                    ACC_raw_csv = dict_pt['ACC_raw_csv']

                if segment_name[0] == '6':
                    # Simband data
                    ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.

            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data
        
            torch_tensor = torch.stack((PPG_tensor, HR_WEPD_30_220_tensor, \
                                        ACC_tensor, HR_WEPD_rescale_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_PPG_ElgendiHR_ACC':
            # 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_Elgendi = dict_pt['PPG_filt_Elgendi']
            PPG_peak_loc_Elgendi_4_2013 = dict_pt['PPG_peak_loc_Elgendi_4_2013']
            HR_Elgendi_4_2013 = dict_pt['HR_Elgendi_4_2013']
            new_PPG_peak_loc_Elgendi = dict_pt['new_PPG_peak_loc_Elgendi']
            new_HR_PPG_Elgendi = dict_pt['new_HR_PPG_Elgendi']
            if dict_pt['ACC_raw_csv'].shape[1] > 1:
                ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
            else:
                ACC_raw_csv = dict_pt['ACC_raw_csv']

            if segment_name[0] == '6':
                # Simband data
                ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.
            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_Elgendi = np.insert(new_PPG_peak_loc_Elgendi, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_Elgendi)):
                start_loc = temp_PPG_peak_loc_Elgendi[ii]
                if ii == len(temp_PPG_peak_loc_Elgendi)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_Elgendi[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = HR_Elgendi_4_2013[0] # Extend the first HR to the beginning.
                elif ii > len(HR_Elgendi_4_2013)-1:
                    PPG_HR_same_len[start_loc:] = HR_Elgendi_4_2013[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = HR_Elgendi_4_2013[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc_Elgendi, PPG_filt_csv[new_PPG_peak_loc_Elgendi,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_Elgendi_4_2013[1:], HR_Elgendi_4_2013, "v-", label='Elgendi HR')
                axs[1].plot(new_PPG_peak_loc_Elgendi[1:], new_HR_PPG_Elgendi, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_ElgendiHR_2024_07_17'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            if PPG_filt_csv.shape[1] > 1:
                PPG_tensor = torch.Tensor(PPG_filt_csv[:,1]).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            else:
                PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs)
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data
        
            torch_tensor = torch.stack((PPG_tensor, HR_tensor, ACC_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_PPG_Elgendi_rescale_HR_ACC':
            # 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_Elgendi = dict_pt['PPG_filt_Elgendi']
            PPG_peak_loc_Elgendi_4_2013 = dict_pt['PPG_peak_loc_Elgendi_4_2013']
            HR_Elgendi_4_2013 = dict_pt['HR_Elgendi_4_2013']
            new_PPG_peak_loc_Elgendi = dict_pt['new_PPG_peak_loc_Elgendi']
            new_HR_PPG_Elgendi = dict_pt['new_HR_PPG_Elgendi']
            if dict_pt['ACC_raw_csv'].shape[1] > 1:
                ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]
            else:
                ACC_raw_csv = dict_pt['ACC_raw_csv']

            if segment_name[0] == '6':
                # Simband data
                ACC_raw_csv = ACC_raw_csv * 9.8 # Convert from gravity to acceleration.
            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_Elgendi = np.insert(new_PPG_peak_loc_Elgendi, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_Elgendi)):
                start_loc = temp_PPG_peak_loc_Elgendi[ii]
                if ii == len(temp_PPG_peak_loc_Elgendi)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_Elgendi[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = HR_Elgendi_4_2013[0] # Extend the first HR to the beginning.
                elif ii > len(HR_Elgendi_4_2013)-1:
                    PPG_HR_same_len[start_loc:] = HR_Elgendi_4_2013[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = HR_Elgendi_4_2013[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc_Elgendi, PPG_filt_csv[new_PPG_peak_loc_Elgendi,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_Elgendi_4_2013[1:], HR_Elgendi_4_2013, "v-", label='Elgendi HR')
                axs[1].plot(new_PPG_peak_loc_Elgendi[1:], new_HR_PPG_Elgendi, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_ElgendiHR_2024_07_17'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.
                
            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            if PPG_filt_csv.shape[1] > 1:
                PPG_tensor = torch.Tensor(PPG_filt_csv[:,1]).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            else:
                PPG_tensor = torch.Tensor(PPG_filt_csv).reshape(1,self._seg_sec * self._seg_fs)
            HR_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_30_220_tensor = self.zero_one_normalization_scaling_HR(HR_tensor)
            HR_rescale_tensor = self.zero_one_normalization_scaling(HR_tensor) # data own min max rescaling.
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_30_220_tensor)',torch.max(HR_30_220_tensor),'torch.min(HR_30_220_tensor)',torch.min(HR_30_220_tensor))
            # print('Debug dataloader after: torch.max(HR_rescale_tensor)',torch.max(HR_rescale_tensor),'torch.min(HR_rescale_tensor)',torch.min(HR_rescale_tensor))
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data
            # print('Debug dataloader after: torch.max(ACC_tensor)',torch.max(ACC_tensor),'torch.min(ACC_tensor)',torch.min(ACC_tensor))
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor),'torch.min(PPG_tensor)',torch.min(PPG_tensor))
            # if torch.any(torch.isnan(PPG_tensor)):
            #     print('Debug PPG_tensor has nan: segment_name',segment_name)
            # if torch.any(torch.isnan(HR_30_220_tensor)):
            #     print('Debug HR_30_220_tensor has nan: segment_name',segment_name)
            # if torch.any(torch.isnan(ACC_tensor)):
            #     print('Debug ACC_tensor has nan: segment_name',segment_name)
            # if torch.any(torch.isnan(HR_rescale_tensor)):
            #     print('HR_30_220_tensor',HR_30_220_tensor)
            #     print('HR_rescale_tensor',HR_rescale_tensor)
            #     print('Debug HR_rescale_tensor has nan: segment_name',segment_name)
            #     # Plot the PPG, peaks, and HR.
            #     import matplotlib.pyplot as plt
            #     fig, axs = plt.subplots(2)
            #     fig.suptitle(segment_name) # Fig title.

            #     axs[0].plot(PPG_filt_csv[:,1])
            #     axs[0].plot(new_PPG_peak_loc_Elgendi, PPG_filt_csv[new_PPG_peak_loc_Elgendi,1], "x", label = 'new peaks')
            #     axs[0].set_ylabel('PPG (a.u.)')
            #     axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

            #     axs[1].plot(PPG_peak_loc_Elgendi_4_2013[1:], HR_Elgendi_4_2013, "v-", label='Elgendi HR')
            #     axs[1].plot(new_PPG_peak_loc_Elgendi[1:], new_HR_PPG_Elgendi, "x-", label='new HR')
            #     axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
            #     axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
            #     axs[1].set_xlabel('30-sec in samples (50 Hz)')
            #     axs[1].set_ylabel('HR (BPM)')

            #     # Save the figure as well.
            #     import pathlib
            #     # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
            #     path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_ElgendiHR_2024_07_17'
            #     UID = segment_name[:3]
            #     pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
            #     filename_fig = segment_name+'_ppg_pk_HR.png'
            #     plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
            #     plt.close() # Close all figs to save memory.
            torch_tensor = torch.stack((PPG_tensor, HR_30_220_tensor, ACC_tensor, HR_rescale_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        elif self.data_dim == '1D_PPG_twoHRs_rescaleHRs_ACC':
            # 1D WEPD PPG and HR are already stored in the same pt file.
            data_path_UID = os.path.join(self.data_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_csv = dict_pt['PPG_filt_csv']
            PPG_filt_WEPD = dict_pt['PPG_filt_WEPD']
            PPG_peak_loc_WEPD = dict_pt['PPG_peak_loc_WEPD']
            PPG_HR_WEPD = dict_pt['PPG_HR_WEPD']
            new_PPG_peak_loc = dict_pt['new_PPG_peak_loc']
            new_HR_PPG = dict_pt['new_HR_PPG']

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_WEPD = np.insert(new_PPG_peak_loc, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_WEPD)):
                start_loc = temp_PPG_peak_loc_WEPD[ii]
                if ii == len(temp_PPG_peak_loc_WEPD)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_WEPD[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = PPG_HR_WEPD[0] # Extend the first HR to the beginning.
                elif ii > len(PPG_HR_WEPD)-1:
                    PPG_HR_same_len[start_loc:] = PPG_HR_WEPD[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = PPG_HR_WEPD[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc, PPG_filt_csv[new_PPG_peak_loc,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_WEPD[1:], PPG_HR_WEPD, "v-", label='WEPD HR')
                axs[1].plot(new_PPG_peak_loc[1:], new_HR_PPG, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_HR_2024_06_26'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            # print('Debug dataloader: PPG_filt_csv',PPG_filt_csv)
            # print('Debug dataloader: PPG_filt_csv.shape',PPG_filt_csv.shape)
            PPG_tensor = torch.Tensor(PPG_filt_csv[:,1]).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500. PPG_filt_csv contains a timestamp column.
            HR_WEPD_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # print('Debug dataloader before: PPG_tensor',PPG_tensor)
            # print('Debug dataloader before: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader before: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # PPG normalization follows the same.
            if self.normalize_type == 'standardize':
                PPG_tensor = self.standard_scaling(PPG_tensor)  # Standardize the data
            elif self.normalize_type == 'zero_one_normalize':
                PPG_tensor = self.zero_one_normalization_scaling(PPG_tensor)  # Standardize the data

            # print('Debug dataloader after: PPG_tensor',PPG_tensor)
            # print('Debug dataloader after: torch.max(PPG_tensor)',torch.max(PPG_tensor))
            # print('Debug dataloader after: torch.min(PPG_tensor)',torch.min(PPG_tensor))

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_WEPD_30_220_tensor = self.zero_one_normalization_scaling_HR(HR_WEPD_tensor)
            HR_WEPD_rescale_tensor = self.zero_one_normalization_scaling(HR_WEPD_tensor) # data own min max rescaling.
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            # 1D Elgendi already saved the filt PPG and ACC.
            data_path_UID = os.path.join(self.ACC_path, segment_name.split('_')[0]) # Get the UID from the segment name.
            seg_path = os.path.join(data_path_UID, segment_name + '_ppg_pk_HR.' + self.data_format) # Go to the UID folder and append the appendix name to it. 
            dict_pt = torch.load(seg_path)

            # Load the HR and peaks.
            PPG_filt_Elgendi = dict_pt['PPG_filt_Elgendi']
            PPG_peak_loc_Elgendi_4_2013 = dict_pt['PPG_peak_loc_Elgendi_4_2013']
            HR_Elgendi_4_2013 = dict_pt['HR_Elgendi_4_2013']
            new_PPG_peak_loc_Elgendi = dict_pt['new_PPG_peak_loc_Elgendi']
            new_HR_PPG_Elgendi = dict_pt['new_HR_PPG_Elgendi']
            ACC_raw_csv = dict_pt['ACC_raw_csv'][:,1]

            # Interpolate the HR with same length of the signal.
            temp_PPG_peak_loc_Elgendi = np.insert(new_PPG_peak_loc_Elgendi, 0, 0, axis=0) # Insert the first index, 0, for pk loc.
            PPG_HR_same_len = np.zeros(len(PPG_filt_csv), dtype = np.float16)
            for ii in range(len(temp_PPG_peak_loc_Elgendi)):
                start_loc = temp_PPG_peak_loc_Elgendi[ii]
                if ii == len(temp_PPG_peak_loc_Elgendi)-1:
                    end_loc = len(PPG_filt_csv)-1 # Same length of PPG, 1499.
                else:
                    end_loc = temp_PPG_peak_loc_Elgendi[ii+1] # The next peak location.
                if ii == 0:
                    PPG_HR_same_len[:end_loc] = HR_Elgendi_4_2013[0] # Extend the first HR to the beginning.
                elif ii > len(HR_Elgendi_4_2013)-1:
                    PPG_HR_same_len[start_loc:] = HR_Elgendi_4_2013[-1] # Extend the last HR to the end.
                else:
                    PPG_HR_same_len[start_loc:end_loc] = HR_Elgendi_4_2013[ii-1]
            if flag_debug_plot:
                # Plot the PPG, peaks, and HR.
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                fig.suptitle(segment_name) # Fig title.

                axs[0].plot(PPG_filt_csv[:,1])
                axs[0].plot(new_PPG_peak_loc_Elgendi, PPG_filt_csv[new_PPG_peak_loc_Elgendi,1], "x", label = 'new peaks')
                axs[0].set_ylabel('PPG (a.u.)')
                axs[0].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.9, 1.1)) # Horizontally display legend.

                axs[1].plot(PPG_peak_loc_Elgendi_4_2013[1:], HR_Elgendi_4_2013, "v-", label='Elgendi HR')
                axs[1].plot(new_PPG_peak_loc_Elgendi[1:], new_HR_PPG_Elgendi, "x-", label='new HR')
                axs[1].plot(PPG_HR_same_len, label='new HR interpolated')
                axs[1].legend(loc="upper right", ncol=2,bbox_to_anchor=(0.5, 1.1)) # Horizontally display legend.
                axs[1].set_xlabel('30-sec in samples (50 Hz)')
                axs[1].set_ylabel('HR (BPM)')

                # Save the figure as well.
                import pathlib
                # path_output_fig = r'/mnt/r/ENGR_Chon/Dong/Python_generated_results/deep_learning_2023/gen_new_PPG_HR_2024_06_26'
                path_output_fig = r'/content/drive/MyDrive/Checkpoint_Colab/TestRNNGRU_HR/gen_new_PPG_ElgendiHR_2024_07_17'
                UID = segment_name[:3]
                pathlib.Path(os.path.join(path_output_fig, UID)).mkdir(parents = True, exist_ok = True)
                filename_fig = segment_name+'_ppg_pk_HR.png'
                plt.savefig(os.path.join(path_output_fig, UID, filename_fig), dpi=100, bbox_inches='tight')
                plt.close() # Close all figs to save memory.

            # Prepare torch tensor.
            HR_Elgendi_tensor = torch.Tensor(PPG_HR_same_len).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.
            ACC_tensor = torch.Tensor(ACC_raw_csv).reshape(1,self._seg_sec * self._seg_fs) # It should be 1500.

            # HR normalization is fixed. 220 BPM to 30 BPM.
            # print('Debug dataloader before: HR_tensor',HR_tensor)
            # print('Debug dataloader before: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader before: torch.min(HR_tensor)',torch.min(HR_tensor))
            HR_Elgendi_30_220_tensor = self.zero_one_normalization_scaling_HR(HR_Elgendi_tensor)
            HR_Elgendi_rescale_tensor = self.zero_one_normalization_scaling(HR_Elgendi_tensor) # data own min max rescaling.
            # print('Debug dataloader after: HR_tensor',HR_tensor)
            # print('Debug dataloader after: torch.max(HR_tensor)',torch.max(HR_tensor))
            # print('Debug dataloader after: torch.min(HR_tensor)',torch.min(HR_tensor))

            # ACC normalization is 
            ACC_tensor = self.zero_one_normalization_scaling_ACC(ACC_tensor)  # Standardize the data
        

            torch_tensor = torch.stack((PPG_tensor, HR_WEPD_30_220_tensor, HR_WEPD_rescale_tensor, \
                                        HR_Elgendi_30_220_tensor, HR_Elgendi_rescale_tensor, ACC_tensor), dim=1) # (none, 1, 1500) -> (none, 3, 1500) Not sure which dimension I should add the tensors together.
            # print('Debug dataloader: torch_tensor', torch_tensor)
            # print('Debug dataloader: PPG_tensor', PPG_tensor)
            # print('Debug dataloader: HR_WEPD_30_220_tensor', HR_WEPD_30_220_tensor)
            # print('Debug dataloader: HR_WEPD_rescale_tensor', HR_WEPD_rescale_tensor)
            # print('Debug dataloader: HR_Elgendi_30_220_tensor', HR_Elgendi_30_220_tensor)
            # print('Debug dataloader: HR_Elgendi_rescale_tensor', HR_Elgendi_rescale_tensor)
            # print('Debug dataloader: ACC_tensor', ACC_tensor)
            # print('Debug dataloader: torch_tensor.shape',torch_tensor.shape)
        return torch_tensor.clone()

    def standard_scaling(self, data):
        #
        # Zero mean and unit variance standardization within each 1-D segment.
        #
        scaler = sklearn.preprocessing.StandardScaler()
        data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        return torch.Tensor(data)
    
    def zero_one_normalization_scaling(self, data):
        #
        # Zero to one normalization within each 1-D segment. DeepBeat method.
        #
        # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        # print('Debug scaler before:',data)
        # # data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        # data = scaler.fit_transform(data)
        # print('Debug scaler after:',data)

        v_min, v_max = data.min(), data.max()
        new_min = 0
        new_max = 1
        if v_min == v_max:
            # Avoid dividing zero.
            print('Debug: avoid dividing zero')
            v_p = (data - v_min)/((v_max - v_min)+ 10**-10)*(new_max - new_min) + new_min
        else:
            v_p = (data - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return torch.Tensor(v_p)
    
    def zero_one_normalization_scaling_HR(self, data):
        #
        # Zero to one normalization within each 1-D segment. DeepBeat method.
        #
        v_min, v_max = 30, 220
        new_min = 0
        new_max = 1
        v_p = (data - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return torch.Tensor(v_p)
    
    def zero_one_normalization_scaling_ACC(self, data):
        """
        ACC is in unit 9.8 m/s^2 in Samsung Galaxy Watch. 
        """
        v_min, v_max = 0, 20 # 0 m/s2 to 20 m/s2, which is -1 G to 2 G. 
        new_min = 0
        new_max = 1
        v_p = (data - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return torch.Tensor(v_p)
    
    @property
    def reset_sec(self):
        # Protect _seg_sec
        return self._seg_sec
    @reset_sec.setter
    def reset_sec(self,new_seg_sec):
        # Assign new value to _seg_sec.
        self._seg_sec = new_seg_sec

    @property
    def reset_fs(self):
        # Protect _seg_fs
        return self._seg_fs
    @reset_fs.setter
    def reset_fs(self,new_seg_fs):
        # Assign new value to _seg_fs.
        self._seg_fs = new_seg_fs

    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'segment_names': self.segment_names,
            'segment_labels': self.segment_labels
            # Save the current batch index if provided
        }
        torch.save(checkpoint, checkpoint_path)
        # print('Debug: dataset checkpoint saved!',checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print('Debug: loaded dataset checkpoint!',checkpoint_path)
        self.segment_names = checkpoint['segment_names']
        self.segment_labels = checkpoint['segment_labels']
        self.refresh_dataset()
        # Load the current batch index if it exists in the checkpoint

def my_PyTorchCustomDatasetDataLoader(fold_name, dict_df_train,
                            dict_paths, normalize_type, data_format, data_dim,
                            batch_size,
                            drop_last=False, num_workers=4,\
                            finished_seg_names = []):
    # 
    # Check if resume from checkpoint segment names or start from beginning.
    #
    # Parameters:
    #    - dict_paths: all the paths that will be used, including 'data_path' and 'labels_path'.
    #    - normalize_type: 'standardize' or 'zero_one_normalize' the signal. 
    #    - data_format: 'csv' for 1-D, or 'pt' for 2-D. 
    # Run the main from the beginning. Load all data into the dataloader.
    data_path = dict_paths['data_path']
    labels_path = dict_paths['labels_path']
    ACC_path = dict_paths['ACC_path']
    aug_path = dict_paths['aug_path']
    # Prep the same name variables.
    segment_names = list(dict_df_train.keys()) # Only getting the segment name.
    dict_labels = dict_df_train
    if len(finished_seg_names) > 0:
        # If any segments have been trained.
        remain_labels = remove_finished_segment_names_and_labels(dict_labels,finished_seg_names)
        segment_names = list(remain_labels.keys())
        dict_labels = remain_labels.copy()
    dataset = CustomDataset(data_path, ACC_path, aug_path, labels_path, normalize_type, data_format, segment_names, dict_labels, data_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, prefetch_factor=2)
    return dataloader, dataset

def my_PyTorchCustomDatasetDataLoader_df(fold_name, df_train_ros_shuffle,
                            dict_paths, normalize_type, data_format, data_dim,
                            batch_size,
                            drop_last=False, num_workers=4,\
                            finished_seg_names = []):
    # 
    # Check if resume from checkpoint segment names or start from beginning.
    #
    # Parameters:
    #    - dict_paths: all the paths that will be used, including 'data_path' and 'labels_path'.
    #    - normalize_type: 'standardize' or 'zero_one_normalize' the signal. 
    #    - data_format: 'csv' for 1-D, or 'pt' for 2-D. 
    # Run the main from the beginning. Load all data into the dataloader.
    data_path = dict_paths['data_path']
    labels_path = dict_paths['labels_path']
    ACC_path = dict_paths['ACC_path']
    aug_path = dict_paths['aug_path']
    # Prep the same name variables.
    segment_names = list(df_train_ros_shuffle['keys']) # Only getting the segment name.
    segment_labels = list(df_train_ros_shuffle['values'])
    if len(finished_seg_names) > 0:
        # If any segments have been trained.
        remain_segment_names = segment_names[len(finished_seg_names)-1:] # pick up where left off.
        remain_segment_labels = segment_labels[len(finished_seg_names)-1:]
        df_train_ros_shuffle_trim = pd.DataFrame({'keys':remain_segment_names, 'values':remain_segment_labels})
    else:
        # No segments were trained before.
        remain_segment_names = segment_names
        remain_segment_labels = segment_labels
        df_train_ros_shuffle_trim = df_train_ros_shuffle.copy()
    dataset = CustomDataset(data_path, ACC_path, aug_path, labels_path, normalize_type, data_format, remain_segment_names, remain_segment_labels, data_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, prefetch_factor=2)
    return dataloader, dataset


# Function to extract and preprocess data
def load_train_valid_dataloader(fold_name, df_train, df_valid,
                              dict_paths, normalize_type, data_format, data_dim,
                              batch_size, finished_seg_names):
    # Extracts paths and loads data into train, validation, and test loaders
    train_loader, train_dataset = my_PyTorchCustomDatasetDataLoader_df(fold_name, df_train, 
                                            dict_paths, normalize_type, data_format, data_dim,
                                            batch_size,
                                            drop_last=False, num_workers=4,\
                                            finished_seg_names = finished_seg_names)
    
    # Usually the validation set will not need to resume training.
    val_loader, val_dataset = my_PyTorchCustomDatasetDataLoader_df(fold_name, df_valid,
                                            dict_paths, normalize_type, data_format, data_dim,
                                            batch_size,
                                            drop_last=False, num_workers=4,\
                                            finished_seg_names = [])
    return train_loader, val_loader, train_dataset, val_dataset

# Function to extract and preprocess data
def load_test_dataloader(fold_name, df_test,
                              dict_paths, normalize_type, data_format, data_dim,
                              batch_size, num_workers):
    # Extracts paths and loads data into train, validation, and test loaders
    test_loader, test_dataset = my_PyTorchCustomDatasetDataLoader_df(fold_name, df_test,
                                            dict_paths, normalize_type, data_format, data_dim,
                                            batch_size,
                                            drop_last=False, num_workers=num_workers,\
                                            finished_seg_names = [])
    return test_loader, test_dataset

def load_Simband_groundtruth(path_train_val_test_split):

    filename_test = 'simband_segments_labels_2024_04_25.csv' # Columns: ['segment_names', 'labels']
    df_test = pd.read_csv(os.path.join(path_train_val_test_split,filename_test))

    df_new_GT_trim = df_test.loc[(df_test['labels'] == 0) | \
                            (df_test['labels'] == 1) | \
                            (df_test['labels'] == 2)]
    df_test = df_new_GT_trim.copy()

    print('Before my_convert_df_to_dict:',df_test['labels'].value_counts())
    print('Before my_convert_df_to_dict:',df_test.shape)
    def my_convert_df_to_dict(df_train):
        # Convert dataframe into dictionary.
        dict_df_train = {}
        for idx,rr in df_train.iterrows():
            dict_df_train[rr['segment_names']] = rr['labels']
        return dict_df_train
    
    dict_df_test = my_convert_df_to_dict(df_test)
    print('After my_convert_df_to_dict:',len(dict_df_test))

    # I have to unify the use of dataframe instead of dictionary.
    df_test = pd.DataFrame({'keys':list(dict_df_test.keys()),'values':list(dict_df_test.values())})

    return df_test

def load_Simband_test_dataloader(fold_name, df_test,
                              dict_paths, normalize_type, data_format, data_dim,
                              batch_size, num_workers):
    # Extracts paths and loads data into train, validation, and test loaders
    test_loader, test_dataset = my_PyTorchCustomDatasetDataLoader_df(fold_name, df_test,
                                            dict_paths, normalize_type, data_format, data_dim,
                                            batch_size,
                                            drop_last=False, num_workers=num_workers,\
                                            finished_seg_names = [])
    return test_loader, test_dataset

def load_MIMICIII_20240809_groundtruth(path_train_val_test_split):

    filename_test = 'MIMICIII_Dong_20240809.csv' # Columns: ['table_file_name', 'mat_file_name','GT_PPG_noisy_AF','GT_PPG_numeric']
    df_test = pd.read_csv(os.path.join(path_train_val_test_split,filename_test))

    df_new_GT_trim = df_test.loc[(df_test['GT_PPG_numeric'] == 0) | \
                            (df_test['GT_PPG_numeric'] == 1) | \
                            (df_test['GT_PPG_numeric'] == 2)]
    df_test = df_new_GT_trim.copy()

    print('Before my_convert_df_to_dict:',df_test['GT_PPG_numeric'].value_counts())
    print('Before my_convert_df_to_dict:',df_test.shape)
    def my_convert_df_to_dict(df_train):
        # Convert dataframe into dictionary.
        dict_df_train = {}
        for idx,rr in df_train.iterrows():
            dict_df_train[rr['table_file_name']] = rr['GT_PPG_numeric']
        return dict_df_train
    
    dict_df_test = my_convert_df_to_dict(df_test)
    print('After my_convert_df_to_dict:',len(dict_df_test))

    # I have to unify the use of dataframe instead of dictionary.
    df_test = pd.DataFrame({'keys':list(dict_df_test.keys()),'values':list(dict_df_test.values())})

    return df_test