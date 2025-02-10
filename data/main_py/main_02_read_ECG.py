# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:20:35 2023

@author: localadmin

Ideas:
    1. Know the PPG search time interval;
    2. Know the ECG patch start time;
    3. Know the ECG non-linear timestamp (from Solo.ASCII.txt);
    4. Search the 150k sample blocks first using the PPG time interval;
    5. Interpolate the ECG non-linear timestamp, locate the precise rows of ECG to load;
    6. Load the ECG from Solo.ECG.txt.
    7. Load the ECG peaks and beat labels.
"""

from itertools import islice
import os
import sys
sys.path.append(r'..\func_py')
from utils import my_func_UID_ECG_final_path
import datetime
import pytz
import re
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

def my_interp_Solo_ECG(start_blc_idx,\
                       fs_ECG,\
                       Solo_time_intv,\
                       ECG_init_datetime):

    if start_blc_idx == 1:
        # Warning: make sure this starts from 1 and not end with start_blc_idx - 1.
        x_Sample_Value = np.array((1,start_blc_idx * fs_ECG*60*10)) # Every 10-min sample.
        v = [float(0),float(Solo_time_intv[start_blc_idx-1])]
    else:
        x_Sample_Value = np.array((start_blc_idx,start_blc_idx+1)) * (fs_ECG*60*10) # Every 10-min sample.
        v = [float(tt) for tt in Solo_time_intv[start_blc_idx:start_blc_idx+2]]
        
    xq_Linear_Interp = np.arange(x_Sample_Value[0],x_Sample_Value[1]+1) # It should be 150001, not 150000.
    
    
    f = interpolate.interp1d(x_Sample_Value, np.array(v), fill_value="extrapolate") # Must put fill_value="extrapolate"
    vq_Linear_Interp = f(xq_Linear_Interp) # interpolated PPG.
    
    sample_datetime = [ECG_init_datetime + datetime.timedelta(milliseconds=float(tt)*1000) for tt in vq_Linear_Interp]
    
    return sample_datetime,xq_Linear_Interp

def my_extract_Solo_ASCII(file_contents):
    lastname_firstname = re.search(r'<Lastname,Firstname>(.*?)</Lastname,Firstname>', file_contents, re.DOTALL).group(1).strip()
    start_date_duration = re.search(r'<Start_Date_Time,Duration\(Sec\)>(.*?)</Start_Date_Time, Duration\(Sec\)>', file_contents, re.DOTALL).group(1).strip()
    sample_interval_time = re.search(r'<150000_Sample_Interval_Time>(.*?)</150000_Sample_Interval_Time>', file_contents, re.DOTALL).group(1).strip()
    patient_events = re.search(r'<Patient_Events>(.*?)</Patient_Events>', file_contents, re.DOTALL).group(1).strip()
    
    sample_interval_time = sample_interval_time.split('\n')
    return lastname_firstname, \
            start_date_duration, \
            sample_interval_time, \
            patient_events
UID = '005'
HPC_flag = False
root_data_path = r'R:\ENGR_Chon\NIH_Pulsewatch_Database'
root_output_path = r'R:\ENGR_Chon\NIH_Pulsewatch_Database'
test_ECG_path_A,\
    Patch_A_start_time,\
    test_ECG_path_B,\
    Patch_B_start_time,\
    test_ECG_path_C,\
    Patch_C_start_time,\
    UMass_type,\
    LinearInterp_root = my_func_UID_ECG_final_path(UID,\
                           HPC_flag,\
                           root_data_path,\
                           root_output_path)

fs_ECG = 250

our_tzone = pytz.timezone('America/New_York') # This line does not exist in MATLAB code. I do not want to pass it as a variable. 01/19/2023.



# =============================================================================
# Load non-linear timestamp from Solo.ASCII.txt
# =============================================================================
PPG_timestamp_start = our_tzone.localize(datetime.datetime.strptime('09/16/2019 14:59:30', '%m/%d/%Y %H:%M:%S'))
PPG_timestamp_end = our_tzone.localize(datetime.datetime.strptime('09/16/2019 15:00:00', '%m/%d/%Y %H:%M:%S'))

path_Solo_ECG = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\DAT_files_for_Cardea_Solo\Mail_Kamran_2022_05_26\Clinical_Trial'
dir_list = os.listdir(path_Solo_ECG)

patch_A_Solo_ECG_name = [x for x in dir_list if UID+'A' in x]
patch_B_Solo_ECG_name = [x for x in dir_list if UID+'B' in x]
patch_C_Solo_ECG_name = [x for x in dir_list if UID+'C' in x]
path_numlines_Solo_ECG = r'R:\ENGR_Chon\Dong\Python_generated_results\Solo_ECG'

df_2 = pd.read_csv(os.path.join(path_numlines_Solo_ECG,'num_lines_Solo_ECG.csv'))

if len(Patch_A_start_time) > 0:
    Patch_A_datetime = our_tzone.localize(datetime.datetime.strptime(Patch_A_start_time, '%m/%d/%Y %H:%M:%S.%f'))
    df_this_Solo = df_2.loc[df_2['folder_name'].str.contains(patch_A_Solo_ECG_name[0], case=False)]
    num_lines = df_this_Solo['num_lines'].values
    test_ECG_path_A = os.path.join(path_Solo_ECG,df_this_Solo['folder_name'].values[0])
    patch_A_dur = num_lines/fs_ECG # Unit: seconds.
    temp_diff_time = PPG_timestamp_start - Patch_A_datetime
    flag_in_patch_A = (temp_diff_time.total_seconds() >= 0 and temp_diff_time.total_seconds() <= patch_A_dur)
else:
    print('Patch A does not exist!')
    patch_A_dur = 0
    flag_in_patch_A = False
    
if len(Patch_B_start_time) > 0:
    Patch_B_datetime = our_tzone.localize(datetime.datetime.strptime(Patch_B_start_time, '%m/%d/%Y %H:%M:%S.%f'))
    df_this_Solo = df_2.loc[df_2['folder_name'].str.contains(patch_B_Solo_ECG_name[0], case=False)]
    num_lines = df_this_Solo['num_lines'].values
    test_ECG_path_B = os.path.join(path_Solo_ECG,df_this_Solo['folder_name'].values[0])
    patch_B_dur = num_lines/fs_ECG # Unit: seconds.
    temp_diff_time = PPG_timestamp_start - Patch_B_datetime
    flag_in_patch_B = (temp_diff_time.total_seconds() >= 0 and temp_diff_time.total_seconds() <= patch_B_dur)
else:
    print('Patch B does not exist!')
    Patch_B_datetime = []
    patch_B_dur = 0
    flag_in_patch_B = False
    
if len(Patch_C_start_time) > 0:
    Patch_C_datetime = our_tzone.localize(datetime.datetime.strptime(Patch_C_start_time, '%m/%d/%Y %H:%M:%S.%f'))
    df_this_Solo = df_2.loc[df_2['folder_name'].str.contains(patch_C_Solo_ECG_name[0], case=False)]
    num_lines = df_this_Solo['num_lines'].values
    test_ECG_path_C = os.path.join(path_Solo_ECG,df_this_Solo['folder_name'].values[0])
    patch_C_dur = num_lines/fs_ECG # Unit: seconds.
    temp_diff_time = PPG_timestamp_start - Patch_C_datetime
    flag_in_patch_C = (temp_diff_time.total_seconds() >= 0 and temp_diff_time.total_seconds() <= patch_C_dur)
else:
    print('Patch C does not exist!')
    Patch_C_datetime = []
    patch_C_dur = 0
    flag_in_patch_C = False
    
if flag_in_patch_A and flag_in_patch_B:
    # Same PPG segment falls in two ECG patches:
    print('Same PPG segment falls in two ECG patches!')
    ECG_path = test_ECG_path_B # for UID 036, 038, 039, just use the second patch ECG.
    ECG_Date_string = Patch_B_start_time # I will assume
else:
    if flag_in_patch_B and flag_in_patch_C:
        # same PPG segment falls in two ECG patches:
        print('Same PPG segment falls in two ECG patches!')
        ECG_path = test_ECG_path_C # for UID 028, just use the third patch ECG.
        ECG_Date_string = Patch_C_start_time # I will assume
    elif flag_in_patch_A:
        # within the duration of patch A. 
        ECG_path = test_ECG_path_A;
        ECG_Date_string = Patch_A_start_time # I will assume
    else:
        # see if patch B exists:
        if isinstance(Patch_B_datetime, datetime.date):
            if flag_in_patch_B:
                ECG_path = test_ECG_path_B
                ECG_Date_string = Patch_B_start_time # I will assume
            else:
                # maybe there is patch C?
                print('PPG falls outside ECG time!') 
                ECG_path = [];
                ECG_Date_string = [];
        else:
            print('PPG falls inside patch B, but patch B does not exist!')
            ECG_path = [] # For AF trial, UID 301 has PPG started before ECG.
            ECG_Date_string = []
    
print(ECG_path)
print(ECG_Date_string)
ECG_init_datetime = our_tzone.localize(datetime.datetime.strptime(ECG_Date_string, '%m/%d/%Y %H:%M:%S.%f'))
# clinical_ECG_root = os.path.join(root_data_path,'DAT_files_for_Cardea_Solo','Mail_Kamran_2022_05_26','Clinical_Trial')
# 
# matches = [x for x in my_dictionary_sentence if x in dir_list]
filename_Solo_ECG = 'Solo.ASCII.txt'
from bs4 import BeautifulSoup
# with open(os.path.join(ECG_path,filename_Solo_ECG)) as f:
#     text = f.read()
# soup = BeautifulSoup(text, 'lxml')

mylines = []
with open(os.path.join(ECG_path,filename_Solo_ECG), 'rt') as myfile:  # 'r' for reading, 't' for text mode.
    file_contents = myfile.read()
    lastname_firstname, \
            start_date_duration, \
            sample_interval_time, \
            patient_events = my_extract_Solo_ASCII(file_contents)
    
Solo_time_intv = sample_interval_time
blocks_datetime = [ECG_init_datetime + datetime.timedelta(milliseconds=float(tt)*1000) for tt in Solo_time_intv] 

# =============================================================================
# Search the block to be used
# =============================================================================
start_blc_idx = 1 # It should start from one, not zero. 
for idx in range(1,len(blocks_datetime)):
    if PPG_timestamp_start >= blocks_datetime[idx-1] and PPG_timestamp_start <= blocks_datetime[idx]:
        start_blc_idx = idx-1
        break

sample_datetime,xq_Linear_Interp = my_interp_Solo_ECG(start_blc_idx,\
                       fs_ECG,\
                       Solo_time_intv,\
                       ECG_init_datetime)

# Check if end PPG time is in another ECG.
start_blc_idx_for_end = 1 # It should start from one, not zero. 
for idx in range(1,len(blocks_datetime)):
    if PPG_timestamp_end >= blocks_datetime[idx-1] and PPG_timestamp_end <= blocks_datetime[idx]:
        start_blc_idx_for_end = idx-1
        break

if not start_blc_idx == start_blc_idx_for_end:
    if start_blc_idx_for_end > start_blc_idx:
        sample_datetime_for_end,\
            xq_Linear_Interp_for_end = my_interp_Solo_ECG(start_blc_idx_for_end,\
                               fs_ECG,\
                               Solo_time_intv,\
                               ECG_init_datetime)
        sample_datetime.append(sample_datetime_for_end)
        xq_Linear_Interp = np.concatenate((xq_Linear_Interp,xq_Linear_Interp_for_end))
        print('PPG span two ECG sample blocks, concatenated two.')
    else:
        print('End PPG index is in a sample block earlier than start PPG index, check!')    
# =============================================================================
# find the ECG segment in the text file.
# =============================================================================
start_spl_idx = 1 # It should start from one, not zero. 
for idx in range(1,len(sample_datetime)):
    if PPG_timestamp_start >= sample_datetime[idx-1] and PPG_timestamp_start <= sample_datetime[idx]:
        start_spl_idx = idx-1
        break

start_spl_idx_txt = start_spl_idx + xq_Linear_Interp[0]

end_spl_idx = 1 # It should start from one, not zero. 
for idx in range(1,len(sample_datetime)):
    if PPG_timestamp_end >= sample_datetime[idx-1] and PPG_timestamp_end <= sample_datetime[idx]:
        end_spl_idx = idx-1
        break
    
end_spl_idx_txt = end_spl_idx + xq_Linear_Interp[0]

filename_Solo_ECG = 'Solo.ECG.txt'
# this_txt_Solo_ECG = []
Valid_Solo_ECG_txt = []
raw_Solo_ECG_txt = []
# Remember, first row start with <Valid_Sample,Data>
with open(os.path.join(ECG_path,filename_Solo_ECG)) as f:
    # for line in islice(f,1, 10): # seq, [start,] stop [, step]
    for line in islice(f,start_spl_idx_txt, end_spl_idx_txt+1): # seq, [start,] stop [, step]
        x = line.strip()
        columns = re.split(',',x)#(r'   ',x)# https://stackoverflow.com/questions/48917121/split-on-more-than-one-space
        if len(columns) > 1:
            Valid_Solo_ECG_txt.append(columns[0])              # Read the entire file to a string
            raw_Solo_ECG_txt.append(int(columns[1]))
        else:
            print('Solo ECG only has one col, check!')
            print('Line between',start_spl_idx_txt,'and',end_spl_idx_txt,', Content:',line)
            Valid_Solo_ECG_txt.append(columns[0])              # Read the entire file to a string
            raw_Solo_ECG_txt.append([])

raw_Solo_ECG_array = np.array(raw_Solo_ECG_txt)
raw_Solo_ECG_datetime = sample_datetime[start_spl_idx:end_spl_idx+1]


# Filter the signal
[b,a] = signal.butter(3,0.4 /(fs_ECG/2),'highpass')
ECG_highpass_data = signal.filtfilt(b,a,raw_Solo_ECG_array)
[b,a] = signal.butter(3,20/(fs_ECG/2),'lowpass') #25 /(fs_ECG/2),'low');
ECG_bandpass_data = signal.filtfilt(b,a,ECG_highpass_data)
ECG_bandpass_data = ECG_bandpass_data - np.mean(ECG_bandpass_data);
ECG_filtered_data = ECG_bandpass_data / abs(np.std(ECG_bandpass_data));

HR1_color = [0, 0.176, 0.8] # Dark blue
HR2_color = [0.098, 0.439, 0] # Dark green
HR1_100BPM = [1, 0.2, 0.231] # Light red
HR2_100BPM = [0.235, 0.839, 0] # Light green
HR1_tachy_color = [0.690, 0.250, 0.266] # Dark red
HR2_tachy_color = [0.019, 1, 0.043] # Bright green
    
figsrc = plt.figure()
ax1 = plt.subplot(211)
# my_title = PPG_file_name
# plt.title(my_title,fontsize=20)
plt.grid()
# line1, = ax1.plot(pk_loc[1:],HR,'o-',color=HR1_color,label=data_path_GT[(-ext+1):]+' HR')
# 
line1, = ax1.plot(raw_Solo_ECG_datetime,ECG_filtered_data,color=HR2_color,label='ECG raw')
ax1.set_ylabel('a.u.',fontsize=20)
        
plt.show()
# %matplotlib auto