# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:49:24 2023

@author: localadmin
"""
import pandas as pd
import os
path_Invalid_time = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\Parquet_Files\Solo_ECG_Invalid_count_time'

get_UID_list = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\mat_for_load_Final_Clinical_Trial_Data'
dir_list_UID = os.listdir(get_UID_list)
first_3_char = list(set([x[:3] for x in dir_list_UID]))
first_3_char_unique = [x for x in first_3_char if x[0] == '0' or x[0] == '1']
first_3_char_unique.sort()

thres_time_invalid = 25
for UID in first_3_char_unique:
    filename_Invalid_time = UID+'_Invalid_count_time.parquet'
    if os.path.isfile(os.path.join(path_Invalid_time,filename_Invalid_time)):
        df_ECG_Valid_time = pd.read_parquet(os.path.join(path_Invalid_time,filename_Invalid_time), engine="fastparquet")
        
        df_ECG_Valid_time_transposed = df_ECG_Valid_time.T
        
        df_ECG_Valid_time_transposed = df_ECG_Valid_time_transposed.rename(columns={0: 'Invalid_time'})
        df_temp = df_ECG_Valid_time_transposed[df_ECG_Valid_time_transposed.Invalid_time >= thres_time_invalid]
        
        percentage_invalid = len(df_temp) / len(df_ECG_Valid_time_transposed) * 100
        print('UID',UID,len(df_temp),'/',len(df_ECG_Valid_time_transposed),'(',"{:.2f}".format(percentage_invalid),'%) has >=',thres_time_invalid,'sec of invalid (noisy) ECG (from SOLO software).')
