# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:29:23 2023

@author: localadmin

You only need to run it once. No need to run it again. If you have file here:
    R:\ENGR_Chon\Dong\Python_generated_results\Solo_ECG
"""
import os
import time
import pandas as pd
# path_all_file = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\DAT_files_for_Cardea_Solo\Mail_Kamran_2022_05_26\Clinical_Trial'
path_all_file = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\DAT_files_for_Cardea_Solo\Mail_Kamran_2022_05_26\AF_Trial'
dir_list = os.listdir(path_all_file)
path_output= r'R:\ENGR_Chon\Dong\Python_generated_results\Solo_ECG'
filename_output = 'num_lines_Solo_ECG_AF_trial.csv'
init_df_flag = True
# num_lines_all= []
# filename_Solo_ECG = []
for folder_name in dir_list:
    start = time.time()
    num_lines = sum(1 for line in open(os.path.join(path_all_file,folder_name,'Solo.ECG.txt')))
    # num_lines_all.append(num_lines)
    # filename_Solo_ECG.append(filename)
    end = time.time()
    print('Count lines for',dir_list[0],' Solo.ECG.txt took',end-start,'second.')
    if init_df_flag:
        df = pd.DataFrame([folder_name],columns=['folder_name'])
        df['num_lines'] = num_lines
        init_df_flag = False
    else:
        df.loc[len(df.index)] = [folder_name,num_lines]
        
    print(df)
    df.to_csv(os.path.join(path_output,filename_output),index=False)
    print('Saved dataframe to csv')
