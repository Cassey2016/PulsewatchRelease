# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:34:53 2023

@author: localadmin
Dong, revised it on 10/12/2024. 
"""

import pandas as pd
import os
# path_Beats_Type_Rhythm = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\Parquet_Files\Solo_Beats_Rhythm_Type_count'
path_Beats_Type_Rhythm = r'/mnt/r/ENGR_Chon/NIH_Pulsewatch_Database/Parquet_Files/Solo_Beats_Rhythm_Type_count'

# get_UID_list = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\mat_for_load_Final_Clinical_Trial_Data'
get_UID_list = r'/mnt/r/ENGR_Chon/NIH_Pulsewatch_Database/mat_for_load_Final_Clinical_Trial_Data'
dir_list_UID = os.listdir(get_UID_list)
first_3_char = list(set([x[:3] for x in dir_list_UID]))
# first_3_char_unique = [x for x in first_3_char if x[0] == '0' or x[0] == '1'] # Dong: I commented out on 05/30/2023 for AF trial UID 408.
first_3_char_unique = [x for x in first_3_char if x[0] == '3' or x[0] == '4'] #
first_3_char_unique.sort()

temp_UIDs = first_3_char_unique[36:37] # Dong, 05/30/2023: only for UID 408.
thres_beats = 3
for UID in temp_UIDs:
# if True:
#     UID = '017'
    filename_Beats_Type_Rhythm = UID+'_Beats_Rhythm_Type.parquet'
    df_input = []
    if os.path.isfile(os.path.join(path_Beats_Type_Rhythm,filename_Beats_Type_Rhythm)):
        df_Beats_Rhythm_Type = pd.read_parquet(os.path.join(path_Beats_Type_Rhythm,filename_Beats_Type_Rhythm), engine="fastparquet")
        
        df_Beats_Rhythm_Type_transposed = df_Beats_Rhythm_Type.T
        
        df_Beats_Rhythm_Type_transposed = df_Beats_Rhythm_Type_transposed.rename(columns={0: 'count_Type_N',\
                                                                                    1: 'count_Type_V',\
                                                                                    2: 'count_Type_A',\
                                                                                    3: 'count_Type_Q',\
                                                                                    4: 'count_Type_Z',\
                                                                                    5: 'count_Type_unseen',\
                                                                                    6: 'count_Rhythm_N',\
                                                                                    7: 'count_Rhythm_A',\
                                                                                    8: 'count_Rhythm_B',\
                                                                                    9: 'count_Rhythm_T',\
                                                                                    10: 'count_Rhythm_S',\
                                                                                    11: 'count_Rhythm_V',\
                                                                                    12: 'count_Rhythm_G',\
                                                                                    13: 'count_Rhythm_H',\
                                                                                    14: 'count_Rhythm_unseen'})
        df_Ventricular_Type = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Type_V >= thres_beats]
        df_PAC_Type = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Type_A >= thres_beats]
        df_Unclassifed_Type = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Type_Q >= thres_beats]
        df_AF_Rhythm = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Rhythm_A >= thres_beats]
        df_Ventricular_Bigeminy_Rhythm = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Rhythm_B >= thres_beats]
        df_Ventricular_Trigeminy_Rhythm = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Rhythm_T >= thres_beats]
        df_SVT_Rhythm = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Rhythm_S >= thres_beats]
        df_Vent_Tachy_Rhythm = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Rhythm_V >= thres_beats]
        df_Atrial_Geminy_Rhythm = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Rhythm_G >= thres_beats]
        df_Heart_Block_Rhythm = df_Beats_Rhythm_Type_transposed[df_Beats_Rhythm_Type_transposed.count_Rhythm_H >= thres_beats]
        
        
        print('------------- UID',UID,'-------------')
        df_input = df_Ventricular_Type.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Type: Ventricular beats:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_PAC_Type.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Type: PAC beats:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_Unclassifed_Type.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Type: Unclassified beats:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_AF_Rhythm.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Rhythm: AF beats:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_Ventricular_Bigeminy_Rhythm.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Rhythm: Ventricular bigeminy:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_Ventricular_Trigeminy_Rhythm.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Rhythm: Ventricular trigeminy:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_SVT_Rhythm.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Rhythm: SVT:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_Vent_Tachy_Rhythm.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Rhythm: Ventricular tachycardia:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')
            
        df_input = df_Atrial_Geminy_Rhythm.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Rhythm: Atrial geminy:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')

        df_input = df_Heart_Block_Rhythm.copy()
        if len(df_input) > 0:
            percentage_temp = len(df_input) / len(df_Beats_Rhythm_Type_transposed) * 100
            print('Rhythm: Heart block:',len(df_input),'/',len(df_Beats_Rhythm_Type_transposed),\
                  '(',"{:.2f}".format(percentage_temp),'%) has >=',thres_beats,'beats from SOLO software.')