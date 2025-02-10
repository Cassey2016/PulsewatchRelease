# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:34:53 2023

@author: localadmin
"""

import pandas as pd
import os
from pathlib import Path
import shutil

def my_func_move_plots(dest_folder,source_folder,png_appendix,df_input):
    # Make sure path exists.
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    for rr_count,rr_filename in enumerate(df_input.index):
        # print(rr_count,rr_filename)
        source_filename = rr_filename[:32]+png_appendix
        source_all = os.path.join(source_folder,source_filename)
        try:
            shutil.copy2(source_all,dest_folder)
        except IOError:
            print("Unable to copy file",source_filename)
            
r_drive_prefix = r'\\grove.ad.uconn.edu\Research'
path_Beats_Type_Rhythm = os.path.join(r_drive_prefix,r'ENGR_Chon\NIH_Pulsewatch_Database\Parquet_Files\Solo_Beats_Rhythm_Type_count')

get_UID_list = os.path.join(r_drive_prefix,r'ENGR_Chon\NIH_Pulsewatch_Database\mat_for_load_Final_Clinical_Trial_Data')
dir_list_UID = os.listdir(get_UID_list)
first_3_char = list(set([x[:3] for x in dir_list_UID]))
# first_3_char_unique = [x for x in first_3_char if x[0] == '0' or x[0] == '1'] # Dong: I commented out on 05/30/2023 for AF trial UID 408.
first_3_char_unique = [x for x in first_3_char if x[0] == '3' or x[0] == '4'] #
first_3_char_unique.sort()

temp_UIDs = first_3_char_unique
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
        
        # Important: input figure folder.
        source_folder = os.path.join(r_drive_prefix,r'ENGR_Chon\NIH_Pulsewatch_Database\Parquet_Files\Solo_Beats_Plot',UID)
        source_folder_2 = os.path.join(r_drive_prefix,r'ENGR_Chon\Darren\Saved_Plots_Vars_ECG\Bashar_1_inverting\ECG_p_plots',UID)
        png_appendix = '_Solo_Beats.png'
        jpg_appendix = '_p_wave_HR.jpg'
            
        print('------------- UID',UID,'-------------')
        df_input = df_Ventricular_Type.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Type_Ventrivular_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Type_Ventrivular_beats_Bashar_1')

            print('Copying Type-Ventricular Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Type-Ventricular Beats Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_PAC_Type.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Type_PAC_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Type_PAC_beats_Bashar_1')

            print('Copying Type-PAC Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Type-PAC Beats Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_Unclassifed_Type.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Type_Unclassified_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Type_Unclassified_beats_Bashar_1')

            print('Copying Type-Unclassified Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Type-Unclassified Beats Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_AF_Rhythm.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_AF_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_AF_beats_Bashar_1')

            print('Copying Rhythm-AF Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Rhythm-AF Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_Ventricular_Bigeminy_Rhythm.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Vtrclr_Bgmny_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Vtrclr_Bgmny_Bashar_1')

            print('Copying Rhythm-Vtrclr Bgmny Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Rhythm-Vtrclr Bgmny Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_Ventricular_Trigeminy_Rhythm.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Vtrclr_Trgmny_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Vtrclr_Trgmny_Bashar_1')

            print('Copying Rhythm-Vtrclr Trgmny Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Rhythm-Vtrclr Trgmny Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_SVT_Rhythm.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_SVT_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_SVT_Bashar_1')

            print('Copying Rhythm-SVT Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Rhythm-SVT Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_Vent_Tachy_Rhythm.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Vtrclr_Tachy_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Vtrclr_Tachy_Bashar_1')

            print('Copying Rhythm-Vtrclr Tachy Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Rhythm-Vtrclr Tachy Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')
            
        df_input = df_Atrial_Geminy_Rhythm.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Atrl_Geminy_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Atrl_Geminy_Bashar_1')

            print('Copying Rhythm-Atrl_Geminy Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Rhythm-Atrl_Geminy Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')

        df_input = df_Heart_Block_Rhythm.copy()
        if len(df_input) > 0:
            dest_folder = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Heart_Block_beats')
            dest_folder_2 = os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\temp_Solo_Adjudication',UID,'Solo_deemed_Rhythm_Heart_Block_Bashar_1')

            print('Copying Rhythm-Heart Block Beats...')
            my_func_move_plots(dest_folder,source_folder,png_appendix,df_input)
                    
            print('Copying Rhythm-Heart Block Bashar_1...')
            my_func_move_plots(dest_folder_2,source_folder_2,jpg_appendix,df_input)
            
            print('Done')