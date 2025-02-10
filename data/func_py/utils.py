# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:01:56 2023

@author: localadmin
"""
import os
from scanf import scanf # In Windows Anaconda, I installed pip install scanf.
import pandas as pd
import numpy as np
import re # For main_01_read_PPG.py
import datetime
import pytz
from scipy import interpolate
import time
from pathlib import Path
import matplotlib.pyplot as plt

def my_func_log_txt_read(test_PPG_path,curr_log_file_name):
    # test_PPG_path = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\Final_Clinical_Trial_Data\110_final'
    # curr_log_file_name = '110_2021_07_01_10_05_02_log.txt'
    # Read a log file, extract the API info.
    mylines = [] 
    
    # Output
    max_seg = 30 # maximum of segment inside one log file is less than 30 segments.
    m_op = [None] * max_seg
    m_HR_DATPD = [None] * max_seg
    m_HR_SWEPD = [None] * max_seg
    m_HR_WEPD = [None] * max_seg
    m_comb = [None] * max_seg
    m_RMSSD = [None] * max_seg
    m_SampEn = [None] * max_seg
    m_IsAF_1 = [None] * max_seg # Mr. Cho has 3 different IsAF sentence.
    m_IsAF_2 = [None] * max_seg
    m_IsAF_3 = [None] * max_seg
    m_PACPVC_pred_1 = [None] * max_seg # Mr. Cho has 2 different PACPVC pred sentence.
    m_PACPVC_pred_2 = [None] * max_seg
    m_FastAF_1 = [None] * max_seg # Mr. Cho has 2 different FastAF sentence.
    m_FastAF_2 = [None] * max_seg
    
    m_HR_1 = [None] * max_seg # Mr. Cho has 3 different HR sentence.
    m_HR_2 = [None] * max_seg
    m_HR_3 = [None] * max_seg
    
    m_seg_num = [None] * max_seg
    m_index = [None] * max_seg
    m_mSensorEnable = [None] * max_seg # msensorenable is double.
    m_seg_start_row_idx = [None] * max_seg
    m_seg_end_row_idx = [None] * max_seg # time inside each segment.
    
    m_API_ver = [None] * max_seg # API is string.
    m_service_var = [None] * max_seg # service is string.
    
    m_storage_total_size = [None] * max_seg
    m_storage_avail_size = [None] * max_seg
    m_battery_perc = [None] * max_seg
        
    m_time = []
    m_time_idx = []
    count_row = 0 # count current row.
    count_seg = -1 # start with one
    my_dictionary_sentence = ['segment number is ',#1
               'heart rate(WEPD) is ',#2
               'comb is ',#3
               'Rmssd_mean is ',#4
               'SampEn is ',#5
               'IsAF is ',#6
               'heart rate(DATPD) is ',#7
               'heart rate(SWEPD) is ',#8
               'PVCPAC flag is ',#9
               'fast_AF_flag is ',#10
               'index:',#11
               'PACPVC_predict_label ',#12
               'Noise ',#13
               'HR is ',#14
               'Fast_AF_flag is ',#15
               'API Version Number is ',#16
               'Service Version is ',#17
               'PPG:hr is ',#18
               'afib is ',#19
               'mSensorEnable is ',#20
               'PPG:mHR is ',#21
               'mAFib is ',#22
               'check_storage : Total : ',#23
               ', Available : ',#24
               'Percentage : ']#25
    
    my_scanf_format = ['%d',#1
                       '%f',#2
                       '%f',#3
                       '%f',#4
                       '%f',#5
                       '%d',#6
                       '%f',#7
                       '%f',#8
                       '%d',#9
                       '%d',#10
                       '%d',#11
                       '%d',#12
                       '%d',#13
                       '%f',#14
                       '%d',#15
                       '%s',#16
                       '%s',#17
                       '%f',#18
                       '%d',#19
                       '%d',#20
                       '%f',#21
                       '%d',#22
                       '%f',#23
                       '%f',#24
                       '%f']#25
    
    with open (os.path.join(test_PPG_path,curr_log_file_name), 'rt') as myfile:  # 'r' for reading, 't' for text mode.
        for myline in myfile:
            mylines.append(myline)              # Read the entire file to a string
            
            matches = [x for x in my_dictionary_sentence if x in myline]
            # any(x in myline for x in my_dictionary_sentence):
            if len(matches): # Not empty
                for dict_str in matches:
                    dict_idx = my_dictionary_sentence.index(dict_str) # Use the index to the scanf
                    temp = scanf(dict_str+my_scanf_format[dict_idx],myline)
                    if temp == None:
                        temp = [np.nan]
                    if dict_idx == 0:
                        count_seg += 1
                        m_seg_num[count_seg] = temp[0]
                        m_seg_start_row_idx[count_seg] = count_row
                        
                    if dict_idx > 0 and count_seg < 0:
                        # no 'segment' sentence before the first segment:
                        count_seg += 1 # I will ignore index is not accurately recorded. 08/20/2020.
                        
                    if dict_idx == 1:
                        m_HR_WEPD[count_seg] = temp[0]
                    if dict_idx == 2:
                        m_comb[count_seg] = temp[0]
                    if dict_idx == 3:
                        m_RMSSD[count_seg] = temp[0]
                    if dict_idx == 4:
                        m_SampEn[count_seg] = temp[0]
                    if dict_idx == 5:
                        m_IsAF_1[count_seg] = temp[0]
                    if dict_idx == 6:
                        m_HR_DATPD[count_seg] = temp[0]
                    if dict_idx == 7:
                        m_HR_SWEPD[count_seg] = temp[0]
                    if dict_idx == 8:
                        m_PACPVC_pred_1[count_seg] = temp[0]
                    if dict_idx == 9:
                        m_FastAF_1[count_seg] = temp[0]
                    if dict_idx == 10:
                        m_index[count_seg] = temp[0]
                    if dict_str == 'PACPVC_predict_label': # 12 (11 in Python)
                        if temp is None:
                            temp = scanf(dict_str+'is '+my_scanf_format[dict_idx],myline)
                        m_PACPVC_pred_2[count_seg] = temp[0]
                    if dict_idx == 12:
                        m_op[count_seg] = temp[0]
                    if dict_idx == 13:
                        m_HR_1[count_seg] = temp[0]
                    if dict_idx == 14:
                        m_FastAF_2[count_seg] = temp[0]
                    if dict_idx == 15:
                        m_API_ver[count_seg] = temp[0]
                    if dict_idx == 16:
                        m_service_var[count_seg] = temp[0]
                    if dict_idx == 17:
                        m_HR_2[count_seg] = temp[0]
                    if dict_idx == 18:
                        m_IsAF_2[count_seg] = temp[0]
                    if dict_idx == 19:
                        m_mSensorEnable[count_seg] = temp[0]
                        m_seg_end_row_idx[count_seg] = count_row
                    if dict_idx == 20:
                        m_HR_3[count_seg] = temp[0]
                    if dict_idx == 21:
                        m_IsAF_3[count_seg] = temp[0]
                    if dict_idx == 22:
                        m_storage_total_size[count_seg] = temp[0]
                        if count_seg == 0:
                            count_seg = -1 # reset the pointer to first index.
                        else:
                            print('Not beginning of log file has count seg, check!')
                    if dict_idx == 23:
                        m_storage_avail_size[count_seg] = temp
                        if count_seg == 0:
                            count_seg = -1 # reset the pointer to first index.
                        else:
                            print('Not beginning of log file has count seg, check!')
                    if dict_idx == 24:
                        m_battery_perc[count_seg] = temp
                        if count_seg == 0:
                            count_seg = -1 # reset the pointer to first index.
                        else:
                            print('Not beginning of log file has count seg, check!')
            
            time_last_idx = myline.find(']') # time ends with ']' like '11:50:00]'
            if time_last_idx >= 0: # If not found str, find() return -1
                m_time.append(myline[:time_last_idx])
                m_time_idx.append(count_row)
            
            count_row += 1
                
    
    
    # clean up arrays:
    del m_op[count_seg+1:] # after these number of seg, remove them.
    
    del m_HR_DATPD[count_seg+1:]
    del m_HR_SWEPD[count_seg+1:]
    del m_HR_WEPD[count_seg+1:]
    del m_comb[count_seg+1:]
    del m_RMSSD[count_seg+1:]
    del m_SampEn[count_seg+1:]
    del m_IsAF_1[count_seg+1:]
    del m_IsAF_2[count_seg+1:]
    del m_IsAF_3[count_seg+1:]
    del m_PACPVC_pred_1[count_seg+1:]
    del m_PACPVC_pred_2[count_seg+1:]
    del m_FastAF_1[count_seg+1:]
    del m_FastAF_2[count_seg+1:]
    
    del m_HR_1[count_seg+1:]
    del m_HR_2[count_seg+1:]
    del m_HR_3[count_seg+1:]
    
    del m_seg_num[count_seg+1:]
    del m_index[count_seg+1:]
    del m_mSensorEnable[count_seg+1:]
    
    del m_API_ver[count_seg+1:] # not sure if cell can work like this.
    del m_service_var[count_seg+1:]
    
    del m_seg_start_row_idx[count_seg+1:]
    del m_seg_end_row_idx[count_seg+1:]
    
    del m_storage_total_size[count_seg+1:]
    del m_storage_avail_size[count_seg+1:]
    del m_battery_perc[count_seg+1:]
    
    m_output_struct = pd.DataFrame(m_op)
    m_output_struct['m_HR_DATPD'] = m_HR_DATPD
    m_output_struct['m_HR_SWEPD'] = m_HR_SWEPD
    m_output_struct['m_HR_WEPD'] = m_HR_WEPD
    m_output_struct['m_comb'] = m_comb
    m_output_struct['m_RMSSD'] = m_RMSSD
    m_output_struct['m_SampEn'] = m_SampEn
    m_output_struct['m_IsAF_1'] = m_IsAF_1
    m_output_struct['m_IsAF_2'] = m_IsAF_2
    m_output_struct['m_IsAF_3'] = m_IsAF_3
    m_output_struct['m_PACPVC_pred_1'] = m_PACPVC_pred_1
    m_output_struct['m_PACPVC_pred_2'] = m_PACPVC_pred_2
    m_output_struct['m_FastAF_1'] = m_FastAF_1
    m_output_struct['m_FastAF_2'] = m_FastAF_2
    m_output_struct['m_HR_1'] = m_HR_1
    m_output_struct['m_HR_2'] = m_HR_2
    m_output_struct['m_HR_3'] = m_HR_3
    m_output_struct['m_seg_num'] = m_seg_num
    m_output_struct['m_index'] = m_index
    m_output_struct['m_mSensorEnable'] = m_mSensorEnable
    m_output_struct['m_seg_start_row_idx'] = m_seg_start_row_idx
    m_output_struct['m_seg_end_row_idx'] = m_seg_end_row_idx
    m_output_struct['m_storage_total_size'] = m_storage_total_size
    m_output_struct['m_storage_avail_size'] = m_storage_avail_size
    m_output_struct['m_battery_perc'] = m_battery_perc
    
    # I decided to put them in the same dataframe, easier to store in pandas H5
    m_output_struct['m_API_ver'] = m_API_ver
    m_output_struct['m_service_var'] = m_service_var
    # m_output_struct['m_time'] = m_time
    # m_output_struct['m_time_idx'] = m_time_idx
    
    return m_output_struct, m_time, m_time_idx, mylines


def my_func_load_acc_txt_after_ppg(path_output,UID):
    # UID='036'
    # path_output = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\py_for_load_Final_Clinical_Trial_Data'
    # store = pd.HDFStore(UID+'_after_ppg_load_acc.h5')
    with pd.HDFStore(os.path.join(path_output,UID+'_after_ppg_load_acc.h5')) as store:
        df = store['df']  # load it
        # df_restored = newstore.select('df')
    # store.close()
    return df


def my_func_read_m_log(PPG_file_name,test_PPG_path):
    curr_log_name = PPG_file_name[:23]
    curr_log_file_name = curr_log_name+'_log.txt'
    print(curr_log_file_name)
    
    if os.path.isfile(os.path.join(test_PPG_path,curr_log_file_name)):
        log_file_empty = False
        m_output_struct,\
            m_time, \
            m_time_idx,\
            mylines = my_func_log_txt_read(test_PPG_path,
                                               curr_log_file_name)
    else:
        # log file does not exists:
        # 045_2020_08_24_12_11_20_log.txt_temp_88
        dir_list = os.listdir(test_PPG_path)
        subs = curr_log_name+'_log'
        log_file_temp = list(filter(lambda x: subs in x, dir_list))
        
        if len(log_file_temp):
            if len(log_file_temp) == 1:
                curr_log_file_name = log_file_temp[0] # I hope it is a str, not the first str of the list.
                log_file_empty = False
                m_output_struct,\
                    m_time, \
                    m_time_idx,\
                    mylines = my_func_log_txt_read(test_PPG_path,
                                                       curr_log_file_name)
            else:
                print('There are more than one log file for this time!')
        else:
            # there is really no such log file exists!
            log_file_empty = True
            
    if log_file_empty == False:
        return m_output_struct, m_time, m_time_idx, mylines
    else:
        m_output_struct = pd.DataFrame()
        m_time = []
        m_time_idx = []
        mylines = []
        return m_output_struct, m_time, m_time_idx, mylines


def my_importdata_as_np(path_file,file_name):
    mylines = []
    
    # Check if file exists.
    if file_name is None:
        # UID 043, seg 1.
        data_buffer = np.empty((0,))#[] # Dong, 03/01/2023: I did not pick np.nan in case I use len() to check empty buffer.
        mylines = np.empty((0,)) #[]
    else:
        with open(os.path.join(path_file,file_name), 'rt') as myfile:  # 'r' for reading, 't' for text mode.
            for myline in myfile:
                x = myline.strip()
                columns = re.split(r'\s{1,}',x)#(r'   ',x)# https://stackoverflow.com/questions/48917121/split-on-more-than-one-space
                mylines.append(columns)              # Read the entire file to a string
        
        # Flatten the text to numpy array
        if len(mylines) > 0:
            data_buffer = np.empty((len(mylines),len(mylines[0])))
            for idx_r, myline in enumerate(mylines):
                for idx_c, mystr in enumerate(myline):
                    res = any(mychr.isdigit() for mychr in mystr)
                    if res:
                        if mystr[-1] == 'e':
                            if idx_r-1 >= 0:
                                print('Debug: mylines[idx_r-1][idx_c]',mylines[idx_r-1][idx_c])
                                prev_str = mylines[idx_r-1][idx_c]
                                mystr = mystr + prev_str.split('e')[1]
                        elif mystr[-1] == '+':
                            if idx_r-1 >= 0:
                                print('Debug: mylines[idx_r-1][idx_c]',mylines[idx_r-1][idx_c])
                                prev_str = mylines[idx_r-1][idx_c]
                                mystr = mystr + prev_str.split('+')[1]
                        data_buffer[idx_r,idx_c] = np.array(mystr,dtype=np.float64)# longdouble
                    else:
                        print('Debug: mystr does not contain any digit:',mystr)
        else:
            data_buffer = np.empty((0,))#[] # Dong, 03/01/2023: I did not pick np.nan in case I use len() to check empty buffer.
            mylines = np.empty((0,)) #[]
    return data_buffer,mylines
    
def my_func_interpolate_timestamp_PPG(diff_PPG_t_msec,\
                                        fs_PPG,\
                                        PPG_t_datetime,\
                                        PPG_timestamp,\
                                        PPG_raw_buffer):
    # Check the type of diff_PPG_t_msec
    if isinstance(diff_PPG_t_msec[0],datetime.timedelta):
        temp_diff = [temp.total_seconds() * 1000 for temp in diff_PPG_t_msec]
    else:
        print('diff_PPG_t_msec content is not datetime')
        temp_diff = [temp * 1000 for temp in diff_PPG_t_msec] # Dong, 04/08/2023: I assume this is in msec so I timed 1000.
    cumsum_PPT_t_msec = np.cumsum(np.array(temp_diff)) # cumulative sum between sampling time (msec)
    rm_idx = np.argwhere(cumsum_PPT_t_msec < 0)
    PPG_t_datetime = np.delete(PPG_t_datetime,rm_idx,axis=0)
    PPG_timestamp = np.delete(PPG_timestamp,rm_idx,axis=0)
    PPG_raw_buffer = np.delete(PPG_raw_buffer,rm_idx,axis=0)
    cumsum_PPT_t_msec = np.delete(cumsum_PPT_t_msec,rm_idx,axis=0)
    # Linear PPG time msec steps:
    linear_t_msec = np.arange(0,cumsum_PPT_t_msec[-1],1,dtype=int) # 01/19/2023: double check if needs to by dtype=int
    
    # Prepare for PPG interpolation:
    x, ix = np.unique(cumsum_PPT_t_msec,return_index=True)
    v = PPG_raw_buffer[1:] # PPG is one sample longer than diff.
    v = v[ix] # keep unique value index.
    xq = linear_t_msec
    
    f = interpolate.interp1d(x, v, fill_value="extrapolate") # Must put fill_value="extrapolate"
    vq1 = f(xq) # interpolated PPG.
    # ideal unisampled PPG msec index:
    ideal_t_msec = np.arange(0,cumsum_PPT_t_msec[-1],1/fs_PPG*1000,dtype=int) # 30sec into msec.
    ideal_PPG = vq1[ideal_t_msec] # downsample interpolated PPG. 01/22/2023: I do not think I should add the 1 index.
    
    # 01/22/2023: Will I have nan entries like ideal_t_msec and ideal_PPG does in MATLAB?
    # I did not translate between lines 164 to 166 in MATLAB.
    # sometimes ideal_PPG is less than 30*fs_PPG long, so need to prune it.
    # 01/30/2023: Like UID 041, seg 20(21 in MATLAB), 041_2020_07_15_15_30_54_ppg_0000
    if ideal_PPG.shape[0] < fs_PPG * 30:
        # have more sample points than expected.
        add_pt = fs_PPG * 30 - ideal_PPG.shape[0]
        ideal_PPG = np.concatenate((ideal_PPG,np.ones((add_pt,))*ideal_PPG[-1]))
        ideal_t_msec = np.concatenate((ideal_t_msec,np.arange(ideal_t_msec[-1]+1,ideal_t_msec[-1] + abs(add_pt),1,dtype=int)))# I am only adding 1 millisecond by a time.
    elif ideal_PPG.shape[0] > fs_PPG * 30:
        # have fewer sample points than expected.
        remove_pt = ideal_PPG.shape[0] - fs_PPG * 30
        ideal_PPG = ideal_PPG[:-abs(remove_pt)] 
        ideal_t_msec = ideal_t_msec[:-abs(remove_pt)] 
        
    return PPG_t_datetime,\
            PPG_raw_buffer,\
            PPG_timestamp,\
            ideal_PPG,\
            ideal_t_msec
    
    
    
def my_func_interpolate_watch_data(PPG_file_buffer,fs_PPG,\
                                   flag_ACC=None,\
                                   PPG_timestamp_ver_2_start_datetime=None,\
                                   PPG_timestamp_ver_2_start_msec=None):

    our_tzone = pytz.timezone('America/New_York') # This line does not exist in MATLAB code. I do not want to pass it as a variable. 01/19/2023.
    
    new_ver_flag = False
    if PPG_file_buffer.shape[1] > 3: # More than one PPG timestamp.
        PPG_timestamp = PPG_file_buffer[:,3] # 4th column.
        PPG_raw_buffer = PPG_file_buffer[:,1] # 2nd column.
        PPG_timestamp_ver_2_raw = PPG_file_buffer[:,0] # 1st column.
        PPG_timestamp_ver_2 = PPG_timestamp_ver_2_raw - PPG_timestamp_ver_2_start_msec # reset the start
        new_ver_flag = True # ver 2.0.1 testing.
    else:
        if not flag_ACC == None: # I did not pass any value in when calling this func.
            if flag_ACC:
                if PPG_file_buffer.shape[1] > 2:
                    PPG_timestamp = PPG_file_buffer[:,2] # 3rd column.
                    PPG_raw_buffer = PPG_file_buffer[:,1] # 2nd column.
                    PPG_timestamp_ver_2_raw = PPG_file_buffer[:,0] # 1st column.
                    PPG_timestamp_ver_2 = PPG_timestamp_ver_2_raw - PPG_timestamp_ver_2_start_msec # reset the start
                    new_ver_flag = True # ver 2.0.1 testing.
                else:
                    PPG_timestamp = PPG_file_buffer[:,0] # 1st column
                    PPG_raw_buffer = PPG_file_buffer[:,1] # 2nd column
                    PPG_timestamp_ver_2_raw = None # 01/19/2023: not sure if I should name it as [] or None.
            else:
                PPG_timestamp = PPG_file_buffer[:,0] # 1st column
                PPG_raw_buffer = PPG_file_buffer[:,1] # 2nd column
                PPG_timestamp_ver_2_raw = None # 01/19/2023: not sure if I should name it as [] or None.
        else: # in case 'flag_ACC' was not input.
            PPG_timestamp = PPG_file_buffer[:,0] # 1st column
            PPG_raw_buffer = PPG_file_buffer[:,1] # 2nd column
            PPG_timestamp_ver_2_raw = None # 01/19/2023: not sure if I should name it as [] or None.
    
    ## Epoch timestamp cleanup:
    # convert Epoch time (msec) to MATLAB datetime (msec):
    PPG_t_datetime = [our_tzone.localize(datetime.datetime.fromtimestamp(tt/1000)) for tt in PPG_timestamp] # From Unix time to datetime.
    diff_PPG_t_msec = [PPG_t_datetime[ii] - PPG_t_datetime[ii-1] for ii in range(1,len(PPG_t_datetime))] # calculate the sampling time (msec) between two samples.
    PPG_t_datetime,PPG_raw_buffer,PPG_timestamp,ideal_PPG,ideal_t_msec = my_func_interpolate_timestamp_PPG(diff_PPG_t_msec,fs_PPG,PPG_t_datetime,PPG_timestamp,PPG_raw_buffer)
    if PPG_t_datetime.shape[0] < ideal_PPG.shape[0]:
        add_pt = fs_PPG * 30 - PPG_t_datetime.shape[0]
        temp_timestamp = np.arange(1,abs(add_pt)+1,dtype=int) # not include stop, so plus one. 03/18/2023.
        temp_add_array = np.array([PPG_t_datetime[-1] + datetime.timedelta(milliseconds=int(tt)) for tt in temp_timestamp]) # I am not sure if PPG_timestamp_start should be added.
        # ((PPG_t_datetime(end)+milliseconds(1)):milliseconds(1):(PPG_t_datetime(end)+milliseconds(abs(add_pt))))'
        PPG_t_datetime = np.concatenate((PPG_t_datetime,temp_add_array))
    elif PPG_t_datetime.shape[0] > ideal_PPG.shape[0]:
        remove_pt = PPG_t_datetime.shape[0] - fs_PPG * 30
        PPG_t_datetime = PPG_t_datetime[:-abs(remove_pt)] # Remove the last few points.
        
    # CLOCK_MONOTONIC timestamp cleanup:
    if new_ver_flag:
        diff_PPG_t_msec_ver_2 = np.diff(PPG_timestamp_ver_2) # the unit is already millisecond.
        # Compare two timestamps
        temp_t_msec = [temp.total_seconds() * 1000 for temp in diff_PPG_t_msec]
        PPG_timestamp_match = np.array_equiv(temp_t_msec,diff_PPG_t_msec_ver_2) # Check if two arrays are equal.
        if not PPG_timestamp_match:
            print('New timestamp does not match old timestamp, check!')
            # for iiii,tttt in enumerate(temp_t_msec):
            #     if not diff_PPG_t_msec_ver_2[iiii] == tttt:
            #         print(iiii,diff_PPG_t_msec_ver_2[iiii],tttt)
        # cumsum_PPT_t_msec_ver_2 = np.cumsum(diff_PPG_t_msec)
        PPG_t_datetime_ver_2 = np.array([PPG_timestamp_ver_2_start_datetime + datetime.timedelta(milliseconds=tt) for tt in PPG_timestamp_ver_2]) # I am not sure if PPG_timestamp_start should be added.
        PPG_t_datetime_ver_2,_,PPG_timestamp_ver_2,ideal_PPG_ver_2,ideal_t_msec_ver_2 = my_func_interpolate_timestamp_PPG(diff_PPG_t_msec_ver_2,fs_PPG,PPG_t_datetime_ver_2,PPG_timestamp_ver_2,PPG_raw_buffer)
        
        if PPG_t_datetime_ver_2.shape[0] < ideal_PPG_ver_2.shape[0]:
            add_pt = fs_PPG * 30 - PPG_t_datetime_ver_2.shape[0]
            temp_timestamp = np.arange(1,abs(add_pt)+1,dtype=int) # 
            temp_add_array = np.array([PPG_t_datetime_ver_2[-1] + datetime.timedelta(milliseconds=int(tt)) for tt in temp_timestamp]) # I am not sure if PPG_timestamp_start should be added.
            # ((PPG_t_datetime(end)+milliseconds(1)):milliseconds(1):(PPG_t_datetime(end)+milliseconds(abs(add_pt))))'
            PPG_t_datetime_ver_2 = np.concatenate((PPG_t_datetime_ver_2,temp_add_array))
        elif PPG_t_datetime_ver_2.shape[0] > ideal_PPG.shape[0]:
            remove_pt = PPG_t_datetime_ver_2.shape[0] - fs_PPG * 30
            PPG_t_datetime_ver_2 = PPG_t_datetime_ver_2[:-abs(remove_pt)] # Remove the last few points. I did differently with MATLAB code line 104. 01/28/2023.
    else:
        ideal_t_msec_ver_2 = []
        PPG_t_datetime_ver_2 = []
        PPG_timestamp_ver_2_raw = []

    return ideal_PPG,\
            ideal_t_msec,\
            PPG_t_datetime,\
            ideal_t_msec_ver_2,\
            PPG_t_datetime_ver_2,\
            PPG_timestamp_ver_2_raw
    
    
def my_func_check_interpolation(PPG_timestamp,\
                                fs_PPG,\
                                PPG_raw_buffer):
    
    our_tzone = pytz.timezone('America/New_York') # This line does not exist in MATLAB code. I do not want to pass it as a variable. 01/19/2023.
    # convert Epoch time (msec) to MATLAB datetime (msec):
    PPG_t_datetime = [our_tzone.localize(datetime.datetime.fromtimestamp(tt/1000)) for tt in PPG_timestamp] # From Unix time to datetime.
    # PPG_t_datetime = datetime(PPG_timestamp./1000,'ConvertFrom','posixTime','Format','dd-MMM-yyyy HH:mm:ss.SSS','TimeZone','America/New_York')#  
    diff_PPG_t_msec = [PPG_t_datetime[ii] - PPG_t_datetime[ii-1] for ii in range(1,len(PPG_t_datetime))] # calculate the sampling time (msec) between two samples.
    
    PPG_t_datetime,\
        PPG_raw_buffer,\
        PPG_timestamp,\
        ideal_PPG,\
        ideal_t_msec = my_func_interpolate_timestamp_PPG(diff_PPG_t_msec,\
                                                            fs_PPG,\
                                                            PPG_t_datetime,\
                                                            PPG_timestamp,\
                                                            PPG_raw_buffer)
    # fixing PPG_t_datetime is shorter than ideal PPG length.
    if PPG_t_datetime.shape[0] < ideal_PPG.shape[0]:
        
        add_pt = fs_PPG * 30 - PPG_t_datetime.shape[0]
        temp_timestamp = np.arange(1,abs(add_pt)+1,dtype=int) # 
        temp_add_array = np.array([PPG_t_datetime[-1] + datetime.timedelta(milliseconds=int(tt)) for tt in temp_timestamp]) # I am not sure if PPG_timestamp_start should be added.
        # ((PPG_t_datetime(end)+milliseconds(1)):milliseconds(1):(PPG_t_datetime(end)+milliseconds(abs(add_pt))))'
        PPG_t_datetime = np.concatenate((PPG_t_datetime,temp_add_array))
    elif PPG_t_datetime.shape[0] > ideal_PPG.shape[0]:
        remove_pt = PPG_t_datetime.shape[0] - fs_PPG * 30
        PPG_t_datetime = PPG_t_datetime[:-abs(remove_pt)] # Remove the last few points. I did differently with MATLAB code line 125 in my_func_interpolate_watch_data_2_1_1. 01/29/2023.
        # R:\ENGR_Chon\Dong\MATLAB\Pulsewatch\Cassey_working\Pulsewatch_alignment\func\my_func_interpolate_watch_data_2_1_1.m

    return PPG_t_datetime,\
            PPG_raw_buffer,\
            PPG_timestamp,\
            ideal_PPG,\
            ideal_t_msec

def my_func_interpolate_watch_data_2_1_1(PPG_file_buffer,fs_PPG,flag_ACC):

    if not flag_ACC:
        PPG_timestamp = PPG_file_buffer[:,3] # 4th column.
        PPG_raw_buffer = PPG_file_buffer[:,1] # 2nd column.
        PPG_timestamp_ver_2_raw = PPG_file_buffer[:,0] # 1st column. Now it is also in Epoch time.
    else:
        ACC_file_buffer = PPG_file_buffer
        ACC_timestamp = ACC_file_buffer[:,4] # 5th column. ACC timestamp
        ACC_x_buffer = ACC_file_buffer[:,1] # In watch service 2.1.1 we have x,y,z.
        ACC_y_buffer = ACC_file_buffer[:,2]
        ACC_z_buffer = ACC_file_buffer[:,3]
        
        ACC_raw_buffer = np.sqrt(np.square(ACC_x_buffer) + np.square(ACC_y_buffer) + np.square(ACC_z_buffer)) # ACC raw signal.
        ACC_timestamp_ver_2_raw = ACC_file_buffer[:,0] # 1st column. unit is millisecond.

        PPG_timestamp = ACC_timestamp
        PPG_raw_buffer = ACC_raw_buffer
        PPG_timestamp_ver_2_raw = ACC_timestamp_ver_2_raw

    # Do the same thing for ver 2 timestamp (Must run it first due to 'PPG_raw_buffer' repeated named):
    PPG_t_datetime_ver_2,\
        PPG_raw_buffer_ver_2,\
        PPG_timestamp_ver_2,\
        ideal_PPG_ver_2,\
        ideal_t_msec_ver_2 = my_func_check_interpolation(PPG_timestamp_ver_2_raw,fs_PPG,PPG_raw_buffer)
    # Epoch timestamp cleanup:
    PPG_t_datetime,\
        PPG_raw_buffer,\
        PPG_timestamp,\
        ideal_PPG,\
        ideal_t_msec = my_func_check_interpolation(PPG_timestamp,fs_PPG,PPG_raw_buffer)
    return ideal_PPG,\
            ideal_t_msec,\
            PPG_t_datetime,\
            ideal_t_msec_ver_2,\
            PPG_t_datetime_ver_2,\
            PPG_timestamp_ver_2_raw
            
def my_func_ver_2_timestamp_start_datetime(UID):
    our_tzone = pytz.timezone('America/New_York') # This line does not exist in MATLAB code. I do not want to pass it as a variable. 01/19/2023.

    if UID == '913_02042021':
        # The start time of first 30-sec segment.
        PPG_timestamp_ver_2_start_datetime_string = '02/04/2021 14:36:26.544' # 24 hours time.
        PPG_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(PPG_timestamp_ver_2_start_datetime_string,\
                                                      '%m/%d/%Y %H:%M:%S.%f'))
        PPG_timestamp_ver_2_start_msec = 1882515 #milisecond.
        
        ACC_timestamp_ver_2_start_datetime_string = '02/04/2021 14:36:26.543' # 24 hours time.
        ACC_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(ACC_timestamp_ver_2_start_datetime_string,\
                                                       '%m/%d/%Y %H:%M:%S.%f'))
        ACC_timestamp_ver_2_start_msec = 1882515 # milisecond. ACC is same as PPG.
    elif UID == '914_2021042601':
        # The start time of first 30-sec segment.
        PPG_timestamp_ver_2_start_datetime_string = '04/26/2021 11:27:00.042' # 24 hours time.
        PPG_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(PPG_timestamp_ver_2_start_datetime_string,\
                                                      '%m/%d/%Y %H:%M:%S.%f'))
        PPG_timestamp_ver_2_start_msec = 42243 #milisecond.
        
        ACC_timestamp_ver_2_start_datetime_string = '04/26/2021 11:27:00.042' # 24 hours time.
        ACC_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(ACC_timestamp_ver_2_start_datetime_string,\
                                                       '%m/%d/%Y %H:%M:%S.%f'))
        ACC_timestamp_ver_2_start_msec = 42243 # milisecond. ACC is same as PPG.
    elif UID == '400':
        # The start time of first 30-sec segment.
        PPG_timestamp_ver_2_start_datetime_string = '05/24/2021 12:17:53.691' # 24 hours time.
        PPG_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(PPG_timestamp_ver_2_start_datetime_string,\
                                                      '%m/%d/%Y %H:%M:%S.%f'))
        PPG_timestamp_ver_2_start_msec = 646596 #milisecond.
        
        ACC_timestamp_ver_2_start_datetime_string = '05/24/2021 12:17:53.681' # 24 hours time.
        ACC_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(ACC_timestamp_ver_2_start_datetime_string,\
                                                       '%m/%d/%Y %H:%M:%S.%f'))
        ACC_timestamp_ver_2_start_msec = 646586 # milisecond. ACC is same as PPG.
    elif UID == '914':
        # The start time of first 30-sec segment.
        PPG_timestamp_ver_2_start_datetime_string = '07/29/2021 14:58:07.866' # 24 hours time.
        PPG_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(PPG_timestamp_ver_2_start_datetime_string,\
                                                      '%m/%d/%Y %H:%M:%S.%f'))
        PPG_timestamp_ver_2_start_msec = 1627585087866 #milisecond.
        
        ACC_timestamp_ver_2_start_datetime_string = '05/24/2021 14:58:07.799' # 24 hours time.
        ACC_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(ACC_timestamp_ver_2_start_datetime_string,\
                                                       '%m/%d/%Y %H:%M:%S.%f'))
        ACC_timestamp_ver_2_start_msec = 1627585087799 # milisecond. ACC is same as PPG.
    elif UID == '117':
        # The start time of first 30-sec segment.
        PPG_timestamp_ver_2_start_datetime_string = '08/16/2021 11:18:34.180' # 24 hours time.
        PPG_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(PPG_timestamp_ver_2_start_datetime_string,\
                                                      '%m/%d/%Y %H:%M:%S.%f'))
        PPG_timestamp_ver_2_start_msec = 1629127114180 #milisecond.
        
        ACC_timestamp_ver_2_start_datetime_string = '08/16/2021 11:18:34.180' # 24 hours time.
        ACC_timestamp_ver_2_start_datetime = our_tzone.localize(datetime.datetime.strptime(ACC_timestamp_ver_2_start_datetime_string,\
                                                       '%m/%d/%Y %H:%M:%S.%f'))
        ACC_timestamp_ver_2_start_msec = 1629127114180 # milisecond. ACC is same as PPG.
    else:
        print('UID was not documented, check!')
    
    return PPG_timestamp_ver_2_start_datetime,\
        PPG_timestamp_ver_2_start_msec,\
        ACC_timestamp_ver_2_start_datetime,\
        ACC_timestamp_ver_2_start_msec
        
def my_func_prep_watch_PPG_ACC(PPG_file_buffer,\
                                ACC_file_buffer,\
                                fs_PPG,\
                                fs_ACC,\
                                PPG_file_name,\
                                test_PPG_path,\
                                UID,\
                                add_1_day_flag,\
                                m_output_struct, \
                                m_time, \
                                m_time_idx,\
                                mylines):
    our_tzone = pytz.timezone('America/New_York') # This line does not exist in MATLAB code. I do not want to pass it as a variable. 01/19/2023.
    if len(m_output_struct) == 0 or (m_output_struct['m_API_ver'].iloc[0] is None):
        # No log file, no API version at all.
        # Try to load PPG
        # 03/19/2023: debug why it is (1500,1) not (1500,)
        flag_sub_API = False
        
        if PPG_file_buffer.ndim > 1:
            
            if PPG_file_buffer.shape[1] == 4: # Four columns
                candidate_UID = ['402','403','405','406','410','411','413','414',\
                             '415','416','417','418','419','420','421','422',\
                             '423']
                
                if PPG_file_name[:3] == '400':
                    # m_output_struct.at[0,'m_API_ver'] = '2.0.1'
                    # m_output_struct = m_output_struct.assign(m_API_ver='2.0.1')
                    if len(m_output_struct) > 0:
                        m_output_struct = m_output_struct.assign(m_API_ver='2.0.1',index=0)
                    else:
                        temp_df = pd.DataFrame({"m_API_ver":'2.0.1',"m_HR_DATPD":None,\
                            "m_HR_SWEPD":None,\
                            "m_HR_WEPD":None,\
                            "m_comb":None,\
                            "m_RMSSD":None,\
                            "m_SampEn":None,\
                            "m_IsAF_1":None,\
                            "m_IsAF_2":None,\
                            "m_IsAF_3":None,\
                            "m_PACPVC_pred_1":None,\
                            "m_PACPVC_pred_2":None,\
                            "m_FastAF_1":None,\
                            "m_FastAF_2":None,\
                            "m_HR_1":None,\
                            "m_HR_2":None,\
                            "m_HR_3":None,\
                            "m_seg_num":None,\
                            "m_index":None,\
                            "m_mSensorEnable":None,\
                            "m_seg_start_row_idx":None,\
                            "m_seg_end_row_idx":None,\
                            "m_storage_total_size":None,\
                            "m_storage_avail_size":None,\
                            "m_battery_perc":None,\
                            "m_service_var":None}, index=[0])
                        # m_output_struct = m_output_struct.append(temp_df)
                        m_output_struct = pd.concat([m_output_struct,temp_df], ignore_index=True)
                elif PPG_file_name[:3] == '914':
                    # m_output_struct.at[0,'m_API_ver'] = '2.1.3'
                    # m_output_struct = m_output_struct.assign(m_API_ver='2.1.3')
                    if len(m_output_struct) > 0:
                        m_output_struct = m_output_struct.assign(m_API_ver='2.1.3',index=0)
                    else:
                        temp_df = pd.DataFrame({"m_API_ver":'2.1.3',"m_HR_DATPD":None,\
                            "m_HR_SWEPD":None,\
                            "m_HR_WEPD":None,\
                            "m_comb":None,\
                            "m_RMSSD":None,\
                            "m_SampEn":None,\
                            "m_IsAF_1":None,\
                            "m_IsAF_2":None,\
                            "m_IsAF_3":None,\
                            "m_PACPVC_pred_1":None,\
                            "m_PACPVC_pred_2":None,\
                            "m_FastAF_1":None,\
                            "m_FastAF_2":None,\
                            "m_HR_1":None,\
                            "m_HR_2":None,\
                            "m_HR_3":None,\
                            "m_seg_num":None,\
                            "m_index":None,\
                            "m_mSensorEnable":None,\
                            "m_seg_start_row_idx":None,\
                            "m_seg_end_row_idx":None,\
                            "m_storage_total_size":None,\
                            "m_storage_avail_size":None,\
                            "m_battery_perc":None,\
                            "m_service_var":None}, index=[0])
                        # m_output_struct = m_output_struct.append(temp_df)
                        m_output_struct = pd.concat([m_output_struct,temp_df], ignore_index=True)
                elif any(x in PPG_file_name[:3] for x in candidate_UID):
                    # # m_output_struct.at[0,'m_API_ver'] = '2.0.1'
                    # m_output_struct = m_output_struct.assign(m_API_ver='2.0.1')
                    # # m_output_struct.at[0,'m_service_var'] = '2.1.1'
                    # m_output_struct = m_output_struct.assign(m_service_var='2.1.1')
                    if len(m_output_struct) > 0:
                        m_output_struct = m_output_struct.assign(m_API_ver='2.0.1',m_service_var='2.1.1',index=0)
                    else:
                        temp_df = pd.DataFrame({"m_API_ver":'2.0.1',"m_HR_DATPD":None,\
                            "m_HR_SWEPD":None,\
                            "m_HR_WEPD":None,\
                            "m_comb":None,\
                            "m_RMSSD":None,\
                            "m_SampEn":None,\
                            "m_IsAF_1":None,\
                            "m_IsAF_2":None,\
                            "m_IsAF_3":None,\
                            "m_PACPVC_pred_1":None,\
                            "m_PACPVC_pred_2":None,\
                            "m_FastAF_1":None,\
                            "m_FastAF_2":None,\
                            "m_HR_1":None,\
                            "m_HR_2":None,\
                            "m_HR_3":None,\
                            "m_seg_num":None,\
                            "m_index":None,\
                            "m_mSensorEnable":None,\
                            "m_seg_start_row_idx":None,\
                            "m_seg_end_row_idx":None,\
                            "m_storage_total_size":None,\
                            "m_storage_avail_size":None,\
                            "m_battery_perc":None,\
                            "m_service_var":'2.1.1'}, index=[0])
                        # m_output_struct = m_output_struct.append(temp_df)
                        m_output_struct = pd.concat([m_output_struct,temp_df], ignore_index=True)
                else:
                    print('func\my_func_prep_watch_PPG_ACC.m: Unknown API, check!')
                
            elif PPG_file_buffer.shape[1] == 3: # Three columns
                # m_output_struct.at[0,'m_API_ver'] = '1.0.15'
                # m_output_struct = m_output_struct.assign(m_API_ver='1.0.15')
                if len(m_output_struct) > 0:
                    m_output_struct = m_output_struct.assign(m_API_ver='1.0.15',index=0)
                else:
                    temp_df = pd.DataFrame({"m_API_ver":'1.0.15',"m_HR_DATPD":None,\
                        "m_HR_SWEPD":None,\
                        "m_HR_WEPD":None,\
                        "m_comb":None,\
                        "m_RMSSD":None,\
                        "m_SampEn":None,\
                        "m_IsAF_1":None,\
                        "m_IsAF_2":None,\
                        "m_IsAF_3":None,\
                        "m_PACPVC_pred_1":None,\
                        "m_PACPVC_pred_2":None,\
                        "m_FastAF_1":None,\
                        "m_FastAF_2":None,\
                        "m_HR_1":None,\
                        "m_HR_2":None,\
                        "m_HR_3":None,\
                        "m_seg_num":None,\
                        "m_index":None,\
                        "m_mSensorEnable":None,\
                        "m_seg_start_row_idx":None,\
                        "m_seg_end_row_idx":None,\
                        "m_storage_total_size":None,\
                        "m_storage_avail_size":None,\
                        "m_battery_perc":None,\
                        "m_service_var":None}, index=[0])
                    # m_output_struct = m_output_struct.append(temp_df)
                    m_output_struct = pd.concat([m_output_struct,temp_df], ignore_index=True)
            # line 74 in MATLAB
            if len(PPG_file_buffer) == 0 and int(PPG_file_name[:3]) >= 36:
                flag_sub_API = True
            else:
                if int(PPG_file_name[:3]) >= 13 and int(PPG_file_name[:3]) < 36:
                    flag_sub_API = True
                elif int(PPG_file_name[:3]) < 13:
                    flag_sub_API = True
        else:
            flag_sub_API = True
            
        if flag_sub_API:
            # line 74 in MATLAB
            if len(PPG_file_buffer) == 0 and int(PPG_file_name[:3]) >= 36:
                # UID 036 and 043 all have empty PPG segment.
                # PPG idx 47888:
                # For PPG idx 51285, I have to remove
                # 'isempty(ACC_file_buffer)' because ACC txt file does not
                # exist.
                # m_output_struct.at[0,'m_API_ver'] = '1.0.15'
                if len(m_output_struct) > 0:
                    m_output_struct = m_output_struct.assign(m_API_ver='1.0.15',index=0)
                else:
                    temp_df = pd.DataFrame({"m_API_ver":'1.0.15',"m_HR_DATPD":None,\
                        "m_HR_SWEPD":None,\
                        "m_HR_WEPD":None,\
                        "m_comb":None,\
                        "m_RMSSD":None,\
                        "m_SampEn":None,\
                        "m_IsAF_1":None,\
                        "m_IsAF_2":None,\
                        "m_IsAF_3":None,\
                        "m_PACPVC_pred_1":None,\
                        "m_PACPVC_pred_2":None,\
                        "m_FastAF_1":None,\
                        "m_FastAF_2":None,\
                        "m_HR_1":None,\
                        "m_HR_2":None,\
                        "m_HR_3":None,\
                        "m_seg_num":None,\
                        "m_index":None,\
                        "m_mSensorEnable":None,\
                        "m_seg_start_row_idx":None,\
                        "m_seg_end_row_idx":None,\
                        "m_storage_total_size":None,\
                        "m_storage_avail_size":None,\
                        "m_battery_perc":None,\
                        "m_service_var":None}, index=[0])
                    # m_output_struct = m_output_struct.append(temp_df)
                    m_output_struct = pd.concat([m_output_struct,temp_df], ignore_index=True)
            else:
                if int(PPG_file_name[:3]) >= 13 and int(PPG_file_name[:3]) < 36:
                    # m_output_struct.at[0,'m_API_ver'] = '1.0.11'
                    # m_output_struct = m_output_struct.assign(m_API_ver='1.0.11',index=0)
                    if len(m_output_struct) > 0:
                        m_output_struct = m_output_struct.assign(m_API_ver='1.0.11',index=0)
                    else:
                        temp_df = pd.DataFrame({"m_API_ver":'1.0.11',"m_HR_DATPD":None,\
                        "m_HR_SWEPD":None,\
                        "m_HR_WEPD":None,\
                        "m_comb":None,\
                        "m_RMSSD":None,\
                        "m_SampEn":None,\
                        "m_IsAF_1":None,\
                        "m_IsAF_2":None,\
                        "m_IsAF_3":None,\
                        "m_PACPVC_pred_1":None,\
                        "m_PACPVC_pred_2":None,\
                        "m_FastAF_1":None,\
                        "m_FastAF_2":None,\
                        "m_HR_1":None,\
                        "m_HR_2":None,\
                        "m_HR_3":None,\
                        "m_seg_num":None,\
                        "m_index":None,\
                        "m_mSensorEnable":None,\
                        "m_seg_start_row_idx":None,\
                        "m_seg_end_row_idx":None,\
                        "m_storage_total_size":None,\
                        "m_storage_avail_size":None,\
                        "m_battery_perc":None,\
                        "m_service_var":None}, index=[0])
                        # m_output_struct = m_output_struct.append(temp_df)
                        m_output_struct = pd.concat([m_output_struct,temp_df], ignore_index=True)
                elif int(PPG_file_name[:3]) < 13:
                    # m_output_struct.at[0,'m_API_ver'] = '1.0.0'
                    if len(m_output_struct) > 0:
                        m_output_struct = m_output_struct.assign(m_API_ver='1.0.0',index=0)
                    else:
                        temp_df = pd.DataFrame({"m_API_ver":'1.0.0',"m_HR_DATPD":None,\
                        "m_HR_SWEPD":None,\
                        "m_HR_WEPD":None,\
                        "m_comb":None,\
                        "m_RMSSD":None,\
                        "m_SampEn":None,\
                        "m_IsAF_1":None,\
                        "m_IsAF_2":None,\
                        "m_IsAF_3":None,\
                        "m_PACPVC_pred_1":None,\
                        "m_PACPVC_pred_2":None,\
                        "m_FastAF_1":None,\
                        "m_FastAF_2":None,\
                        "m_HR_1":None,\
                        "m_HR_2":None,\
                        "m_HR_3":None,\
                        "m_seg_num":None,\
                        "m_index":None,\
                        "m_mSensorEnable":None,\
                        "m_seg_start_row_idx":None,\
                        "m_seg_end_row_idx":None,\
                        "m_storage_total_size":None,\
                        "m_storage_avail_size":None,\
                        "m_battery_perc":None,\
                        "m_service_var":None}, index=[0])
                        # m_output_struct = m_output_struct.append(temp_df)
                        m_output_struct = pd.concat([m_output_struct,temp_df], ignore_index=True)
                        
    if m_output_struct['m_API_ver'].iloc[0] == '1.0.0' or \
        (m_output_struct['m_service_var'].iloc[0] is None \
         and m_output_struct['m_API_ver'].iloc[0] == '1.0.11') or \
        m_output_struct['m_API_ver'].iloc[0] == '1.0.13':
        # Line 99 in MATLAB code.
        temp_curr_seg = int(PPG_file_name[28:32]) # 28, 29, 30, 31, '0000'
        # Use file name to know time stamp.
        PPG_timestamp_start = datetime.timedelta(seconds=30*temp_curr_seg) + \
            our_tzone.localize(datetime.datetime.strptime(PPG_file_name[4:23], '%Y_%m_%d_%H_%M_%S')) # yyyy_MM_dd_HH_mm_ss
        PPG_timestamp_end = PPG_timestamp_start + datetime.timedelta(seconds=PPG_file_buffer.shape[0]/fs_PPG)
        
        if len(PPG_file_buffer) > 0: # PPG txt file is not empty.
            temp_timestamp = np.arange(0,PPG_file_buffer.shape[0])/fs_PPG # 01/17/2023: I started from sample 0, not 1. Different from my MATLAB code.
            PPG_t_datetime = [PPG_timestamp_start + datetime.timedelta(seconds=tt) for tt in temp_timestamp]
            ACC_t_datetime = PPG_t_datetime
            
            PPG_t_msec = [datetime.timedelta(milliseconds=tt) for tt in temp_timestamp] # I am not sure if PPG_timestamp_start should be added.
            ACC_t_msec = PPG_t_msec
            
            PPG_raw_buffer = PPG_file_buffer[:,0]
            if len(ACC_file_buffer) > 0:
                # if ACC txt file is not empty.
                ACC_raw_buffer = ACC_file_buffer[:,0]
            else:
                # ACC file is empty like UID 026, index 8614.
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
        else: # PPG txt file is empty.
            PPG_raw_buffer = np.zeros((fs_PPG * 30,)); # 11/20/2020: I don't know if I should give zero to here.
            ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape);
            PPG_t_datetime = [None] * fs_PPG * 30 # 01/17/2023: I cannot find NaT variable in Python
            ACC_t_datetime = [None] * fs_PPG * 30
            PPG_t_msec = [None] * fs_PPG * 30 # np.zeros(PPG_raw_buffer.shape);
            ACC_t_msec = [None] * fs_PPG * 30 # np.zeros(PPG_raw_buffer.shape);
        
        # Get segment index 0000:
        PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
        if PPG_4_dig_str == '0000':
            # it is segment 0000:
            # Worked: Plan 1: test cut first few seconds:
            cut_sec = 2;
            PPG_raw_buffer = PPG_raw_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
            PPG_t_msec = PPG_t_msec[cut_sec*fs_PPG:]
            PPG_t_datetime = PPG_t_datetime[cut_sec*fs_PPG:]
            PPG_file_buffer = PPG_file_buffer[cut_sec*fs_PPG:]
            SamsungHR_buffer = np.zeros(PPG_raw_buffer.shape) # no Samsung HR.
            SamsungHR_t_msec = PPG_t_msec # no Samsung HR t_msec.
            SamsungHR_t_datetime = PPG_t_datetime # no Samsung HR t datetime.
            # print('Check size for seg 0000',PPG_raw_buffer.shape)
        else:
            # I only want to replot all seg 0000 now.
            # ---Bug fixing for UID 032, seg 0001, idx 32. PPG file only has
            # 1366 samples, not 1500. 01/02/2022. ---
            if PPG_raw_buffer.shape[0] < fs_PPG * 30:
                # I have no way to figure out if I should fill the
                # beginning or the end, so I decided to fill the end with
                # the value of last sample to avoid sudden jump of the
                # signal.
                compensate_samples = fs_PPG * 30 - PPG_raw_buffer.shape[0]
                PPG_raw_buffer = np.concatenate((PPG_raw_buffer,np.ones((compensate_samples,))*PPG_raw_buffer[-2])) # -1 is incomplete.
                # MATLAB line 163.
                temp_timestamp = np.arange(0,PPG_raw_buffer.shape[0])/fs_PPG # 01/17/2023: I started from sample 0, not 1. Different from my MATLAB code.
                PPG_t_msec = [datetime.timedelta(milliseconds=tt) for tt in temp_timestamp] # I am not sure if PPG_timestamp_start should be added.
                PPG_t_datetime = [PPG_timestamp_start + datetime.timedelta(seconds=tt) for tt in temp_timestamp]
                PPG_file_buffer = np.concatenate((PPG_file_buffer[:,0],np.ones((compensate_samples,))*PPG_file_buffer[-2]))
                # PPG_color = [0, 0.176, 0.8] # Dark blue
                # ECG_color = [0,0,0] # Black
                # my_fig, my_ax = plt.subplots(figsize=(20, 20))
                # my_ax.plot(PPG_raw_buffer,color=PPG_color)
                # Path(r'R:\ENGR_Chon\Dong\Python_generated_results\Debug_Plot').mkdir(parents=True, exist_ok=True)
                # my_fig.savefig(os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\Debug_Plot',\
                #                             PPG_file_name[:32]+'_raw_buffer.png'), dpi=96, bbox_inches='tight') # Resolution is 300 dpi.
                # my_fig, my_ax = plt.subplots(figsize=(20, 20))
                # my_ax.plot(PPG_file_buffer,color=ECG_color)
                # Path(r'R:\ENGR_Chon\Dong\Python_generated_results\Debug_Plot').mkdir(parents=True, exist_ok=True)
                # my_fig.savefig(os.path.join(r'R:\ENGR_Chon\Dong\Python_generated_results\Debug_Plot',\
                #                             PPG_file_name[:32]+'_file_buffer.png'), dpi=96, bbox_inches='tight') # Resolution is 300 dpi.
                if ACC_raw_buffer.shape[0] < fs_ACC * 30:
                    ACC_raw_buffer = np.concatenate((ACC_raw_buffer,np.ones((compensate_samples,))*ACC_raw_buffer[-1]))
                    
                if len(ACC_t_msec) < fs_ACC * 30: # Dong, 03/18/2023: it is a list.
                    ACC_t_msec = PPG_t_msec
                    
                if len(ACC_t_datetime) < fs_ACC * 30:# Dong, 03/18/2023: it is a list.
                    ACC_t_datetime = PPG_t_datetime
                
            SamsungHR_buffer = np.zeros(PPG_raw_buffer.shape) # no Samsung HR.
            SamsungHR_t_msec = PPG_t_msec # no Samsung HR t_msec.
            SamsungHR_t_datetime = PPG_t_datetime # no Samsung HR t datetime.
        PPG_t_msec_ver_2 = [] # For ver 2.0.1 testing. 03/02/2021.
        PPG_t_datetime_ver_2 = []
        ACC_t_msec_ver_2 = []
        ACC_t_datetime_ver_2 = []
    elif m_output_struct['m_API_ver'].iloc[0] == '1.0.11' or \
         m_output_struct['m_API_ver'].iloc[0] == '1.0.13' and \
         len(m_time_idx) > 0:
        # UID 013
        # There is time stamp in log file.
        # 1. extract the time stamp cell for each segment.
        m_seg_start_row_idx = m_output_struct['m_seg_start_row_idx']
        m_seg_end_row_idx = m_output_struct['m_seg_end_row_idx']
        m_seg_num = m_output_struct['m_seg_num']
        m_index = m_output_struct['m_index']
        
        m_seg_time = [None] * m_output_struct.shape[0]
        for index_t, row_t in m_output_struct.iterrows():
            if row_t['m_seg_start_row_idx'] == None:
                if index_t > 0:
                    # not missing the beginning row index:
                    print('middle seg missed row index, check!')
                else:
                    # the first segment is missing a start row
                    # index.
                    temp_start = m_time_idx[0]
                    temp_end = m_time_idx.index(row_t['m_seg_end_row_idx']) # m_time_idx already stores int.
                    # I am assuming the index in the m_output_struct is from zero and continuous.
                    m_seg_time[index_t] = m_time[temp_start:temp_end+1]
            else:
                temp_start = m_time_idx.index(row_t['m_seg_start_row_idx']) # m_time_idx already stores int.
                if pd.isna(row_t['m_seg_end_row_idx']):
                    temp_end = temp_start
                    m_seg_time[index_t] = [] #m_time[temp_start:temp_end+1]
                else:
                    temp_end = m_time_idx.index(row_t['m_seg_end_row_idx']) # m_time_idx already stores int.
                    # I am assuming the index in the m_output_struct is from zero and continuous.
                    m_seg_time[index_t] = m_time[temp_start:temp_end+1]
        
        # Line 224 in MATLAB:
        # get the time for this segment of PPG and ACC:
        temp_curr_seg = int(PPG_file_name[28:32]) # 28, 29, 30, 31, '0000'
        m_index = list(m_index) # Convert Series to list.
        try:
            temp_index_idx = m_index.index(temp_curr_seg)
        except:
            temp_index_idx = None
        
        if temp_index_idx is None:
            # no matching segment index
            if temp_curr_seg > 0:
                # there is previous segment at least not empty.
                # load the time of last PPG segment.
                # Dong, 01/18/2023, I will not save the time_aim23 file as they are same as this func.
                temp_curr_seg = int(PPG_file_name[28:32]) # 28, 29, 30, 31, '0000'
                PPG_timestamp_start = datetime.timedelta(seconds=1*(temp_curr_seg+1)) + \
                    datetime.timedelta(seconds=30*temp_curr_seg) + \
                    our_tzone.localize(datetime.datetime.strptime(PPG_file_name[4:23], '%Y_%m_%d_%H_%M_%S')) # yyyy_MM_dd_HH_mm_ss
                PPG_timestamp_end = PPG_timestamp_start + datetime.timedelta(seconds=PPG_file_buffer.shape[0]/fs_PPG)
            elif temp_curr_seg == 0:
                PPG_timestamp_start = datetime.timedelta(seconds=1) + \
                    our_tzone.localize(datetime.datetime.strptime(PPG_file_name[4:23], '%Y_%m_%d_%H_%M_%S')) # yyyy_MM_dd_HH_mm_ss
                PPG_timestamp_end = PPG_timestamp_start + datetime.timedelta(seconds=PPG_file_buffer.shape[0]/fs_PPG)
            else:
                print('unseen type of PPG file name seg, check!')
        else:
            temp_m_seg_time = sorted(list(set(m_seg_time[temp_index_idx]))) # Equal to unique. I sorted and not sure if this will cause a problem
            if len(temp_m_seg_time) < 1:
                # no time in log file, probably caused by power off the
                # watch early.s
                if temp_curr_seg > 0:
                    # there is previous segment at least not empty.
                    # load the time of last PPG segment. 
                    temp_curr_seg = int(PPG_file_name[28:32]) # 28, 29, 30, 31, '0000'
                    PPG_timestamp_start = datetime.timedelta(seconds=1*(temp_curr_seg+1)) + \
                        datetime.timedelta(seconds=30*temp_curr_seg) + \
                        our_tzone.localize(datetime.datetime.strptime(PPG_file_name[4:23], '%Y_%m_%d_%H_%M_%S')) # yyyy_MM_dd_HH_mm_ss
                    PPG_timestamp_end = PPG_timestamp_start + datetime.timedelta(seconds=PPG_file_buffer.shape[0]/fs_PPG)
                elif temp_curr_seg == 0:
                    PPG_timestamp_start = datetime.timedelta(seconds=1) + \
                        our_tzone.localize(datetime.datetime.strptime(PPG_file_name[4:23], '%Y_%m_%d_%H_%M_%S')) # yyyy_MM_dd_HH_mm_ss
                    PPG_timestamp_end = PPG_timestamp_start + datetime.timedelta(seconds=PPG_file_buffer.shape[0]/fs_PPG)
                else:
                    print('unseen type of PPG file name seg, check!')
            else:
                # it is fine to have more than two. I will still use
                # the 2nd time.
                temp_date = PPG_file_name[4:14]
                temp_time = temp_m_seg_time[0]
                # 03/11/2021: fix bug for ver 1.0.11 no timestamp PPG
                # spanning two days.
                span_two_days_flag = False # Mark from 23:59 to 00:00.
                this_m_seg_time = m_seg_time[0]
                if len(this_m_seg_time) == 0:
                    if len(m_seg_time) > 1:
                        this_m_seg_time = m_seg_time[1] # For UID 310, the first cell is empty.
                    else:
                        print('my_func_prep_watch_PPG_ACC: m_seg_time is empty, check!\n');
                first_m_seg_time = this_m_seg_time[0]
                this_hour_char = first_m_seg_time[:2] # 0,1
                if PPG_file_name[15:17] == '23' and this_hour_char == '00': # 017_2019_12_04_23_59_49_ppg_0000.txt
                    # The file name starts before the 00:00 AM.
                    span_two_days_flag = True
                else:
                    temp_start_flag = True
                    for this_m_seg_time in m_seg_time:
                        if len(this_m_seg_time) == 0:
                            continue
                        else:
                            first_m_seg_time = this_m_seg_time[0]
                            if temp_start_flag: # Initial previous hour char.
                                prev_hour_char = first_m_seg_time[:2]
                                temp_start_flag = False
                            this_hour_char = first_m_seg_time[:2]
                            if prev_hour_char == '23' and this_hour_char == '00':
                                span_two_days_flag = True
                                break
                            prev_hour_char = this_hour_char # Update previous hour char.
                print('temp_time',temp_time)
                print('len(temp_time)',len(temp_time))
                if len(temp_time) > 8:
                    # with millisecond.
                    if len(temp_time) == 14: # UID 018, index 1, time: 12:18:16:100:0
                        PPG_timestamp_end = our_tzone.localize(datetime.datetime.strptime(temp_date+','+temp_time[:12], '%Y_%m_%d,%H:%M:%S:%f'))
                    elif len(temp_time) == 13:
                        PPG_timestamp_end = our_tzone.localize(datetime.datetime.strptime(temp_date+','+temp_time[:11], '%Y_%m_%d,%H:%M:%S:%f'))
                    elif len(temp_time) == 12:
                        PPG_timestamp_end = our_tzone.localize(datetime.datetime.strptime(temp_date+','+temp_time[:10], '%Y_%m_%d,%H:%M:%S:%f'))
                    else:
                        print('Unseen num of digit, check!')
                elif len(temp_time) == 8:
                    # without millisecond.
                    # 03/11/2021: check if it is a 5-min spanning 2 days.
                    PPG_timestamp_end = our_tzone.localize(datetime.datetime.strptime(temp_date+','+temp_time, '%Y_%m_%d,%H:%M:%S'))
                else:
                    print('Unseen number of time digit, check!')
                
                print('Loc 1: PPG_timestamp_end',PPG_timestamp_end)
                prev_time = PPG_file_name[15:17] # 15, 16
                if temp_index_idx > 0:
                    prev_m_seg_time = sorted(list(set(m_seg_time[temp_index_idx-1])))
                    if len(prev_m_seg_time) > 0:
                        prev_time = prev_m_seg_time[0] # For UID 310, the m_seg_time{1,1} is empty. 07/12/2021.
                        
                if span_two_days_flag and temp_time[:2] =='00' and prev_time[:2] == '23': # This is the transit 30-sec segment.
                    PPG_timestamp_end = PPG_timestamp_end + datetime.timedelta(days=1)
                    add_1_day_flag = True
                elif add_1_day_flag: # the 30-sec segment after the span in this 5-min period.
                    PPG_timestamp_end = PPG_timestamp_end + datetime.timedelta(days=1)
                    
                print('Loc 2: PPG_timestamp_end',PPG_timestamp_end)
                print('span_two_days_flag',span_two_days_flag)
                print('add_1_day_flag',add_1_day_flag)
                print('PPG_file_buffer:', PPG_file_buffer)
                PPG_timestamp_start = PPG_timestamp_end - datetime.timedelta(seconds=PPG_file_buffer.shape[0]/fs_PPG) # test with 30 sec.
        # print('test, PPG_timestamp_start',PPG_timestamp_start)
        # Line 387 in MATLAB:
        if PPG_file_buffer.shape[0] > 0: # PPG txt file is not empty.
            temp_timestamp = np.arange(0,PPG_file_buffer.shape[0])/fs_PPG # 01/17/2023: I started from sample 0, not 1. Different from my MATLAB code.
            PPG_t_datetime = [PPG_timestamp_start + datetime.timedelta(seconds=tt) for tt in temp_timestamp]
            ACC_t_datetime = PPG_t_datetime
            
            PPG_t_msec = [datetime.timedelta(milliseconds=tt) for tt in temp_timestamp] # I am not sure if PPG_timestamp_start should be added.
            ACC_t_msec = PPG_t_msec
            
            PPG_raw_buffer = PPG_file_buffer[:,0]
            if ACC_file_buffer.shape[0] > 0:
                ACC_raw_buffer = ACC_file_buffer[:,0]
            else:
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
        else: # PPG txt file is empty.
            PPG_raw_buffer = np.zeros((fs_PPG * 30,)); # 11/20/2020: I don't know if I should give zero to here.
            ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape);
            PPG_t_datetime = [None] * fs_PPG * 30 # 01/17/2023: I cannot find NaT variable in Python
            ACC_t_datetime = [None] * fs_PPG * 30
            PPG_t_msec = [None] * fs_PPG * 30 # np.zeros(PPG_raw_buffer.shape);
            ACC_t_msec = [None] * fs_PPG * 30 # np.zeros(PPG_raw_buffer.shape);
        # tonight, test the timestamp. Try to align with ECG
        # time??? 09/01/2020.
        # === Added 06/28/2020: ====
        # Get segment index 0000:
        PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
        if PPG_4_dig_str == '0000':
            # print('test,PPG_t_datetime[0]',PPG_t_datetime[0])
            # it is segment 0000:
            # Worked: Plan 1: test cut first few seconds:
            cut_sec = 2;
            PPG_raw_buffer = PPG_raw_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
            PPG_t_msec = PPG_t_msec[cut_sec*fs_PPG:]
            PPG_t_datetime = PPG_t_datetime[cut_sec*fs_PPG:]
            PPG_file_buffer = PPG_file_buffer[cut_sec*fs_PPG:]
            SamsungHR_buffer = np.zeros(PPG_raw_buffer.shape) # no Samsung HR.
            SamsungHR_t_msec = PPG_t_msec # no Samsung HR t_msec.
            SamsungHR_t_datetime = PPG_t_datetime # no Samsung HR t datetime.
            # print('Check size for seg 0000',PPG_raw_buffer.shape)
        else:
            # I only want to replot all seg 0000 now.
            SamsungHR_buffer = np.zeros(PPG_raw_buffer.shape) # no Samsung HR.
            SamsungHR_t_msec = PPG_t_msec # no Samsung HR t_msec.
            SamsungHR_t_datetime = PPG_t_datetime # no Samsung HR t datetime.
        PPG_t_msec_ver_2 = [] # For ver 2.0.1 testing. 03/02/2021.
        PPG_t_datetime_ver_2 = []
        ACC_t_msec_ver_2 = []
        ACC_t_datetime_ver_2 = []
        
        # print('test,PPG_t_datetime[0]',PPG_t_datetime[0])
    elif m_output_struct['m_API_ver'].iloc[0] == '1.0.14' or \
         m_output_struct['m_API_ver'].iloc[0] == '1.0.15' or \
         m_output_struct['m_API_ver'].iloc[0] == '2.0.0':
         # both API version should have independent PPG time stamp.
         # interpolate PPG based on its timestamp:
        if PPG_file_buffer.shape[0] > 0:
            PPG_raw_buffer,\
                PPG_t_msec,\
                PPG_t_datetime,_,_,_ = my_func_interpolate_watch_data(PPG_file_buffer,fs_PPG)
            PPG_Samsung_HR = PPG_file_buffer[:,2] # get Samsung HR.
            if ACC_file_buffer.shape[0] > 0 and np.ndim(ACC_file_buffer) > 1: 
                if ACC_file_buffer.shape[1] > 1:
                    ACC_raw_buffer,\
                        ACC_t_msec,\
                        ACC_t_datetime,_,_,_ = my_func_interpolate_watch_data(ACC_file_buffer,fs_ACC)
                else:
                    # ACC file does not exist.
                    ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
                    ACC_t_msec = PPG_t_msec
                    ACC_t_datetime = PPG_t_datetime
            else:
                # Copied from above else, 01/28/2023.
                # ACC file does not exist.
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
                ACC_t_msec = PPG_t_msec
                ACC_t_datetime = PPG_t_datetime
           
        else:
            if ACC_file_buffer.shape[0] > 0 and np.ndim(ACC_file_buffer) > 1: 
                if ACC_file_buffer.shape[1] > 1:
                    ACC_raw_buffer,\
                        ACC_t_msec,\
                        ACC_t_datetime,_,_,_ = my_func_interpolate_watch_data(ACC_file_buffer,fs_ACC)
                        
                    PPG_raw_buffer = np.zeros(ACC_raw_buffer.shape)
                    PPG_t_msec = ACC_t_msec
                    PPG_t_datetime = ACC_t_datetime
        
                    PPG_Samsung_HR = np.zeros(ACC_raw_buffer.shape)
                else:
                    print('both PPG and ACC are empty.')
                    PPG_raw_buffer = np.zeros((fs_PPG * 30,1))
                    ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
        
                    PPG_t_msec = np.zeros(PPG_raw_buffer.shape)
                    ACC_t_msec = np.zeros(PPG_raw_buffer.shape)
        
                    PPG_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/28/2023: I cannot find NaT variable in Python
                    ACC_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/28/2023: I cannot find NaT variable in Python
        
                    PPG_Samsung_HR = np.zeros(ACC_raw_buffer.shape)
            else:
                # Copied from above else, 01/28/2023
                print('both PPG and ACC are empty.')
                PPG_raw_buffer = np.zeros((fs_PPG * 30,1))
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
        
                PPG_t_msec = np.zeros(PPG_raw_buffer.shape)
                ACC_t_msec = np.zeros(PPG_raw_buffer.shape)
        
                PPG_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/28/2023: I cannot find NaT variable in Python
                ACC_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/28/2023: I cannot find NaT variable in Python
        
                PPG_Samsung_HR = np.zeros(ACC_raw_buffer.shape)
                 
        # --- Only for UID 119A, watch was idle. 12/16/2021 ---
        # 1st PPG file datetime: 01/01/2018, 10:49:53.432 AM.
        # ECG start time: 08/03/2021, 09:04 AM.
        # 1st PPG file name: 08/03/2021, 10:49:51 AM.
        if PPG_file_name[:3] == '119':
            PPG_year_this = PPG_t_datetime[0].year # This is int, not str as in MALAB.
            if PPG_year_this == '2018':
                PPG_wrong_datetime = our_tzone.localize(datetime.datetime.strptime('01/01/2018', '%m/%d/%Y'))
                ECG_right_datetime = our_tzone.localize(datetime.datetime.strptime('08/03/2021', '%m/%d/%Y'))
                
                PPG_add_sec = ECG_right_datetime - PPG_wrong_datetime
                PPG_t_datetime = PPG_t_datetime + PPG_add_sec
                ACC_t_datetime = ACC_t_datetime + PPG_add_sec
        # === Added 06/28/2020: ====
        # Get segment index 0000:
        PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
        if PPG_4_dig_str == '0000':
            # it is segment 0000:
            # Worked: Plan 1: test cut first few seconds:
            cut_sec = 2;
            PPG_raw_buffer = PPG_raw_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
            PPG_t_msec = PPG_t_msec[cut_sec*fs_PPG:]
            PPG_t_datetime = PPG_t_datetime[cut_sec*fs_PPG:]
            PPG_file_buffer = PPG_file_buffer[cut_sec*fs_PPG:]
            PPG_Samsung_HR = PPG_Samsung_HR[cut_sec*fs_PPG:]
        else:
            pass
        # === end of Added 06/28/2020 ===
        PPG_timestamp_start = PPG_t_datetime[0]
        PPG_timestamp_end = PPG_t_datetime[-1]
        if PPG_file_buffer.shape[0] > 0:
            temp_HR_file_buffer = np.stack([PPG_file_buffer[:,0],PPG_Samsung_HR],axis=1)
            SamsungHR_buffer,\
              SamsungHR_t_msec,\
              SamsungHR_t_datetime,_,_,_ = my_func_interpolate_watch_data(temp_HR_file_buffer,fs_PPG)
        else:
            if ACC_file_buffer.shape[0] > 0:
                SamsungHR_t_msec = ACC_t_msec
                SamsungHR_t_datetime = ACC_t_datetime

                SamsungHR_buffer = np.zeros(ACC_raw_buffer.shape)
                # === Added 06/28/2020: ====
                # Get segment index 0000:
                PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
                if PPG_4_dig_str == '0000':
                    # it is segment 0000:
                    # Worked: Plan 1: test cut first few seconds:
                    cut_sec = 2
                    SamsungHR_buffer = SamsungHR_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
                    SamsungHR_t_msec = SamsungHR_t_msec[cut_sec*fs_PPG:]
                    SamsungHR_t_datetime = SamsungHR_t_datetime[cut_sec*fs_PPG:]
                else:
                    pass
                # === end of Added 06/28/2020 ===
            else:
                print('both PPG and ACC are empty.')
                SamsungHR_buffer = np.zeros(ACC_raw_buffer.shape)
                SamsungHR_t_msec = np.zeros(ACC_raw_buffer.shape)
                SamsungHR_t_datetime = np.zeros(ACC_raw_buffer.shape)
        
        PPG_t_msec_ver_2 = [] # For ver 2.0.1 testing. 03/02/2021.
        PPG_t_datetime_ver_2 = []
        ACC_t_msec_ver_2 = []
        ACC_t_datetime_ver_2 = [] # Line 596 in MATLAB.
    elif (m_output_struct['m_API_ver'].iloc[0] == '2.0.1') and not \
         m_output_struct['m_service_var'].iloc[0] == '2.1.1':
        
        # Get the initial recording time for PPG, it was hand-copied
        # from the 1st segment of the entire recording.
        PPG_timestamp_ver_2_start_datetime,\
            PPG_timestamp_ver_2_start_msec,\
            ACC_timestamp_ver_2_start_datetime,\
            ACC_timestamp_ver_2_start_msec = my_func_ver_2_timestamp_start_datetime(UID)
        if PPG_file_buffer.shape[0] > 0:
            flag_ACC = False # this input is PPG data.
            PPG_raw_buffer,\
                PPG_t_msec,\
                PPG_t_datetime,\
                PPG_t_msec_ver_2,\
                PPG_t_datetime_ver_2,\
                PPG_timestamp_ver_2_raw = my_func_interpolate_watch_data(PPG_file_buffer,\
                                                                        fs_PPG,\
                                                                        flag_ACC,\
                                                                        PPG_timestamp_ver_2_start_datetime,\
                                                                        PPG_timestamp_ver_2_start_msec)
                    
            PPG_Samsung_HR = PPG_file_buffer[:,2] # get Samsung HR.
            if ACC_file_buffer.shape[0] > 0 and np.ndim(ACC_file_buffer) > 1: 
                if ACC_file_buffer.shape[1] > 1:
                    flag_ACC = True # this input is ACC data.
                    ACC_raw_buffer,\
                        ACC_t_msec,\
                        ACC_t_datetime,\
                        ACC_t_msec_ver_2,\
                        ACC_t_datetime_ver_2,\
                        ACC_timestamp_ver_2_raw = my_func_interpolate_watch_data(ACC_file_buffer,fs_ACC,\
                                                                                 flag_ACC,\
                                                                                 ACC_timestamp_ver_2_start_datetime,\
                                                                                 ACC_timestamp_ver_2_start_msec)
                else:
                    # ACC file does not exist.
                    ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
                    ACC_t_msec = PPG_t_msec
                    ACC_t_datetime = PPG_t_datetime
            else:
                # Copied from above else, 01/29/2023
                # ACC file does not exist.
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
                ACC_t_msec = PPG_t_msec
                ACC_t_datetime = PPG_t_datetime
        else:
            if ACC_file_buffer.shape[0] > 0: 
                flag_ACC = True # this input is ACC data.
                ACC_raw_buffer,\
                    ACC_t_msec,\
                    ACC_t_datetime,\
                    ACC_t_msec_ver_2,\
                    ACC_t_datetime_ver_2,\
                    ACC_timestamp_ver_2_raw = my_func_interpolate_watch_data(ACC_file_buffer,fs_ACC,\
                                                                             flag_ACC,\
                                                                             ACC_timestamp_ver_2_start_datetime,\
                                                                             ACC_timestamp_ver_2_start_msec)
                PPG_raw_buffer = np.zeros(ACC_raw_buffer.shape)
                PPG_t_msec = ACC_t_msec
                PPG_t_datetime = ACC_t_datetime

                PPG_Samsung_HR = np.zeros(ACC_raw_buffer.shape)
            else:
                print('both PPG and ACC are empty.')

                PPG_raw_buffer = np.zeros((fs_PPG * 30,1))
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)

                PPG_t_msec = np.zeros(PPG_raw_buffer.shape)
                ACC_t_msec = np.zeros(PPG_raw_buffer.shape)

                PPG_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/29/2023: I cannot find NaT variable in Python
                ACC_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/29/2023: I cannot find NaT variable in Python

                PPG_Samsung_HR = np.zeros(ACC_raw_buffer.shape)
                
        PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
        if PPG_4_dig_str == '0000': # Line 663 in MATLAB.
            # it is segment 0000:
            # Worked: Plan 1: test cut first few seconds:
            cut_sec = 2;
            PPG_raw_buffer = PPG_raw_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
            PPG_t_msec = PPG_t_msec[cut_sec*fs_PPG:]
            PPG_t_datetime = PPG_t_datetime[cut_sec*fs_PPG:]
            PPG_file_buffer = PPG_file_buffer[cut_sec*fs_PPG:]
            PPG_Samsung_HR = PPG_Samsung_HR[cut_sec*fs_PPG:]
            
            PPG_t_msec_ver_2 = PPG_t_msec_ver_2[cut_sec*fs_PPG:]
            PPG_t_datetime_ver_2 = PPG_t_datetime_ver_2[cut_sec*fs_PPG:]
        else:    
            pass
        PPG_timestamp_start = PPG_t_datetime[0]
        PPG_timestamp_end = PPG_t_datetime[-1]
        # check file name time lapse vs timestamp lapse:
        # file name time lapse is in PPG_struct;
        if PPG_file_buffer.shape[0] > 0:
            temp_HR_file_buffer = np.stack([PPG_file_buffer[:,3],PPG_Samsung_HR],axis=1) # 4th column is the old timestamp.
            
            SamsungHR_buffer,\
             SamsungHR_t_msec,\
             SamsungHR_t_datetime,_,_,_ = my_func_interpolate_watch_data(temp_HR_file_buffer,fs_PPG)
        else:
            if ACC_file_buffer.shape[0] > 0:
                SamsungHR_t_msec = ACC_t_msec
                SamsungHR_t_datetime = ACC_t_datetime

                SamsungHR_buffer = np.zeros(ACC_raw_buffer.shape)

                PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
                if PPG_4_dig_str == '0000':
                    # it is segment 0000:
                    # Worked: Plan 1: test cut first few seconds:
                    cut_sec = 2;
                    SamsungHR_buffer = SamsungHR_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
                    SamsungHR_t_msec = SamsungHR_t_msec[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
                    SamsungHR_t_datetime = SamsungHR_t_datetime[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
            else:
                print('both PPG and ACC are empty.')
                SamsungHR_buffer = np.zeros(ACC_raw_buffer.shape)
                SamsungHR_t_msec = np.zeros(ACC_raw_buffer.shape)
                SamsungHR_t_datetime = np.zeros(ACC_raw_buffer.shape)
                
    elif (m_output_struct['m_API_ver'].iloc[0] == '2.0.1' and \
         m_output_struct['m_service_var'].iloc[0] == '2.1.1') or \
         (m_output_struct['m_API_ver'].iloc[0] == '2.0.2' and \
         m_output_struct['m_service_var'].iloc[0] == '2.1.1'): # Line 710 in MATLAB. 
        
        if PPG_file_buffer.shape[0] > 0:
            flag_ACC = False # this input is PPG data.
            PPG_raw_buffer,\
                PPG_t_msec,\
                PPG_t_datetime,\
                PPG_t_msec_ver_2,\
                PPG_t_datetime_ver_2,\
                PPG_timestamp_ver_2_raw = my_func_interpolate_watch_data_2_1_1(PPG_file_buffer,\
                                                                               fs_PPG,\
                                                                               flag_ACC)
            PPG_Samsung_HR = PPG_file_buffer[:,2] # get Samsung HR.
            if ACC_file_buffer.shape[0] > 0 and np.ndim(ACC_file_buffer) > 1: 
                if ACC_file_buffer.shape[1] > 1:
                    flag_ACC = True # this input is ACC data.
                    
                    ACC_raw_buffer,\
                        ACC_t_msec,\
                        ACC_t_datetime,\
                        ACC_t_msec_ver_2,\
                        ACC_t_datetime_ver_2,\
                        ACC_timestamp_ver_2_raw = my_func_interpolate_watch_data_2_1_1(ACC_file_buffer,\
                                                                                       fs_ACC,\
                                                                                       flag_ACC)
            else:
                # ACC file does not exist.
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
                ACC_t_msec = PPG_t_msec
                ACC_t_datetime = PPG_t_datetime
            
        else:
            if ACC_file_buffer.shape[0] > 0:
                flag_ACC = True # this input is ACC data.
                ACC_raw_buffer,\
                    ACC_t_msec,\
                    ACC_t_datetime,\
                    ACC_t_msec_ver_2,\
                    ACC_t_datetime_ver_2,\
                    ACC_timestamp_ver_2_raw = my_func_interpolate_watch_data_2_1_1(ACC_file_buffer,\
                                                                                   fs_ACC,\
                                                                                   flag_ACC)         
                PPG_raw_buffer = np.zeros(ACC_raw_buffer.shape)
                PPG_t_msec = ACC_t_msec
                PPG_t_datetime = ACC_t_datetime
            
                PPG_Samsung_HR = np.zeros(ACC_raw_buffer.shape)
            else:
                print('both PPG and ACC are empty.')
            
                PPG_raw_buffer = np.zeros((fs_PPG * 30,1))
                ACC_raw_buffer = np.zeros(PPG_raw_buffer.shape)
            
                PPG_t_msec = np.zeros(PPG_raw_buffer.shape)
                ACC_t_msec = np.zeros(PPG_raw_buffer.shape)
            
                PPG_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/29/2023: I cannot find NaT variable in Python
                ACC_t_datetime = [None] * PPG_raw_buffer.shape[0] # 01/29/2023: I cannot find NaT variable in Python
            
                PPG_Samsung_HR = np.zeros(ACC_raw_buffer.shape)
                
        PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
        if PPG_4_dig_str == '0000': # Line 663 in MATLAB.
            # it is segment 0000:
            # Worked: Plan 1: test cut first few seconds:
            cut_sec = 2;
            PPG_raw_buffer = PPG_raw_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
            PPG_t_msec = PPG_t_msec[cut_sec*fs_PPG:]
            PPG_t_datetime = PPG_t_datetime[cut_sec*fs_PPG:]
            PPG_file_buffer = PPG_file_buffer[cut_sec*fs_PPG:]
            PPG_Samsung_HR = PPG_Samsung_HR[cut_sec*fs_PPG:]
            
            PPG_t_msec_ver_2 = PPG_t_msec_ver_2[cut_sec*fs_PPG:]
            PPG_t_datetime_ver_2 = PPG_t_datetime_ver_2[cut_sec*fs_PPG:]
        else:    
            pass
        
        PPG_timestamp_start = PPG_t_datetime[0]
        PPG_timestamp_end = PPG_t_datetime[-1]
        # check file name time lapse vs timestamp lapse:
        # file name time lapse is in PPG_struct;
        if PPG_file_buffer.shape[0] > 0:
            
            temp_HR_file_buffer = np.stack([PPG_file_buffer[:,3],PPG_Samsung_HR],axis=1) # 4th column is the old timestamp.
            
            SamsungHR_buffer,\
             SamsungHR_t_msec,\
             SamsungHR_t_datetime,_,_,_ = my_func_interpolate_watch_data(temp_HR_file_buffer,fs_PPG)
        else:
            if ACC_file_buffer.shape[0] > 0:
                SamsungHR_t_msec = ACC_t_msec
                SamsungHR_t_datetime = ACC_t_datetime

                SamsungHR_buffer = np.zeros(ACC_raw_buffer.shape)

                PPG_4_dig_str = PPG_file_name[28:32] # 28, 29, 30, 31. 916_2020_06_06_00_12_24_ppg_0000.txt
                if PPG_4_dig_str == '0000':
                    # it is segment 0000:
                    # Worked: Plan 1: test cut first few seconds:
                    cut_sec = 2;
                    SamsungHR_buffer = SamsungHR_buffer[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
                    SamsungHR_t_msec = SamsungHR_t_msec[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
                    SamsungHR_t_datetime = SamsungHR_t_datetime[cut_sec*fs_PPG:] # From cut_sec*fs_PPG to the end
            else:
                print('both PPG and ACC are empty.')
                SamsungHR_buffer = np.zeros(ACC_raw_buffer.shape)
                SamsungHR_t_msec = np.zeros(ACC_raw_buffer.shape)
                SamsungHR_t_datetime = np.zeros(ACC_raw_buffer.shape)
                        
    elif m_output_struct['m_API_ver'].iloc[0] == '2.1.3':
        PPG_raw_buffer = PPG_file_buffer[:,1]
        ACC_raw_buffer = ACC_file_buffer[:,1]
        PPG_t_msec = []
        PPG_t_datetime = []
        ACC_t_msec = []
        ACC_t_datetime = []
        SamsungHR_buffer = []
        SamsungHR_t_msec = []
        SamsungHR_t_datetime = []
        PPG_timestamp_start = []
        PPG_timestamp_end = []
        add_1_day_flag = []
        PPG_t_msec_ver_2 = []
        PPG_t_datetime_ver_2 = []
        ACC_t_msec_ver_2 = []
        ACC_t_datetime_ver_2 = []
        PPG_timestamp_ver_2_raw = []
        ACC_timestamp_ver_2_raw = []
    else:
        # unknown API version occurred.
        print('unknown API occurred!')
        # keyboard;
        
    if not 'ACC_timestamp_ver_2_raw' in locals(): # exists var in MATLAB.
        PPG_timestamp_ver_2_raw = []
        ACC_timestamp_ver_2_raw = []

    return PPG_raw_buffer,\
        PPG_t_msec,\
        PPG_t_datetime,\
        ACC_raw_buffer,\
        ACC_t_msec,\
        ACC_t_datetime,\
        SamsungHR_buffer,\
        SamsungHR_t_msec,\
        SamsungHR_t_datetime,\
        PPG_timestamp_start,\
        PPG_timestamp_end,\
        add_1_day_flag,\
        PPG_t_msec_ver_2,\
        PPG_t_datetime_ver_2,\
        ACC_t_msec_ver_2,\
        ACC_t_datetime_ver_2,\
        PPG_timestamp_ver_2_raw,\
        ACC_timestamp_ver_2_raw
        
def my_func_UID_ECG_final_path(UID,\
                               HPC_flag,\
                               root_data_path,\
                               root_output_path):
    if HPC_flag:
        clinical_ECG_root = os.path.join(root_data_path,'DAT_files_for_Cardea_Solo','Mail_Kamran_2022_05_26','Clinical_Trial')
        AF_ECG_root = os.path.join(root_data_path,'AF_trial','Patch_ECG_Converted_Data')
        LinearInterp_root = os.path.join(root_data_path,'UMass_Patch_Adjuciation','Linear_Interp_Timestamp')
    else:
        clinical_ECG_root = os.path.join(root_data_path,'DAT_files_for_Cardea_Solo','Mail_Kamran_2022_05_26','Clinical_Trial')
        AF_ECG_root = os.path.join(root_data_path,'AF_trial','Patch_ECG_Converted_Data')
        LinearInterp_root = os.path.join(root_output_path,'UMass_Patch_Adjuciation','Linear_Interp_Timestamp')

    if UID == '001':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = '09/03/2019 10:22:52.000' # 24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '09/10/2019 17:57:00.000' #'08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '002':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/03/2019 15:45:45.000'#  24 hours time.  
        Patch_A_start_time = '09/03/2019 15:43:45.000' # 24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '09/11/2019 09:37:55.200'#  24 hours time.
        Patch_B_start_time = '09/11/2019 09:27:31.200' # 24 hours time.
        UMass_type = 'NSR'
    elif UID == '003':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/10/2019 14:52:06.000'#  24 hours time.
        Patch_A_start_time = '09/10/2019 14:51:55.000' # 24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '09/17/2019 10:57:16.000'#  24 hours time.
        Patch_B_start_time = '09/17/2019 10:56:57.000' # 24 hours time.
        UMass_type = 'NSR'
    elif UID == '004':
        # keyboard;
        pass
    elif UID == '005':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/16/2019 13:52:00.000'#  24 hours time.
        Patch_A_start_time = '09/16/2019 13:53:06.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '09/23/2019 10:02:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '006':
        # keyboard;
        pass
    elif UID == '007':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/20/2019 10:35:49.000'#  24 hours time.
        Patch_A_start_time = '09/20/2019 10:59:29.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '09/27/2019 07:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '009':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '10/07/2019 11:00:45.000'#  24 hours time.
        Patch_A_start_time = '10/07/2019 10:59:30.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '10/14/2019 10:30:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '011':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '10/21/2019 12:26:09.000'#  24 hours time.
        Patch_A_start_time = '10/21/2019 12:25:33.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '10/28/2019 18:58:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '012':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '10/31/2019 11:52:30.500'#  24 hours time.
        Patch_A_start_time = '10/31/2019 11:51:38.500'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '11/07/2019 09:14:44.000'#  24 hours time.
        Patch_B_start_time = '11/07/2019 09:14:27.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '013':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '11/06/2019 11:41:35.000'#  24 hours time.
        Patch_A_start_time = '11/06/2019 11:41:25.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '11/14/2019 13:45:31.000'#  24 hours time.
        Patch_B_start_time = '11/14/2019 13:45:24.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '014':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '11/12/2019 10:46:00.000'#  24 hours time.
        Patch_A_start_time = '11/12/2019 10:45:55.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '11/19/2019 15:57:58.000'#  24 hours time.
        Patch_B_start_time = '11/19/2019 15:56:41.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '017':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '12/04/2019 16:32:00.000'#  24 hours time.
        Patch_A_start_time = '12/04/2019 16:32:21.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '12/11/2019 09:15:00.000'#  24 hours time.
        Patch_B_start_time = '12/11/2019 09:18:29.000'#  24 hours time. # Testing on 05/10/2022
        UMass_type = 'NSR'
    elif UID == '018':
        # keyboard;
        pass
    elif UID == '019':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '01/03/2020 15:06:00.000'#  24 hours time.
        Patch_A_start_time = '01/03/2020 15:07:00.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '01/10/2020 20:00:00.000'#  24 hours time.
        Patch_B_start_time = '01/10/2020 19:59:57.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '020':
        test_ECG_path_A = []# [clinical_ECG_root,'\Pulsewatch',UID,'A'];
        Patch_A_start_time = ''# '01/07/2020 14:47:00.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '01/14/2020 15:31:00.000'#  24 hours time.
        Patch_B_start_time = '01/14/2020 15:30:32.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '021':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '01/14/2020 16:36:31.500'#  24 hours time.
        Patch_A_start_time = '01/14/2020 16:35:39.500'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '01/21/2020 17:31:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '022':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '01/22/2020 15:26:32.000'#  24 hours time.
        Patch_A_start_time = '01/22/2020 15:26:11.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '01/29/2020 15:25:39.594'#  24 hours time.
        Patch_B_start_time = '01/29/2020 15:25:10.594'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '024':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '01/29/2020 13:06:31.500'#  24 hours time.
        Patch_A_start_time = '01/29/2020 13:05:47.500'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '02/05/2020 18:08:51.600'#  24 hours time.
        Patch_B_start_time = '02/05/2020 18:08:31.600'#  24 hours time.
        UMass_type = 'NSR'    
    elif UID == '026':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '02/06/2020 14:49:00.000'#  24 hours time.
        Patch_A_start_time = '02/06/2020 14:49:31.900'#  24 hours time.
        test_ECG_path_B = []# [clinical_ECG_root,'\Pulsewatch',UID,'B'];
        Patch_B_start_time = ''# '02/13/2020 11:45:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '027':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '02/11/2020 14:00:11.200'#  24 hours time.
        Patch_A_start_time = '02/11/2020 13:59:48.200'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '02/18/2020 10:14:38.660'#  24 hours time.
        Patch_B_start_time = '02/18/2020 10:14:35.660'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '028':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '02/13/2020 13:31:05.000'#  24 hours time.
        Patch_A_start_time = '02/13/2020 13:30:47.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '02/20/2020 23:13:25.000'#  24 hours time.
        Patch_B_start_time = '02/20/2020 23:13:24.000'#  24 hours time.
        test_ECG_path_C = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'C')
#         Patch_C_start_time = '02/21/2020 11:47:30.000'#  24 hours time.
        Patch_C_start_time = '02/21/2020 11:47:17.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '029':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '02/14/2020 11:35:46.000'#  24 hours time.
        Patch_A_start_time = '02/14/2020 11:35:45.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '02/21/2020 11:19:31.000'#  24 hours time.
        Patch_B_start_time = '02/21/2020 11:18:08.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '030':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '02/19/2020 17:10:24.000'#  24 hours time.
        Patch_A_start_time = '02/19/2020 17:10:05.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '02/26/2020 18:13:21.000'#  24 hours time.
        Patch_B_start_time = '02/26/2020 18:13:09.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '032':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '02/26/2020 13:58:00.000'#  24 hours time.
        Patch_A_start_time = '02/26/2020 13:57:32.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '03/04/2020 20:44:48.000'#  24 hours time.
        Patch_B_start_time = '03/04/2020 20:44:28.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '034':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '03/02/2020 14:00:01.000'#  24 hours time.
        Patch_A_start_time = '03/02/2020 13:58:38.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '03/09/2020 08:01:03.000'#  24 hours time.
        Patch_B_start_time = '03/09/2020 8:00:23.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '035':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = '07/24/2020 09:31:59.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '036':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '06/26/2020 13:11:11.273'#  24 hours time.
        Patch_A_start_time = '06/26/2020 13:10:42.273'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '07/03/2020 08:06:17.141'#  24 hours time.
        Patch_B_start_time = '07/03/2020 08:05:26.141'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '037':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = ''#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '07/10/2020 15:00:00.000'#  24 hours time.
        Patch_B_start_time = '07/03/2020 16:25:12.578'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '038':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '06/24/2020 11:20:23.495'#  24 hours time.
        Patch_A_start_time = '06/24/2020 11:13:19.495'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '07/01/2020 07:50:52.451'#  24 hours time.
        Patch_B_start_time = '07/01/2020 7:49:25.451'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '039':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '06/29/2020 16:58:50.000'#  24 hours time.
        Patch_A_start_time = '06/29/2020 16:58:32.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '07/06/2020 20:57:22.267'#  24 hours time.
        Patch_B_start_time = '07/06/2020 20:57:17.267'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '040':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = '08/07/2020 13:05:00.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '041':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '07/15/2020 15:09:31.163'#  24 hours time.
        Patch_A_start_time = '07/15/2020 15:09:19.163'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '07/22/2020 17:15:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '042':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '07/28/2020 10:36:57.039'#  24 hours time.
        Patch_A_start_time = '07/28/2020 10:36:27.039'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '08/04/2020 10:15:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '043':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = '07/28/2020 14:39:00.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '08/04/2020 18:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '044':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '08/10/2020 20:22:42.000'#  24 hours time.
        Patch_A_start_time = '08/10/2020 20:19:23.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = ''# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '045':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '08/07/2020 13:43:44.793'#  24 hours time.
        Patch_A_start_time = '08/07/2020 13:43:11.793'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '08/14/2020 13:58:24.000'#  24 hours time.
        Patch_B_start_time = '08/14/2020 13:57:38.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '047':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/01/2020 13:07:00.000'#  24 hours time.
        Patch_A_start_time = '09/01/2020 13:06:59.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '09/08/2020 13:51:16.400'# '08/18/2020 22:00:00.000'#  24 hours time.
        Patch_B_start_time = '09/08/2020 13:36:24.400'# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '049':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/04/2020 13:36:31.375'#  24 hours time.
        Patch_A_start_time = '09/04/2020 13:36:35.375'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '09/11/2020 20:46:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '050':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = '09/25/2020 11:58:00.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = ''# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '051':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = ''#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '10/05/2020 10:30:00.000'# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR' 
    elif UID == '052':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/24/2020 09:22:32.000'#  24 hours time.
        Patch_A_start_time = '09/24/2020 09:17:30.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '10/03/2020 10:55:09.000'# '08/18/2020 22:00:00.000'#  24 hours time.
        Patch_B_start_time = '10/03/2020 10:51:42.000'# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '053':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '09/25/2020 10:07:56.600'#  24 hours time.
        Patch_A_start_time = '09/25/2020 10:07:54.600'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '10/02/2020 11:47:48.890'# '08/18/2020 22:00:00.000'#  24 hours time.
        Patch_B_start_time = '10/02/2020 11:47:06.890'# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '054':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '10/02/2020 08:51:27.000'#  24 hours time.
        Patch_A_start_time = '10/02/2020 08:50:56.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '10/09/2020 09:16:30.000'# '08/18/2020 22:00:00.000'#  24 hours time.
        Patch_B_start_time = '10/09/2020 09:13:34.000'# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '055':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '10/05/2020 10:50:29.000'#  24 hours time.
        Patch_A_start_time = '10/05/2020 10:50:16.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '10/08/2020 11:47:00.000'# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '056':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '10/20/2020 15:09:05.000'#  24 hours time.
        Patch_A_start_time = '10/20/2020 15:08:45.000'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = ''# '08/18/2020 22:00:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '057':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '10/19/2020 14:57:33.560'#  24 hours time.
        Patch_A_start_time = '10/19/2020 14:55:27.560'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '10/26/2020 12:20:10.810'#  24 hours time.
        Patch_B_start_time = '10/26/2020 12:18:53.810'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '058':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '10/29/2020 09:51:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '11/06/2020 17:43:22.140'#  24 hours time.
#          Patch_B_start_time = '11/06/2020 17:43:22.140'#  24 hours time.
# I commented because UMass swtiched patch A and B. I am not sure if I have
# to rename all patch A and patch B 1-hour files.
         UMass_type = 'NSR'
    elif UID == '062':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '11/20/2020 11:02:51.100'#  24 hours time.
        Patch_A_start_time = '11/20/2020 10:57:29.100'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '11/27/2020 14:50:30.330'#  24 hours time.
        Patch_B_start_time = '11/27/2020 14:43:02.330'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '063':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '12/03/2020 10:39:38.400'#  24 hours time.
        Patch_A_start_time = '12/03/2020 10:39:07.400'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '064':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '12/08/2020 11:39:13.082'#  24 hours time.
        Patch_A_start_time = '12/08/2020 11:38:43.082'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '12/15/2020 07:51:56.705'#  24 hours time.
        Patch_B_start_time = '12/15/2020 7:49:31.705'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '065':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = ''#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '12/17/2020 12:00:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '067':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '12/15/2020 15:51:01.780'#  24 hours time.
        Patch_A_start_time = '12/15/2020 15:45:46.780'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '12/22/2020 19:34:36.025'#  24 hours time.
        Patch_B_start_time = '12/22/2020 19:34:33.025'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '068':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '12/22/2020 12:36:01.000'#  24 hours time.
        Patch_A_start_time = '12/22/2020 12:35:59.500'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '12/29/2020 11:06:04.881'#  24 hours time.
        Patch_B_start_time = '12/29/2020 11:05:58.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '069':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
        Patch_A_start_time = ''#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
        Patch_B_start_time = '01/06/2021 19:30:00.000'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '070':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '01/05/2021 12:23:00.497'#  24 hours time.
        Patch_A_start_time = '01/05/2021 12:22:54.497'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '01/15/2021 10:34:59.275'#  24 hours time.
        Patch_B_start_time = '01/15/2021 10:33:17.275'#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '071':
        test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = '01/07/2021 16:10:05.393'#  24 hours time.
        Patch_A_start_time = '01/07/2021 16:10:04.393'#  24 hours time.
        test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = '01/15/2021 21:53:26.000'#  24 hours time.
        Patch_B_start_time = '01/15/2021 21:53:07.000'#  24 hours time.
        UMass_type = 'NSR'
#     elif UID == '072':
#         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#         Patch_A_start_time = ''#  24 hours time.
#         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#         Patch_B_start_time = ''#  24 hours time.
#        UMass_type = 'NSR'
    elif UID == '073':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '01/21/2021 11:54:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '074':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '01/28/2021 11:08:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '02/04/2021 11:00:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '075':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '02/05/2021 14:07:05.000'#  24 hours time.
         Patch_A_start_time = '02/05/2021 14:07:03.500'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '077':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '02/10/2021 08:19:48.561'#  24 hours time.
         Patch_A_start_time = '02/10/2021 8:09:25.561'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '02/17/2021 08:52:22.000'#  24 hours time.
         Patch_B_start_time = '02/17/2021 8:52:12.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '078':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '02/11/2021 22:46:10.000'#  24 hours time.
         Patch_A_start_time = '02/11/2021 10:41:13.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '02/17/2021 12:20:40.000'#  24 hours time.
         Patch_B_start_time = '02/17/2021 12:18:58.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '080':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '02/22/2021 11:42:03.000'#  24 hours time.
         Patch_A_start_time = '02/22/2021 11:39:34.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '03/01/2021 10:27:31.300'#  24 hours time.
#          Patch_B_start_time = '03/01/2021 10:27:29.300'#  24 hours time.
         Patch_B_start_time = '03/01/2021 10:09:05.300'#  
         UMass_type = 'NSR'
    elif UID == '082':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '03/05/2021 11:45:35.000'#  24 hours time.
         Patch_A_start_time = '03/05/2021 11:44:44.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '03/12/2021 10:11:14.416'#  24 hours time.
         Patch_B_start_time = '03/12/2021 10:11:04.416'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '083':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '03/11/2021 11:41:44.000'#  24 hours time.
         Patch_A_start_time = '03/11/2021 11:41:52.500'#  24 hours time.
         test_ECG_path_B = [''];
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '084':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '03/17/2021 10:10:41.000'#  24 hours time.
         Patch_A_start_time = '03/17/2021 10:08:49.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '03/27/2021 09:56:39.000'#  24 hours time.
         Patch_B_start_time = '03/27/2021 9:53:05.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '086':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '03/22/2021 10:44:05.000'#  24 hours time.
         Patch_A_start_time = '03/22/2021 10:41:09.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '03/29/2021 10:34:55.700'#  24 hours time.
         Patch_B_start_time = '03/29/2021 10:26:52.700'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '087':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '03/23/2021 09:47:53.000'#  24 hours time.
         Patch_A_start_time = '03/23/2021 9:46:16.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '03/30/2021 10:15:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '088':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '03/23/2021 16:38:28.000'#  24 hours time.
         Patch_A_start_time = '03/23/2021 16:37:10.500'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '03/30/2021 23:32:42.900'#  24 hours time.
         Patch_B_start_time = '03/30/2021 23:32:32.900'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '089':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = ''#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '04/06/2021 10:16:55.000'#  24 hours time.
         Patch_B_start_time = '04/06/2021 09:58:52.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '090':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '03/30/2021 12:15:45.000'#  24 hours time.
         Patch_A_start_time = '03/30/2021 12:12:27.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '04/06/2021 07:38:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '091':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '04/05/2021 08:53:41.000'#  24 hours time.
         Patch_A_start_time = '04/05/2021 08:42:09.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '04/12/2021 16:29:06.500'#  24 hours time.
         Patch_B_start_time = '04/12/2021 16:15:22.500'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '093':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '04/22/2021 11:04:05.000'#  24 hours time.
         Patch_A_start_time = '04/22/2021 11:01:54.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '04/29/2021 09:19:00.000'#  24 hours time.
         Patch_B_start_time = '04/29/2021 9:16:52.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '094':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '04/23/2021 12:20:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '096':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = ''#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '05/17/2021 09:55:00.000'#  24 hours time.
         test_ECG_path_C = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'C')
         Patch_C_start_time = '05/20/2021 08:38:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '098':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '05/12/2021 16:12:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '05/19/2021 08:00:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '099':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '05/19/2021 12:38:33.000'#  24 hours time.
         Patch_A_start_time = '05/19/2021 12:36:42.500'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '100':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '05/27/2021 09:54:34.700'#  24 hours time.
         Patch_A_start_time = '05/27/2021 9:51:21.700'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '06/03/2021 07:21:43.000'#  24 hours time.
         Patch_B_start_time = '06/03/2021 07:16:49.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '101':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '06/01/2021 10:16:55.000'#  24 hours time.
         Patch_A_start_time = '06/01/2021 10:16:51.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '06/08/2021 11:00:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '102':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '06/02/2021 11:45:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '06/10/2021 05:50:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '104':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '06/09/2021 14:41:17.000'#  24 hours time.
         Patch_A_start_time = '06/09/2021 14:40:13.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '06/16/2021 11:20:24.200'#  24 hours time.
         Patch_B_start_time = '06/16/2021 11:19:12.200'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '105':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '06/09/2021 14:41:17.000'#  24 hours time.
         Patch_A_start_time = '06/16/2021 14:24:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '06/16/2021 11:20:24.200'#  24 hours time.
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '106':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '06/21/2021 12:28:19.000'#  24 hours time.
         Patch_A_start_time = '06/21/2021 12:28:14.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '06/27/2021 18:30:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '108':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '06/28/2021 10:03:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '07/05/2021 13:06:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '109':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '06/29/2021 09:53:10.000'#  24 hours time.
         Patch_A_start_time = '06/29/2021 09:52:54.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '110':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '07/01/2021 10:03:01.000'#  24 hours time.
         Patch_A_start_time = '07/01/2021 10:02:27.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '07/08/2021 07:54:22.000'#  24 hours time.
         Patch_B_start_time = '07/08/2021 7:54:04.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '111':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '07/02/2021 11:16:55.000'#  24 hours time.
         Patch_A_start_time = '07/02/2021 11:16:40.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '07/09/2021 09:20:00.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '112':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '07/06/2021 10:32:54.000'#  24 hours time.
         Patch_A_start_time = '07/06/2021 10:32:25.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '07/13/2021 10:45:10.000'#  24 hours time.
         Patch_B_start_time = '07/13/2021 10:43:22.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '113':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '07/15/2021 11:48:33.000'#  24 hours time.
         Patch_A_start_time = '07/15/2021 11:47:30.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '07/22/2021 05:59:39.000'#  24 hours time.
         Patch_B_start_time = '07/22/2021 5:56:20.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '116':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = '07/30/2021 12:58:00.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = ''#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '118':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '08/02/2021 16:08:26.000'#  24 hours time.
         Patch_A_start_time = '08/02/2021 16:07:20.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '08/09/2021 20:28:09.000'#  24 hours time.
         Patch_B_start_time = '08/09/2021 20:26:49.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '119':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
#          Patch_A_start_time = '08/04/2021 04:55:39.000'#  24 hours time.
         Patch_A_start_time = '08/04/2021 04:43:25.000'#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
#          Patch_B_start_time = '08/10/2021 09:44:16.000'#  24 hours time.
         Patch_B_start_time = '08/10/2021 09:35:21.000'#  24 hours time.
         UMass_type = 'NSR'
    elif UID == '120':
         test_ECG_path_A = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'A')
         Patch_A_start_time = ''#  24 hours time.
         test_ECG_path_B = os.path.join(clinical_ECG_root,'Pulsewatch'+UID+'B')
         Patch_B_start_time = '08/12/2021 05:36:14.900'#  24 hours time.
         UMass_type = 'NSR'

 # AF trial 300-330
    elif UID == '300':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '01/03/2020 16:26:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '301':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '01/06/2020 09:42:29.800'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '302':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '01/06/2020 11:06:48.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '303':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '01/06/2020 16:53:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '304':
        # keyboard;
        pass
    elif UID == '305':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '01/07/2020 14:31:09.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '306':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '01/13/2020 09:28:15.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '307':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '01/13/2020 12:14:33.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '308':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '02/19/2020 09:23:29.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '309':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '02/21/2020 15:20:16.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '310':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '03/04/2020 13:56:34.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '311':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '03/06/2020 14:31:55.500'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '312':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '03/12/2020 08:40:36.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '313':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '07/30/2020 14:43:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '314':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '07/31/2020 16:43:00.000'#  24 hours time.
        Patch_A_start_time = '07/31/2020 16:22:02.500'#  09/06/2020: try with moving the start time earlier.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '315':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/03/2020 15:38:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '316':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/03/2020 16:29:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '317':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/04/2020 12:42:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '318':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/04/2020 14:11:04.952'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '319':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '08/05/2020 11:28:00.000'#  24 hours time.
        Patch_A_start_time = '08/05/2020 11:25:02.600'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '320':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/05/2020 13:12:35.736'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '321':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '08/06/2020 10:04:00.000'#  24 hours time.
        Patch_A_start_time = '08/06/2020 10:04:31.163'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '322':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '08/06/2020 14:39:00.000'#  24 hours time.
        Patch_A_start_time = '08/06/2020 14:35:17.021'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '323':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '08/06/2020 14:39:00.000'#  24 hours time.
        Patch_A_start_time = '08/06/2020 15:09:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Unknown'];
    elif UID == '324':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/06/2020 16:13:20.400'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '325':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/06/2020 16:43:07.800'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '326':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/11/2020 12:13:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '327':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '08/11/2020 14:27:00.000'#  24 hours time.
        Patch_A_start_time = '08/11/2020 14:27:01.757'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'NSR'
    elif UID == '328':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '08/11/2020 15:47:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['NSR w AFlutter'];
    elif UID == '329':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '08/12/2020 10:41:00.000'#  24 hours time.
        Patch_A_start_time = '08/12/2020 10:40:55.574'#  09/08/2020, Dong found this from all signal trace plot
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '330':   
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '05/24/2021 12:41:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['NSR w AFlutter'];
    elif UID == '400':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '05/24/2021 12:11:00.000'#  24 hours time.
        Patch_A_start_time = '05/24/2021 12:10:10.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '402':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '07/01/2021 14:16:23.000'#  24 hours time.
        Patch_A_start_time = '07/01/2021 14:14:31.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '403':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '09/13/2021 14:52:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '404':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '09/16/2021 11:30:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '405':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '09/21/2021 15:05:44.000'#  24 hours time.
        Patch_A_start_time = '09/21/2021 15:03:58.500'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '406':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '09/23/2021 11:14:21.000'#  24 hours time.
        Patch_A_start_time = '09/23/2021 11:11:54.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '407':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '09/28/2021 10:22:53.000'#  24 hours time.
        Patch_A_start_time = '09/28/2021 10:21:52.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '408':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '09/30/2021 10:16:09.000'#  24 hours time.
        Patch_A_start_time = '09/30/2021 10:13:25.750'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '409':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#          Patch_A_start_time = '10/05/2021 13:45:36.000'#  24 hours time.
        Patch_A_start_time = '10/05/2021 13:44:42.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '410':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '10/07/2021 16:04:35.000'#  24 hours time.
        Patch_A_start_time = '10/07/2021 16:03:29.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '411':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '10/12/2021 14:09:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '412':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '10/19/2021 11:26:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '413':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '10/19/2021 14:31:07.600'#  24 hours time.
        Patch_A_start_time = '10/19/2021 14:29:18.100'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '414':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '10/22/2021 16:30:11.000'#  24 hours time.
        Patch_A_start_time = '10/22/2021 16:29:47.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '415':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '10/25/2021 10:56:31.000'#  24 hours time.
        Patch_A_start_time = '10/25/2021 10:54:59.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '416':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '10/28/2021 11:45:48.000'#  24 hours time.
        Patch_A_start_time = '10/28/2021 11:43:52.250'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '417':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '11/02/2021 11:23:30.000'#  24 hours time.
        Patch_A_start_time = '11/02/2021 11:21:29.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '418':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
        Patch_A_start_time = '11/05/2021 11:56:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '419':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '11/08/2021 11:26:52.000'#  24 hours time.
        Patch_A_start_time = '11/08/2021 11:26:03.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '420':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '11/30/2021 10:23:26.000'#  24 hours time.
        Patch_A_start_time = '11/30/2021 10:21:56.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '421':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '12/06/2021 10:34:55.000'#  24 hours time.
        Patch_A_start_time = '12/06/2021 10:34:40.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '422':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '12/13/2021 09:57:54.000'#  24 hours time.
        Patch_A_start_time = '12/13/2021 09:57:47.545'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'
    elif UID == '423':
        test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = '12/16/2021 08:36:08.000'#  24 hours time.
        Patch_A_start_time = '12/16/2021 08:36:03.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = 'AF'

#     elif UID == ''
#         test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = ''#  24 hours time.
#         test_ECG_path_B = [];
#         Patch_B_start_time = ''#  24 hours time.
#         UMass_type = 'AF'
#     elif UID == ''
#         test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = ''#  24 hours time.
#         test_ECG_path_B = [];
#         Patch_B_start_time = ''#  24 hours time.
#         UMass_type = 'AF'
#     elif UID == ''
#         test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = ''#  24 hours time.
#         test_ECG_path_B = [];
#         Patch_B_start_time = ''#  24 hours time.
#         UMass_type = 'AF'
#     elif UID == ''
#         test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = ''#  24 hours time.
#         test_ECG_path_B = [];
#         Patch_B_start_time = ''#  24 hours time.
#         UMass_type = 'AF'
#     elif UID == ''
#         test_ECG_path_A = os.path.join(AF_ECG_root,UID)
#         Patch_A_start_time = ''#  24 hours time.
#         test_ECG_path_B = [];
#         Patch_B_start_time = ''#  24 hours time.
#         UMass_type = 'AF'
    elif UID == '913_02042021':
        test_ver_2_path = r'R:\ENGR_Chon\Dong\myGalaxyWatchDatabase\Test_APP_ver_2_0\Dong_concatenated_ECG';
        test_ECG_path_A = [test_ver_2_path];
        Patch_A_start_time = '02/04/2021 12:59:02.500'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Ver2.0.1 Testing'];
    elif UID == '914_2021042601':
        if HPC_flag:
            test_ver_2_path = r'/scratch/doh16101-lab/Test_Up_Time_2_1_0/Dong_concatenated_ECG_protected';
        else:
            test_ver_2_path = r'R:\ENGR_Chon\Dong\myGalaxyWatch3Database\Test_Up_Time_2_1_0\Dong_concatenated_ECG_protected';
        test_ECG_path_A = [test_ver_2_path];
        Patch_A_start_time = '04/26/2021 11:31:02.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Ver2.1.0 Testing'];
    elif UID == '999_2021042601':
        if HPC_flag:
            test_ver_2_path = r'/scratch/doh16101-lab/Test_Up_Time_2_1_0/Andrew_concatenated_ECG_protected/Andrew_concatenated_ECG_0426';           
        else:
            test_ver_2_path = r'R:\ENGR_Chon\Dong\myGalaxyWatchDatabase\Test_Up_Time_2_1_0\Andrew_concatenated_ECG_0426';
        test_ECG_path_A = [test_ver_2_path];
        Patch_A_start_time = '04/26/2021 11:30:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Ver2.1.0 Testing'];
    elif UID == '999_2021042901':
        if HPC_flag:
            test_ver_2_path = r'/scratch/doh16101-lab/Test_Up_Time_2_1_0/Andrew_concatenated_ECG_protected/999_2021042901';
        else:
            test_ver_2_path = r'R:\ENGR_Chon\Dong\myGalaxyWatchDatabase\Test_Up_Time_2_1_0\Andrew_1_hour_ECG\999_2021042901';
        test_ECG_path_A = [test_ver_2_path];
        Patch_A_start_time = '04/29/2021 10:35:15.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Ver2.1.0 Testing'];
    elif UID == '999_2021050101':
        if HPC_flag:
            test_ver_2_path = r'/scratch/doh16101-lab/Test_Up_Time_2_1_0/Andrew_concatenated_ECG_protected/Andrew_concatenated_ECG_0501';
        else:
            test_ver_2_path = r'R:\ENGR_Chon\Dong\myGalaxyWatchDatabase\Test_Up_Time_2_1_0\Andrew_concatenated_ECG_0501';
        test_ECG_path_A = [test_ver_2_path];
        Patch_A_start_time = '05/01/2021 10:34:20.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Ver2.1.0 Testing'];
    elif UID == '999_2021050501':
        if HPC_flag:
            test_ver_2_path = r'/scratch/doh16101-lab/Test_Up_Time_2_1_0/Andrew_concatenated_ECG_protected/Andrew_concatenated_ECG_0505';
        else:
            test_ver_2_path = r'R:\ENGR_Chon\Dong\myGalaxyWatchDatabase\Test_Up_Time_2_1_0\Andrew_concatenated_ECG_0505';
        test_ECG_path_A = [test_ver_2_path];
        Patch_A_start_time = '05/05/2021 14:37:00.000'#  24 hours time.
        test_ECG_path_B = [];
        Patch_B_start_time = ''#  24 hours time.
        UMass_type = ['Ver2.1.0 Testing'];
    else:
        print('not finished pruning!');
        # keyboard;
    if not 'test_ECG_path_C' in locals(): # exists var in MATLAB. only UID 028 has patch C.
        test_ECG_path_C = []
        Patch_C_start_time = []
    return test_ECG_path_A,\
            Patch_A_start_time,\
            test_ECG_path_B,\
            Patch_B_start_time,\
            test_ECG_path_C,\
            Patch_C_start_time,\
            UMass_type,\
            LinearInterp_root
            
            
def my_interp_Solo_ECG(start_blc_idx,\
                       fs_ECG,\
                       Solo_time_intv,\
                       ECG_init_datetime):

    if start_blc_idx == 0:
        # Warning: make sure this starts from 1 and not end with start_blc_idx - 1.
        x_Sample_Value = np.array((1,(start_blc_idx+1) * fs_ECG*60*10)) # Every 10-min sample.
        v = [float(0),float(Solo_time_intv[start_blc_idx])]
    else:
        x_Sample_Value = np.array((start_blc_idx,start_blc_idx+1)) * (fs_ECG*60*10) # Every 10-min sample.
        v = [float(tt) for tt in Solo_time_intv[start_blc_idx-1:start_blc_idx+1]]
        
    xq_Linear_Interp = np.arange(x_Sample_Value[0],x_Sample_Value[1]+1) # It should be 150001, not 150000.
    
    
    f = interpolate.interp1d(x_Sample_Value, np.array(v), fill_value="extrapolate") # Must put fill_value="extrapolate"
    vq_Linear_Interp = f(xq_Linear_Interp) # interpolated PPG.
    
    sample_datetime = [ECG_init_datetime + datetime.timedelta(milliseconds=float(tt)*1000) for tt in vq_Linear_Interp]
    
    return sample_datetime, xq_Linear_Interp, vq_Linear_Interp

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
            
def my_extract_Solo_Beats(file_contents):
    Beats_info = re.search(r'<Beat_Location,Beat_Type,Rhythm>(.*?)</Beat_Location,Beat_Type,Rhythm>', file_contents, re.DOTALL).group(1).strip()
    Beats_info = Beats_info.split('\n')
    return Beats_info

def my_find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def my_func_Cardea_SOLO_know_patch_A_B(patch_A_Solo_ECG_name,\
                                        Patch_A_start_time,\
                                        patch_B_Solo_ECG_name,\
                                        Patch_B_start_time,\
                                        patch_C_Solo_ECG_name,\
                                        Patch_C_start_time,\
                                        PPG_timestamp_start,\
                                        fs_ECG,\
                                        df_2,\
                                        path_Solo_ECG):

    # =============================================================================
    # Load non-linear timestamp from Solo.ASCII.txt
    # =============================================================================
    # PPG_timestamp_start = our_tzone.localize(datetime.datetime.strptime('09/16/2019 14:59:30', '%m/%d/%Y %H:%M:%S'))
    # PPG_timestamp_end = our_tzone.localize(datetime.datetime.strptime('09/16/2019 15:00:00', '%m/%d/%Y %H:%M:%S'))
    
    our_tzone = pytz.timezone('America/New_York') # This line does not exist in MATLAB code. I do not want to pass it as a variable. 01/19/2023.
    
    if PPG_timestamp_start == None:
        print('PPG_timestamp_start is None')
        flag_in_patch_A = False
        flag_in_patch_B = False
        Patch_B_datetime = []
        flag_in_patch_C = False
        Patch_C_datetime = []
    else:
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
    
    return ECG_path,ECG_Date_string

def my_func_load_Solo_Beats(ECG_path):
    with open(os.path.join(ECG_path,'Solo.Beats.txt'), 'rt') as myfile:  # 'r' for reading, 't' for text mode.
        file_contents = myfile.read()
        Beats_info = my_extract_Solo_Beats(file_contents)
        
    Solo_Beat_Location = []
    Solo_Beat_Type = []
    Solo_Beat_Rhythm = []
    for rr, line in enumerate(Beats_info):
        columns = re.split(',',line)#(r'   ',x)# https://stackoverflow.com/questions/48917121/split-on-more-than-one-space
        if len(columns) > 2:
            Solo_Beat_Location.append(float(columns[0]))              # Read the entire file to a string
            Solo_Beat_Type.append(columns[1])
            Solo_Beat_Rhythm.append(columns[2])
        elif len(columns) == 2:
            print('Solo Beat only has two cols, check!')
            print('Line:',rr,', Content:',line)
            Solo_Beat_Location.append(float(columns[0]))              # Read the entire file to a string
            Solo_Beat_Type.append(columns[1])
            Solo_Beat_Rhythm.append([])
        elif len(columns) == 1:
            print('Solo Beat only has one col, check!')
            print('Line:',rr,', Content:',line)
            Solo_Beat_Location.append(float(columns[0]))              # Read the entire file to a string
            Solo_Beat_Type.append([])
            Solo_Beat_Rhythm.append([])
    
    Solo_Beat_Location = np.array(Solo_Beat_Location)
    
    return Solo_Beat_Location, Solo_Beat_Type, Solo_Beat_Rhythm

def my_func_load_Solo_ASCII(ECG_path):
    with open(os.path.join(ECG_path,'Solo.ASCII.txt'), 'rt') as myfile:  # 'r' for reading, 't' for text mode.
        file_contents = myfile.read()
        lastname_firstname, \
            start_date_duration, \
            Solo_time_intv, \
            patient_events = my_extract_Solo_ASCII(file_contents)
    return lastname_firstname,\
            start_date_duration,\
            Solo_time_intv,\
            patient_events
            
def my_func_extend_array(target_len,input_array):
    if isinstance(input_array,np.ndarray):
        remain_len = target_len - input_array.shape[0]
        empty_array = np.empty((remain_len,))
        empty_array[:] = np.nan
        return_array = np.concatenate((input_array,empty_array))
    elif isinstance(input_array,list):
        return_array = input_array[:target_len] + [np.nan]*(target_len - len(input_array))
        
    return return_array

def my_func_add_var_to_df(file_beats_loc,df_buffer,\
                          PPG_file_name):
    if df_buffer.shape[0] == 0:
        df_buffer = pd.DataFrame({PPG_file_name: file_beats_loc})
    else:
        df_buffer[PPG_file_name] = file_beats_loc
    return df_buffer

def my_func_list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def my_func_save_buffer_df_to_parquet(path_beats_loc,filename_beats_loc,\
                                      df_buffer):
    # Save the dataframe to parquet
    temp_file = os.path.join(path_beats_loc,filename_beats_loc)
    Path(path_beats_loc).mkdir(parents=True, exist_ok=True)
    
    flag_save_to_new_parquet = False
    if not os.path.isfile(temp_file):
        # File does not exist.
        flag_save_to_new_parquet = True
    else:
        try:
            df_file_beats_loc = pd.read_parquet(temp_file, engine="fastparquet")
            flag_save_to_new_parquet = False
        except OSError:
            print('Parquet file broken:',temp_file)
            flag_save_to_new_parquet = True
        
    if flag_save_to_new_parquet:
        start = time.time()
        df_buffer.to_parquet(temp_file, compression=None)
        end = time.time()
        print('Save to parquet:',end - start,'sec')
    else:
        start = time.time()
        df_file_beats_loc = pd.read_parquet(temp_file, engine="fastparquet")
        end = time.time()
        read_sec = end - start
        # df_file_beats_loc = df_file_beats_loc.append(df_buffer)
        df_file_beats_loc = pd.concat([df_file_beats_loc,df_buffer], axis=1)
        df_temp = df_file_beats_loc.T
        df_temp = df_temp.drop_duplicates()
        df_temp = df_temp.T # Transpose it back.
        
        if any(df_temp.columns.duplicated()):
            # Making sure the dataframe has no duplicated columns before saving to parquet file.
            list_all_col = list(df_temp.columns.values)
            
            newlist = []
            duplist = []
            for i in list_all_col:
                if i not in newlist:
                    newlist.append(i)
                else:
                    duplist.append(i)
                    
            all_idx = list(np.arange(0,len(df_temp.columns))) # Generate all the index.
            remain_idx = all_idx # Keep the column in the dataframe.
            for ii in duplist:
                temp_col_idx = my_func_list_duplicates_of(list(df_temp.columns),ii)
            
                # remain_idx = remain_idx.remove(temp_col_idx[0:-1])
                for jj in temp_col_idx[0:-1]:
                    if jj in remain_idx:
                        remain_idx.remove(jj)
            
            df_remain = df_temp.iloc[:,remain_idx]
            # print('Case 1: df_remain.columns',df_remain.columns)
            df_remain.to_parquet(temp_file, compression=None)
        else:
            # print('Case 2: df_temp.columns',df_temp.columns)
            df_temp.to_parquet(temp_file, compression=None)
            
        
        end = time.time()
        print('Read: ',read_sec,'sec, and save to parquet:',end - start,'sec')
    return

# df = my_func_load_acc_txt_after_ppg(all_PPG_ACC_file_name_path,UID)

def my_func_return_row_idx(path_beats_loc,list_of_files_loc,df):
    if len(list_of_files_loc) == 0:
        temp_start_idx_loc = [0]
    else:
        filename_beats_loc = list_of_files_loc[-1]
        temp_file = os.path.join(path_beats_loc,filename_beats_loc)
        flag_save_to_new_parquet = False
        if len(list_of_files_loc) < 2:
            flag_save_to_new_parquet = True
        for ii in range(2,len(list_of_files_loc)+1):
            try:
                df_check_length = pd.read_parquet(temp_file, engine="fastparquet")
                flag_save_to_new_parquet = False
                break
            except OSError:
                print('Parquet file broken:',temp_file)
                flag_save_to_new_parquet = True
            
            filename_beats_loc = list_of_files_loc[-ii]
            temp_file = os.path.join(path_beats_loc,filename_beats_loc)
        # 110_2021_07_08_10_59_02_Solo_Beats_Loc.parquet
        if flag_save_to_new_parquet:
            # Looped all parquet files but still cannot find a valid file.
            temp_start_idx_loc = [0]
        else:
            if len(df_check_length) > 0:
                temp_col_name = df_check_length.columns
                temp_check_filename = temp_col_name[-1]
                temp_start_idx_loc = df.index[df['All_PPG_file_name'].str.contains(temp_check_filename, case=False)].tolist()
            else:
                # UID 032, seg 0, 03/18/2023:
                temp_check_filename = filename_beats_loc[:24]
                temp_start_idx_loc = df.index[df['All_PPG_file_name'].str.contains(temp_check_filename, case=False)].tolist()
                # temp_start_idx_loc = temp_start_idx_loc[0]
    return temp_start_idx_loc[0]
    
def my_func_return_temp_idx(path_beats_loc,df):
    isExist = os.path.exists(path_beats_loc)
    if not isExist:
        temp_start_idx_loc = 0
    else:
        list_of_files_loc = sorted( filter( lambda x: os.path.isfile(os.path.join(path_beats_loc, x)),
                                os.listdir(path_beats_loc) ) )
        
        temp_start_idx_loc = my_func_return_row_idx(path_beats_loc,list_of_files_loc,df)
        
    return temp_start_idx_loc
def my_func_check_start_idx(path_beats_loc,\
                            path_beats_rhythm,\
                            path_beats_type,\
                            path_ECG_valid,\
                            path_ECG_raw,\
                            path_ECG_timestamp,\
                            df):
    
# First, comment out the check if parquet file exists part.
# Second, try to load the parquet file. 
# Skip the failed parquet file. 
#     
    temp_start_idx_loc = my_func_return_temp_idx(path_beats_loc,df)
    temp_start_idx_rhythm = my_func_return_temp_idx(path_beats_rhythm,df)
    temp_start_idx_type = my_func_return_temp_idx(path_beats_type,df)
    temp_start_idx_valid = my_func_return_temp_idx(path_ECG_valid,df)
    
    temp_start_idx_raw = my_func_return_temp_idx(path_ECG_raw,df)
    temp_start_idx_timestamp = my_func_return_temp_idx(path_ECG_timestamp,df)

    start_idx = np.min([int(temp_start_idx_loc),\
                        int(temp_start_idx_rhythm),\
                        int(temp_start_idx_type),\
                        int(temp_start_idx_valid),\
                        int(temp_start_idx_raw),\
                        int(temp_start_idx_timestamp)])
    
    start_idx = start_idx + 1
    return start_idx
