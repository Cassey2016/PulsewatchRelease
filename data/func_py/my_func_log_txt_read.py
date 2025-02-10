# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:01:56 2023

@author: localadmin
"""
import os
from scanf import scanf # In Windows Anaconda, I installed pip install scanf.
path_log_file = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\Final_Clinical_Trial_Data\110_final'
file_name_log = '110_2021_07_01_10_05_02_log.txt'
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

with open (os.path.join(path_log_file,file_name_log), 'rt') as myfile:  # 'r' for reading, 't' for text mode.
    for myline in myfile:
        mylines.append(myline)              # Read the entire file to a string
        
        matches = [x for x in my_dictionary_sentence if x in myline]
        # any(x in myline for x in my_dictionary_sentence):
        if len(matches): # Not empty
            for dict_str in matches:
                dict_idx = my_dictionary_sentence.index(dict_str) # Use the index to the scanf
                temp = scanf(dict_str+my_scanf_format[dict_idx],myline)

                if dict_idx == 0:
                    count_seg += 1
                    m_seg_num[count_seg] = temp
                    m_seg_start_row_idx[count_seg] = count_row
                    
                if dict_idx > 0 and count_seg < 0:
                    # no 'segment' sentence before the first segment:
                    count_seg += 1 # I will ignore index is not accurately recorded. 08/20/2020.
                    
                if dict_idx == 1:
                    m_HR_WEPD[count_seg] = temp
                if dict_idx == 2:
                    m_comb[count_seg] = temp
                if dict_idx == 3:
                    m_RMSSD[count_seg] = temp
                if dict_idx == 4:
                    m_SampEn[count_seg] = temp
                if dict_idx == 5:
                    m_IsAF_1[count_seg] = temp
                if dict_idx == 6:
                    m_HR_DATPD[count_seg] = temp
                if dict_idx == 7:
                    m_HR_SWEPD[count_seg] = temp
                if dict_idx == 8:
                    m_PACPVC_pred_1[count_seg] = temp
                if dict_idx == 9:
                    m_FastAF_1[count_seg] = temp
                if dict_idx == 10:
                    m_index[count_seg] = temp
                if dict_str == 'PACPVC_predict_label': # 12 (11 in Python)
                    if temp is None:
                        temp = scanf(dict_str+'is '+my_scanf_format[dict_idx],myline)
                    m_PACPVC_pred_2[count_seg] = temp
                if dict_idx == 12:
                    m_op[count_seg] = temp
                if dict_idx == 13:
                    m_HR_1[count_seg] = temp
                if dict_idx == 14:
                    m_FastAF_2[count_seg] = temp
                if dict_idx == 15:
                    m_API_ver[count_seg] = temp
                if dict_idx == 16:
                    m_service_var[count_seg] = temp
                if dict_idx == 17:
                    m_HR_2[count_seg] = temp
                if dict_idx == 18:
                    m_IsAF_2[count_seg] = temp
                if dict_idx == 19:
                    m_mSensorEnable[count_seg] = temp
                    m_seg_end_row_idx[count_seg] = count_row
                if dict_idx == 20:
                    m_HR_3[count_seg] = temp
                if dict_idx == 21:
                    m_IsAF_3[count_seg] = temp
                if dict_idx == 22:
                    m_storage_total_size[count_seg] = temp
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