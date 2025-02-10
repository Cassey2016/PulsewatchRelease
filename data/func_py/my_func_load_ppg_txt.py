# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:02:09 2023

@author: localadmin

Implemented func my_func_load_ppg_txt
And func my_func_load_acc_txt_after_ppg

I save the 'All_PPG_file_name','ACC_for_All_PPG_file_name' as dataframe and H5.
I will not call this 'main' code like MATLAB to load the H5. To save time.

"""
# =============================================================================
# Step 1: find all files that contain 'ppg' in the file name.
# =============================================================================
import os
root_data_path = r'R:\ENGR_Chon\NIH_Pulsewatch_Database'

# Get all the UIDs that were generated by MATLAB.
get_UID_list = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\mat_for_load_Final_Clinical_Trial_Data'
dir_list_UID = os.listdir(get_UID_list)

first_3_char = list(set([x[:3] for x in dir_list_UID]))
# first_3_char_unique = [x for x in first_3_char if x[0] == '0' or x[0] == '1']
first_3_char_unique = [x for x in first_3_char if x[0] == '3' or x[0] == '4']
first_3_char_unique.sort()

for UID in first_3_char_unique:
# UID = '036'
# if UID == '036':
    if UID[0] == '0' or UID[0] == '1':
        test_PPG_path = os.path.join(root_data_path,'Final_Clinical_Trial_Data',UID+'_final')
    elif UID[0] == '3' or UID[0] == '4':
        test_PPG_path = os.path.join(root_data_path,'AF_trial','Final_AF_Trial_Data',UID+'_final')
        
    try:
        dir_list = os.listdir(test_PPG_path)
    except FileNotFoundError:
        continue
        # pass
    
    subs = 'ppg'
    all_PPG_file_name = list(filter(lambda x: subs in x, dir_list))
    subs = 'accel'
    ACC_file_listing = list(filter(lambda x: subs in x, dir_list))
    
    # =============================================================================
    # Step 2: find the file index when PPG file has a long file name.
    # =============================================================================
    long_file_name = [x for x in all_PPG_file_name if len(x) > 36]
    temp_long_file_idx = [all_PPG_file_name.index(x) for x in long_file_name]
    
    # =============================================================================
    # Step 3: find match file name, compare content, delete index.
    # =============================================================================
    
    
    
    def my_importdata(path_file,file_name):
        mylines = []
        with open(os.path.join(path_file,file_name), 'rt') as myfile:  # 'r' for reading, 't' for text mode.
            for myline in myfile:
                mylines.append(myline)              # Read the entire file to a string
        
        return mylines
    
    # I was trying to implement MATLAB to Python, but it is not needed.
    # 
    # def common_member(a, b):
    #     a_set = set(a)
    #     b_set = set(b)
    # 
    #     if len(a_set.intersection(b_set)) > 0:
    #         return list(a_set.intersection(b_set))
    # 
    # def my_cmp_PPG_same_file_name_content(keep_file_name,
    #                                       all_PPG_file_name,
    #                                       temp_long_file_name,
    #                                       folder_path,
    #                                       temp_long_file_idx):
    #     # % find if there is same file in the PPG file name cell.
    #     matches = [x for x in keep_file_name if x in all_PPG_file_name]
    #     match_idx = [all_PPG_file_name.index(x) for x in matches]
    #     if len(match_idx):
    #         # % remove match index that is same as long file name.
    #         rm_idx = common_member(match_idx, temp_long_file_idx)
    #         match_idx_2 = [item for item in match_idx if item not in rm_idx]
    #         match_idx = match_idx_2
            
    #     delete_idx = []
    #     if len(match_idx): # Not empty
    #         # there is one or more files has same name?
    #         if len(match_idx) > 1:
    #             print('more than one file have same name, check!')
    #             # keyboard in MATLAB.
    #         else:
    #             # Only one file has same name:
    #             ori_txt = my_importdata(os.path.join(folder_path,all_PPG_file_name{match_idx,1}))
                
    #     else:
    #         # Empty matching index, just return.
    #         return delete_idx
    
    # Trim the long file name?
    # keep_file_name = []
    for temp_long_name in long_file_name:
        # print(temp_long_name)
        if temp_long_name[32:36] == '.txt': # Indice 32, 33, 34, 35.
            # % type 2: 002_2019_09_14_07_57_00_ppg_0007.txt_temp_10031
            # % discard the name after .txt
            # keep_file_name.append(temp_long_name[:36])
            keep_file_name = temp_long_name[:36]
        elif temp_long_name[-4:] == '.txt':
            # % type 1: 002_2019_09_03_15_48_17_ppg_00011567540732795.txt
            # % keep the first four digits.
            # keep_file_name.append(temp_long_name[:32]+'.txt')
            keep_file_name = temp_long_name[:32]+'.txt'
        else:
            print('unseen long file name, check! 1',temp_long_name)
        # print(keep_file_name)
        matches = keep_file_name in all_PPG_file_name
        # print(matches)
        if matches: # Not empty
            this_matches = keep_file_name
            # 1. load orginal file content:
            ori_txt = my_importdata(test_PPG_path,this_matches)
            
            # 2. load long file content:
            long_txt = my_importdata(test_PPG_path,temp_long_name)
            
            # 3. check if any one of the array is empty:
            if len(ori_txt): # Orig txt not empty.
                if len(long_txt): # Long file name txt not empty.
                    if ori_txt == long_txt:
                        # if both file are eqaul content, delete the long file:
                        # print('before: len(all_PPG_file_name):',len(all_PPG_file_name))
                        all_PPG_file_name.remove(temp_long_name)
                        # print('after: len(all_PPG_file_name):',len(all_PPG_file_name))
                    else:
                        
                        print('unseen unequal content, check!')
                        print(this_matches)
                        print(temp_long_name)
                else:
                    # Long file name txt is empty.
                    # origin txt has content, but long file name txt is empty:
                    # print('before: len(all_PPG_file_name):',len(all_PPG_file_name))
                    all_PPG_file_name.remove(temp_long_name)
                    # print('after: len(all_PPG_file_name):',len(all_PPG_file_name))
            else:# Orig txt is empty.
                if len(long_txt): # Long file name txt not empty.
                    # print('before: len(all_PPG_file_name):',len(all_PPG_file_name))
                    all_PPG_file_name.remove(this_matches)
                    # print('after: len(all_PPG_file_name):',len(all_PPG_file_name))
                else:
                    # both ori and long file are empty:
                    print('Both files are empty, please check!')
    
                        
    # print('UID:',UID,', len(all_PPG_file_name):',len(all_PPG_file_name))
    
    
    subs = 'accel'
    ACC_file_listing = list(filter(lambda x: subs in x, dir_list))
    # =============================================================================
    # Step 2: find the file index when PPG file has a long file name.
    # =============================================================================
    long_file_name = [x for x in ACC_file_listing if len(x) > 38]
    temp_long_file_idx = [ACC_file_listing.index(x) for x in long_file_name]
    
    # =============================================================================
    # Step 3: find match file name, compare content, delete index.
    # =============================================================================
    # Trim the long file name?
    # keep_file_name = []
    for temp_long_name in long_file_name:
        if temp_long_name[34:38] == '.txt': # 34, 35, 36, 37
            # % type 2: 002_2019_09_14_07_57_00_ppg_0007.txt_temp_10031
            # % discard the name after .txt
            # keep_file_name.append(temp_long_name[:35]+'.txt')
            keep_file_name = temp_long_name[:38]
        elif temp_long_name[-4:] == '.txt':
            # % type 1: 002_2019_09_03_15_48_17_ppg_00011567540732795.txt
            # % keep the first four digits.
            # keep_file_name.append(temp_long_name[:32]+'.txt')
            keep_file_name = temp_long_name[:34]+'.txt'
        else:
            print('unseen long file name, check! 2',temp_long_name)
            
        matches = keep_file_name in ACC_file_listing
        if matches: # Not empty
            this_matches = keep_file_name
            # 1. load orginal file content:
            ori_txt = my_importdata(test_PPG_path,this_matches)
            
            # 2. load long file content:
            long_txt = my_importdata(test_PPG_path,temp_long_name)
            
            # 3. check if any one of the array is empty:
            if len(ori_txt): # Orig txt not empty.
                if len(long_txt): # Long file name txt not empty.
                    if ori_txt == long_txt:
                        # if both file are eqaul content, delete the long file:
                        ACC_file_listing.remove(temp_long_name)
                    else:
                        flag_delete_long = 0
                        if len(long_txt) <= len(ori_txt):
                            # 036_2020_07_05_10_51_15_accel_0003.txt_temp_8846
                            flag_delete_long = 1 # Entered the if condition
                            for count,value in enumerate(long_txt):
                                if value == ori_txt[count]:
                                    continue
                                else:
                                    flag_delete_long = 2
                                    
                        if flag_delete_long == 2:
                            ACC_file_listing.remove(temp_long_name)
                        elif flag_delete_long == 1:
                            print('long file has some part that original file does not have.')
                        else:
                            # flag_delete_long == 0
                            print('long file has longer content, check!')
                else:
                    # Long file name txt is empty.
                    # origin txt has content, but long file name txt is empty:
                    ACC_file_listing.remove(temp_long_name)
            else:# Orig txt is empty.
                if len(long_txt): # Long file name txt not empty.
                    ACC_file_listing.remove(this_matches)
                else:
                    # both ori and long file are empty:
                    print('Both files are empty, please check!')
                        
    # print('UID:',UID,', len(ACC_file_listing):',len(ACC_file_listing))
    # Save the all_PPG_file_name to some format?
    
    # =============================================================================
    # Step 4: pair ACC with same time and seg PPG, if no matched file, keep the cell empty.
    # =============================================================================
    ACC_for_All_PPG_file_name = [None] * len(all_PPG_file_name)
    for ii,temp_PPG_name in enumerate(all_PPG_file_name):
        if temp_PPG_name[32:36] == '.txt': # 32,33,34,35
            keep_file_name = temp_PPG_name[:36]
        elif temp_PPG_name[-4:] == '.txt':
            keep_file_name = temp_PPG_name[:32]+'.txt'
        else:
            print('unseen long file name, check! 3',temp_PPG_name)
            
        reformat_name = keep_file_name[:24]+'accel_'+keep_file_name[28:32]
        # matches = any(reformat_name in x for x in ACC_file_listing) # Substring finding
        res = [x for x in ACC_file_listing if reformat_name in x]
        if not res: 
            # No ACC file, skip this PPG match ACC cell.
            print(reformat_name,'not in ACC_file_listing')
            continue
        else:
            # List not empty.
            ACC_for_All_PPG_file_name[ii] = res[0] # Use list index.
            
    print('UID:',UID,', len(ACC_for_All_PPG_file_name):',len(ACC_for_All_PPG_file_name))

    # Save all_PPG_file_name and ACC_for_All_PPG_file_name
    # I should be able to use a Pandas dataframe to store it.
    path_output = r'R:\ENGR_Chon\NIH_Pulsewatch_Database\py_for_load_Final_Clinical_Trial_Data'
    import pandas as pd # I have to install conda install pytables to use HDFStore.
    df = pd.DataFrame(list(zip(all_PPG_file_name,ACC_for_All_PPG_file_name)),columns=['All_PPG_file_name','ACC_for_All_PPG_file_name'])
    
    # df['ACC_for_All_PPG_file_name'] = ACC_for_All_PPG_file_name
    
    store = pd.HDFStore(os.path.join(path_output,UID+'_after_ppg_load_acc.h5'))
    
    store['df'] = df  # save it
    store.close()
    # store['df']  # load it
