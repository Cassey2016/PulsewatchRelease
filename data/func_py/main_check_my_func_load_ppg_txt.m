clear;
close all;
clc;

path_file = ['R:\ENGR_Chon\NIH_Pulsewatch_Database\mat_for_load_Final_Clinical_Trial_Data'];

temp_folder_path = [path_file,filesep,'*_focus_PPG_only*']; % visit all files have 'ppg' in the file name.
PPG_file_listing = dir(temp_folder_path);

for ii = 1:size(PPG_file_listing,1)
    this_mat_file_name = PPG_file_listing(ii).name;
%     disp(this_mat_file_name);
    load([path_file,filesep,this_mat_file_name]);
    fprintf('UID: %s, len(PPG_name_cell): %d\n',this_mat_file_name(1:3),size(PPG_name_cell,1));
end