import tarfile
import glob
import os

def extract_all_files(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(extract_to)

# --- Example usage ---
# tar_file_path = '/content/drive/MyDrive/Pulsewatch_Data/tar_PPG_filt_30sec_csv/003.tar'
# extract_to = '/content/PPG_filt_30sec_csv/' # It does not need to be created beforehand.
# extract_all_files(tar_file_path, extract_to)

def my_untar_PPG_filt_30sec_csv(path_tar_file,path_extract_to) -> None:
    #
    # Untar the tarred 1D filtered PPG csv files into folders.
    # 
    # Parameters:
    #    - path_tar_file: the source folder stores tar files.
    #    - path_extract_to: the destination folder contains untarred files.
    # Output:
    #    None.
    # path_tar_file = '/content/drive/MyDrive/Pulsewatch_Data/tar_PPG_filt_30sec_csv'
    # Get all the UIDs in the tar path.
    list_tar_filenames = glob.glob(os.path.join(path_tar_file,'*.tar'), recursive=True)

    for this_tar_file in list_tar_filenames:
        # Check if destination folder exists:
        UID = this_tar_file.split('/')[-1][:3]
        path_tar_folder = os.path.join(path_extract_to,UID) # The first three digit in the tar file name is the UID.
        if os.path.isdir(path_tar_folder):
            print('Tarred folder exists, skipped untar:',path_tar_folder)
        else:
            print('Untarring:',path_tar_folder)
            extract_all_files(this_tar_file, path_extract_to)
    
    return None