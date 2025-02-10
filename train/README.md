# Training model 

## For conference paper published in ICASSP 2025

In the folder `train`:

You may run these Notebooks on Google Colab, with `T4 GPU, High RAM` resource.

1. Run `train/main_train_model_four_channels.ipynb` for training our 1D-bi-GRU with four channels of input (**PPG, HR, ACC, magHR**).

   - It will first call the setting code `train/run_train_model_four_channels.py`.
   
   - In the last line of the setting code, it will call the modular code `train/my_main.py` to train our 1D-bi-GRU with different settings.

   - Our saved models are on Google Drive: [TestRNNGRU_HR_ACC_rescaleHR](https://drive.google.com/drive/folders/107pCHkm7K8vcmIlLU5XHABjcIuMnu3vT?usp=drive_link).
     
     It is also mentioned in this func: `utils/my_pathdef.py`, `my_ckpt_path_2024_07_16`.

2. Run `train/main_train_model_HR_ACC.ipynb` for training our 1D-bi-GRU with two channels of input (**HR, ACC**).

   - It will first call the setting code `train/run_train_model_HR_ACC.py`.

   - Our saved models are on Google Drive: [TestRNNGRU_HR_ACC_noPPG](https://drive.google.com/drive/folders/1-xYFqtmFS3QlqJJUf-tZ2GvJfdZL8ji3?usp=drive_link).

     It is also mentioned in this func: `utils/my_pathdef.py`, `my_ckpt_path_2024_07_12`.

3. Run `train/main_train_model_PPG_only.ipynb` for training our 1D-bi-GRU with one channel of input (**PPG only**).

   - It will first call the setting code `train/run_train_model_PPG_only.py`.

   - Our saved models are on Google Drive: [TestRNNGRU_batch32](https://drive.google.com/drive/folders/10UeAzp4sYzxCic5vuxhYNNazLsqn6fpw?usp=drive_link)

     It is also mentioned in this func: `utils/my_pathdef.py`, `my_ckpt_path_2024_07_16_batch32`.

Note:

- The training data will be untarred into your temporary Google Colab drive. 

  - 1D filtered PPG and the heart rate calculated from WEPD algorithm stored in PyTorch binary format: [tar_PT_1D_PPG_HR_Pulsewatch](https://drive.google.com/drive/folders/130XzyOjix3-Sb8dGt_0Kgc_lZeF66JdL?usp=drive_link)

  - 1D raw ACC (also includes the heart rate calculated from Elgendi et al. algorithm) stored in PyTorch binary format: [tar_PPG_Elgendi_30sec_HR_ACC_pt](https://drive.google.com/drive/folders/1A0iLeSs_pySDuKulNC-xWmUdswqs8WIP?usp=drive_link)

  - The reference ECG, ECG loading points, plots of 30-sec aligned ECG and PPG have been uploaded to [Google drive "Synapse" folder](https://drive.google.com/drive/folders/1kfeEN8WDp1xhypB5VVze_0W6FUocPfxU?usp=drive_link). The loading code in MATLAB and Python will be provided after the documentation of deep learning model have been completed. (Dong, last modified at 17:46 EST on 01/16/2025).

- The split of the dataset is on Google drive: [SMOTE_everything_fold_2_2024_03_30](https://drive.google.com/drive/folders/12FfPZRf5wDzexFGxKPXqGojtP-d1N5jq?usp=drive_link)

- The adjudication is on Google drive: [Adjudication_UConn](https://drive.google.com/drive/folders/182R7-Q_lDawf6u2RqJJwZUzex0VKUlhP?usp=drive_link)

  - You might see me loading the ground truth during training from this folder [final_attemp_4_1_Dong_Ohm](https://drive.google.com/drive/folders/1a4XJw3ANN0b-Q7wjNYxNI3phl9u5V2r9?usp=drive_link),

  - and loading the ground truth during testing from this folder [final_attemp_4_1_Dong_Ohm_2024_02_18_copy](https://drive.google.com/drive/folders/1RArra_FLG9GTwX2rsmKM_YSRGZ5QAxU-?usp=drive_link). This folder should contain the same adjudication labels but remove the repeated segments (due to summarizing everyone's adjudication).

## For journal paper published unknown:

Model 4, `PPG only`: train/main_train_model_PPG_only.ipynb

Model 5, `PPG + ACC`: train/main_train_model_PPG_ACC.ipynb

Model 6, `HR only`: train/main_train_model_HR_only.ipynb

Model 7, `HR + ACC`: train/main_train_model_HR_ACC.ipynb

Model 8, `HR + magHR`: train/main_train_model_HR_magHR.ipynb

Model 9, `HR + magHR + ACC`: train/main_train_model_HR_magHR_ACC.ipynb

Model 10, `PPG + HR`: train/main_train_model_PPG_HR.ipynb

Model 11, `PPG + HR + ACC`: train/main_train_model_PPG_HR_ACC.ipynb

Model 12, `PPG + HR + ACC + magHR (best model)`: train/main_train_model_four_channels.ipynb