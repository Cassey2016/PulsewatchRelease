# PulsewatchRelease
The pretrained models, codes, and data loading documentation for the Pulsewatch dataset.

Acronym:

- PPG: photoplethysmography;
- HR: heart rate;
- ACC: accelerometer;
- magHR: magnified HR;
- ECG: electrocardiogram.

## ICASSP 2025
Dong, last edit 02/10/2025.
### Training model

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

  - The reference ECG, ECG loading points, plots of 30-sec aligned ECG and PPG have been uploaded to [Synapse](https://drive.google.com/drive/folders/1kfeEN8WDp1xhypB5VVze_0W6FUocPfxU?usp=drive_link). The loading code in MATLAB and Python will be provided after the documentation of deep learning model have been completed. (Dong, last modified at 17:46 EST on 01/16/2025).

- The split of the dataset is on Google drive: [SMOTE_everything_fold_2_2024_03_30](https://drive.google.com/drive/folders/12FfPZRf5wDzexFGxKPXqGojtP-d1N5jq?usp=drive_link)

- The adjudication is on Google drive: [Adjudication_UConn](https://drive.google.com/drive/folders/182R7-Q_lDawf6u2RqJJwZUzex0VKUlhP?usp=drive_link)

  - You might see me loading the ground truth during training from this folder [final_attemp_4_1_Dong_Ohm](https://drive.google.com/drive/folders/1a4XJw3ANN0b-Q7wjNYxNI3phl9u5V2r9?usp=drive_link),

  - and loading the ground truth during testing from this folder [final_attemp_4_1_Dong_Ohm_2024_02_18_copy](https://drive.google.com/drive/folders/1RArra_FLG9GTwX2rsmKM_YSRGZ5QAxU-?usp=drive_link). This folder should contain the same adjudication labels but remove the repeated segments (due to summarizing everyone's adjudication).

### Re-train comparison model(s)

In the folder `traincomparison`:

#### Liu et al. 2022 JAHA model

doi: 10.1161/JAHA.121.023555
https://github.com/zdzdliu/PPGArrhythmiaDetection
@article{liu2022multiclass,
  title={Multiclass Arrhythmia Detection and Classification From Photoplethysmography Signals Using a Deep Convolutional Neural Network},
  author={Liu, Zengding and Zhou, Bin and Jiang, Zhiming and Chen, Xi and Li, Ye and Tang, Min and Miao, Fen},
  journal={Journal of the American Heart Association},
  volume={11},
  number={7},
  pages={e023555},
  year={2022},
  publisher={Am Heart Assoc}
}

1. Run `traincomparison/main_train_Liu_model_PPG_only.ipynb` for training 1D VGG-16 model with one channel of input (**PPG only**).

   - It will call the code `traincomparison/run_train_Liu_model_PPG_only.py`.

   - This re-trained model is saved on Google Drive: [TestRNNGRU_batch32](https://drive.google.com/drive/folders/10UeAzp4sYzxCic5vuxhYNNazLsqn6fpw?usp=drive_link)

     It is also mentioned in this func: `utils/my_pathdef.py`, `my_ckpt_path_2024_08_21_PPG_only`.

#### Chen et al. 2024 BSN model

Will be provided in Darren Chen's GitHub page soon.

### Testing model

In the folder `test`:

Again, you may run the Notebooks on Google Colab with `T4 GPU, High RAM` resource.

1. Run `test/main_test_model_four_channels.ipynb` for testing our 1D-bi-GRU with four channels of input (**PPG, HR, ACC, magHR**).

   - It will call the code `test/run_test_model_four_channels.py`, which includes the setting code, loading the test set, and saving the predicted labels.

2. Run `test/main_test_model_HR_ACC.ipynb` for our 1D-bi-GRU with two channels of input (**HR, ACC**).

   - It will call the code `test/run_test_model_HR_ACC.py`.

3. Run `test/main_test_model_PPG_only.ipynb` for our 1D-bi-GRU with one channel of input (**PPG only**).

   - It will call the code `test/run_test_model_PPG_only.py`.

### Testing comparison (re-trained) model(s)

In the folder `testcomparison`:

1. Run `testcomparison/main_test_Liu_model_PPG_only.ipynb` for testing re-trained 1D VGG-16 model with one channel of input (**PPG only**).

   - It will call the code `testcomparison/run_test_Liu_model_PPG_only.py`.

### Evaluating models and calculate metrics

In the folder `evalperformance`:

Details on how to run the code are written in the `README.md` file in that folder.
