# Re-train comparison model(s)
Dong, last edit 02/10/2025.

## 1. For conference paper published on ICASSP 2025:

### 1.1 Liu et al. 2022 JAHA model

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

### 1.2 Chen et al. 2024 BSN model

Will be provided in Darren Chen's GitHub page soon.

## 2. For journal paper published on unknown:

Model 1, `1D-VGG-16 (PPG only)`: traincomparison/main_train_Liu_model_PPG_only.ipynb

Model 2, `1D-VGG-16 (four channels)`: traincomparison/main_train_Liu_model_four_channels.ipynb

Model 3, `2D DenseNet (2D TFS)`: will contact 2D DenseNet paper author to upload it.