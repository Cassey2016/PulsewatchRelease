# Testing model
Dong, last edit 02/10/2025.

## 1. For the conference paper published on ICASSP 2025:

In the folder `test`:

Again, you may run the Notebooks on Google Colab with `T4 GPU, High RAM` resource.

1. Run `test/main_test_model_four_channels.ipynb` for testing our 1D-bi-GRU with four channels of input (**PPG, HR, ACC, magHR**).

   - It will call the code `test/run_test_model_four_channels.py`, which includes the setting code, loading the test set, and saving the predicted labels.

2. Run `test/main_test_model_HR_ACC.ipynb` for our 1D-bi-GRU with two channels of input (**HR, ACC**).

   - It will call the code `test/run_test_model_HR_ACC.py`.

3. Run `test/main_test_model_PPG_only.ipynb` for our 1D-bi-GRU with one channel of input (**PPG only**).

   - It will call the code `test/run_test_model_PPG_only.py`.

## 2. For the journal paper published on unknown:

Model 4, `PPG only`: test/main_test_model_PPG_only.ipynb

Model 5, `PPG + ACC`: test/main_test_model_PPG_ACC.ipynb

Model 6, `HR only`: test/main_test_model_HR_only.ipynb

Model 7, `HR + ACC`: test/main_test_model_HR_ACC.ipynb

Model 8, `HR + magHR`: test/main_test_model_HR_magHR.ipynb

Model 9, `HR + magHR + ACC`: test/main_test_model_HR_magHR_ACC.ipynb

Model 10, `PPG + HR`: test/main_test_model_PPG_HR.ipynb

Model 11, `PPG + HR + ACC`: test/main_test_model_PPG_HR_ACC.ipynb

Model 12, `PPG + HR + ACC + magHR (best model)`: test/main_test_model_four_channels.ipynb