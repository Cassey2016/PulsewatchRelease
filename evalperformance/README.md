# The output files for evaluating models
Dong, last edit 02/10/2025.
Root path on Google drive: [GitHub/evalperformance](https://drive.google.com/drive/folders/1ff_xm1AdSgNJ4lwQS21d49kEF128fSW8?usp=drive_link)


## 1. Same-subject testing results (not included in the journal paper, three-class):
Run first: evalperformance/main_01_same_subject_test.py
Run second: evalperformance/main_01_same_subject_test_plot_journal.ipynb

### Files used/output:
Google drive: [01_same_subject_test](https://drive.google.com/drive/folders/1pnRiF3lhzXRUe6wOy5_C1IgPU2S1gMAQ?usp=drive_link)

## 2. Subject-independent testing results for folds 1 and 2 (three-class):
Run first: evalperformance/main_02_fold_1_2.py
Run second: evalperformance/main_02_fold_1_2_plot_journal.ipynb

### Files used/output:
Google drive: [02_fold_1_2_combined_subject_independent_test](https://drive.google.com/drive/folders/1czrLwlT3vCciRl_QDj7Lo17EgcMnGPgK?usp=drive_link)

## 3. Subject-independent testing results for folds 1 and 2 (binary-class):
evalperformance/main_03_fold_1_2_binary.py
evalperformance/main_03_fold_1_2_binary_plot.ipynb

### Files used/output:
Google drive: [03_fold_1_2_binary_combined_subject_independent_test](https://drive.google.com/drive/folders/1kRvE-x4wyX7PdfoKdkCc1iGBosavIanZ?usp=drive_link)

## 4. Subject-independent testing results for NSR subset (three-class):
evalperformance/main_04_independent_NSR.py
evalperformance/main_04_independent_NSR_plot.ipynb

### Files used/output:
Google drive: [04_remain_NSR_combined_subject_independent_test](https://drive.google.com/drive/folders/1JMg6Y2ZZiT8Fej9SP9Y61MXcLwcQ8Iph?usp=drive_link)

## 5. Subject-independent testing results for NSR subset (binary-class):
evalperformance/main_04_independent_NSR_binary.py
evalperformance/main_04_independent_NSR_binary_plot.ipynb

### Files used/output:
Google drive: [05_remain_NSR_binary_combined_subject_independent_test](https://drive.google.com/drive/folders/1bIg2pHGdiLCFwjTtQAX9H-UuzSU1YASD?usp=drive_link)

## 6. Average the metrics for folds 1 and 2 and NSR subset (three-class):
evalperformance/main_05_merge_fold12_independent_NSR.ipynb

### Files used/output:
Google drive: [06_merge_fold12_indpNSR](https://drive.google.com/drive/folders/1Zf1Q7LtVD5g45LINOzzJXtOOGpUpgNjH?usp=drive_link)

## 7. Average the metrics for folds 1 and 2 and NSR subset (binary-class):
evalperformance/main_05_merge_binary_fold12_independent_NSR.ipynb

### Files used/output:
Google drive: [07_merge_binary_fold12_indpNSR](https://drive.google.com/drive/folders/1mDYbPAACpjtbgOyAEbB6qFfajMmyDbfg?usp=drive_link)

## 8. Calculate macro-AUROC for folds 1, 2 and NSR subset (three-class):
evalperformance/main_07_avg_AUROC.ipynb

### Files used/output:
Google drive: [08_AUROC_all_pulsewatch_subject_independent_test](https://drive.google.com/drive/folders/1uxkVWGkGffrO4yuAxvRhTf346157gYJ6?usp=drive_link)

## 9. Calculate macro-AUROC for folds 1,2, and NSR subset (binary-class):
evalperformance/main_07_avg_AUROC_binary.ipynb

### Files used/output:
Google drive: [09_AUROC_binary_all_pulsewatch_subject_independent_test](https://drive.google.com/drive/folders/1zULVD1Gqa4x7RvrQnMfc-CWdNguJPw8E?usp=drive_link)

## 10. Calculate the confidence interval of macro-AUROC (three-class):
evalperformance/main_08_confidence_interval_bootstrap.ipynb

### Files used/output:
Google drive: [10_AUROC_all_pulsewatch_subject_independent_test_CI](https://drive.google.com/drive/folders/1S2F5TYCvhzOJ7-8AvDVRW4OunB7EuOWR?usp=drive_link)
[11_AUROC_MIMICIII_CI](https://drive.google.com/drive/folders/1yF-RFbv0CpyQmruytI3oGTTXW6CvVjNW?usp=drive_link)
[12_AUROC_Simband_CI](https://drive.google.com/drive/folders/1mZiQTKc6v-iwvQ4bD21rWGZXkXnOMIhH?usp=drive_link)

## 11. Plot the macro-AUROC with confidence interval, for Pulsewatch, Simband, and MIMIC-III:
evalperformance/main_08_CI_plot.ipynb

(Output is a figure.)

## 12. Plot the 27 segments from 27 subjects:
evalperformance/main_09_plot_Simband_MIMIC.ipynb

### Files used/output:
Google drive: [13_plot_PPG_three_modalities](https://drive.google.com/drive/folders/18jP6X5BwX2oTF2PqGuYRq3EOewTY8vo-?usp=drive_link)