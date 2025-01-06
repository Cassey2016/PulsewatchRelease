"""
Evaluate a saved model.

Dong, 07/17/2024.
"""

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import log_loss, precision_recall_fscore_support, roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_onehotencode_np_array(y_true):
    encoded_arr = np.zeros((y_true.size, y_true.max()+1), dtype=int)
    encoded_arr[np.arange(y_true.size),y_true] = 1
    return encoded_arr

def my_test_model(model, test_loader, n_classes, metrics_test):
    # Stochastic validation
    model.eval()
    # likelihood.eval()
    test_correct = 0
    with torch.no_grad():
        # Randomize the validation indices.
        print('Debug: inside test loop...')
        test_indices = torch.randperm(len(test_loader.dataset))[:int(1 * len(test_loader.dataset))]
        print('Debug: len(test_indices)',len(test_indices))

        y_true = []
        y_pred = []
        y_pred_prob = []
        y_pred_logit = []
        segment_names_all = []
        for batch_index, test_batch in enumerate(test_loader):
            if test_batch['data'].dim() == 3:
                # Only 1D.
                X_test = torch.tensor(test_batch['data']).to(device)
            else:
                X_test = torch.squeeze(test_batch['data'], dim=1).to(device)  # Use reshape here
            y_test_true = test_batch['label'].to(device)
            segment_names = test_batch['segment_name']

            # # print('Debug test model: X_test',X_test)
            # print('Debug test model: X_test.shape',X_test.shape)
            # print('Debug: segment_names',segment_names)
            temp_X_test = X_test.detach().cpu().numpy()
            # print('Debug: temp_X_test.max', np.max(temp_X_test))
            # print('Debug: temp_X_test.min', np.min(temp_X_test))
            # print('Debug: torch.any(torch.isnan(X_test))',torch.any(torch.isnan(X_test)))
            # print('Debug: torch.any(torch.isinf(X_test))',torch.any(torch.isinf(X_test)))
            assert not torch.any(torch.isnan(X_test))
            assert not torch.any(torch.isinf(X_test))
            # print('Debug: y_test_true',y_test_true)
            y_test_true = y_test_true.type(torch.int64)
            y_test_true_onehot = F.one_hot(y_test_true, num_classes=n_classes)
            y_test_pred_logit, y_test_pred_prob = model(X_test)
            y_test_pred = torch.argmax(y_test_pred_prob, dim=1)
            all_params = torch.cat([x.view(-1) for x in model.parameters()]) # x.view(-1) convert tensor to 1D.

            temp_test_pred = y_test_pred.detach().cpu().numpy()
            temp_test_true = y_test_true.detach().cpu().numpy()
            temp_y_test_pred_prob = y_test_pred_prob.detach().cpu().numpy()
            temp_y_test_pred_logit = y_test_pred_logit.detach().cpu().numpy()
            y_true.append(temp_test_true)
            y_pred.append(temp_test_pred)
            y_pred_prob.append(temp_y_test_pred_prob)
            y_pred_logit.append(temp_y_test_pred_logit)
            segment_names_all.append(segment_names)

            # For debugging purpose
            boolarr = (temp_test_true == temp_test_pred)
            boolarr = np.array(boolarr,dtype=bool)
            test_correct = boolarr.sum()
            # test_correct += torch.sum(test_predictions == test_labels) #(test_predictions == test_labels).type(torch.float).sum().item()
            test_accuracy = 100 * test_correct / len(temp_test_true)
            print(f'Debug test {batch_index}/{len(test_loader)}: {test_accuracy}')

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_pred_prob = np.concatenate(y_pred_prob)
        y_pred_logit = np.concatenate(y_pred_logit)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        # I guess sklearn roc does not need one hot encoded labels.
        y_true_onehot = my_onehotencode_np_array(y_true)
        y_pred_onehot = my_onehotencode_np_array(y_pred)

        # auc_roc = roc_auc_score(y_true_onehot,
        #                         y_pred_prob,
        #                         multi_class='ovr') # I will calculate it after collecting all the predicted labels and probabilities.
        print('Debug: segment_names_all',segment_names_all)
        metrics_test['segment_names'].append(segment_names_all)
        metrics_test['precision'].append(precision)
        metrics_test['recall'].append(recall)
        metrics_test['f1_score'].append(f1)
        # metrics_test['auc_roc'].append(auc_roc)
        metrics_test['y_true'].append(y_true)
        metrics_test['y_pred'].append(y_pred)
        metrics_test['y_pred_prob'].append(y_pred_prob)
        metrics_test['y_pred_logit'].append(y_pred_logit)

        print('Debug test: len(test_indices)', len(test_indices))

    # Get one accuracy at the end of training.
    boolarr = (y_pred == y_true)
    boolarr = np.array(boolarr,dtype=bool)
    test_correct += boolarr.sum()
    # test_correct += torch.sum(test_predictions == test_labels) #(test_predictions == test_labels).type(torch.float).sum().item()
    test_accuracy = 100 * test_correct / len(test_indices)
    metrics_test['test_accuracy'].append(test_accuracy)
    
    print("test_accuracy = {}".format(test_accuracy))

    return metrics_test
