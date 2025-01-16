import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
import time
import copy
import my_dataloader
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myRNNGRUmodel(nn.Module):
    def __init__(self, input_size_L, in_channels_d, num_classes):
        """
        
        Parameter:
            - input: dimension (none, d, L), d is the 1 for 1-D PPG. L is 1500 for 30-sec PPG sampled at 50 Hz.
        """
        super(myRNNGRUmodel, self).__init__()
        self.input_size_L = input_size_L
        self.in_channels_d = in_channels_d
        self.num_classes = num_classes
        
        # For 1D PPG or 1D PPG+HR:
        out_channels = 4 * in_channels_d # 4 x d filters.
        self.out_channels = out_channels
        kernel_size = 5
        stride = 1 # It seems like their input and output dimension remained the same.
        padding = 2 # I think to make the input and output the same size, I need to put padding = 2 for kernel size 5.
        dilation = 1 # Default is 1.
        groups = 1 # Default is 1.
        bias = True # Default is True.
        padding_mode = 'zeros' # Default 'zeros'
        self.Conv1d = nn.Conv1d(in_channels_d, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # self.Flatten = nn.Flatten(1,2) # If input is in shape (none, 4d, L), output is (none, 4d*L).

        # input_size = (None, input_size_L, out_channels) # (none, L, 4*d)
        # input_size = (int(out_channels), int(input_size_L))
        input_size = self.out_channels # Input features. https://discuss.pytorch.org/t/input-shape-to-gru-layer/171318/3
        hidden_size = 128 # Two layers, bidirectional.
        self.hidden_size = hidden_size # For initializing state.
        num_layers = 1 # One bidirectional GRU (B-GRU).
        bias = True # Default is True.
        batch_first = True # Input and output tensors have shape (batch, seq, feature).
        dropout = 0 # No dropout.
        bidirectional = True # Bidirectional.
        print('Debug init: GRU input_size',input_size,'hidden_size',hidden_size,'num_layers',num_layers,'bias',bias,'batch_first',batch_first,'dropout',dropout,'bidirectional',bidirectional)
        if bidirectional:
            self.h0_shape_0 = 2 * num_layers
        else:
            self.h0_shape_0 = num_layers
        self.GRU = nn.GRU(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)
        # num_features = (None, hidden_size * 2, input_size_L) # (N, C, L), N=batch size, C=num of features or channels, L=sequence length.
        # num_features = (hidden_size * 2, input_size_L)
        num_features = self.input_size_L
        self.BN = nn.BatchNorm1d(num_features)
        self.Dropout = nn.Dropout1d(p = 0.2, inplace = False) # 20% dropout.
        self.Dense = nn.Linear(hidden_size * 2, 1)
        self.Dense2 = nn.Linear(self.input_size_L,self.num_classes) # I should switch the dimension.

    def forward(self, x):
        # print('Debug: input x.shape',x.shape)
        # x = x.permute(0,2,1) # (none, L, d) -> (none, d, L), switch dimension with signal length.
        # print('Debug forward after permute: x.shape',x.shape)
        x = self.Conv1d(x) # input (none, d, L)
        # print('Debug forward after Conv1d: x.shape',x.shape)
        x = x.permute(0,2,1) # (none, d, L) -> (none, L, d), switch back.
        # print('Debug: after Conv1d x.shape',x.shape)
        # x = self.Flatten(x)
        # print('Debug: after Flatten x.shape',x.shape)
        # h0 = torch.zeros(x.size(1), x.size(0), x.size(2)).to(device)
        # h0 = torch.zeros(x.size(2)//2, self.hidden_size, x.size(0))
        h0 = torch.zeros(self.h0_shape_0,x.size(0),self.hidden_size, device=device,requires_grad=True)
        # print('Debug forward: h0.shape',h0.shape)
        x,_ = self.GRU(x, h0)
        # print('Debug: after GRU x.shape',x.shape)
        # print('Debug: dir(x) after GRU', dir(x))
        x = self.BN(x)
        # print('Debug: after BN x.shape',x.shape)
        x = self.Dropout(x)
        # print('Debug: after Dropout x.shape',x.shape)
        x = self.Dense(x)
        # print('Debug: after Dense x.shape',x.shape)

        # --- Original model's output layer ---
        # output = F.softmax(x, dim=1) # Not sure why dimension has to be 1.
        # --- My newly added dense layer with new output layer ---
        x = x.permute(0,2,1) # (none, L, 1) -> (none, 1, L)
        x = self.Dense2(x) # (none, 1, L) -> (none, 1, 3)
        x = x.permute(0,2,1) # (none, 1, 3) -> (none, 3, 1), switch back to the same dimension 
        # Pytorch nn.CrossEntropyLoss apply softmax automatically, so no need to apply it.
        output = torch.squeeze(x,dim=-1) # (none, 3, 1) -> (none, 3)
        output_prob = F.softmax(x, dim=1) # Along the second dim (3) in (none, 3, 1) dimension.
        output_prob = torch.squeeze(output_prob,dim=-1) # (none, 3, 1) -> (none, 3)
        # print('Debug: after softmax output.shape',output.shape)
        # print('Debug model: output.grad_fn',output.grad_fn)
        return output, output_prob
    
def train_RNN_GRU_model(PARAMS_train):
    # Unload the input parameters.
    model = PARAMS_train['model']
    optimizer = PARAMS_train['optimizer']
    criterion_train = PARAMS_train['criterion_train']
    metrics = PARAMS_train['metrics']
    checkpoint_path = PARAMS_train['checkpoint_path']
    epoch = PARAMS_train['epoch']
    best_val_loss = PARAMS_train['best_val_loss']
    flag_reload_dataloader = PARAMS_train['flag_reload_dataloader']
    train_loader = PARAMS_train['train_loader']
    train_dataset = PARAMS_train['train_dataset']
    batch_size = PARAMS_train['batch_size']
    fold_name = PARAMS_train['fold_name']
    df_train = PARAMS_train['df_train']
    df_valid = PARAMS_train['df_valid']
    dict_paths = PARAMS_train['dict_paths']
    normalize_type = PARAMS_train['normalize_type']
    data_format = PARAMS_train['data_format']
    data_dim = PARAMS_train['data_dim']
    n_classes = PARAMS_train['n_classes']
    modelckpt_name = PARAMS_train['modelckpt_name']
    loss_type = PARAMS_train['loss_type']
    PARAMS_config = PARAMS_train['PARAMS_config']
    
    # Resume the ckpt for finished segments.
    finished_idx = []
    finished_seg_names = []
    y_train_pred = [] # Store the predicted training results.
    y_train_true_all = []
    train_loss_Darren = 0.0
    train_loss_Cassey = 0.0
    train_correct = 0
    # from torchsummary import summary
    # summary(model, (1500,1))
    for batch_index, train_batch in enumerate(train_loader):
        # print(f'Debug: now in a new batch of data! {batch_index}/{len(train_loader)}') # train_batch is the image data.
        model.train()
        # likelihood.train()
        optimizer.zero_grad()

        # X_train = train_batch['data'].reshape(train_batch['data'].size(0), -1).to(device)  # Use reshape here
        # print('Debug: train_batch.shape',train_batch['data'].shape)
        # print('Debug: train_batch.dim()',train_batch['data'].dim())
        if train_batch['data'].dim() == 3:
            # Only 1D.
            X_train = torch.tensor(train_batch['data']).to(device)
        else:
            X_train = torch.squeeze(train_batch['data'], dim=1).to(device)  # 1D PPG: (none, 1, d, L) -> (none, d, L)
        # print('Debug: X_train.shape',X_train.shape)
        # print('Debug train: torch.any(torch.isnan(X_train))',torch.any(torch.isnan(X_train)))
        assert not torch.any(torch.isnan(X_train))
        assert not torch.any(torch.isinf(X_train))
        y_train_true = train_batch['label'].to(device)
        assert not torch.any(torch.isnan(y_train_true))
        assert not torch.any(torch.isinf(y_train_true))
        y_train_true_onehot = F.one_hot(y_train_true, num_classes=n_classes)
        # Save the finished segment index and segment names from this batch.
        temp_finished_idx = train_batch['idx']
        temp_finished_seg_names = train_batch['segment_name']
        # print('Debug: temp_finished_idx:',temp_finished_idx)
        # print('Debug: temp_finished_segment_name:',temp_finished_seg_names)
        finished_idx.append(temp_finished_idx)
        finished_seg_names.append(temp_finished_seg_names)

        # X_train = X_train.unsqueeze(2) # (none, 1500) -> (none, 1500, 1)
        # print('Debug model: X_train',X_train)
        # print('Debug model: torch.max(X_train)',torch.max(X_train))
        # print('Debug model: torch.min(X_train)',torch.min(X_train))
        train_outputs_logit, train_outputs_prob = model(X_train)
        # train_outputs_logit = torch.squeeze(train_outputs_logit,dim=-1) # (none, 3, 1) -> (none, 3)
        # print('Debug train model: train_outputs_prob.shape',train_outputs_prob.shape) # (none, 3)
        # print('Debug train model: y_train_true_onehot.shape',y_train_true_onehot.shape) # (none, 3)
        # print('Debug train model: train_outputs_prob',train_outputs_prob)
        train_outputs = torch.argmax(train_outputs_prob,dim=1) # (none, 3) -> (none, 1)
        # train_outputs = torch.squeeze(train_outputs,dim=1) # (none, 1) -> (none,)
        # print('Debug train model after: train_outputs',train_outputs)
        # print('Debug: train_outputs.shape',train_outputs.shape)
        # print('Debug: y_train_true.shape',y_train_true.shape)
        # print('Debug: train_outputs[:10]',train_outputs[:10])
        # print('Debug: y_train_true[:10] ',y_train_true[:10])
        # print('Debug: type(train_outputs)',type(train_outputs))
        # print('Debug: type(y_train_true)',type(y_train_true))
        # print('Debug: train_outputs.dtype',train_outputs.dtype)
        # print('Debug: y_train_true.dtype',y_train_true.dtype)
        # print('Debug: criterion_train(train_outputs, y_train_true)',criterion_train(train_outputs.to(torch.float32), y_train_true.to(torch.float32)))

        # Regularization, refer: https://stackoverflow.com/questions/44641976/pytorch-how-to-add-l1-regularizer-to-activations
        lambda_l1 = 0.01
        lambda_l2 = 0.01
        # for name, param in model.named_parameters():
        #     print(name,'param.requires_grad',param.requires_grad,'param.data.shape',param.data.shape)

        # for x in model.parameters():
        #     print('Debug model parameters x.shape',x.shape)
        #     if x.shape[0] > 10:
        #         if len(x.shape) > 1:
        #             print('Debug model parameters x[:3,:3]',x[:3,:3])
        #         else:
        #             print('Debug model parameters x[:3]',x[:3])
        #     elif len(x.shape) > 1:
        #         if x.shape[1] > 10:
        #             print('Debug model parameters x[:,:3]',x[:,:3])
        #         else:
        #             print('Debug model parameters x',x)
        #     else:
        #         print('Debug model parameters x',x)
        all_params = torch.cat([x.view(-1) for x in model.parameters()]) # x.view(-1) convert tensor to 1D.
        l1_regularization = lambda_l1 * torch.norm(all_params, 1)
        l2_regularization = lambda_l2 * torch.norm(all_params, 2)
        if loss_type == 'default':
            loss = criterion_train(train_outputs_logit.to(torch.float32), y_train_true.long()) # It is single value torch tensor.
        elif loss_type == 'l1':
            loss = criterion_train(train_outputs_logit.to(torch.float32), y_train_true.long()) + l1_regularization
        elif loss_type == 'l2':
            loss = criterion_train(train_outputs_logit.to(torch.float32), y_train_true.long()) + l2_regularization
        elif loss_type == 'l1+l2':
            loss = criterion_train(train_outputs_logit.to(torch.float32), y_train_true.long()) + l1_regularization + l2_regularization
        
        # print('Debug train: loss_type',loss_type,'loss',loss)
        # print('Debug train: l1_regularization',l1_regularization)
        # print('Debug train: l2_regularization',l2_regularization)
        # loss = criterion_train(train_outputs_logit.to(torch.float32), y_train_true.long()) + l1_regularization + l2_regularization
        # print('Debug train model: loss', loss)
        # print('Debug: outputs',train_outputs)
        # print('Debug: dir(outputs)',dir(train_outputs))
        # print('Debug: outputs.mean',train_outputs.mean)
        # train_predicted = train_outputs.mean.argmax(dim=-1)
        # _, train_predicted = torch.max(train_outputs.mean, 1)
        # print('Debug: predicted',train_predicted)
        # print('Debug: predicted.shape',train_predicted.shape)
        # print('Debug: type(predicted)',type(train_predicted))
        temp_train_pred = train_outputs.detach().cpu().numpy()
        temp_train_true = y_train_true.detach().cpu().numpy()
        # print('train match',np.array((temp_train_pred == temp_train_true),dtype=bool).sum())
        y_train_pred.append(temp_train_pred) # Convert one-hot encoded probability to class number. 
        y_train_true_all.append(temp_train_true)
        train_loss_Darren += loss.item()
        train_loss_Cassey += loss.item() * len(train_batch['label'])

        loss.backward()
        optimizer.step()

        # Save the dataloader for this batch.
        save_ckpt_model_path = os.path.join(checkpoint_path,modelckpt_name)
        # print('Debug train model: save_ckpt_model_path',save_ckpt_model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'likelihood_state_dict': likelihood.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'finished_seg_names':finished_seg_names,
            'finished_idx':finished_idx,
            # Include other metrics as needed
            'metrics':metrics,
            'PARAMS_config':PARAMS_config
        }, save_ckpt_model_path)

    # Get one accuracy at the end of training.
    y_train_true = y_train_true_all
    y_train_pred = np.concatenate(y_train_pred)
    y_train_true = np.concatenate(y_train_true)
    # print('Debug train model: y_train_pred.shape',y_train_pred.shape)
    # print('Debug train model: y_train_true.shape',y_train_true.shape)
    boolarr = (y_train_pred == y_train_true)
    boolarr = np.array(boolarr,dtype=bool)
    train_correct += boolarr.sum() #(y_train_pred == y_train_true).type(torch.float).sum().item()
    len_trainset = len(train_dataset.segment_labels)
    print('Debug: train_correct',train_correct)
    train_accuracy = 100 * train_correct / len_trainset

    metrics['train_accuracy'].append(train_accuracy)  # Store the training loss
    train_loss_Darren_final = train_loss_Darren / len(train_loader) # Number of batches.
    train_loss_Cassey_final = train_loss_Cassey / len(train_loader.dataset) # Number of all training samples.
    print("train_accuracy = {}".format(train_accuracy))
    print("train_loss_Darren_final = {}".format(train_loss_Darren_final))
    print("train_loss_Cassey_final = {}".format(train_loss_Cassey_final))
    metrics['train_loss_Darren'].append(train_loss_Darren_final)  # Store the training loss
    metrics['train_loss_Cassey'].append(train_loss_Cassey_final)  # Store the training loss
    metrics['epoch'].append(epoch)

    # Reset the finished segments again because we finished one epoch.
    finished_idx = [] 
    finished_seg_names = []
    print('Debug: flag_reload_dataloader', flag_reload_dataloader)
    if flag_reload_dataloader:
        print('Debug: reset the train_loader now...')
        # Reset the train dataloader now.
        startTime_for_tictoc = time.time()
        # --- Dong, 02/15/2024:
        # train_loader,_,_ = preprocess_data_train_val(data_format, clinical_trial_train, clinical_trial_test, batch_size, finished_seg_names,\
                        #   read_all_labels=False)
        
        train_loader, _, train_dataset, _ = my_dataloader.load_train_valid_dataloader(fold_name, df_train, df_valid,
                                    dict_paths, normalize_type, data_format, data_dim,
                                    batch_size, finished_seg_names = finished_seg_names)
        endTime_for_tictoc = time.time() - startTime_for_tictoc
        print(f'Debug: took {endTime_for_tictoc} to reset the train_loader')
        flag_reload_dataloader = False # Turn off the flag for reseting train dataloader.

    dict_return_value = {
        'model':model,
        'optimizer':optimizer,
        'metrics':metrics,
        'lambda_l1':lambda_l1,
        'lambda_l2':lambda_l2,
        'finished_seg_names':finished_seg_names,
        'finished_idx':finished_idx,
        'flag_reload_dataloader':flag_reload_dataloader
    }
    return dict_return_value

def valid_RNN_GRU_model(PARAMS_valid):
    
    flag_break_train = PARAMS_valid['flag_break_train']
    lambda_l1 = PARAMS_valid['lambda_l1']
    lambda_l2 = PARAMS_valid['lambda_l2']
    criterion_val = PARAMS_valid['criterion_val']
    finished_seg_names = PARAMS_valid['finished_seg_names']
    finished_idx = PARAMS_valid['finished_idx']
    model = PARAMS_valid['model']
    optimizer = PARAMS_valid['optimizer']
    metrics = PARAMS_valid['metrics']
    checkpoint_path = PARAMS_valid['checkpoint_path']
    epoch = PARAMS_valid['epoch']
    best_val_loss = PARAMS_valid['best_val_loss']
    train_loader = PARAMS_valid['train_loader']
    val_loader = PARAMS_valid['val_loader']
    loss_type = PARAMS_valid['loss_type']
    n_classes = PARAMS_valid['n_classes']
    patience = PARAMS_valid['patience']
    epochs_no_improve = PARAMS_valid['epochs_no_improve']
    datackpt_name = PARAMS_valid['datackpt_name']
    modelckpt_name = PARAMS_valid['modelckpt_name']
    best_model_ckpt_name = PARAMS_valid['best_model_ckpt_name']
    best_model_state = PARAMS_valid['best_model_state']
    best_metrics = PARAMS_valid['best_metrics']
    save_ckpt_best_model_path = PARAMS_valid['save_ckpt_best_model_path']
    PARAMS_config = PARAMS_valid['PARAMS_config']
    filename_output = PARAMS_config['filename_output']
    # Stochastic validation
    model.eval()
    # likelihood.eval()
    valid_correct = 0
    with torch.no_grad():
        # Randomize the validation indices.
        print('Debug: inside validation loop...')
        val_indices = torch.randperm(len(val_loader.dataset))[:int(1 * len(val_loader.dataset))]
        print('Debug: len(val_indices)',len(val_indices))
        val_loss = 0.0
        val_labels = []
        val_predictions = []
        for batch_index, val_batch in enumerate(val_loader):
            # print(f'Debug: now in a new batch of data! {batch_index}/{len(val_loader)}') # train_batch is the image data.
            # print('Debug: val_batch[''data''].shape',val_batch['data'].shape)
            # X_valid = val_batch['data'].reshape(val_batch['data'].size(0), -1).to(device)  # Use reshape here
            if val_batch['data'].dim() == 3:
                # Only 1D.
                X_valid = torch.tensor(val_batch['data']).to(device)
            else:
                X_valid = torch.squeeze(val_batch['data'], dim=1).to(device)  # Use reshape here
            # print('Debug: X_valid.shape',X_valid.shape)
            # X_valid = X_valid.unsqueeze(2)
            # print('Debug: X_valid.shape',X_valid.shape)
            y_valid_true = val_batch['label'].to(device)
            y_valid_true_onehot = F.one_hot(y_valid_true, num_classes=n_classes)
            y_valid_pred_logit, y_valid_pred_prob = model(X_valid)
            # print('Debug valid model: y_valid_pred_prob.shape',y_valid_pred_prob.shape)
            # print('Debug valid model: y_valid_true_onehot.shape',y_valid_true_onehot.shape)
            y_valid_pred = torch.argmax(y_valid_pred_prob, dim=1)
            # y_valid_pred = torch.squeeze(y_valid_pred, dim=1)
            # val_loss_batch = -mll(y_valid_pred, y_valid_true).sum()
            # print('Debug: y_valid_pred[:10]',y_valid_pred[:10])
            # print('Debug: y_valid_true[:10]',y_valid_true[:10])
            all_params = torch.cat([x.view(-1) for x in model.parameters()]) # x.view(-1) convert tensor to 1D.
            l1_regularization = lambda_l1 * torch.norm(all_params, 1)
            l2_regularization = lambda_l2 * torch.norm(all_params, 2)
            # val_loss_batch = criterion_val(y_valid_pred.to(torch.float32), y_valid_true.to(torch.float32)).sum() + l1_regularization + l2_regularization
            if loss_type == 'default':
                val_loss_batch = criterion_val(y_valid_pred_logit.to(torch.float32), y_valid_true.long())
            elif loss_type == 'l1':
                val_loss_batch = criterion_val(y_valid_pred_logit.to(torch.float32), y_valid_true.long()) + l1_regularization
            elif loss_type == 'l2':
                val_loss_batch = criterion_val(y_valid_pred_logit.to(torch.float32), y_valid_true.long()) + l2_regularization
            elif loss_type == 'l1+l2':
                val_loss_batch = criterion_val(y_valid_pred_logit.to(torch.float32), y_valid_true.long()) + l1_regularization + l2_regularization
            
            # val_loss_batch = criterion_val(y_valid_pred_logit.to(torch.float32), y_valid_true.long()).sum() + l1_regularization + l2_regularization
            val_loss += val_loss_batch.item() * len(val_batch['label'])
            temp_valid_pred = y_valid_pred.detach().cpu().numpy()
            temp_valid_true = y_valid_true.detach().cpu().numpy()
            val_labels.append(temp_valid_true)
            val_predictions.append(temp_valid_pred)

        val_labels = np.concatenate(val_labels)
        val_predictions = np.concatenate(val_predictions)
        # print('Debug: val_labels.shape', val_labels.shape)
        # print('Debug: val_predictions.shape', val_predictions.shape)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='macro')
        # auc_roc = roc_auc_score(label_binarize(val_labels, classes=np.arange(n_classes)),
        #                         label_binarize(val_predictions, classes=np.arange(n_classes)),
        #                         multi_class='ovr')

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        # metrics['auc_roc'].append(auc_roc)
        val_loss /= len(val_indices)
        print('Debug val: len(val_indices)', len(val_indices))
    
    metrics['valid_loss'].append(val_loss)  # Store the training loss
    # Get one accuracy at the end of training.
    boolarr = (val_predictions == val_labels)
    boolarr = np.array(boolarr,dtype=bool)
    valid_correct += boolarr.sum()
    # valid_correct += torch.sum(val_predictions == val_labels) #(val_predictions == val_labels).type(torch.float).sum().item()
    valid_accuracy = 100 * valid_correct / len(val_indices)
    metrics['valid_accuracy'].append(valid_accuracy)
    
    print("valid_loss = {}".format(val_loss))
    print("valid_accuracy = {}".format(valid_accuracy))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        PARAMS_valid['best_val_loss'] = best_val_loss # Update to the main code. 
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
        # best_likelihood_state = copy.deepcopy(likelihood.state_dict())
        best_metrics = metrics
        # Since I save the checkpoint at the end of each epoch, so I do not need to save the model again.
        str_epoch = '{epoch:{fill}{width}}'.format(epoch=epoch, fill='0', width=4)
        str_val_loss = '{val_loss:.{prec}f}'.format(val_loss=best_val_loss, prec=4)
        best_model_ckpt_name = filename_output.split('.')[0]+'_epoch_'+str_epoch+'_val_loss_'+str_val_loss+'.pt'
        save_ckpt_best_model_path = os.path.join(checkpoint_path,best_model_ckpt_name)
        PARAMS_valid['best_model_ckpt_name'] = best_model_ckpt_name
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'likelihood_state_dict': likelihood.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'finished_seg_names':finished_seg_names,
            'finished_idx':finished_idx,
            # Include other metrics as needed
            'metrics':metrics,
            'PARAMS_config':PARAMS_config
        }, save_ckpt_best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            flag_break_train = True
        
    # Save checkpoint at the end of each epoch
    save_ckpt_model_path = os.path.join(checkpoint_path,modelckpt_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'likelihood_state_dict': likelihood.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'finished_seg_names':finished_seg_names,
        'finished_idx':finished_idx,
        # Include other metrics as needed
        'metrics':metrics,
        'PARAMS_config':PARAMS_config
    }, save_ckpt_model_path)
    print('Debug: saved model checkpoint with epoch.',save_ckpt_model_path)
    
    # Optionally, save the dataset state at intervals or after certain conditions
    save_ckpt_dataset_path = os.path.join(checkpoint_path,datackpt_name)
    train_loader.dataset.save_checkpoint(save_ckpt_dataset_path)  # Finished all batches, so start from zero.

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        flag_break_train = True

    return best_model_state, best_metrics, flag_break_train, save_ckpt_best_model_path, PARAMS_valid

def train_valid_RNN_GRU_model_all(PARAMS_all):
    # Unload all the parameters.
    train_loader = PARAMS_all['train_loader']
    val_loader = PARAMS_all['val_loader']
    train_dataset = PARAMS_all['train_dataset']
    val_dataset = PARAMS_all['val_dataset']
    batch_size = PARAMS_all['batch_size']
    fold_name = PARAMS_all['fold_name']
    df_train = PARAMS_all['df_train']
    df_valid = PARAMS_all['df_valid']
    dict_paths = PARAMS_all['dict_paths']
    normalize_type = PARAMS_all['normalize_type']
    data_format = PARAMS_all['data_format']
    data_dim = PARAMS_all['data_dim']
    loss_type = PARAMS_all['loss_type']
    learning_rate = PARAMS_all['learning_rate']
    num_iterations = PARAMS_all['num_iterations']
    n_classes = PARAMS_all['n_classes']
    patience = PARAMS_all['patience']
    flag_resume_training = PARAMS_all['flag_resume_training']
    datackpt_name = PARAMS_all['datackpt_name']
    modelckpt_name = PARAMS_all['modelckpt_name']
    best_model_ckpt_name = PARAMS_all['best_model_ckpt_name']
    PARAMS_config = PARAMS_all['PARAMS_config']

    checkpoint_path = dict_paths['saving_path']
    print(f'Debug: resume_training:{flag_resume_training}, checkpoint_path: {checkpoint_path}')
    # Get input dimension.
    if data_dim[:3] == '1D_':
        # Input is 1-D, so 30x50.
        print('Debug: dir(train_dataset)',dir(train_dataset))
        input_h = train_dataset.reset_sec # Without passing variable is fetching the private variable.
        input_w = train_dataset.reset_fs
        if data_dim == '1D_PPG_HR' or data_dim == '1D_HR_ACC' or data_dim == '1D_PPG_ACC_only' or data_dim == '1D_HR_rescaleHR_only':
            input_d = 2 # Input PPG and HR at the same time. Or input HR and ACC without PPG.
        elif data_dim == '1D_PPG_HR_ACC' or data_dim == '1D_PPG_ElgendiHR_ACC' or data_dim == '1D_HR_rescaleHR_ACC_noPPG':
            input_d = 3 # Input PPG, HR, and ACC at the same time.
            print('Debug: input_d',input_d)
        elif data_dim == '1D_PPG_HR_ACC_rescaleHR' or data_dim == '1D_PPG_Elgendi_rescale_HR_ACC' or data_dim == '1D_PPG_aug_HR_ACC_rescaleHR'\
            or data_dim == '1D_PPG_aug5k_HR_ACC_rescaleHR':
            input_d = 4 # Input PPG, HR, rescaled HR, and ACC at the same time.
            print('Debug: input_d',input_d)
        elif data_dim == '1D_PPG_twoHRs_rescaleHRs_ACC':
            input_d = 6 # Input PPG, WEPD HR, rescaled WEPD HR, Elgendi HR, rescaled Elgendi HR, and ACC at the same time.
        else:
            input_d = 1
            print('Debug: input_d',input_d)
        input_size_L = input_h * input_w
        in_channels_d = input_d
    else:
        # Input is 2D, so 128x128.
        input_h = train_dataset.reset_img_size # 128.
        input_w = input_h
        if data_dim[-3:] == '_HR':
            input_d = 2
        elif data_dim[-3:] == 'ACC':
            input_d = 3 # Input PPG, HR, and ACC at the same time.
        else:
            input_d = 1
        input_size_L = input_h
        in_channels_d = input_w
    # Initialize the model.
    # model = MultitaskGPModel(input_h,input_w).to(device) # Dong added the input dimension.
    # likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=4, num_classes=4).to(device)

    num_classes = n_classes
    if data_dim[:2] == '1D':
        model = myRNNGRUmodel(input_size_L, in_channels_d, num_classes).to(device)
    elif data_dim[:2] == '2D':
        model = myRNNGRUmodel_2D(input_size_L, in_channels_d, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Dong comment: not sure if 1e-4 is too big learning rate. 
    criterion_train = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
    print('Debug: len(train_loader.dataset):',len(train_loader.dataset),', len(train_loader):',len(train_loader))
    # mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))

    # Load checkpoint if resuming training for gp model.
    start_epoch = 0
    flag_reload_dataloader = False # We do not need to reset train loader in the new epoch.
    metrics = {
        'epoch': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        #'auc_roc': [],
        'train_loss_Cassey': [], # Without scaling with batch size
        'train_loss_Darren': [], # Scaling with batch size and then devide by training set size.
        'valid_loss': [],
        'train_accuracy':[],
        'valid_accuracy':[],
    }
    ckpt_model_file = os.path.join(checkpoint_path,modelckpt_name)
    print(f'Debug 01/15/2025: ckpt path: {ckpt_model_file}')
    if flag_resume_training and os.path.exists(ckpt_model_file):
        print(f'Debug 01/15/2025: loading ckpt: {ckpt_model_file}')
        checkpoint = torch.load(ckpt_model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        # likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] # Resume from the same epoch because you did not finished it.
        print('Debug resume ckpt: start_epoch',start_epoch)
        metrics = checkpoint['metrics'] # Load the metrics.
        # Update the dataloader if there are segments finished.
        finished_seg_names = checkpoint['finished_seg_names']

        if len(finished_seg_names) > 0:
            # There were segments used in training. Only update the train loader.
            flag_reload_dataloader = True
            print('Debug: renewing train_loader now...')
            startTime_for_tictoc = time.time()
            # ---- Dong, 02/15/2024: I want to test training on large dataset and resume training. ----
            # train_loader,_,_ = preprocess_data_train_val(data_format, clinical_trial_train, clinical_trial_test, batch_size, finished_seg_names,\
            #                   read_all_labels=False)
            # import my_dataloader
            train_loader, val_loader, train_dataset, val_dataset = my_dataloader.load_train_valid_dataloader(fold_name, df_train, df_valid,
                                        dict_paths, normalize_type, data_format, data_dim,
                                        batch_size, finished_seg_names = finished_seg_names)
            endTime_for_tictoc = time.time() - startTime_for_tictoc
            print(f'Debug: took {endTime_for_tictoc} to renew the train_loader')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    flag_break_train = False # Breaking the epoch for loop from validation process.
    flag_new_val = True # 06/29/2024: I should change it into, check if best model ckpt exist, if so, load it.
    for epoch in tqdm(range(start_epoch,num_iterations), desc='Training', unit='epoch', leave=False):
        PARAMS_train = {'model':model,
                        'optimizer':optimizer,
                        'criterion_train':criterion_train,
                        'metrics':metrics,
                        'checkpoint_path':checkpoint_path,
                        'epoch':epoch,
                        'best_val_loss':best_val_loss,
                        'flag_reload_dataloader':flag_reload_dataloader,
                        'train_loader':train_loader,
                        'train_dataset':train_dataset,
                        'batch_size':batch_size,
                        'fold_name':fold_name,
                        'df_train':df_train,
                        'df_valid':df_valid,
                        'dict_paths':dict_paths,
                        'normalize_type':normalize_type,
                        'data_format':data_format,
                        'data_dim':data_dim,
                        'n_classes':n_classes,
                        'modelckpt_name':modelckpt_name,
                        'loss_type':loss_type,
                        'PARAMS_config':PARAMS_config}
        
        dict_return_value = train_RNN_GRU_model(PARAMS_train)

        model = dict_return_value['model']
        optimizer = dict_return_value['optimizer']
        metrics = dict_return_value['metrics']
        lambda_l1 = dict_return_value['lambda_l1']
        lambda_l2 = dict_return_value['lambda_l2']
        finished_seg_names = dict_return_value['finished_seg_names']
        finished_idx = dict_return_value['finished_idx']
        flag_reload_dataloader = dict_return_value['flag_reload_dataloader']
        
        if flag_new_val:
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            save_ckpt_best_model_path = os.path.join(checkpoint_path,best_model_ckpt_name)
            flag_new_val = False
        PARAMS_valid = {'flag_break_train':flag_break_train,
                        'lambda_l1':lambda_l1,
                        'lambda_l2':lambda_l2,
                        'criterion_val':criterion_val,
                        'finished_seg_names':finished_seg_names,
                        'finished_idx':finished_idx,
                        'model':model,
                        'optimizer':optimizer,
                        'metrics':metrics,
                        'checkpoint_path':checkpoint_path,
                        'epoch':epoch,
                        'best_val_loss':best_val_loss,
                        'train_loader':train_loader,
                        'val_loader':val_loader,
                        'loss_type':loss_type,
                        'n_classes':n_classes,
                        'patience':patience,
                        'epochs_no_improve':epochs_no_improve,
                        'datackpt_name':datackpt_name,
                        'modelckpt_name':modelckpt_name,
                        'best_model_ckpt_name':best_model_ckpt_name,
                        'best_model_state':best_model_state,
                        'best_metrics':best_metrics,
                        'save_ckpt_best_model_path':save_ckpt_best_model_path,
                        'PARAMS_config':PARAMS_config}

        best_model_state, \
            best_metrics, \
            flag_break_train, \
            save_ckpt_best_model_path, PARAMS_valid = valid_RNN_GRU_model(PARAMS_valid)
        
        best_val_loss = PARAMS_valid['best_val_loss'] # Update the best loss.
        best_model_state = PARAMS_valid['best_model_state']
        best_metrics = PARAMS_valid['best_metrics']
        save_ckpt_best_model_path = PARAMS_valid['save_ckpt_best_model_path']

        # Clean cache.
        torch.cuda.empty_cache()
        gc.collect()
        
        if flag_break_train:
            # Validation loss did not increase.
            break

    # Optionally, load the best model at the end of training
    if os.path.exists(save_ckpt_best_model_path):
        checkpoint = torch.load(save_ckpt_best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        best_model_state = copy.deepcopy(model.state_dict())
        # best_likelihood_state = copy.deepcopy(likelihood.state_dict())
        best_metrics = metrics

    return best_model_state, best_metrics