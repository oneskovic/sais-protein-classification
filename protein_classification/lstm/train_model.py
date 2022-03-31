import pickle
import pandas as pd
import numpy as np
from torch import nn
import torch
from torchmetrics import Accuracy, F1Score, CohenKappa
from preprocess import preprocess
import torch.nn.functional as F
import optuna
from sklearn.model_selection import train_test_split
import pickle

data_path = 'data/klasifikacija-proteina.csv'
class_cnt = 11

class LSTMPredictor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.hidden(padded)
        batch_size = tag_space.shape[0]

        preds = torch.zeros((batch_size,tag_space.shape[-1]))
        for i in range(batch_size):
            preds[i] = tag_space[i, lengths[i]-1, :]
        tag_scores = F.log_softmax(preds, dim=0)
        return tag_scores


def eval_metrics(y_pred, y_true, should_print = False):
    metric_acc = Accuracy()
    metric_kappa = CohenKappa(num_classes=class_cnt)
    metric_f1 = F1Score(num_classes=class_cnt, average='macro')
    acc = metric_acc(y_pred, y_true)
    kappa = metric_kappa(y_pred, y_true)
    f1 = metric_f1(y_pred, y_true)
    if should_print:
        print(f'Accuracy: {acc:.2f}')
        print(f'Kappa: {kappa:.2f}')
        print(f'F1: {f1:.2f}')
    return acc, kappa, f1

def get_packed_sequence(batch):
    padded_len = batch[0][0].shape[0]
    embedding_size = batch[0][0].shape[1]
    batch_size = len(batch)

    inputs = torch.zeros((batch_size, padded_len, embedding_size))
    lengths = []
    i = 0
    for x,length in batch:
        inputs[i] = x
        lengths.append(length)
        i += 1
    return torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
def train_model(model, x_train, x_test, y_train, y_test, hparams, trial = None, should_print = False):
    learning_rate = hparams['learning_rate']
    epoch_cnt = 20
    batch_size = 32
    # Define loss function
    criterion = nn.NLLLoss()
    metric = Accuracy()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    
    # Train model
    for epoch in range(epoch_cnt):
        model_file = open(f'models/lstm/trained_models/{trial._trial_id}.pkl', 'wb+')
        pickle.dump(model, model_file)
        model_file.close()

        # perm = torch.randperm(x_train.shape[0])
        # x_train = x_train[perm]
        # y_train = y_train[perm]
        avg_loss = 0.0
        optimize_acc = 0.0
        for i in range(0,len(x_train),batch_size):
            optimizer.zero_grad()
            model.zero_grad()
            x = get_packed_sequence(x_train[i:i+batch_size])
            y = y_train[i:i+batch_size]
            y_pred = model(x)
            loss = criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                current_acc = metric(y_pred,y)
                percent_done = i / len(x_train) * 100.0
                print(f'{percent_done:.2f}% done, current accuracy: {current_acc:.4f}        ', end='\r')
                optimize_acc += current_acc / (len(x_train) / batch_size)
                avg_loss += loss.item() / (len(x_train) / batch_size)
        
        print(f'Loss: {avg_loss:.4f}')
        print(f'Accuracy: {optimize_acc:.4f}')
        # Test model
        with torch.no_grad():
            x = get_packed_sequence(x_test)
            outputs = model(x)
            loss = criterion(outputs, y_test)
            optimize_acc = metric(outputs, y_test)
            print(f'Validation accuracy: {optimize_acc:.4f}')
        if trial is not None:
            trial.report(optimize_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        #     if should_print:
        #         print(f'Test accuracy: {optimize_acc:.2f}       ', end='\n')
                #eval_metrics(outputs, y_test, should_print=True)
        
    return optimize_acc

def create_model(hparams, input_size):
    hidden_dim = hparams['neuron_cnt1']
    model = LSTMPredictor(input_size, hidden_dim, class_cnt)
    return model

def objective(trial):
    # Hyperparameters
    neuron_cnt1 = trial.suggest_int('neuron_cnt1', 50, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Load data
    data = pd.read_csv(data_path)
    # Preprocess data
    x, y = preprocess(data)
    # Split data
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_opt, y_val, y_opt = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)
    # Convert to torch tensors
    # x_train = torch.tensor(x_train, dtype=torch.float32)
    # x_opt = torch.tensor(x_opt, dtype=torch.float32)
    # x_val = torch.tensor(x_val, dtype=torch.float32)

    # y_train = torch.tensor(y_train, dtype=torch.long)
    # y_opt = torch.tensor(y_opt, dtype=torch.long)
    # y_val = torch.tensor(y_val, dtype=torch.long)

    hparams = {'neuron_cnt1': neuron_cnt1, 'learning_rate': learning_rate}
    model = create_model(hparams, x[0][0].shape[1])
    optimze_acc = train_model(model, x_train, x_opt, y_train, y_opt, hparams, trial, True)
    return optimze_acc

data = pd.read_csv(data_path)    
study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction='maximize')
study.optimize(objective, n_trials=5)

pickle.dump(study, open('models/lstm/study.pkl', 'wb+'))
