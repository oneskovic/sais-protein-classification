import pickle
import pandas as pd
import numpy as np
from torch import nn
import torch
from torchmetrics import Accuracy, F1Score, CohenKappa
from utils.preprocess import preprocess_encode_ngram
import optuna
from sklearn.model_selection import train_test_split

data_path = 'data/klasifikacija-proteina-large.csv'
class_cnt = 200

def eval_model(x, y, model):
    # Define loss fn and metric
    criterion = nn.CrossEntropyLoss()
    metric = Accuracy()
    pred = model(x)
    loss = criterion(pred, y)
    acc = metric(pred, y)
    return loss, acc

def create_model(hparams, input_shape):
    neuron_cnt1 = hparams['neuron_cnt1']
    neuron_cnt2 = hparams['neuron_cnt2']
    model = nn.Sequential(
        nn.Linear(input_shape, neuron_cnt1),
        nn.ReLU(),
        nn.Linear(neuron_cnt1, neuron_cnt2),
        nn.ReLU(),
        nn.Linear(neuron_cnt2, class_cnt),
    )
    return model

def eval_metrics(y_pred, y_true, should_print = False):
    metric_acc = Accuracy()
    metric_kappa = CohenKappa(num_classes=class_cnt)
    metric_f1 = F1Score(average='macro')
    acc = metric_acc(y_pred, y_true)
    kappa = metric_kappa(y_pred, y_true)
    f1 = metric_f1(y_pred, y_true)
    if should_print:
        print(f'Accuracy: {acc:.2f}')
        print(f'Kappa: {kappa:.2f}')
        print(f'F1: {f1:.2f}')
    return acc, kappa, f1

def train_model(model, x_train, x_test, y_train, y_test, hparams, trial = None, should_print = False):
    learning_rate = hparams['learning_rate']
    epoch_cnt = 4
    batch_size = 100
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    metric = Accuracy()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    for epoch in range(epoch_cnt):
        perm = torch.randperm(x_train.shape[0])
        for i in range(0,x_train.shape[0],batch_size):
            indices = perm[i:i+batch_size]
            x_batch = x_train[indices]
            y_batch = y_train[indices]
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                #print(f'Train Loss: {loss}')
                train_acc = metric(outputs, y_batch)
                #print(f'Train Accuracy: {train_acc}')
            if should_print:
                percent_done = i / x_train.shape[0] / batch_size * 100
                print(f'{percent_done:.2f}% Train accuracy: {train_acc:.2f}        ', end='\r')
        # Test model
        with torch.no_grad():
            outputs = model(x_test)
            loss = criterion(outputs, y_test)
            optimize_acc = metric(outputs, y_test)
            if trial is not None:
                trial.report(optimize_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if should_print:
                print(f'Test accuracy: {optimize_acc:.2f}       ', end='\n')
                eval_metrics(outputs, y_test, should_print=True)
        
    return optimize_acc

def validate(hparams, model):
        # Load data
    data = pd.read_csv(data_path)
    # Preprocess data
    x, y = preprocess(data)
    # Split data
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_opt, y_val, y_opt = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)
    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_opt = torch.tensor(x_opt, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_opt = torch.tensor(y_opt, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_model(model, x_train, x_opt, y_train, y_opt, hparams, should_print=True)
    return eval_model(x_val, y_val, model)

def objective(trial):
    # Hyperparameters
    neuron_cnt1 = trial.suggest_int('neuron_cnt1', 50, 700)
    neuron_cnt2 = trial.suggest_int('neuron_cnt2', 20, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Load data
    data = pd.read_csv(data_path)
    # Preprocess data
    x, y = preprocess_encode_ngram(data)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    # Split data
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_opt, y_val, y_opt = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)
    # Convert to torch tensors

    hparams = {'neuron_cnt1': neuron_cnt1, 'neuron_cnt2': neuron_cnt2, 'learning_rate': learning_rate}
    model = create_model(hparams, x.shape[1])
    optimze_acc = train_model(model, x_train, x_opt, y_train, y_opt, hparams, trial, True)
    return optimze_acc

def main():
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction='maximize')
    study.optimize(objective, n_trials=50)

    model = create_model(study.best_params, x.shape[1])
    val_loss, val_acc = validate(study.best_params, model)
    print('Validation loss: ', val_loss)
    print('Validation accuracy: ', val_acc)

    pickle.dump(model, open('best_model.pkl', 'wb+'))
    pickle.dump(study, open('study.pkl', 'wb+'))
