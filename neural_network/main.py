from copyreg import pickle
import pandas as pd
import numpy as np
from torch import nn
import torch
from torchmetrics import Accuracy
from preprocess import preprocess
import optuna
from sklearn.model_selection import train_test_split

data_path = '/home/ognjen/dev/sais-protein-classification/data/Klasifikacija-proteina.csv'

def validate(hparams):
    neuron_cnt1 = hparams['neuron_cnt1']
    neuron_cnt2 = hparams['neuron_cnt2']
    learning_rate = hparams['learning_rate']
    class_cnt = 11

    # Load data
    data = pd.read_csv('/home/ognjen/dev/sais-protein-classification/data/Klasifikacija-proteina.csv')
    # Preprocess data
    x, y = preprocess(data)
    # Split data
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, test_size=0.2, random_state=42)
    x_opt, x_val, y_opt, y_val = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)
    # Convert to torch tensors
    x_train = torch.from_numpy(x_train).float()
    x_val = torch.from_numpy(x_val).float()
    x_opt = torch.from_numpy(x_opt).float()

    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_opt = torch.from_numpy(y_opt).long()

    # Define model
    model = nn.Sequential(
        nn.Linear(x.shape[1], neuron_cnt1),
        nn.ReLU(),
        nn.Linear(neuron_cnt1, neuron_cnt2),
        nn.ReLU(),
        nn.Linear(neuron_cnt2, class_cnt),
    )
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    metric = Accuracy()
    pred = model(x_val)
    val_loss = criterion(pred, y_val)
    val_acc = metric(pred, y_val)
    return val_loss, val_acc

def objective(trial):
    # Hyperparameters
    neuron_cnt1 = trial.suggest_int('neuron_cnt1', 50, 500)
    neuron_cnt2 = trial.suggest_int('neuron_cnt2', 20, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    class_cnt = 11

    # Load data
    data = pd.read_csv('/home/ognjen/dev/sais-protein-classification/data/Klasifikacija-proteina.csv')
    # Preprocess data
    x, y = preprocess(data)
    # Split data
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, test_size=0.2, random_state=42)
    x_opt, x_val, y_opt, y_val = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)
    # Convert to torch tensors
    x_train = torch.from_numpy(x_train).float()
    x_val = torch.from_numpy(x_val).float()
    x_opt = torch.from_numpy(x_opt).float()

    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_opt = torch.from_numpy(y_opt).long()

    # Define model
    model = nn.Sequential(
        nn.Linear(x.shape[1], neuron_cnt1),
        nn.ReLU(),
        nn.Linear(neuron_cnt1, neuron_cnt2),
        nn.ReLU(),
        nn.Linear(neuron_cnt2, class_cnt),
    )
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    metric = Accuracy()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train model
    for epoch in range(30):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            #print(f'Train Loss: {loss}')
            train_acc = metric(outputs, y_train)
            #print(f'Train Accuracy: {train_acc}')
            
        # Test model
        with torch.no_grad():
            outputs = model(x_opt)
            loss = criterion(outputs, y_opt)
            # Get accuracy
            # _, predicted = torch.max(outputs.data, 1)
            # accuracy = (predicted == y_opt).sum().item() / y_opt.size(0)
            optimize_acc = metric(outputs, y_opt)
            #print(f'Test Loss: {loss}')
            #print(f'Test Accuracy: {optimize_acc}')
            trial.report(optimize_acc, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
        
        percent_done = epoch / 30.0 * 100
        print(f'{percent_done:.2f}% Train accuracy: {train_acc:.2f} Test accuracy: {optimize_acc:.2f}       ', end='\r')
    return optimize_acc

data = pd.read_csv('/home/ognjen/dev/sais-protein-classification/data/Klasifikacija-proteina.csv')
x, y = preprocess(data)
study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction='maximize')
study.optimize(objective, n_trials=100)

val_loss, val_acc = validate(study.best_params)
print('Validation loss: ', val_loss)
print('Validation accuracy: ', val_acc)

pickle.dump(study, open('study.pkl', 'wb+'))
