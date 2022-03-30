import pickle
import pandas as pd
import numpy as np
from torch import nn
import torch
from torchmetrics import Accuracy, F1Score, CohenKappa
from preprocess import preprocess
import optuna
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def eval_metrics(y_pred, y_true, should_print = False):
    metric_acc = Accuracy()
    metric_kappa = CohenKappa(num_classes=200)
    metric_f1 = F1Score(num_classes=200,average='macro')
    metric_f1_per_class = F1Score(num_classes=200,average=None)
    acc = metric_acc(y_pred, y_true)
    kappa = metric_kappa(y_pred, y_true)
    f1 = metric_f1(y_pred, y_true)
    f1_per_class = metric_f1_per_class(y_pred, y_true)
    if should_print:
        print(f'Accuracy: {acc:.2f}')
        print(f'Kappa: {kappa:.2f}')
        print(f'F1: {f1:.2f}')
    return acc, kappa, f1, f1_per_class

def eval_on_train_set(model):
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

    y_pred = model(x_val)
    return eval_metrics(y_pred, y_val, should_print=True)

data_path = 'data/klasifikacija-proteina-novo.csv'

data = pd.read_csv(data_path)
data['prot_Pfam'][data['prot_Pfam'] == 'None'] = None
data = data.dropna()
x,y = preprocess(data)
best_model = pickle.load(open(r'models\neural_network_more_classes\best_model.pkl', 'rb'))


m1,m2,m3,f = eval_on_train_set(model=best_model)
f = f.detach().numpy()
from preprocess import label_dict
labels = np.array([x for x in label_dict.keys()])
mask = f < 0.9
f_to_show = f[mask]
labels_to_show = labels[mask]
sorted_ind = np.array([x for x in reversed(f_to_show.argsort())])
f_to_show = f_to_show[sorted_ind]
labels_to_show = labels_to_show[sorted_ind]

plt.bar(labels_to_show, f_to_show)
plt.xlabel('Klasa')
plt.ylabel('F1')
plt.xticks(rotation='vertical')
plt.title('F1 score za klase gde F1 < 0.9')
plt.show()



print(m1)
# x = torch.tensor(x, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.long)

# y_pred = best_model(x)
# eval_metrics(y_pred,y, should_print = True)
