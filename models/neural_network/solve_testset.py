import pandas as pd
import numpy as np
from torch import nn
import torch
from torchmetrics import Accuracy
from preprocess import preprocess
import optuna
from sklearn.model_selection import train_test_split
import pickle

data_path = 'data/test_no_labels.csv'

data = pd.read_csv(data_path)
x = preprocess(data)
x = torch.tensor(x).float()
best_model = pickle.load(open('best_model.pkl', 'rb'))

y_pred = best_model(x)
predicted_classes = np.argmax(y_pred.detach().numpy(), axis=1)

from preprocess import label_dict
inv_label_dict = {v: k for k, v in label_dict.items()}
class_names = [inv_label_dict[x] for x in predicted_classes]
data['prot_Pfam'] = class_names
data.to_csv('data/test_no_labels_predicted.csv', index=False)