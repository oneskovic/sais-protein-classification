import numpy as np
import pandas as pd

def onehot_encode_aa(x, codes_dict):
    x = x.upper()
    onehot_x = np.zeros((len(x), len(codes_dict)))
    for i, letter in enumerate(x):
        onehot_x[i][codes_dict[letter]] = 1
    return onehot_x

def onehot_encode_labels(labels, label_dict):
    onehot_labels = np.zeros((len(labels), len(label_dict)))
    for i, label in enumerate(labels):
        onehot_labels[i][label_dict[label]] = 1
    return onehot_labels

# Padd a list of onehot sequences with zeros
def zero_padd(x, max_len):
    n = len(x[0][0])
    x_padded = np.zeros((len(x), n, max_len))
    for i, seq in enumerate(x):
        x_padded[i][:,:seq.shape[0]] = seq.T

    return x_padded


def make_dict(values):
    d = dict()
    for i, value in enumerate(values):
        d[value] = i
    return d

def preprocess(data):
    aa_seqs = data['AA_sequence']
    max_len = max([len(x) for x in aa_seqs])
    labels = data['prot_Pfam']
    
    distinct_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    codes_dict = make_dict(distinct_aas)
    onehot_seqs = zero_padd([onehot_encode_aa(seq, codes_dict) for seq in aa_seqs], max_len)
    onehot_seqs = onehot_seqs.reshape(onehot_seqs.shape[0],onehot_seqs.shape[1]*onehot_seqs.shape[2])
    distinct_labels = labels.unique()
    label_dict = make_dict(distinct_labels)
    encoded_labels = np.array([label_dict[label] for label in labels])

    return onehot_seqs, encoded_labels
