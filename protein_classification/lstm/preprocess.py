import numpy as np
import pandas as pd
import torch

label_dict = {'PF17482': 0, 'PF02665': 1, 'PF01888': 2, 'PF04749': 3, 'PF00639': 4, 'PF07819': 5, 'PF13618': 6, 'PF01395': 7, 'PF00890': 8, 'PF09721': 9, 'PF04279': 10}

def onehot_encode_aa(x, codes_dict):
    x = x.upper()
    onehot_x = torch.zeros((len(x), len(codes_dict)))
    for i, letter in enumerate(x):
        onehot_x[i][codes_dict[letter]] = 1
    return onehot_x

def generate_ngrams(n, current_str, i, res):
    if i == n:
        res[''.join(x for x in current_str)] = len(res)
        return
    for c in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
        current_str[i] = c
        generate_ngrams(n, current_str, i+1, res)

def ngram_encode_aa(x, ngram_map):
    x = x.upper()
    ngram_x = np.zeros(len(ngram_map))
    for i in range(len(x)-2):
        ngram_x[ngram_map[x[i:i+3]]] = 1
    return ngram_x

def onehot_encode_labels(labels, label_dict):
    onehot_labels = np.zeros((len(labels), len(label_dict)))
    for i, label in enumerate(labels):
        onehot_labels[i][label_dict[label]] = 1
    return onehot_labels

# Padd a list of onehot sequences with zeros
def zero_padd(x, max_len):
    n = len(x[0][0])
    x_padded = torch.zeros((len(x), max_len, n))
    for i, seq in enumerate(x):
        x_padded[i][:seq.shape[0]] = seq

    return x_padded


def make_dict(values):
    d = dict()
    for i, value in enumerate(values):
        d[value] = i
    return d

def preprocess(data):
    aa_seqs = data['AA_sequence']
    lengths = [len(x) for x in aa_seqs]
    max_len = max(lengths)
    
    distinct_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    codes_dict = make_dict(distinct_aas)
    encoded_seqs = zero_padd([onehot_encode_aa(seq, codes_dict) for seq in aa_seqs], max_len)
    encoded_seqs = [(seq, lengths[i]) for i, seq in enumerate(encoded_seqs)]

    # encoded_seqs = encoded_seqs.reshape(encoded_seqs.shape[0],encoded_seqs.shape[1]*encoded_seqs.shape[2])
    
    # ngram_map = dict()
    # generate_ngrams(3, [None]*3, 0, ngram_map)
    # encoded_seqs = np.array([ngram_encode_aa(seq, ngram_map) for seq in aa_seqs])

    if 'prot_Pfam' in data.columns:        
        labels = data['prot_Pfam']
        # distinct_labels = labels.unique()
        # label_dict = make_dict(distinct_labels)
        encoded_labels = np.array([label_dict[label] for label in labels])
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)
        return encoded_seqs, encoded_labels
    else:
        return encoded_seqs
