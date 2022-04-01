import numpy as np
import pandas as pd

label_dict = {'PF04598': 0, 'PF02081': 1, 'PF14124': 2, 'PF03354': 3, 'PF10187': 4, 'PF11222': 5, 'PF08885': 6, 'PF07406': 7, 'PF02560': 8, 'PF03871': 9, 'PF06239': 10, 'PF11011': 11, 'PF15371': 12, 'PF07863': 13, 'PF08708': 14, 'PF03702': 15, 'PF08701': 16, 'PF16579': 17, 'PF07564': 18, 'PF12253': 19, 'PF08344': 20, 'PF16385': 21, 'PF11449': 22, 'PF14610': 23, 'PF10240': 24, 'PF14347': 25, 'PF15884': 26, 'PF04404': 27, 'PF08356': 28, 'PF12367': 29, 'PF09270': 30, 'PF04871': 31, 'PF09848': 32, 'PF04712': 33, 'PF07034': 34, 'PF02750': 35, 'PF09090': 36, 'PF17517': 37, 'PF10486': 38, 'PF09717': 39, 'PF08807': 40, 'PF06889': 41, 'PF12842': 42, 'PF13574': 43, 'PF07052': 44, 'PF11874': 45, 'PF04721': 46, 'PF05106': 47, 'PF08698': 48, 'PF04062': 49, 'PF11006': 50, 'PF09068': 51, 'PF04420': 52, 'PF08618': 53, 'PF09538': 54, 'PF05030': 55, 'PF12619': 56, 'PF14937': 57, 'PF05219': 58, 'PF00870': 59, 'PF03885': 60, 'PF15239': 61, 'PF06395': 62, 'PF14438': 63, 'PF08861': 64, 'PF04621': 65, 'PF16173': 66, 'PF10998': 67, 'PF14454': 68, 'PF06147': 69, 'PF06050': 70, 'PF08167': 71, 'PF07040': 72, 'PF14606': 73, 'PF07297': 74, 'PF05345': 75, 'PF03957': 76, 'PF10558': 77, 'PF15785': 78, 'PF06930': 79, 'PF04949': 80, 'PF05827': 81, 'PF05499': 82, 'PF09440': 83, 'PF03125': 84, 'PF06552': 85, 'PF05527': 86, 'PF11267': 87, 'PF14457': 88, 'PF11777': 89, 'PF17261': 90, 'PF14966': 91, 'PF12257': 92, 'PF15106': 93, 'PF16103': 94, 'PF09738': 95, 'PF10176': 96, 'PF05404': 97, 'PF13934': 98, 'PF12013': 99, 'PF02010': 100, 'PF05361': 101, 'PF02505': 102, 'PF10210': 103, 'PF10220': 104, 'PF15955': 105, 'PF07231': 106, 'PF10738': 107, 'PF07896': 108, 'PF11594': 109, 'PF08293': 110, 'PF15383': 111, 'PF01905': 112, 'PF07424': 113, 'PF07165': 114, 'PF03192': 115, 'PF16046': 116, 'PF04113': 117, 'PF16507': 118, 'PF10675': 119, 'PF05735': 120, 'PF15261': 121, 'PF14390': 122, 'PF14773': 123, 'PF13963': 124, 'PF01235': 125, 'PF09392': 126, 'PF15262': 127, 'PF08769': 128, 'PF07807': 129, 'PF09968': 130, 'PF04100': 131, 'PF10455': 132, 'PF04310': 133, 'PF08854': 134, 'PF12754': 135, 'PF03378': 136, 'PF06191': 137, 'PF14007': 138, 'PF09029': 139, 'PF14385': 140, 'PF10233': 141, 'PF12070': 142, 'PF07930': 143, 'PF16584': 144, 'PF08216': 145, 'PF00477': 146, 'PF12957': 147, 'PF16378': 148, 'PF01165': 149, 'PF12100': 150, 'PF16516': 151, 'PF14208': 152, 'PF04090': 153, 'PF10744': 154, 'PF15249': 155, 'PF12631': 156, 'PF11034': 157, 'PF02486': 158, 'PF13798': 159, 'PF09415': 160, 'PF11347': 161, 'PF11371': 162, 'PF10077': 163, 'PF08690': 164, 'PF16911': 165, 'PF06730': 166, 'PF08832': 167, 'PF09797': 168, 'PF04634': 169, 'PF12917': 170, 'PF12527': 171, 'PF13204': 172, 'PF07830': 173, 'PF16750': 174, 'PF05659': 175, 'PF08853': 176, 'PF06005': 177, 'PF12688': 178, 'PF16266': 179, 'PF14677': 180, 'PF14902': 181, 'PF12444': 182, 'PF11201': 183, 'PF02233': 184, 'PF12634': 185, 'PF14977': 186, 'PF06124': 187, 'PF16498': 188, 'PF03887': 189, 'PF14089': 190, 'PF03759': 191, 'PF10260': 192, 'PF15013': 193, 'PF17264': 194, 'PF11677': 195, 'PF12899': 196, 'PF07224': 197, 'PF10546': 198, 'PF14354': 199}

def onehot_encode_aa(x, codes_dict):
    x = x.upper()
    onehot_x = np.zeros((len(x), len(codes_dict)))
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
    
    distinct_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    codes_dict = make_dict(distinct_aas)
    # encoded_seqs = zero_padd([onehot_encode_aa(seq, codes_dict) for seq in aa_seqs], max_len)
    # encoded_seqs = encoded_seqs.reshape(encoded_seqs.shape[0],encoded_seqs.shape[1]*encoded_seqs.shape[2])
    
    ngram_map = dict()
    generate_ngrams(3, [None]*3, 0, ngram_map)
    encoded_seqs = np.array([ngram_encode_aa(seq, ngram_map) for seq in aa_seqs])

    if 'prot_Pfam' in data.columns:        
        labels = data['prot_Pfam']
        encoded_labels = np.array([label_dict[label] for label in labels])
        return encoded_seqs, encoded_labels
    else:
        return encoded_seqs
