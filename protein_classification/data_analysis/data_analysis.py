import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def count_aa_occurance(data):
    letter_cnts = dict()
    for seq in data['AA_sequence']:
        for letter in seq:
            if letter not in letter_cnts:
                letter_cnts[letter] = 1
            else:
                letter_cnts[letter] += 1
    return letter_cnts

data = pd.read_csv('data/klasifikacija-proteina-small.csv')

lengths = np.array([len(x) for x in data['AA_sequence']])
plt.title('Raspodela duzina sekvenci amino kiselina')
plt.hist(lengths, bins=50)
plt.show()
longest_seq = max(lengths)
print('Najduza sekvenca: ', longest_seq)

letter_cnts = count_aa_occurance(data)
plt.title('Raspodela broja pojavljivanja svake aminokiseline')
plt.bar(letter_cnts.keys(), letter_cnts.values())
plt.show()

categories = data['prot_Pfam'].unique()
print('Broj razlicitih kategorija: ', len(categories))
