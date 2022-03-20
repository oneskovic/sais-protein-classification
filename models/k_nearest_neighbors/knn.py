from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from leven import levenshtein

row_cnt = 1000
data = pd.read_csv("data/Klasifikacija-proteina.csv")
aa_sequence = data["AA_sequence"].head(row_cnt)
protein_family = data["prot_Pfam"].head(row_cnt)

def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return levenshtein(aa_sequence[i], aa_sequence[j])

X = np.arange(len(aa_sequence)).reshape(-1, 1)
y = protein_family

def test_ks():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0, random_state = 69)
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k, metric=lev_metric)
        knn.fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
        print(f"K: {k}")
        print(f"Train accuracy: {train_accuracy[i]}")
        print(f"Test accuracy: {test_accuracy[i]}")
        print("====================")

def generate_predictions():
    global aa_sequence
    k = 6
    knn = KNeighborsClassifier(n_neighbors=k, metric=lev_metric)
    knn.fit(X, y)

    unlabeled_data = pd.read_csv("data/test_no_labels.csv")

    predictions_file = open("predictions.csv", "w")
    predictions_file.write("prot_ID,AA_sequence,prot_Pfam\n") # CSV header
    for ind, row in unlabeled_data.iterrows():
        seq = row["AA_sequence"]
        aa_sequence = pd.concat([aa_sequence, pd.Series(data = seq, index = [len(aa_sequence)])])
        prediction = knn.predict(np.full(1, [len(aa_sequence) - 1]).reshape(-1, 1))[0]
        prot_id = row["prot_ID"]
        print(f"Protein ID: {prot_id} (Sequence {seq[:7]}...)  ==>  Predicted family: {prediction}")
        predictions_file.write(f"{prot_id},{seq},{prediction}\n")
        aa_sequence = aa_sequence[:-1]
    predictions_file.close()

generate_predictions()