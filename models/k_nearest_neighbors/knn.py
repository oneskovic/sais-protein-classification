import pickle
from sklearn.neighbors import KNeighborsClassifier
from models.utils.preprocess import preprocess
from sklearn.model_selection import train_test_split
from collections import namedtuple
import numpy as np

Model = namedtuple('Model', ['k', 'train_accuracy', 'test_accuracy', 'bin'])

def __prepare_data(data):
    preprocessed = preprocess(data)
    inputs = np.array(preprocessed[0], dtype=bool)
    if (len(preprocessed) > 1):
        labels = preprocessed[1]
        return inputs, labels
    else:
        return inputs

def get_best(data, metric, max_k=6):
    inputs, labels = __prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size = 0.2, random_state = 69)
    
    best_model = Model(None, 0.0, 0.0, None)
    # Try all k values from 1 to max_k
    for k in range(2, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        train_accuracy = knn.score(X_train, y_train)
        test_accuracy = knn.score(X_test, y_test)
        # Save model if it's better than the previous best
        if test_accuracy > best_model.test_accuracy:
            best_model = Model(k, train_accuracy, test_accuracy, knn)
        # Logging to console
        print(f"K: {k}")
        print(f"Train accuracy: {train_accuracy}")
        print(f"Test accuracy: {test_accuracy}")
        print("====================")
    
    # Save best model
    pickle.dump(best_model.bin, open(r'/home/ognjen/dev/sais-protein-classification/models/best_model_knn.pkl', 'wb'))
    return best_model


def generate_predictions(data, model, write_dir):
    from models.utils.preprocess import inv_label_dict
    inputs = __prepare_data(data)
    predictions_file = open(write_dir, "w")
    predictions_file.write("prot_ID,AA_sequence,prot_Pfam\n") # CSV header
    for ind, row in data.iterrows():
        prediction = model.predict(inputs[ind])
        prot_id = row["prot_ID"]
        seq = row["AA_sequence"]
        prot_pfam = inv_label_dict[prediction]
        print(f"Protein ID: {prot_id} (Sequence {seq[:7]}...)  ==>  Predicted family: {prot_pfam}")
        predictions_file.write(f"{prot_id},{seq},{prot_pfam}\n")
    predictions_file.close()

# def generate_predictions():
#     global aa_sequence
#     k = 6
#     knn = KNeighborsClassifier(n_neighbors=k, metric="jaccard")
#     knn.fit(X, y)

#     unlabeled_data = pd.read_csv("data/test_no_labels.csv")

#     predictions_file = open("predictions.csv", "w")
#     predictions_file.write("prot_ID,AA_sequence,prot_Pfam\n") # CSV header
#     for ind, row in unlabeled_data.iterrows():
#         seq = row["AA_sequence"]
#         aa_sequence = pd.concat([aa_sequence, pd.Series(data = seq, index = [len(aa_sequence)])])
#         prediction = knn.predict(np.full(1, [len(aa_sequence) - 1]).reshape(-1, 1))[0]
#         prot_id = row["prot_ID"]
#         print(f"Protein ID: {prot_id} (Sequence {seq[:7]}...)  ==>  Predicted family: {prediction}")
#         predictions_file.write(f"{prot_id},{seq},{prediction}\n")
#         aa_sequence = aa_sequence[:-1]
#     predictions_file.close()