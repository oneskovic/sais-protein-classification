from copyreg import pickle
import models.k_nearest_neighbors.knn as knn
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("data/Klasifikacija-proteina.csv")
knn.get_best(data, "jaccard")
unlabeled_data = pd.read_csv("data/test_no_labels.csv")
best_model = pickle.load(open("models/best_model_knn.pkl", 'rb'))
knn.generate_predictions(unlabeled_data, best_model, "temp/knn_predictions.csv")