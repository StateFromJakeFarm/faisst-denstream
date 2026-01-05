import numpy as np

from faisst_denstream.DenStream import DenStream
from random import randint
from sys import stderr
from loguru import logger
from sklearn.model_selection import ParameterGrid
from dbcv import dbcv
from collections import Counter

def clustering_grid_search(model_class, X, param_grid):
    best_score = -1
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        try:
            model = model_class(**params)
            labels = model.fit_predict(X)

            assigned_idx = np.where(labels != -1)[0]
            labels = labels[assigned_idx]
            num_clusters = np.unique(labels).shape[0] - 1 # one of them is -1
        except:
            continue

        # Silhouette score requires at least 2 clusters
        if len(np.unique(labels)) > 1:
            score = dbcv(X, labels)
        else:
            score = -1


        print(f"DBCV: {score:.4f}\tnum clusters: {num_clusters}")

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    return best_model, best_params, best_score


logger.remove()

test_dataset_size = 1000
test_dataset_dim = 2

X = np.random.uniform(randint(0, 10), randint(1, 10), size=(test_dataset_size, test_dataset_dim))

grid = {
    "lamb": [0.01, 0.05, 0.1],
    "beta": [0.3, 0.5, 0.7],
    "mu": [3, 5, 10, 20, 40, 50],
    "epsilon": [0.1, 0.3, 0.5, 1, 1.5, 2, 5],
    "n_init_points": [int(test_dataset_size * i * 0.1) for i in range(1, 6)],
    "stream_speed": [1, 5],
}

best_model, best_params, best_score = clustering_grid_search(DenStream, X, grid)
print(best_score)
print(best_params)
