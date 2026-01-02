import numpy as np

from faisst_denstream.DenStream import DenStream
from random import randint
from sys import stderr
from loguru import logger


logger.remove()
logger.add(stderr, level="INFO")

test_dataset_size = 10000
test_dataset_dim = 3
num_test_datasets = 10

# Create model
lamb = 0.05
beta = 0.5
mu = 10
epsilon = 0.5
n_init_points = int(test_dataset_size * 0.25)
stream_speed = 10

model = DenStream(lamb, beta, mu, epsilon, n_init_points, stream_speed)

for i in range(num_test_datasets):
    X = np.random.normal(loc=randint(0, 10), scale=randint(1, 5), size=(test_dataset_size, test_dataset_dim))
    model.partial_fit(X)

    # Get full clusters
    t = 0
    for cluster_id, points in model._get_clusters():
        t += len(points)

    print(f"{i+1}: {t}/{test_dataset_size * (i+1)} ({t / (test_dataset_size * (i+1)) * 100:.1f}%) points belong to clusters")
