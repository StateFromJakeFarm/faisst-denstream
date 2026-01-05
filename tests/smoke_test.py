import numpy as np

from faisst_denstream.DenStream import DenStream
from random import randint
from sys import stdout
from loguru import logger
from collections import Counter


logger.remove()
logger.add(stdout, level="INFO")

test_dataset_size = 1000
test_dataset_dim = 2
num_test_datasets = 10

# Create model
lamb = 0.01
beta = 0.8
mu = 4
epsilon = 3
n_init_points = int(test_dataset_size * 0.25)
stream_speed = 1

model = DenStream(lamb, beta, mu, epsilon, n_init_points, stream_speed)
print(model.get_params())

# Run on a stream of random data
for i in range(num_test_datasets):
    X = np.random.normal(randint(0, 3), 10, size=(test_dataset_size, test_dataset_dim))

    preds = model.fit_predict(X)
    print("Points per Cluster:")
    print(Counter(preds))
    print()
