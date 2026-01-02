import numpy as np

from faisst_denstream.DenStream import DenStream
from random import randint
from sys import stderr
from loguru import logger


logger.remove()
logger.add(stderr, level="INFO")

test_dataset_size = 1000
test_dataset_dim = 3
num_test_datasets = 10

# Create model
lamb = 0.05
beta = 0.5
mu = 5
epsilon = 1
n_init_points = int(test_dataset_size * 0.25)
stream_speed = 1

model = DenStream(lamb, beta, mu, epsilon, n_init_points, stream_speed)
print(model.get_params())

# Run on a stream of random data
for i in range(num_test_datasets):
    X = np.random.normal(randint(0, 10), randint(1, 10), size=(test_dataset_size, test_dataset_dim))

    preds = model.fit_predict(X)
    num_assigned = np.where(preds != -1)[0].shape[0]
    print(f"{num_assigned}/{test_dataset_size} ({num_assigned / test_dataset_size * 100:.1f}%) points landed in clusters")
