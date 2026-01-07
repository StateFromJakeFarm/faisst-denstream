# FAISSt DenStream
---
### A performant implementation of the DenStream algorithm that relies heavily on [FAISS](https://github.com/facebookresearch/faiss).

## Installation
Without GPU acceleration:
`pip install faisst-denstream`

With GPU acceleration:
`pip install faisst-denstream[gpu]`


## Basic Usage
```python
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

model = DenStream(lamb, mu, beta, epsilon, n_init_points, stream_speed)

# Multiple datasets to simulate fitting model to stream
X1 = np.random.normal(loc=randint(0, 10), scale=randint(1, 5), size=(10_000, 3))
X2 = np.random.normal(loc=randint(0, 10), scale=randint(1, 5), size=(10_000, 3))

# As long as model has consumed at least n_init_points points, `predict` and `fit_predict`
# can be called to get cluster labels for each point
model.fit(X1)
x1_labels = model.predict(X1)

x2_labels = model.fit_predict(X2)
```
