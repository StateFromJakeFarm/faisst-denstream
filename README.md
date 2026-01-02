# FAISSt DenStream
---
### A performant implementation of the DenStream algorithm that relies heavily on [FAISS](https://github.com/facebookresearch/faiss).

## Installation
`pip install git+https://github.com/StateFromJakeFarm/faisst-denstream.git`


## Basic Usage
```python
from faisst_denstream import DenStream

# Create model
lamb = 0.05
beta = 0.5
mu = 10
epsilon = 0.5
n_init_points = int(test_dataset_size * 0.25)
stream_speed = 10

model = DenStream(lamb, beta, mu, epsilon, n_init_points, stream_speed)

# Multiple datasets to simulate fitting model to stream
X1 = np.random.normal(loc=randint(0, 10), scale=randint(1, 5), size=(10_000, 3))
X2 = np.random.normal(loc=randint(0, 10), scale=randint(1, 5), size=(10_000, 3))

# Fit once and get points in each cluster
model.partial_fit(X1)
model.get_clusters()

# Fit again and get points in each cluster
model.partial_fit(X2)
model.get_clusters()
```
