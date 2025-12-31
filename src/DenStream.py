import numpy as np
import faiss

from MicroCluster import MicroCluster


class DenStream:
    def __init__(
            self,
            lamb, # lambda is a keyword
            beta,
            mu,
            epsilon,
            n_samples_init,
            stream_speed):

        # Hyperparameters
        self.lamb = lamb
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.n_samples_init = n_samples_init
        self.stream_speed = stream_speed

        # Internal components
        self.pmc = []
        self.omc = []
        self.tc = 1
        self.Tp = np.ceil(1/lamb * np.log(beta * mu / (beta * mu - 1)))
        self.init_points = None
        self.initialized = False


    def partial_fit(
            self,
            X):

        if self.initialized:
            # Everything is all set, so run normal DenStream algo
            pass
        elif self.init_points is None:
            # This is the first batch of points we've seen
            self.init_points = X[:self.n_samples_init]
            X = X[self.n_samples_init:]
        elif self.init_points.shape[0] < self.n_samples_init:
            # We need more points before we can initialize this model, so just add
            # these new points to the pool
            remaining = self.n_samples_init - self.init_points.shape[0]
            self.init_points = np.concatenate((self.init_points, X[:remaining]), axis=0)
            X = X[remaining:]

        if not self.initialized and self.init_points.shape[0] == self.n_samples_init:
            # It's time to initialize our p-micro-clusters
            index = faiss.IndexFlatL2(self.init_points.shape[1])
            unassigned = set(range(self.n_samples_init))

            for point_idx in unassigned:
                query = np.array([self.init_points[point_idx]])
                lims, dists, inds = index.range_search(query, epsilon)
                if len(inds) > 0:
                    print(lims)
                    print(dists)
                    print(inds)
                    print()


if __name__ == "__main__":
    # Create model
    lamb = 0.01
    beta = 0.5
    mu = 2.5
    epsilon = 0.5
    n_samples_init = 500
    stream_speed = 10

    model = DenStream(lamb, beta, mu, epsilon, n_samples_init, stream_speed)

    # Create points
    X = np.random.random((500, 3)).astype('float32')

    # Test fit
    model.partial_fit(X)
