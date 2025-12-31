import numpy as np
import faiss

from MicroCluster import MicroCluster
from loguru import logger


class DenStream:
    def __init__(
            self,
            lamb, # lambda is a keyword
            beta,
            mu,
            epsilon,
            n_init_points,
            stream_speed):

        # Hyperparameters
        self.lamb = lamb
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.n_init_points = n_init_points
        self.stream_speed = stream_speed

        # Internal components
        self.pmc = []
        self.omc = []
        self.tc = 1
        self.Tp = np.ceil(1/lamb * np.log(beta * mu / (beta * mu - 1)))
        self.init_points = None
        self.initialized = False

        logger.debug(
            "Model Params:"
            f"\n\tlambda        = {self.lamb}"
            f"\n\tbeta          = {self.beta}"
            f"\n\tmu            = {self.mu}"
            f"\n\tepsilon       = {self.epsilon}"
            f"\n\tn_init_points = {self.n_init_points}"
            f"\n\tstream_speed  = {self.stream_speed}"
        )


    @logger.catch
    def partial_fit(
            self,
            X):

        if self.initialized:
            # Everything is all set, so run normal DenStream algo
            pass
        elif self.init_points is None:
            # This is the first batch of points we've seen
            self.init_points = X[:self.n_init_points]
            X = X[self.n_init_points:]
        elif self.init_points.shape[0] < self.n_init_points:
            # We need more points before we can initialize this model, so just add
            # these new points to the pool
            remaining = self.n_init_points - self.init_points.shape[0]
            self.init_points = np.concatenate((self.init_points, X[:remaining]), axis=0)
            X = X[remaining:]

        if not self.initialized and self.init_points.shape[0] == self.n_init_points:
            logger.debug("Got enough points to initialize. Initializing p-micro-clusters...")

            # It's time to initialize our p-micro-clusters
            index = faiss.IndexFlatL2(self.init_points.shape[1])
            index.add(self.init_points)

            # Iterate over all points that have not been assigned to a cluster
            assigned = set()
            for point_idx, point in enumerate(self.init_points):
                if point_idx in assigned:
                    # Point already belongs to a p-micro-cluster
                    continue

                # Find points in epsilon neighborhood
                query = np.array([self.init_points[point_idx]])
                lims, dists, inds = index.range_search(query, epsilon)

                if len(inds) >= self.beta * self.mu:
                    # This point and its epsilon neighborhood are heavy enough to be a p-micro-cluster
                    new_pmc = MicroCluster(self.init_points[inds], 1, self.lamb)
                    self.pmc.append(new_pmc)

                    # The points of this new p-micro-cluster are now off the table
                    assigned.update(inds)

            logger.debug(f"Found {len(self.pmc)} potential-micro-clusters after initialization")
            logger.debug(f"{self.n_init_points - len(assigned)}/{self.n_init_points} points did not get assigned to a potential-micro-cluster")


if __name__ == "__main__":
    # Create model
    lamb = 0.01
    beta = 0.5
    mu = 2.5
    epsilon = 0.04
    n_init_points = 200
    stream_speed = 10

    model = DenStream(lamb, beta, mu, epsilon, n_init_points, stream_speed)

    # Create points
    X = np.random.random((500, 3)).astype('float32')

    # Test fit
    model.partial_fit(X)
