import numpy as np
import faiss

from loguru import logger
from collections import Counter
from faisst_denstream.MicroCluster import MicroCluster
from inspect import signature
from sklearn.base import BaseEstimator


# This is sys.maxsize on my machine
MAX_CLUSTER_ID = 9223372036854775807 


class DenStream(BaseEstimator):
    def __init__(
            self,
            lamb=0.05,
            beta=0.7,
            mu=10,
            epsilon=0.3,
            n_init_points=300,
            stream_speed=10):

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
        self.speed_tracker = 1
        self.Tp = int(np.ceil(1/lamb * np.log(beta * mu / (beta * mu - 1))))
        self.init_points = None
        self.initialized = False
        self.next_cluster_id = 0

        logger.debug(
            "Model Params:"
            f"\n\tlambda        = {self.lamb}"
            f"\n\tbeta          = {self.beta}"
            f"\n\tmu            = {self.mu}"
            f"\n\tepsilon       = {self.epsilon}"
            f"\n\tTp            = {self.Tp}"
            f"\n\tn_init_points = {self.n_init_points}"
            f"\n\tstream_speed  = {self.stream_speed}"
        )


    def _merge_new_point(
            self,
            point,
            current_time):

        if len(point.shape) == 1:
            # Needs to be 2-dim
            point = np.array([point])

        # Find out which p-micro-cluster center the new point is closest to
        if len(self.pmc) > 0:
            pmc_centers = np.vstack([p.center for p in self.pmc])
            index = faiss.IndexFlatL2(pmc_centers.shape[1])
            index.add(pmc_centers)

            dist, pmc_idx = index.search(point, 1)
            dist = np.sqrt(dist[0][0])
            pmc_idx = pmc_idx[0][0]

            # Get would-be radius if this point is added to its nearest p-micro-cluster
            would_be_radius = self.pmc[pmc_idx]._get_radius_if_new_point_added(point)
            if would_be_radius <= self.epsilon:
                # Found a home!
                logger.debug('point added to existing p-micro-cluster')
                print(self.pmc[pmc_idx].radius, would_be_radius)
                self.pmc[pmc_idx].add_point(point)

                return self.pmc[pmc_idx]

        # We could not find a p-micro-cluster to accept the new point, so now we need
        # to check the o-micro-clusters
        if len(self.omc) == 0:
            # This point will become our first o-micro-cluster
            logger.debug('point became new o-micro-cluster')
            new_omc = MicroCluster(point, self.tc, self.lamb)
            self.omc.append(new_omc)

            return new_omc

        omc_centers = np.vstack([o.center for o in self.omc])
        index = faiss.IndexFlatL2(omc_centers.shape[1])
        index.add(omc_centers)

        dist, omc_idx = index.search(point, 1)
        dist = np.sqrt(dist[0][0])
        omc_idx = omc_idx[0][0]

        would_be_radius = self.omc[omc_idx]._get_radius_if_new_point_added(point)
        if would_be_radius <= self.epsilon:
            # Found a fixer-upper home!
            logger.debug('point added to o-micro-cluster')
            self.omc[omc_idx].add_point(point)

            if self.omc[omc_idx].weight >= self.beta * self.mu:
                logger.debug('o-micro-cluster upgraded to p-micro-cluster')
                self.pmc.append(self.omc[omc_idx])
                self.omc.pop(omc_idx)

                return self.pmc[-1]

            return self.omc[omc_idx]
        else:
            # This point doesn't fit into any p or o-micro-clusters, so it becomes the start of a new o-micro-cluster
            logger.debug('point became new o-micro-cluster')
            new_omc = MicroCluster(point, self.tc, self.lamb)
            self.omc.append(new_omc)

            return new_omc


    def _DenStream(
            self,
            X):

        for point in X:
            # Add this point to a p or o-micro-cluster
            winner = self._merge_new_point(point, self.tc)

            if self.tc % self.Tp == 0:
                logger.debug(f"Removing potential and outlier micro-clusters whose weights have fallen too far")

                # Remove any p-micro-clusters whose weights have fallen below the threshold
                to_delete = []
                for idx, pmc in enumerate(self.pmc):
                    if pmc.weight < self.beta * self.mu:
                        to_delete.append(idx)

                logger.debug(f"\t{len(to_delete)}/{len(self.pmc)} potential-micro-clusters were deleted")

                for idx in reversed(to_delete):
                    del self.pmc[idx]

                # Remove any o-micro-clusters whose weights have fallen below their custom thresholds
                to_delete = []
                for idx, omc in enumerate(self.omc):
                    if omc.weight < omc.get_xi(self.tc, self.Tp):
                        to_delete.append(idx)

                logger.debug(f"\t{len(to_delete)}/{len(self.omc)} outlier-micro-clusters were deleted")

                for idx in reversed(to_delete):
                    del self.omc[idx]

            if self.speed_tracker == self.stream_speed:
                # Move forward in time
                self.tc += 1
                self.speed_tracker = 1


    def partial_fit(
            self,
            X,
            y=None):

        if self.initialized:
            # Everything is all set, so run normal DenStream algo
            self._DenStream(X)
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
                lims, dists, inds = index.range_search(query, np.square(self.epsilon)) # L2 gives squared distances

                # Exclude points that are already part of other p-micro-clusters
                inds = list(set([int(x) for x in inds]) - assigned)

                if len(inds) >= self.beta * self.mu:
                    # This point and its epsilon neighborhood are heavy enough to be a p-micro-cluster
                    new_pmc = MicroCluster(self.init_points[inds], 1, self.lamb)
                    self.pmc.append(new_pmc)

                    # The points of this new p-micro-cluster are now off the table
                    assigned.update(inds)

            logger.debug(f"Found {len(self.pmc)} potential-micro-clusters after initialization")
            if len(self.pmc) == 0:
                raise ValueError("Did not find any potential-micro-clusters during initialization! Product of beta and mu is likely too large")

            logger.debug(f"{self.n_init_points - len(assigned)}/{self.n_init_points} points did not get assigned to a potential-micro-cluster")

        if not self.initialized and self.init_points.shape[0] == self.n_init_points and len(self.pmc) > 0:
            self.initialized = True

        if self.initialized and X.shape[0] > 0:
            # There are still some points left and we've finished initializing, so we can run this on the remaining points
            self._DenStream(X)

        return self


    def fit(
            self,
            X,
            y=None):

        self.partial_fit(X, y)

        return self


    def _generate_clusters(self):

        if len(self.pmc) == 0:
            # Can't have clusters without p-micro-clusters
            return

        # Find directly-densely-connected groups of p-micro-clusters 
        pmc_centers = np.vstack([p.center for p in self.pmc])
        index = faiss.IndexFlatL2(pmc_centers.shape[1])
        index.add(pmc_centers)

        pmc_cluster_ids = [-1 for _ in range(len(self.pmc))]
        for cur_idx, cur_center in enumerate(pmc_centers):
            if pmc_cluster_ids[cur_idx] != -1:
                # This p-micro-cluster has already been assigned to a cluster
                continue

            # Find centers 2 or fewer epsilons apart (max distance between two pmc)
            query = np.array([cur_center])
            lims, dists, double_epsilon_indeces = index.range_search(query, np.square(2 * self.epsilon)) # L2 gives squared distances

            # Two micro clusters can be 2*epsilon apart and still not be densely connected because the
            # actual radii of the micro clusters themselves might not be touching
            dists = np.sqrt(dists)
            ddc = []
            for dist, neighb_idx in zip(dists, double_epsilon_indeces):
                if neighb_idx != cur_idx and dist <= self.pmc[cur_idx].radius + self.pmc[neighb_idx].radius:
                    # Radii are actually touching
                    ddc.append(neighb_idx)

            # Check if any of the points we're directly-density-connected to are already
            # part of another cluster
            neighbor_cluster_ids = [pmc_cluster_ids[n] for n in ddc if pmc_cluster_ids[n] != -1]
            if len(neighbor_cluster_ids) > 0:
                # At least one neighbor is already in a cluster, so assign this micro cluster, all neighbors,
                # and all points belonging to clusters of which neighbors are members to the neighbor
                # cluster with the smallest ID
                cluster_id = min(neighbor_cluster_ids)
            else:
                # None of our neighbors belong to any clusters, so let's start a new one
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1

            pmc_cluster_ids[cur_idx] = cluster_id

            for neighb_idx in ddc:
                if pmc_cluster_ids[neighb_idx] != -1 and pmc_cluster_ids[neighb_idx] != cluster_id:
                    # This neighbor belongs to another cluster that needs to be subsumed into the oldest neighbor cluster
                    for idx, cur_cluster_id in enumerate(pmc_cluster_ids):
                        if idx != neighb_idx and cur_cluster_id == pmc_cluster_ids[neighb_idx]:
                            pmc_cluster_ids[idx] = cluster_id

                    pmc_cluster_ids[neighb_idx] = cluster_id

                else:
                    # This neighbor does not belong to a cluster yet
                    pmc_cluster_ids[neighb_idx] = cluster_id

        # This is a slight change from how the algo works in the paper. Instead of requiring at least one of the
        # micro clusters in a cluster to be a core-micro-cluster, we will just check if the sum of the weights
        # of the potential-micro-clusters in the cluster is above mu.
        total_weights = {}
        for pmc_idx, cluster_id in enumerate(pmc_cluster_ids):
            if cluster_id in total_weights:
                total_weights[cluster_id] += self.pmc[pmc_idx].weight
            else:
                total_weights[cluster_id] = self.pmc[pmc_idx].weight

        not_heavy_enough = set()
        for cluster_id, total_weight in total_weights.items():
            if total_weight < self.mu:
                not_heavy_enough.add(cluster_id)

        last_cluster_ids = {c: [] for c in set(pmc_cluster_ids)}
        for pmc_idx, cluster_id in enumerate(pmc_cluster_ids):
            if cluster_id in not_heavy_enough:
                # The cluster this p-micro-cluster was part of isn't actually heavy enough to be a cluster
                pmc_cluster_ids[pmc_idx] = -1
            else:
                # Record the last cluster that this PM
                last_cluster_ids[cluster_id].append(self.pmc[pmc_idx].last_cluster_id)

        # The micro-clusters of a cluster could split apart to become part of another cluster
        # over time and maybe even re-merge together later on. In order to provide some consistency
        # in the cluster IDs, we will hold a vote.
        last_id_map = {}
        for cluster_id, last_ids_this_cluster in last_cluster_ids.items():
            last_cluster_id_counts = Counter(last_ids_this_cluster)
        
            # most_common() is giving me an error?
            most_common_last_cluster_id = -1
            max_count = 0
            for last_cluster_id, num_pmc in last_cluster_id_counts.items():
                if num_pmc > max_count:
                    most_common_last_cluster_id = last_cluster_id
                    max_count = num_pmc

            if most_common_last_cluster_id != -1:
                # Use the most common cluster ID from last time
                last_id_map[cluster_id] = most_common_last_cluster_id

        for pmc_idx, cluster_id in enumerate(pmc_cluster_ids):
            self.pmc[pmc_idx].last_cluster_id = last_id_map.get(cluster_id, cluster_id)

        # TODO: address this!
        # There is an edge case where cluster A gets split up into two clusters and cluster A's core-micro-clusters
        # form the majority in both of those new clusters. In this case, the largest of the duplicates will maintain
        # cluster A's ID, and other clusters will get new IDs.
        #cluster_id_uses = {}
        #for idx, (cluster_id, _) in enumerate(clusters):
        #    if cluster_id in cluster_id_uses:
        #        cluster_id_uses[cluster_id].append(idx)
        #    else:
        #        cluster_id_uses[cluster_id] = [idx]

        #print(cluster_id_uses)

        # Update last cluster assignments for each core-micro-cluster

        logger.info(
            "Clustering Request:"
            f"\n\toutlier-micro-clusters:   {len(self.omc)}"
            f"\n\tpotential-micro-clusters: {len(self.pmc)}"
            f"\n\tclusters:                 {len(last_cluster_ids)}"
        )


    def predict(
            self,
            X):

        outputs = [-1 for _ in X]
        if len(self.pmc) == 0:
            # Can't have clusters without p-micro-clusters
            return outputs

        self._generate_clusters()

        # TODO: should this map to any pmc, or only cmcs?
        pmc_centers = np.vstack([p.center for p in self.pmc]).astype(np.float32)
        index = faiss.IndexFlatL2(pmc_centers.shape[1])
        index.add(pmc_centers)

        dists, indeces = index.search(X, k=1)

        indeces = indeces[:,0]
        for point_idx, neighbor_idx in zip(range(X.shape[0]), indeces):
            outputs[point_idx] = self.pmc[neighbor_idx].last_cluster_id

        return outputs


    def fit_predict(
            self,
            X,
            y=None):

        # Fit new points
        self.partial_fit(X)

        if not self.initialized:
            raise ValueError(f"Model has not yet consumed enough points to finish initializing ({self.init_points.shape[0]}/{self.n_init_points})!")

        return self.predict(X)


if __name__ == "__main__":
    from random import randint
    from sys import stderr

    logger.remove()
    logger.add(stderr, level="INFO")

    test_dataset_size = 1000
    test_dataset_dim = 3
    num_test_datasets = 10

    # Create model
    lamb = 0.05
    beta = 0.5
    mu = 5
    epsilon = 0.75
    n_init_points = int(test_dataset_size * 0.25)
    stream_speed = 1

    model = DenStream(lamb, beta, mu, epsilon, n_init_points, stream_speed)
    print(model.get_params())

    for i in range(num_test_datasets):
        X = np.random.normal(randint(0, 10), randint(1, 10), size=(test_dataset_size, test_dataset_dim))

        preds = model.fit_predict(X)
        num_assigned = np.where(preds != -1)[0].shape[0]
        print(f"{num_assigned}/{test_dataset_size} ({num_assigned / test_dataset_size * 100:.1f}%) points landed in clusters")
