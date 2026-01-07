import numpy as np
import faiss

from loguru import logger
from collections import Counter
from faisst_denstream.MicroCluster import MicroCluster
from inspect import signature
from sklearn.base import BaseEstimator
from collections import deque


class DenStream(BaseEstimator):
    def __init__(
            self,
            lamb,
            mu,
            beta,
            epsilon,
            n_init_points,
            stream_speed=1,
            radius_multiplier=1):

        """
        Density-Based Clustering over an Evolving Data Stream with Noise (DenStream)
        implemented using the FAISS library for fast vector search

        Parameters
        ----------
        lamb: float (required)
            The "fading" or "forgetting" factor determines how quickly past data
            points are forgotten (i.e. lose their influence on current clusters).
            The larger the value, the faster points are forgotten. Must be greater
            than 0.

        mu: int or float (required)
            The minimum weight required to form a cluster. Each point's weight begins
            as 1 and then fades with time at a rate determined by lamb. Mu must be
            greater than 0.

        beta: float (required)
            The fraction of mu (weight) that is required for a micro-cluster to be a
            candidate for merging into an actual cluster (potential-micro-cluster).
            Beta must be in the range (0, 1].

        epsilon: int or float (required)
            The maximum distance a point is allowed to be from its micro-cluster's
            center. Epsilon must be greater than 0.

        n_init_points: int (required)
            The number of points used to initialize the potential-micro-clusters.
            The model must "fit" to at least this many points before any labels
            can be predicted.

        stream_speed: int (optional, default 1)
            The number of points that must be fitted in order for the model's
            current time to be incremented. Because points' influence fade with
            increasing time, higher values of stream_speed lead to the past having
            a greater effect on current clusters. stream_speed must be greater
            than 0.

        radius_multiplier: float (optional, default 1)
            A multiplier applied to potential-micro-clusters' radii during the
            merging process used to create the real clusters. The real clusters
            are created by combining all potential-micro-clusters whose radii
            overlap. Sometimes long, dense blobs of points can create lots of
            heavy potential-micro-clusters whose radii are very small. You'd
            typically like all of the potential-micro-clusters in these long
            stretches of points to be combined into a single cluster, but this won't
            happen due to their small radii. This multiplier is NOT in the original
            paper and is added here for your experimentation. radius_multiplier
            must be greater than 0.
        """

        # Check passed params
        if lamb <= 0:
            raise ValueError("lamb (lambda) must be greater than 0")
        if mu <= 0:
            raise ValueError("mu must be greater than 0")
        if beta <= 0 or beta > 1:
            raise ValueError("beta must be in the range 0 < beta <= 1")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0")
        if n_init_points < 1:
            raise ValueError("n_init_points must be greater than 0")
        if stream_speed <= 0:
            raise ValueError("stream_speed must be greater than 0")
        if radius_multiplier <= 0:
            raise ValueError("radius_multiplier must be greater than 0")

        # Hyperparameters
        self.lamb = lamb
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.n_init_points = n_init_points
        self.stream_speed = stream_speed
        self.radius_multiplier = radius_multiplier

        # Internal components
        self.pmc = []
        self.omc = []
        self.tc = 1
        self.speed_tracker = 1
        self.Tp = int(np.ceil(1/lamb * np.log(beta * mu / (beta * mu - 1))))
        self.init_points = None
        self.initialized = False
        self.next_cluster_id = 0


    def __sklearn_is_fitted__(self):
        return self.initialized


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
                self.pmc[pmc_idx].add_point(point)

                return self.pmc[pmc_idx]

        # We could not find a p-micro-cluster to accept the new point, so now we need
        # to check the o-micro-clusters
        if len(self.omc) == 0:
            # This point will become our first o-micro-cluster
            new_omc = MicroCluster(point, self.tc, self.lamb, radius_multiplier=self.radius_multiplier)
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
            self.omc[omc_idx].add_point(point)

            if self.omc[omc_idx].weight >= self.beta * self.mu:
                self.pmc.append(self.omc[omc_idx])
                self.omc.pop(omc_idx)

                return self.pmc[-1]

            return self.omc[omc_idx]
        else:
            # This point doesn't fit into any p or o-micro-clusters, so it becomes the start of a new o-micro-cluster
            new_omc = MicroCluster(point, self.tc, self.lamb, radius_multiplier=self.radius_multiplier)
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

                for pmc in self.pmc:
                    if pmc != winner:
                        pmc.degrade()

                for omc in self.omc:
                    if omc != winner:
                        omc.degrade()


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
                    new_pmc = MicroCluster(self.init_points[inds], 1, self.lamb, radius_multiplier=self.radius_multiplier)
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

        # Find centers 2 or fewer epsilons apart (max distance between two pmc)
        lims, dists, idx = index.range_search(pmc_centers, np.square(2*self.epsilon)) # L2 gives squared distances
        dists = np.sqrt(dists)

        edges = []
        edge_dists = []
        for cur, start_idx in enumerate(lims):
            edges.append([])
            edge_dists.append([])

            if cur == len(lims) - 1:
                end_idx = len(lims)
            else:
                end_idx = lims[cur+1]

            for neighbor, dist in zip(idx[start_idx:end_idx], dists[start_idx:end_idx]):
                radii_sum = self.pmc[cur].radius + self.pmc[neighbor].radius
                if neighbor == cur or dist > radii_sum:
                    continue

                edges[cur].append(neighbor)
                edge_dists[cur].append(dist)

        # Find connected components
        pmc_clusters = [-1 for _ in range(len(self.pmc))]
        local_next_cluster_id = 0
        cluster_member_last_ids = []
        for pmc_idx in range(len(self.pmc)):
            if pmc_clusters[pmc_idx] != -1:
                # Already visited
                continue

            # Cluster starts out with just one point
            pmc_clusters[pmc_idx] = local_next_cluster_id
            cluster_member_last_ids.append([self.pmc[pmc_idx].last_cluster_id])

            # BFS on reachable neighbors
            Q = deque([pmc_idx])
            while len(Q) > 0:
                cur = Q.popleft()

                pmc_clusters[cur] = local_next_cluster_id
                for neighbor in edges[cur]:
                    if pmc_clusters[neighbor] != -1:
                        # Already visited
                        continue

                    # New point in this cluster
                    Q.append(neighbor)
                    cluster_member_last_ids[local_next_cluster_id].append(self.pmc[neighbor].last_cluster_id)

            local_next_cluster_id += 1

        # This is a slight change from how the algo works in the paper. Instead of requiring at least one of the
        # micro clusters in a cluster to be a core-micro-cluster, we will just check if the sum of the weights
        # of the potential-micro-clusters in the cluster is above mu.
        total_weights = {}
        for pmc_idx, cluster_id in enumerate(pmc_clusters):
            if cluster_id in total_weights:
                total_weights[cluster_id] += self.pmc[pmc_idx].weight
            else:
                total_weights[cluster_id] = self.pmc[pmc_idx].weight

        not_heavy_enough = set()
        for cluster_id, total_weight in total_weights.items():
            if total_weight < self.mu:
                not_heavy_enough.add(cluster_id)

        for pmc_idx, cluster_id in enumerate(pmc_clusters):
            if cluster_id in not_heavy_enough:
                # The cluster this p-micro-cluster was part of isn't actually heavy enough to be a cluster
                pmc_clusters[pmc_idx] = -1

        # If the majority of this cluster was previously part of another cluster, rename it after
        # that other cluster so that there is continuity in the cluster IDs
        old_id_map = {}
        for cluster_id, member_last_ids in enumerate(cluster_member_last_ids):
            last_id_counts = Counter(member_last_ids)
            most_common_old = last_id_counts.most_common()[0][0]
            if most_common_old != -1:
                old_id_map[cluster_id] = most_common_old

        for pmc_idx, cur_cluster in enumerate(pmc_clusters):
            self.pmc[pmc_idx].last_cluster_id = old_id_map.get(cur_cluster, cur_cluster)

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
            f"\n\tclusters:                 {local_next_cluster_id}"
        )


    def predict(
            self,
            X,
            max_dist_outside_radius=np.inf):

        if not self.initialized:
            raise BaseException("predict called before model finished initializing")

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
        dists = np.sqrt(dists) # Index returns squared distances

        indeces = indeces[:,0]
        for point_idx, closest_pmc_idx, closest_pmc_dist in zip(range(X.shape[0]), indeces, dists):
            dist_outside_radius = closest_pmc_dist - self.pmc[closest_pmc_idx].radius
            if dist_outside_radius <= max_dist_outside_radius:
                outputs[point_idx] = self.pmc[closest_pmc_idx].last_cluster_id

        return outputs


    def fit_predict(
            self,
            X,
            y=None,
            max_dist_outside_radius=np.inf):

        # Fit new points
        self.partial_fit(X)

        if not self.initialized:
            raise ValueError(f"Model has not yet consumed enough points to finish initializing ({self.init_points.shape[0]}/{self.n_init_points})!")

        return self.predict(X, max_dist_outside_radius=max_dist_outside_radius)


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from random import randint
    from sys import stderr
    from sklearn.datasets import make_moons

    # Create moons with noise
    test_dataset_size = 1000
    data, _ = make_moons(n_samples=test_dataset_size, noise=0.1)
    data *= 10
    np.random.shuffle(data)

    # Create model
    lamb = 0.01
    beta = 0.2
    mu = 8
    epsilon = 3
    n_init_points = int(test_dataset_size * 0.5)
    stream_speed = 1
    radius_multiplier = 2

    model = DenStream(lamb, mu, beta, epsilon, n_init_points, stream_speed, radius_multiplier)
    print(model.get_params())

    # Train model
    preds = model.fit_predict(data, max_dist_outside_radius=1)

    # Plot results and micro clusters
    points = []
    clusters = []
    for point, pred in zip(data, preds):
        points.append(point)
        clusters.append(pred)

    out = pd.DataFrame(data=points, columns=['x', 'y'])
    out['c'] = clusters

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=out, x='x', y='y', hue='c', palette='tab10')

    for pmc in model.pmc:
        ax.add_patch(plt.Circle(pmc.center, pmc.radius, color='green', fill=False, lw=2))

    for omc in model.omc:
        ax.add_patch(plt.Circle(omc.center, omc.radius, color='red', fill=False, lw=2))

    ax.set_aspect('equal', adjustable='datalim')

    plt.show()
