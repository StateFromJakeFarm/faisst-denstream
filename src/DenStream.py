import numpy as np
import faiss

from loguru import logger
from MicroCluster import MicroCluster
from collections import Counter

# This is sys.maxsize on my machine
MAX_POINT_ID = 9223372036854775807
MAX_CLUSTER_ID = 9223372036854775807 


class DenStream:
    @logger.catch(reraise=True)
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
        self.speed_tracker = 1
        self.Tp = int(np.ceil(1/lamb * np.log(beta * mu / (beta * mu - 1))))
        self.init_points = None
        self.initialized = False
        self.next_point_id = n_init_points # because we only use this variable after initialization
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
            point_id,
            current_time):

        # Find out which p-micro-cluster center the new point is closest to
        pmc_centers = np.vstack([p.center for p in self.pmc])
        index = faiss.IndexFlatL2(pmc_centers.shape[1])
        index.add(pmc_centers)

        if len(point.shape) == 1:
            # Needs to be 2-dim
            point = np.array([point])

        dist, pmc_idx = index.search(point, 1)
        dist = dist[0][0]
        pmc_idx = pmc_idx[0][0]

        # Get would-be radius if this point is added to its nearest p-micro-cluster
        would_be_radius = self.pmc[pmc_idx]._get_radius_if_new_point_added(point)
        if would_be_radius <= self.epsilon:
            # Found a home!
            logger.debug(f"Added new point to potential-micro-cluster #{pmc_idx}")
            self.pmc[pmc_idx].add_point(point, point_id, self.tc)

            return

        # We could not find a p-micro-cluster to accept the new point, so now we need
        # to check the o-micro-clusters
        if len(self.omc) == 0:
            # This point will become our first o-micro-cluster
            logger.debug(f"Created our first outlier-micro-cluster")
            self.omc.append(MicroCluster(point, [point_id], self.tc, self.lamb))

            return

        omc_centers = np.vstack([o.center for o in self.omc])
        index = faiss.IndexFlatL2(omc_centers.shape[1])
        index.add(omc_centers)

        dist, omc_idx = index.search(point, 1)
        dist = dist[0][0]
        omc_idx = omc_idx[0][0]

        would_be_radius = self.omc[omc_idx]._get_radius_if_new_point_added(point)
        if would_be_radius <= self.epsilon:
            # Found a fixer-upper home!
            logger.debug(f"Added new point to outlier-micro-cluster #{omc_idx}")
            self.omc[omc_idx].add_point(point, point_id, self.tc)

            if self.omc[omc_idx].weight >= self.beta * self.mu:
                self.pmc.append(self.omc[omc_idx])
                self.omc.pop(omc_idx)

                logger.debug(
                    f"Outlier-micro-cluster #{omc_idx} has been upgraded to a potential-micro-cluster."
                    f" There are now {len(self.pmc)} p-micro-clusters and {len(self.omc)} o-micro-clusters."
                )
        else:
            # This point doesn't fit into any p or o-micro-clusters, so it becomes the start of a new o-micro-cluster
            self.omc.append(MicroCluster(point, [point_id], self.tc, self.lamb))
            logger.debug(f"Creating new outlier-micro-cluster. There are now {len(self.omc)} outlier-micro-clusters.")


    @logger.catch(reraise=True)
    def partial_fit(
            self,
            X):

        if self.initialized:
            # Everything is all set, so run normal DenStream algo
            for point in X:
                # Add this point to a p or o-micro-cluster
                self._merge_new_point(point, self.next_point_id, self.tc)

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

                # Increment the stream speed tracker
                self.speed_tracker += 1
                if self.speed_tracker == self.stream_speed:
                    # Move forward in time
                    self.tc += 1
                    self.speed_tracker = 1

                # Because this is an online/streaming clustering algorithm, it could be fed lots of data over a long
                # period of time if used in production. Point IDs will loop around after the MAX_POINT_ID-th point
                # is fed to the algorithm. This means that we will only be able to retrieve the cluster assignments
                # for the most recent MAX_POINT_ID points we've fit.
                if self.next_point_id == MAX_POINT_ID - 1:
                    self.next_point_id = 0
                else:
                    self.next_point_id += 1

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

                # Exclude points that are already part of other p-micro-clusters
                inds = list(set([int(x) for x in inds]) - assigned)

                if len(inds) >= self.beta * self.mu:
                    # This point and its epsilon neighborhood are heavy enough to be a p-micro-cluster
                    new_pmc = MicroCluster(self.init_points[inds], [point_idx], 1, self.lamb)
                    self.pmc.append(new_pmc)

                    # The points of this new p-micro-cluster are now off the table
                    assigned.update(inds)

            logger.debug(f"Found {len(self.pmc)} potential-micro-clusters after initialization")
            if len(self.pmc) == 0:
                raise ValueError("Did not find any potential-micro-clusters during initialization! Product of beta and mu is likely too large")

            logger.debug(f"{self.n_init_points - len(assigned)}/{self.n_init_points} points did not get assigned to a potential-micro-cluster")

            # Now that we've initialized, calling this function again will just run the regular DenStream algo
            self.initialized = True
            self.partial_fit(X)


    def _get_clusters(self):
        # Find all p-micro-clusters that qualify as c-micro-clusters
        cmcs = [p for p in self.pmc if p.weight >= self.mu]
        cmc_centers = np.vstack([c.center for c in cmcs])

        # Find groups of c-micro-clusters 
        index = faiss.IndexFlatL2(cmc_centers.shape[1])
        index.add(cmc_centers)

        clusters = []
        assigned = set()
        for cmc_idx, cmc_center in enumerate(cmc_centers):
            if cmc_idx in assigned:
                continue

            # Find centers 2 or fewer epsilons apart (radii of micro-clusters touching)
            query = np.array([cmc_center])
            lims, dists, inds = index.range_search(query, 2 * self.epsilon)

            # Exclude points that are already part of other clusters
            inds = list(set(inds) - assigned)
            assigned.update(inds)

            # Add all points from each c-micro-cluster into bucket
            cluster_points = []
            last_cluster_ids = []
            for idx in inds:
                cmc = cmcs[idx]
                cluster_points.extend(zip(cmc.point_ids, cmc.points.tolist()))
                last_cluster_ids.append(cmc.last_cluster_id)

            # The micro-clusters of a cluster could split apart to become part of another cluster
            # over time and maybe even re-merge together later on. In order to provide some consistency
            # in the cluster IDs, we will hold a vote.
            last_cluster_id_counts = Counter(last_cluster_ids)
            
            # most_common() is giving me an error?
            most_common_last_cluster_id = -1
            max_count = 0
            for last_cluster_id, num_pmc in last_cluster_id_counts.items():
                if num_pmc > max_count:
                    most_common_last_cluster_id = last_cluster_id
                    max_count = num_pmc

            if most_common_last_cluster_id == -1:
                # Because majority of core-micro-clusters in this cluster have never been part of a cluster
                # before, this will be a new cluster.
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
            else:
                # Use the most common cluster ID from last time
                cluster_id = most_common_last_cluster_id

            for idx in inds:
                cmc = cmcs[idx]
                cmc.last_cluster_id = cluster_id

            clusters.append((cluster_id, cluster_points))

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
            f"\n\tcore-micro-clusters:      {len(cmcs)}"
            f"\n\tclusters:                 {len(clusters)}"
        )

        return clusters


if __name__ == "__main__":
    from random import randint
    from sys import stderr

    logger.remove()
    logger.add(stderr, level="INFO")

    test_dataset_size = 1000
    test_dataset_dim = 3
    num_test_datasets = 1000

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

        print(f"{i+1}: {t}/{test_dataset_size * (i+1)} points belong to clusters")
