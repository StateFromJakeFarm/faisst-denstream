import numpy as np


class MicroCluster:
    def __init__(
            self,
            points,
            creation_time,
            lamb):

        points = points
        point_arrival_times = np.array([creation_time for _ in range(points.shape[0])])
        self.lamb = lamb
        self.t0 = creation_time
        self.last_cluster_id = -1
        self.degrade_factor = np.pow(2, -self.lamb)

        # Init weight
        self.weight = points.shape[0]

        # Init radius
        self.linear_sum = np.sum(points, axis=0)
        self.squared_sum = np.sum(np.square(points), axis=0)
        variance_per_dim = (self.squared_sum / self.weight) - np.square(self.linear_sum / self.weight)
        self.radius = np.sqrt(np.sum(variance_per_dim))

        # Init Center
        self.center = self.linear_sum / self.weight


    def add_point(
            self,
            point):

        point = point[0]

        # See property 3.1 in paper
        self.linear_sum += point
        self.squared_sum += np.square(point)

        self.weight += 1

        self.center = self.linear_sum / self.weight

        variance_per_dim = (self.squared_sum / self.weight) - np.square(self.linear_sum / self.weight)
        self.radius = np.sqrt(np.sum(variance_per_dim))

    def degrade(self):
        self.linear_sum *= self.degrade_factor
        self.squared_sum *= self.degrade_factor

        self.weight *= self.degrade_factor

        self.center = self.linear_sum / self.weight

        variance_per_dim = (self.squared_sum / self.weight) - np.square(self.linear_sum / self.weight)
        self.radius = np.sqrt(np.sum(variance_per_dim))


    def _get_radius_if_new_point_added(self, point):
        # See property 3.1 in paper
        new_linear_sum = self.linear_sum + point
        new_squared_sum = self.squared_sum + np.square(point)
        new_weight = self.weight + 1
        new_variance_per_dim = (new_squared_sum / new_weight) - np.square(new_linear_sum / new_weight)

        return np.sqrt(np.sum(new_variance_per_dim))


    def get_xi(self, tc, Tp):
        return (
            (2**(-self.lamb * (tc - self.t0 + Tp)) - 1)
            / (2**(-self.lamb * Tp) - 1)
        )
