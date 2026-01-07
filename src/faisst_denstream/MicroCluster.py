import numpy as np


class MicroCluster:
    def __init__(
            self,
            point,
            creation_time,
            lamb,
            radius_multiplier=2):

        if len(point.shape) == 2:
            point = point[0]

        self.lamb = lamb
        self.t0 = creation_time
        self.radius_multiplier = radius_multiplier
        self.last_cluster_id = -1
        self.degrade_factor = np.pow(2, -self.lamb)

        # Init weight
        self.weight = 1

        # Init radius
        self.linear_sum = point
        self.squared_sum = np.square(point)
        variance_per_dim = (self.squared_sum / self.weight) - np.square(self.linear_sum / self.weight)
        variance_per_dim = np.maximum(variance_per_dim, 0)
        self.radius = 0

        # Init Center
        self.center = self.linear_sum / self.weight


    def add_point(
            self,
            point):

        if len(point.shape) == 2:
            point = point[0]

        # See property 3.1 in paper
        self.linear_sum += point
        self.squared_sum += np.square(point)

        self.weight += 1

        self.center = self.linear_sum / self.weight

        variance_per_dim = (self.squared_sum / self.weight) - np.square(self.linear_sum / self.weight)
        variance_per_dim = np.maximum(variance_per_dim, 0)
        self.radius = np.sqrt(np.sum(variance_per_dim)) * self.radius_multiplier


    def degrade(self):
        self.linear_sum *= self.degrade_factor
        self.squared_sum *= self.degrade_factor

        self.weight *= self.degrade_factor

        self.center = self.linear_sum / self.weight

        variance_per_dim = (self.squared_sum / self.weight) - np.square(self.linear_sum / self.weight)
        variance_per_dim = np.maximum(variance_per_dim, 0)
        self.radius = np.sqrt(np.sum(variance_per_dim)) * self.radius_multiplier


    def _get_radius_if_new_point_added(
            self,
            point):

        if len(point.shape) == 2:
            point = point[0]

        # See property 3.1 in paper
        new_linear_sum = self.linear_sum + point
        new_squared_sum = self.squared_sum + np.square(point)
        new_weight = self.weight + 1
        new_variance_per_dim = (new_squared_sum / new_weight) - np.square(new_linear_sum / new_weight)
        new_variance_per_dim = np.maximum(new_variance_per_dim, 0)

        return np.sqrt(np.sum(new_variance_per_dim)) * self.radius_multiplier


    def get_xi(
            self,
            tc,
            Tp):

        return (
            (2**(-self.lamb * (tc - self.t0 + Tp)) - 1)
            / (2**(-self.lamb * Tp) - 1)
        )
