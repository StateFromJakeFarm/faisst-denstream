import numpy as np


class MicroCluster:
    def __init__(
            self,
            points,
            point_ids,
            creation_time,
            lamb):

        self.points = points
        self.point_arrival_times = np.array([creation_time for _ in range(points.shape[0])])
        self.point_ids = point_ids
        self.lamb = lamb
        self.t0 = creation_time

        self._update_weight()
        self._update_center()
        self._update_radius()


    def add_point(
            self,
            point,
            point_id,
            arrival_time):

        self.points = np.concatenate((self.points, point), axis=0)
        self.point_ids.append(point_id)
        point_arrival_time = np.array([arrival_time])
        self.point_arrival_times = np.concatenate((self.point_arrival_times, point_arrival_time), axis=0)

        # Need to keep stats current
        self._update_weight()
        self._update_center()
        self._update_radius()


    def _get_point_fades(self):
        return np.pow(2, -self.lamb * (self.point_arrival_times - self.t0))


    def _update_weight(self):
        self.weight = np.sum(self._get_point_fades())


    def _update_center(self):
        self.center = np.matmul(self._get_point_fades(), self.points) / self.weight


    def _update_radius(self):
        self.faded_center_l1 = np.matmul(self._get_point_fades(), self.points)

        squared_points = np.pow(self.points, 2)
        self.faded_center_l2 = np.matmul(self._get_point_fades(), squared_points)

        diff_of_faded_norms = (
            # |CF2|/w
            np.linalg.norm(self.faded_center_l2, ord=1) / self.weight

            # - (|CF1|/w)^2
            - np.pow(np.linalg.norm(self.faded_center_l1, ord=1) / self.weight, 2)
        )

        self.radius = np.sqrt(diff_of_faded_norms) if diff_of_faded_norms > 0 else 0


    def _get_radius_if_new_point_added(self, point):
        # See property 3.1 in paper
        new_faded_center_l1 = self.faded_center_l1 + point
        new_faded_center_l2 = self.faded_center_l2 + np.pow(point, 2)
        new_weight = self.weight + 1

        # Calculate radius
        diff_of_faded_norms = (
            # |CF2|/w
            np.linalg.norm(new_faded_center_l2, ord=1) / new_weight

            # - (|CF1|/w)^2
            - np.pow(np.linalg.norm(new_faded_center_l1, ord=1) / new_weight, 2)
        )

        return np.sqrt(diff_of_faded_norms) if diff_of_faded_norms > 0 else 0


    def get_xi(self, tc, Tp):
        return (
            (2**(-self.lamb * (tc - self.t0 + Tp)) - 1)
            / (2**(-self.lamb * Tp) - 1)
        )
