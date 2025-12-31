import numpy as np


class MicroCluster:
    def __init__(
            self,
            points,
            creation_time,
            lamb):

        self.points = points
        self.point_arrival_times = np.array([creation_time for _ in range(points.shape[0])])
        self.lamb = lamb
        self.t0 = creation_time

        self._update_weight()
        self._update_center()


    def add_points(
            self,
            points,
            point_arrival_time):

        self.points = np.concatenate((self.points, points), axis=0)
        point_arrival_times = np.array([point_arrival_time for _ in range(points.shape[0])])
        self.point_arrival_times = np.concatenate((self.point_arrival_times, point_arrival_times), axis=0)

        # Need to keep stats current
        self._update_weight()
        self._update_center()
        #self._update_radius()


    def _get_point_fades(self):
        return np.pow(2, -self.lamb * (self.point_arrival_times - self.t0))


    def _update_weight(self):
        self.weight = np.sum(self._get_point_fades())


    def _update_center(self):
        print(self._get_point_fades())
        print(self.points)
        print(np.matmul(self._get_point_fades(), self.points))
        self.center = np.matmul(self._get_point_fades(), self.points) / self.weight
        print(self.center)
        exit(0)
