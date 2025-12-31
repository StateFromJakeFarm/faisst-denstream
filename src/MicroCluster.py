import numpy as np


class MicroCluster:
    def __init__(
            self,
            first_point,
            creation_time):

        self.points = np.array([first_point])
        self.t0 = creation_time
