import numpy as np
from abc import abstractmethod
from src.backend.point_cloud.point_cloud import PointCloud


class WeightCalculatorOutliersBase:

    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    @abstractmethod
    def calculate_outlier_weights(
        self,
        old_point_cloud: PointCloud,
        inliers: np.ndarray[bool],
    ):
        pass
