import numpy as np

from src.backend.point_cloud.point_cloud import PointCloud


class InlierPredictorBase():
    
    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    def predict_inliers(
        self,
        old_point_cloud: PointCloud,
        final_predictions: np.ndarray,
    ) -> np.ndarray[bool]:
        return np.array([True] * len(old_point_cloud.cloud_points), dtype=bool)