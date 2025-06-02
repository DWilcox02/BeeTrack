import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class InlierPredictorBase():
    
    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    def predict_inliers_rotations(self, old_point_clouds: List[PointCloud], final_positions: np.ndarray):
        return [
            (np.array([True] * len(pc.cloud_points), dtype=bool), 0)
            for pc in old_point_clouds
        ]