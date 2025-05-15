import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class InlierPredictorBase():
    
    def predict_inliers_rotations(self, old_point_clouds: List[PointCloud], final_positions: np.ndarray):
        return np.array([
            ([True] * len(pc.cloud_points), 0) 
            for pc in old_point_clouds
        ], dtype=bool)