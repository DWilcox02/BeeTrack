import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class WeightCalculatorBase():
    
    def calculate_weights_errors(self, old_point_clouds: List[PointCloud], inliers_rotations=List[tuple[np.ndarray, float]], query_point_reconstructions=List[np.ndarray]):
        return [
            pc.weights for pc in old_point_clouds
        ]