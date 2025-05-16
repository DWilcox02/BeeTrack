import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class WeightCalculatorBase():
    
    def calculate_distance_weights(
        self, 
        predicted_point_clouds: List[PointCloud], 
        inliers_rotations: List[tuple[np.ndarray, float]], 
        true_query_points: List[np.ndarray]
    ):
        return [
            pc.weights for pc in predicted_point_clouds
        ]
    
    def calculate_outlier_weights(
        self, 
        predicted_point_clouds: List[PointCloud], 
        inliers_rotations: List[tuple[np.ndarray, float]], 
        true_query_points: List[np.ndarray]            
    ):
        return [
            pc.weights for pc in predicted_point_clouds
        ]
        