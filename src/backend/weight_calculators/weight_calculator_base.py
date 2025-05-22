import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class WeightCalculatorBase():

    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)
    
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
        