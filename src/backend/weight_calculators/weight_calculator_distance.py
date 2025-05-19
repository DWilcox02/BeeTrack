import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud
from .weight_calculator_base import WeightCalculatorBase


ERROR_SIGMA = 0.5
OUTLIER_PENALTY = 0.5

class WeightCalculatorDistance(WeightCalculatorBase):

    def calculate_distance_weights(
        self, 
        predicted_point_clouds: List[PointCloud], 
        inliers_rotations: List[tuple[np.ndarray, float]], 
        true_query_points: List[np.ndarray],
        initial_positions: List[np.ndarray]
    ):
        return [
            self.calculate_weights_errors_for_point(opc, ir, qpr) 
            for opc, ir, qpr in zip(predicted_point_clouds, inliers_rotations, true_query_points)
        ]

    def calculate_weights_errors_for_point(
        self, 
        predicted_point_cloud: PointCloud, 
        inliers_rotation: tuple[np.ndarray, float], 
        true_query_point: np.ndarray
    ):
        qp_predictions = predicted_point_cloud.query_point_predictions()
        distances = self.distances_to_predictions(
            true_query_point=true_query_point, 
            predicted_query_points=qp_predictions
        )
        distace_threshold = predicted_point_cloud.radius
        
        accuracy_weights = np.exp(-distances / distace_threshold)
        new_weights = predicted_point_cloud.weights * (1 - ERROR_SIGMA) + accuracy_weights * ERROR_SIGMA
        weight_sum = np.sum(new_weights)
        new_weights /= weight_sum
        
        assert(np.sum(new_weights) - 1 < 0.001)
        return new_weights


    def distances_to_predictions(self, true_query_point: np.ndarray, predicted_query_points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(predicted_query_points - true_query_point, axis=1)