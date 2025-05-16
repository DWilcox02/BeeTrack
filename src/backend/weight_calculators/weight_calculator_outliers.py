import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud
from .weight_calculator_base import WeightCalculatorBase

WEIGHT_PENALTY = 0.5

class WeightCalculatorOutliers(WeightCalculatorBase):

    def calculate_outlier_weights(
        self,
        old_point_clouds: List[PointCloud],
        inliers_rotations: List[tuple[np.ndarray, float]],
    ):
        return [
            self.calculate_outlier_weights_for_point(opc, ir)
            for opc, ir in zip(old_point_clouds, inliers_rotations)
        ]

    def calculate_outlier_weights_for_point(
        self,
        predicted_point_cloud: PointCloud,
        inliers_rotation: tuple[np.ndarray, float],
    ):
        new_weights = []
        for old_weight, inlier in zip(predicted_point_cloud.weights, inliers_rotation[0]):
            if inlier:
                new_weights.append(old_weight)
            else:
                new_weights.append(old_weight * WEIGHT_PENALTY)
        new_weights = np.array(new_weights)
        new_weight_sum = np.sum(new_weights)
        new_weights /= new_weight_sum

        assert np.sum(new_weights) - 1 < 0.001
        return new_weights