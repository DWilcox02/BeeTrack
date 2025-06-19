import numpy as np

from src.backend.point_cloud.point_cloud import PointCloud
from .weight_calculator_outliers_base import WeightCalculatorOutliersBase

WEIGHT_PENALTY = 0.5

class WeightCalculatorOutliersPenalty(WeightCalculatorOutliersBase):

    def calculate_outlier_weights(
        self,
        old_point_cloud: PointCloud,
        inliers: np.ndarray[bool],
    ):
        new_weights = []
        for old_weight, inlier in zip(old_point_cloud.weights, inliers):
            # Penalize outliers
            if inlier:
                new_weights.append(old_weight)
            else:
                new_weights.append(old_weight * WEIGHT_PENALTY)

        # Re-normalize weights
        new_weights = np.array(new_weights)
        new_weight_sum = np.sum(new_weights)
        new_weights /= new_weight_sum

        # Threshold in case of floating point errors
        assert np.sum(new_weights) - 1 < 0.001
        return new_weights