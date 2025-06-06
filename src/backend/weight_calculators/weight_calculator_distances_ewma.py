import numpy as np
from src.backend.point_cloud.point_cloud import PointCloud
from .weight_calculator_distances_base import WeightCalculatorDistancesBase


ERROR_SIGMA = 0.5
OUTLIER_PENALTY = 0.5

class WeightCalculatorDistancesEWMA(WeightCalculatorDistancesBase):

    def calculate_distance_weights(
        self,
        predicted_point_cloud: PointCloud,
        true_query_point: np.ndarray,
    ) -> np.ndarray:
        qp_predictions = predicted_point_cloud.query_point_predictions()
        distances = self.distances_to_predictions(
            true_query_point=true_query_point, predicted_query_points=qp_predictions
        )
        distace_threshold = predicted_point_cloud.radius

        # Update according to exponentially-weighted moving average
        accuracy_weights = np.exp(-distances / distace_threshold)
        new_weights = predicted_point_cloud.weights * (1 - ERROR_SIGMA) + accuracy_weights * ERROR_SIGMA

        # Re-normalize weights
        weight_sum = np.sum(new_weights)
        new_weights /= weight_sum

        # Threshold in case of floating point errors
        assert np.sum(new_weights) - 1 < 0.001
        return new_weights
