import numpy as np
from abc import abstractmethod
from src.backend.point_cloud.point_cloud import PointCloud


class WeightCalculatorDistancesBase:
    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    @abstractmethod
    def calculate_distance_weights(
        self,
        predicted_point_cloud: PointCloud,
        true_query_point: np.ndarray,
    ) -> np.ndarray:
        pass

    def distances_to_predictions(self, true_query_point: np.ndarray, predicted_query_points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(predicted_query_points - true_query_point, axis=1)
