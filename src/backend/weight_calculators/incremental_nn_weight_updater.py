import numpy as np

from typing import List

from src.backend.models.query_point_prediction_model import QueryPointPredictionModel
from src.backend.point_cloud.point_cloud import PointCloud
from .weight_calculator_base import WeightCalculatorBase




class IncrementalNNWeightUpdater(WeightCalculatorBase):

    def __init__(self, prediction_models: List[QueryPointPredictionModel]):
        self.prediction_models = prediction_models


    def calculate_distance_weights(
        self, 
        predicted_point_clouds: List[PointCloud], 
        inliers_rotations: List[tuple[np.ndarray, float]], 
        true_query_points: List[np.ndarray]
        ):
        for prediction_model, ppc, ir, tqp in zip(self.prediction_models, predicted_point_clouds, inliers_rotations, true_query_points):
            prediction_model.incremental_fit(ppc, ir, tqp)

        return [ppc.weights for ppc in predicted_point_clouds]