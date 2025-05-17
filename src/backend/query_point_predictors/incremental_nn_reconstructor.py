import numpy as np
from typing import List

from src.backend.point_cloud.point_cloud import PointCloud
from .query_point_reconstructor_base import QueryPointReconstructorBase
from src.backend.models.query_point_prediction_model import QueryPointPredictionModel



class IncrementalNNReconstructor(QueryPointReconstructorBase):

    def __init__(self, num_point_clouds: int):
        self.prediction_models = []
        for i in range(num_point_clouds):
            self.prediction_models.append(QueryPointPredictionModel())

    def get_prediction_models(self) -> List[QueryPointPredictionModel]:
        return self.prediction_models

    def reconstruct_query_points(
        self,
        old_point_clouds: List[PointCloud],
        final_positions: np.ndarray,
        inliers_rotations: List[tuple[np.ndarray, float]],
    ):
        return [
            self.predict_query_point_nn(opc, fps, ir, qppm) 
            for opc, fps, ir, qppm in zip(old_point_clouds, final_positions, inliers_rotations, self.prediction_models)
        ]
    
    def predict_query_point_nn(
            self,
            old_point_cloud: PointCloud,
            final_positions: np.ndarray,
            inliers_rotation: tuple[np.ndarray, float],
            prediction_model: QueryPointPredictionModel
    ):
        inliers, rotation = inliers_rotation
        return prediction_model.predict(old_point_cloud, final_positions, inliers, rotation)