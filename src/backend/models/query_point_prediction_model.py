import numpy as np

from src.backend.point_cloud.point_cloud import PointCloud


class QueryPointPredictionModel():
    

    def predict(
            self,
            old_point_cloud: PointCloud,
            final_positions: np.ndarray,
            inliers_rotation: tuple[np.ndarray, float],
        ):
        pass

    def incremental_fit(
        self, 
        predicted_point_cloud: PointCloud,
        inliers_rotation: tuple[np.ndarray, float],
        true_query_point: np.ndarray
    ):
        pass