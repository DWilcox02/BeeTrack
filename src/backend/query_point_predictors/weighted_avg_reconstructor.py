
import numpy as np

from src.backend.point_cloud.point_cloud import PointCloud
from .query_point_reconstructor_base import QueryPointReconstructorBase

class WeightedAvgReconstructor(QueryPointReconstructorBase):

    def reconstruct_query_point(
        self, point_cloud: PointCloud, final_positions: np.ndarray, inliers_rotation: tuple[np.ndarray, float]
    ):
        inliers, rotation = inliers_rotation
        weights = point_cloud.weights

        final_predictions = []
        for vec_qp_to_cp, pos in zip(point_cloud.vectors_qp_to_cp, final_positions):
            rotated_vec = self.rotate_vector(vec_qp_to_cp, rotation)
            final_predictions.append(pos - rotated_vec)
        final_predictions = np.array(final_predictions)
        

        weighted_avg = np.array([0.0, 0.0])
        for pred, weight in zip(final_predictions, weights):
            weighted_avg += weight * pred

        return weighted_avg