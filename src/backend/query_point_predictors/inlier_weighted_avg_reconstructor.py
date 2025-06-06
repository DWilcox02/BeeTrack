
import numpy as np

from src.backend.point_cloud.point_cloud import PointCloud
from .query_point_reconstructor_base import QueryPointReconstructorBase

class InlierWeightedAvgReconstructor(QueryPointReconstructorBase):
    
    def reconstruct_query_point(
        self, 
        point_cloud: PointCloud, 
        final_predictions: np.ndarray[np.float32], 
        inliers: np.ndarray[bool]
    ) -> np.ndarray:
        if len([x for x in inliers if x]) < 2:
            # If very few outliers, just take weighted average of all
            inliers = np.array([True] * len(inliers), dtype=bool)
            # Important to do this here since the true number of inliers (< 2) is influential for confidence calculation

        weights = point_cloud.weights

        inlier_predictions = final_predictions[inliers]
        inlier_weights = weights[inliers]

        weight_sum = np.sum(inlier_weights)

        normalized_weights = inlier_weights / weight_sum
        weighted_avg = np.array([0.0, 0.0])
        for pred, weight in zip(inlier_predictions, normalized_weights):
            weighted_avg += weight * pred

        return weighted_avg