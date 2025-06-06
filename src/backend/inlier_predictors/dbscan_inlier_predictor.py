import numpy as np
from sklearn.cluster import DBSCAN
from typing import List
import copy

from src.backend.point_cloud.point_cloud import PointCloud

from .inlier_predictor_base import InlierPredictorBase

class DBSCANInlierPredictor(InlierPredictorBase):

    def __init__(self, dbscan_epsilon: float):
        self.dbscan_epsilon = dbscan_epsilon

    def predict_inliers(
        self,
        old_point_cloud: PointCloud,
        final_predictions: np.ndarray,
    ):
        inliers = None

        best_final_predictions_masked = final_predictions[old_point_cloud.inliers]
        eps = old_point_cloud.radius * self.dbscan_epsilon

        min_samples = len(best_final_predictions_masked) // 2
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(best_final_predictions_masked)
        inlier_idxs = clustering.core_sample_indices_
        # self.log(f"Inlier idxs: {len(inlier_idxs)}")

        if len(inlier_idxs) > 0:
            inlier_mask = np.zeros_like(clustering.labels_, dtype=bool)
            inlier_mask[inlier_idxs] = True
            inliers = inlier_mask

        inliers = np.array(inliers, dtype=bool)
        return inliers
