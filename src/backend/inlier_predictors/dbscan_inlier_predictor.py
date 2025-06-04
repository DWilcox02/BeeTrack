import numpy as np
from sklearn.cluster import DBSCAN
from typing import List
import copy

from src.backend.point_cloud.point_cloud import PointCloud

from .inlier_predictor_base import InlierPredictorBase

class DBSCANInlierPredictor(InlierPredictorBase):

    def __init__(self, dbscan_epsilon: float):
        self.dbscan_epsilon = dbscan_epsilon

    def predict_inliers_rotations(
        self, old_point_clouds: List[PointCloud], final_positions: np.ndarray
    ) -> List[tuple[np.ndarray, float]]:
        return [
            self.predict_for_point_cloud(opc, fps) for opc, fps in zip(old_point_clouds, final_positions)
        ]

    
    def predict_for_point_cloud(self, old_point_cloud: PointCloud, final_positions: np.ndarray):
        best_deformity = np.inf
        best_rotation = None

        # Determine rotation
        for r in range(0, 360, 10):
            final_predictions = old_point_cloud.query_point_predictions(final_positions=final_positions, rotation=r)
            rotation_deformity = old_point_cloud.deformity(points=final_predictions)
            if rotation_deformity < best_deformity:
                best_deformity = rotation_deformity
                best_rotation = r

        inliers = None
        # self.log(f"Old inliers: {len([x for x in old_point_cloud.inliers if x])}")

        final_predictions = old_point_cloud.query_point_predictions(
            final_positions=final_positions, 
            rotation=best_rotation
        )
        j = 0
        mapping = {}
        for i in range(len(final_predictions)):
            if old_point_cloud.inliers[i]:
                mapping[j] = i
                j += 1

        # self.log(f"Mapping: {mapping}")
        final_predictions_masked = final_predictions[old_point_cloud.inliers]
        # self.log(f"Len final predictions masked: {len(final_predictions_masked)}")
        eps = old_point_cloud.radius * self.dbscan_epsilon

        min_samples = len(final_predictions_masked) // 2
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(final_predictions_masked)
        inlier_idxs = clustering.core_sample_indices_
        # self.log(f"Inlier idxs: {len(inlier_idxs)}")

        if len(inlier_idxs) > 0:
            inlier_mask = np.zeros_like(clustering.labels_, dtype=bool)
            inlier_mask[inlier_idxs] = True

            # self.log(f"Inlier mask: {len(inlier_mask)}")
            combined_mask = copy.deepcopy(old_point_cloud.inliers)
            for j in range(len(inlier_mask)):
                i = mapping[j]
                combined_mask[i] = combined_mask[i] and inlier_mask[j]
            inliers = combined_mask

        inliers = np.array(inliers, dtype=bool)
        num_inliers = np.sum(inliers)
        self.log(f"Best error and rotation: {best_deformity}, rotation: {best_rotation}, inliers: {num_inliers}/{len(inliers)}")
        return inliers, best_rotation, best_deformity
