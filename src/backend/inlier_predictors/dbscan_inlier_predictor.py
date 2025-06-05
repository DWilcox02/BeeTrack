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
        best_final_predictions = None

        # print(f"Len final positions: {len(final_positions)}")

        # Determine rotation
        for r in range(0, 360, 10):
            final_predictions = old_point_cloud.query_point_predictions(final_positions=final_positions, rotation=r)
            rotation_deformity = old_point_cloud.deformity(points=final_predictions)
            if rotation_deformity < best_deformity:
                best_deformity = rotation_deformity
                best_rotation = r
                best_final_predictions = final_predictions

        inliers = None
        # self.log(f"Old inliers: {len([x for x in old_point_cloud.inliers if x])}")

        # self.log(f"Mapping: {mapping}")
        # print(f"Len best final predictions: {len(best_final_predictions)}")
        # print(f"Len old point cloud inliers: {len(old_point_cloud.inliers)}")
        best_final_predictions_masked = best_final_predictions[old_point_cloud.inliers]
        # self.log(f"Len final predictions masked: {len(best_final_predictions_masked)}")
        eps = old_point_cloud.radius * self.dbscan_epsilon

        min_samples = len(best_final_predictions_masked) // 2
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(best_final_predictions_masked)
        inlier_idxs = clustering.core_sample_indices_
        # self.log(f"Inlier idxs: {len(inlier_idxs)}")

        if len(inlier_idxs) > 0:
            inlier_mask = np.zeros_like(clustering.labels_, dtype=bool)
            inlier_mask[inlier_idxs] = True

            # self.log(f"Inlier mask: {len(inlier_mask)}")
            inliers = inlier_mask

        inliers = np.array(inliers, dtype=bool)
        num_inliers = np.sum(inliers)
        self.log(f"Best error and rotation: {best_deformity}, rotation: {best_rotation}, inliers: {num_inliers}/{len(inliers)}")
        return inliers, best_rotation, best_deformity
