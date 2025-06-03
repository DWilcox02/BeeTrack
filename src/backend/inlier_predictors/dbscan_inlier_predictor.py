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
        best_error = np.inf
        best_inliers = None
        best_rotation = None

        # self.log(f"Old inliers: {len([x for x in old_point_cloud.inliers if x])}")

        for r in range(0, 360, 10):
            final_predictions = old_point_cloud.query_point_predictions(
                final_positions=final_positions, 
                rotation=r
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


                inlier_predictions = final_predictions[combined_mask]
                mean = np.mean(inlier_predictions, axis=0)
                error = old_point_cloud.deformity(mean, inlier_predictions)
                
                if error < best_error:
                    best_error = error
                    best_inliers = combined_mask
                    best_rotation = r
        best_inliers = np.array(best_inliers, dtype=bool)
        # TODO: Check for when inliers and rotation is None
        num_inliers = np.sum(best_inliers)
        self.log(f"Best error and rotation: {best_error}, rotation: {best_rotation}, inliers: {num_inliers}/{len(best_inliers)}")
        return best_inliers, best_rotation
