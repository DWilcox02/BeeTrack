import numpy as np
from sklearn.cluster import HDBSCAN
from typing import List

from src.backend.point_cloud.point_cloud import PointCloud

from .inlier_predictor_base import InlierPredictorBase


class HDBSCANInlierPredictor(InlierPredictorBase):

    def predict_inliers_rotations(
        self, old_point_clouds: List[PointCloud], final_positions: np.ndarray
    ) -> List[tuple[np.ndarray, float]]:
        return [self.predict_for_point_cloud(opc, fps) for opc, fps in zip(old_point_clouds, final_positions)]

    def predict_for_point_cloud(self, old_point_cloud: PointCloud, final_positions: np.ndarray):
        best_error = np.inf
        best_inliers = None
        best_rotation = None

        for r in range(0, 360, 10):
            final_predictions = []
            for vec_qp_to_cp, pos in zip(old_point_cloud.vectors_qp_to_cp, final_positions):
                rotated_vec = self.rotate_vector(vec_qp_to_cp, r)
                final_predictions.append(pos - rotated_vec)

            final_predictions = np.array(final_predictions, dtype=np.float32)
            # self.log(f"Old Inliers: {old_point_cloud.inliers}")
            j = 0
            mapping = {}
            for i in range(len(final_predictions)):
                if old_point_cloud.inliers[i]:
                    mapping[j] = i
                    j += 1

            # self.log(f"Mapping: {mapping}")
            # self.log(f"Old inliers: {len(old_point_cloud.inliers)}")
            final_predictions_masked = final_predictions[old_point_cloud.inliers]
            # self.log(f"Len final predictions masked: {len(final_predictions_masked)}")

            min_cluster_size = len(final_predictions_masked) // 2
            clustering = HDBSCAN().fit(final_predictions_masked)
            inlier_idxs = [i for i in range(len(clustering.labels_)) if clustering.labels_[i] >= 0]
            self.log(f"Cluster labels: {clustering.labels_}")

            if len(inlier_idxs) > 0:
                inlier_mask = np.zeros_like(clustering.labels_, dtype=bool)
                inlier_mask[inlier_idxs] = True

                # self.log(f"Inlier mask: {len(inlier_mask)}")
                combined_mask = old_point_cloud.inliers
                for j in range(len(inlier_mask)):
                    i = mapping[j]
                    combined_mask[i] = combined_mask[i] and inlier_mask[j]

                inlier_predictions = final_predictions[combined_mask]
                mean = np.mean(inlier_predictions, axis=0)
                error = np.sum(np.linalg.norm(final_predictions - mean, axis=1))
                if error < best_error:
                    best_error = error
                    best_inliers = combined_mask
                    best_rotation = r
        best_inliers = np.array(best_inliers, dtype=bool)
        # TODO: Check for when inliers and rotation is None
        self.log(f"Best error and rotation: {best_error}, rotation: {best_rotation}, inliers: {best_inliers}")
        return best_inliers, best_rotation

    def rotate_vector(self, vector, angle_degrees):
        """Rotate a 2D vector by the given angle in degrees"""
        angle_rad = np.radians(angle_degrees)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Create rotation matrix
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        return np.dot(rotation_matrix, vector)