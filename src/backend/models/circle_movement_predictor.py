import numpy as np
from sklearn.cluster import DBSCAN
from typing import List

from src.backend.models.circle_movement_model import CircleMovementModel
from src.backend.models.circle_movement_result import CircleMovementResult
from src.backend.point_cloud.point_cloud import PointCloud


class CircleMovementPredictor:
    def recalc_query_points_rotations(self, point_clouds: List[PointCloud], final_positions_lists):
        return [
            self.predict_circle_x_y_r(point_cloud=point_cloud, final_positions=final_positions)
            for point_cloud, final_positions in zip(point_clouds, final_positions_lists)
        ]

    def predict_circle_x_y_r(self, point_cloud: PointCloud, final_positions) -> CircleMovementResult:
        # NEXT:
        # Use clustering, rotations, and error minimization to determine the best cluster
        # and its rotation
        best_error = np.inf
        best_result = None

        for r in range(0, 360, 10):
            final_predictions = []
            for vec_qp_to_cp, pos in zip(point_cloud.vectors_qp_to_cp, final_positions):
                # Rotate the vec_qp_to_cp by the current rotation r
                rotated_vec = self.rotate_vector(vec_qp_to_cp, r)
                final_predictions.append(pos - rotated_vec)

            final_predictions = np.array(final_predictions, dtype=np.float32)

            eps = point_cloud.radius

            min_samples = len(final_positions) // 2
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(final_predictions)
            # for c, fp in zip(clustering.labels_, final_predictions):
            #     print(f"Cluster label {c} for {fp}")
            inlier_idxs = clustering.core_sample_indices_

            inlier_mask = np.zeros_like(clustering.labels_, dtype=bool)
            inlier_mask[inlier_idxs] = True
            outlier_idxs = [i for i in range(len(final_positions)) if i not in inlier_idxs]

            inlier_weights = point_cloud.weights[inlier_mask]
            total_inlier_weight = np.sum(inlier_weights)  # <= 1

            inlier_final_predictions = final_predictions[inlier_mask]
            weighted_mean = np.array([0, 0], dtype=np.float32)
            for weight, point in zip(inlier_weights, inlier_final_predictions):
                temp_weight = weight / total_inlier_weight  # normalized to inliers only
                weighted_mean += temp_weight * point

            error = np.sum(np.linalg.norm(final_predictions - weighted_mean, axis=1))
            if error < best_error:
                best_error = error
                best_result = CircleMovementResult(
                    x=weighted_mean[0],
                    y=weighted_mean[1],
                    r=r,
                    inlier_idxs=inlier_idxs,
                    outlier_idxs=outlier_idxs,
                    final_predictions=final_predictions,
                )
        print(f"Best error and rotation: {best_error}, rotation: {best_result.r}")
        return best_result

    def get_new_point(self, model: CircleMovementModel, query_point_start):
        delta_x = model.delta_translation[0].item()
        delta_y = model.delta_translation[1].item()
        rotation = model.rotation_angle.item()
        new_center = (query_point_start[0] + delta_x, query_point_start[1] + delta_y)
        print(f"New center position: {new_center}")
        print(f"Circle rotated by {rotation} (degrees or radians idk which)")
        return new_center, rotation
