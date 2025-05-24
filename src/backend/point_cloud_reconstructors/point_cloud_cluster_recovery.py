import random
import numpy as np
from typing import List

from src.backend.point_cloud.point_cloud import PointCloud
from .point_cloud_reconstructor_base import PointCloudReconstructorBase


class PointCloudClusterRecovery(PointCloudReconstructorBase):
    def reconstruct_point_clouds(
        self,
        old_point_clouds: List[PointCloud],
        final_positions: np.ndarray,
        inliers_rotations: List[tuple[np.ndarray, float]],
        query_point_reconstructions: List[np.ndarray],
        weights: List[np.ndarray],
    ) -> List[PointCloud]:
        return [
            self.recover_cluster(opc, fps, irs, qp, ws)
            for opc, fps, irs, qp, ws in zip(
                old_point_clouds, final_positions, inliers_rotations, query_point_reconstructions, weights
            )
        ]

    def recover_cluster(
        self,
        predicted_point_cloud: PointCloud,
        final_positions: np.ndarray,
        inliers_rotation: tuple[np.ndarray, float],
        true_query_point: np.ndarray,
        weights: np.ndarray,
    ) -> PointCloud:
        predicted_query_point = predicted_point_cloud.query_point_array()
        validation_distance = np.linalg.norm(predicted_query_point - true_query_point)
        if validation_distance > predicted_point_cloud.radius:
            _, rotation = inliers_rotation
            weights = np.array([1 / len(predicted_point_cloud.cloud_points)] * len(predicted_point_cloud.cloud_points), dtype=np.float32)
            return self.reconstruct_with_center_rotation(
                query_point=true_query_point,
                rotation=rotation,
                old_point_cloud=predicted_point_cloud,
                weights=weights
            )
        else:
            return self.reconstruct_inliers(
                old_point_cloud=predicted_point_cloud,
                final_positions=final_positions,
                inliers_rotation=inliers_rotation,
                query_point=true_query_point,
                weights=weights
            )

    def reconstruct_with_center_rotation(
        self, query_point: np.ndarray, rotation: float, old_point_cloud: PointCloud, weights: np.ndarray
    ) -> PointCloud:
        # Create rotation matrix
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)

        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Rotate the offset vectors directly
        rotated_vectors = np.matmul(old_point_cloud.vectors_qp_to_cp, rotation_matrix.T)

        # Calculate final positions
        reconstructed_points = query_point + rotated_vectors

        formatted_new_query_point = old_point_cloud.format_new_query_point(query_point)

        return PointCloud(
            query_point=formatted_new_query_point,
            cloud_points=reconstructed_points,
            radius=old_point_cloud.radius,
            rotation=rotation,
            weights=weights,
            vectors_qp_to_cp=old_point_cloud.vectors_qp_to_cp,
            orig_vectors=old_point_cloud.orig_vectors,
            log_fn=old_point_cloud.log_fn
        )

    def reconstruct_inliers(
        self,
        old_point_cloud: PointCloud,
        final_positions: np.ndarray,
        inliers_rotation: tuple[np.ndarray, float],
        query_point: np.ndarray,
        weights: np.ndarray,
    ) -> PointCloud:
        formatted_point = old_point_cloud.format_new_query_point(query_point)
        radius = old_point_cloud.radius
        inliers, rotation = inliers_rotation
        orig_vectors = old_point_cloud.orig_vectors

        random.seed(42)

        cloud_points = []
        for i in range(len(orig_vectors)):
            if inliers[i]:
                cloud_points.append(final_positions[i])
            else:
                redrawn_outlier = query_point + random.choice(orig_vectors)
                cloud_points.append(redrawn_outlier)

        return PointCloud(
            query_point=formatted_point,
            cloud_points=cloud_points,
            radius=radius,
            rotation=rotation,
            weights=weights,
            orig_vectors=old_point_cloud.orig_vectors,
            log_fn=old_point_cloud.log_fn,
        )
