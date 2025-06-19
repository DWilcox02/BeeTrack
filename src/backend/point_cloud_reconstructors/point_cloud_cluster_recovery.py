import random
import numpy as np
from typing import List

from src.backend.point_cloud.point_cloud import PointCloud
from .point_cloud_reconstructor_base import PointCloudReconstructorBase
from src.backend.utils.reconstruction_helper import reconstruct_with_center_rotation


class PointCloudClusterRecovery(PointCloudReconstructorBase):
    def reconstruct_point_cloud(
        self,
        old_point_cloud: PointCloud,
        final_positions: np.ndarray[np.float32],
        inliers: np.ndarray[bool],
        rotation: float,
        query_point_reconstruction: np.ndarray,
        weights: np.ndarray,
    ) -> PointCloud:
        predicted_query_point = old_point_cloud.query_point_array()
        validation_distance = np.linalg.norm(predicted_query_point - query_point_reconstruction)
        if validation_distance > old_point_cloud.radius:
            weights = np.array([1 / len(old_point_cloud.cloud_points)] * len(old_point_cloud.cloud_points), dtype=np.float32)
            return reconstruct_with_center_rotation(
                old_point_cloud=old_point_cloud,
                rotation=rotation,
                query_point_reconstruction=query_point_reconstruction,
                weights=weights
            )
        else:
            return self.reconstruct_outliers(
                old_point_cloud=old_point_cloud,
                final_positions=final_positions,
                inliers=inliers,
                rotation=rotation,
                query_point=query_point_reconstruction,
                weights=weights
            )

    def reconstruct_outliers(
        self,
        old_point_cloud: PointCloud,
        final_positions: np.ndarray[np.float32],
        inliers: np.ndarray[bool],
        rotation: float,
        query_point: np.ndarray[np.float32],
        weights: np.ndarray[np.float32],
    ) -> PointCloud:
        formatted_point = old_point_cloud.format_new_query_point(query_point)
        radius = old_point_cloud.radius
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
