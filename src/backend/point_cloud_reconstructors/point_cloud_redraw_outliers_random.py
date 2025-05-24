import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud

from .point_cloud_reconstructor_base import PointCloudReconstructorBase

import random


class PointCloudRedrawOutliersRandom(PointCloudReconstructorBase):
    def reconstruct_point_clouds(
        self,
        old_point_clouds: List[PointCloud],
        final_positions: np.ndarray,
        inliers_rotations: List[tuple[np.ndarray, float]],
        query_point_reconstructions: List[np.ndarray],
        weights: List[np.ndarray],
    ) -> List[PointCloud]:
        return [
            self.reconstruct_inliers(opc, fps, irs, qp, ws)
            for opc, fps, irs, qp, ws in zip(
                old_point_clouds, final_positions, inliers_rotations, query_point_reconstructions, weights
            )
        ]

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
