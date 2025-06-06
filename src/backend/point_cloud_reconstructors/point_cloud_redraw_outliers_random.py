import random
import numpy as np

from src.backend.point_cloud.point_cloud import PointCloud
from .point_cloud_reconstructor_base import PointCloudReconstructorBase



class PointCloudRedrawOutliersRandom(PointCloudReconstructorBase):
    def reconstruct_point_cloud(
        self,
        old_point_cloud: PointCloud,
        final_positions: np.ndarray[np.float32],
        inliers: np.ndarray[bool],
        rotation: float,
        query_point_reconstruction: np.ndarray,
        weights: np.ndarray,
    ) -> PointCloud:
        formatted_point = old_point_cloud.format_new_query_point(query_point_reconstruction)
        radius = old_point_cloud.radius
        orig_vectors = old_point_cloud.orig_vectors

        random.seed(42)

        cloud_points = []
        for i in range(len(orig_vectors)):
            if inliers[i]:
                cloud_points.append(final_positions[i])
            else:
                redrawn_outlier = random.choice(orig_vectors) + query_point_reconstruction
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