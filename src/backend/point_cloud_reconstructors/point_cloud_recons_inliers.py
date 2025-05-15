import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud

from .point_cloud_reconstructor_base import PointCloudReconstructorBase


class PointCloudReconsInliers(PointCloudReconstructorBase):

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
            for opc, fps, irs, qp, ws in zip(old_point_clouds, final_positions, inliers_rotations, query_point_reconstructions, weights)
        ]
    
    def reconstruct_inliers(
        self, 
        old_point_cloud: PointCloud,
        final_positions: np.ndarray,
        inliers_rotation: tuple[np.ndarray, float],
        query_point: np.ndarray,
        weights: np.ndarray
    ) -> PointCloud:
        # mask = np.zeros_like(len(final_positions), dtype=bool)
        # mask[inliers] = True
        
        formatted_point = old_point_cloud.format_new_query_point(query_point)
        cloud_points = final_positions
        radius = old_point_cloud.radius
        inliers, rotation = inliers_rotation

        return PointCloud(
            query_point=formatted_point,
            cloud_points=cloud_points,
            radius=radius,
            rotation=rotation,
            weights=weights,
            inliers=inliers
        )