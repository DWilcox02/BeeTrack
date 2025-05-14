import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class PointCloudReconstructorBase():

    def reconstruct_all_clouds_from_vectors(
            self, 
            query_points: List[np.ndarray], 
            rotations: List[float], 
            point_clouds: List[PointCloud],
            weights: List[np.ndarray]
        ) -> List[PointCloud]:
        # TODO: Redraw all OUTLIERS and use weights to lerp between reconstructed point and final position
        # Therefore good weights mean the point is doing well so we keep it,
        # bad weights mean the point is not doing well so we reconstruct
        return [
            self.reconstruct_with_center_rotation(qp, r, pc, w)
            for qp, r, pc, w in zip(query_points, rotations, point_clouds, weights)
        ]
    
    def reconstruct_with_center_rotation(
            self, 
            query_point: np.ndarray, 
            rotation: float, 
            point_cloud: PointCloud,
            weights: np.ndarray
        ) -> PointCloud:
        # Create rotation matrix
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)

        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Rotate the offset vectors directly
        rotated_vectors = np.matmul(point_cloud.vectors_qp_to_cp, rotation_matrix.T)

        # Calculate final positions
        reconstructed_points = query_point + rotated_vectors

        formatted_new_query_point = point_cloud.format_new_query_point(query_point)

        return PointCloud(
            query_point=formatted_new_query_point,
            cloud_points=reconstructed_points,
            radius=point_cloud.radius,
            rotation=rotation,
            weights=point_cloud.weights,
            vectors_qp_to_cp=point_cloud.vectors_qp_to_cp,
        )


    def reconstruct_point_clouds(
            self, 
            old_point_clouds: List[PointCloud], 
            inliers_rotations=List[tuple[List[int], float]], 
            query_point_reconstructions=List[np.ndarray],
            weights=List[np.ndarray]
        ) -> List[PointCloud]:
        rotations = [r for _, r in inliers_rotations]
        return self.reconstruct_all_clouds_from_vectors(
            query_points=query_point_reconstructions, 
            rotations=rotations,
            point_clouds=old_point_clouds,
            weights=weights
        )