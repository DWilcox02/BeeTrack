import numpy as np
from typing import Optional
from src.backend.point_cloud.point_cloud import PointCloud



def reconstruct_with_center_rotation(
    old_point_cloud: PointCloud,
    rotation: float,
    query_point_reconstruction: np.ndarray,
    weights: Optional[np.ndarray[float]] = None
) -> PointCloud:

    # Rotate the offset vectors directly
    angle_rad = np.radians(rotation)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_vectors = np.matmul(old_point_cloud.vectors_qp_to_cp, rotation_matrix.T)
    reconstructed_points = rotated_vectors + query_point_reconstruction

    # Formatting
    formatted_new_query_point = old_point_cloud.format_new_query_point(query_point_reconstruction)
    if weights is None:
        weights = old_point_cloud.weights

    return PointCloud(
        query_point=formatted_new_query_point,
        cloud_points=reconstructed_points,
        radius=old_point_cloud.radius,
        rotation=rotation,
        weights=weights,
        vectors_qp_to_cp=old_point_cloud.vectors_qp_to_cp,
        orig_vectors=old_point_cloud.orig_vectors,
        log_fn=old_point_cloud.log_fn,
    )