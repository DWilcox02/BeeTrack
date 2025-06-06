import numpy as np
from src.backend.point_cloud.point_cloud import PointCloud

def calculate_rotation_deformity_predictions(
        old_point_cloud: PointCloud,
        final_positions: np.ndarray
) -> tuple[float, float, np.ndarray]:
    best_deformity = np.inf
    best_rotation = None
    best_final_predictions = None

    # Determine rotation
    for r in range(0, 360, 10):
        final_predictions = old_point_cloud.query_point_predictions(final_positions=final_positions, rotation=r)
        rotation_deformity = old_point_cloud.deformity(points=final_predictions)
        if rotation_deformity < best_deformity:
            best_deformity = rotation_deformity
            best_rotation = r
            best_final_predictions = final_predictions

    return (best_rotation, best_deformity, best_final_predictions)