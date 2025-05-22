import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class QueryPointReconstructorBase():

    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)
    
    def reconstruct_query_points(self, old_point_clouds: List[PointCloud], final_positions: np.ndarray, inliers_rotations: List[tuple[np.ndarray, float]]):
        return [
            self.reconstruct_query_point(pc, fps, irs)
            for pc, fps, irs in zip(old_point_clouds, final_positions, inliers_rotations)
        ]
    
    def reconstruct_query_point(self, point_cloud: PointCloud, final_positions: np.ndarray, inliers_rotation: tuple[np.ndarray, float]):
        return np.mean(final_positions, axis=0)
    
    def rotate_vector(self, vector, angle_degrees):
        """Rotate a 2D vector by the given angle in degrees"""
        angle_rad = np.radians(angle_degrees)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Create rotation matrix
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        return np.dot(rotation_matrix, vector)