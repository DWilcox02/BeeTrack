import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class QueryPointReconstructorBase():
    
    def reconstruct_query_points(self, old_point_clouds: List[PointCloud], final_positions: np.ndarray, inliers_rotations: List[tuple[List[int], float]]):
        return [
            self.reconstruct_query_point(pc, fps, irs)
            for pc, fps, irs in zip(old_point_clouds, final_positions, inliers_rotations)
        ]
    
    def reconstruct_query_point(self, point_cloud: PointCloud, final_positions: np.ndarray, inliers_rotations: tuple[List[int], float]):
        return np.mean(final_positions, axis=0)