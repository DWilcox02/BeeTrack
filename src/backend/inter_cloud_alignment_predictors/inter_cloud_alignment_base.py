import numpy as np

from typing import List

from src.backend.point_cloud.point_cloud import PointCloud


class InterCloudAlignmentBase():
    
    def align_query_points(self, query_point_reconstructions: List[np.ndarray], inter_point_cloud_matrix: np.ndarray):
        return query_point_reconstructions