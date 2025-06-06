import numpy as np
from src.backend.point_cloud.point_cloud import PointCloud
from src.backend.utils.reconstruction_helper import reconstruct_with_center_rotation

class PointCloudReconstructorBase():

    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    def reconstruct_point_cloud(
        self,
        old_point_cloud: PointCloud,
        final_positions: np.ndarray[np.float32],
        inliers: np.ndarray[bool],
        rotation: float,
        query_point_reconstruction: np.ndarray,
        weights: np.ndarray,
    ) -> PointCloud:
        return reconstruct_with_center_rotation(
            old_point_cloud=old_point_cloud,
            rotation=rotation,
            query_point_reconstruction=query_point_reconstruction
        )