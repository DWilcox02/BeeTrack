import numpy as np
from typing import List

from src.backend.models.circle_movement_model import CircleMovementModel
from src.backend.models.circle_movement_result import CircleMovementResult
from src.backend.point_cloud.point_cloud import PointCloud

class CircleMovementPredictor:
    def recalc_query_points_rotations(self, point_clouds: List[PointCloud], final_positions_lists):
        return [
            self.predict_circle_x_y_r(point_cloud=point_cloud, final_positions=final_positions) 
            for point_cloud, final_positions in zip(point_clouds, final_positions_lists)
        ]

    def predict_circle_x_y_r(self, point_cloud: PointCloud, final_positions) -> CircleMovementResult:
        # NEXT: 
        # Use clustering, rotations, and error minimization to determine the best cluster
        # and its rotation
        weighted_mean = np.array([0, 0], dtype=np.float32)
        for weight, point in zip(point_cloud.weights, final_positions):
            weighted_mean += weight * point
        return CircleMovementResult(
            x=weighted_mean[0],
            y=weighted_mean[1],
            r=0,
            inlier_idxs=[i for i in range(len(final_positions))],
            outlier_idxs=[]
        )
        
    def get_new_point(self, model: CircleMovementModel, query_point_start):
        delta_x = model.delta_translation[0].item()
        delta_y = model.delta_translation[1].item()
        rotation = model.rotation_angle.item()
        new_center = (query_point_start[0] + delta_x, query_point_start[1] + delta_y)
        print(f"New center position: {new_center}")
        print(f"Circle rotated by {rotation} (degrees or radians idk which)")
        return new_center, rotation
    
    