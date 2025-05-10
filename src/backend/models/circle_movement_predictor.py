import numpy as np
from sklearn.cluster import DBSCAN
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
        print(f"Predicting final location of initial point {point_cloud.query_point}")
        final_predictions = np.array([
            pos - vec_qp_to_cp
            for vec_qp_to_cp, pos in zip(point_cloud.vectors_qp_to_cp, final_positions)
        ], dtype=np.float32)

        eps = point_cloud.radius
        print(f"Using eps: {eps}")
        min_samples = len(final_positions) // 2
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(final_predictions)
        # for c, fp in zip(clustering.labels_, final_predictions):
        #     print(f"Cluster label {c} for {fp}")
        inlier_idxs = clustering.core_sample_indices_
        print(f"Inlier indices: {inlier_idxs}")
        inlier_mask = np.zeros_like(clustering.labels_, dtype=bool)
        inlier_mask[inlier_idxs] = True
        outlier_idxs = [i for i in range(len(final_positions)) if i not in inlier_idxs]
        print(f"Outlier idxs: {outlier_idxs}")

        inlier_weights = point_cloud.weights[inlier_mask]
        total_inlier_weight = np.sum(inlier_weights) # <= 1
        print(total_inlier_weight)
        inlier_final_predictions = final_predictions[inlier_mask]
        weighted_mean = np.array([0, 0], dtype=np.float32)
        for weight, point in zip(inlier_weights, inlier_final_predictions):
            temp_weight = weight / total_inlier_weight # normalized to inliers only
            weighted_mean += temp_weight * point
        print(f"Cluster finished with weighted mean {weighted_mean}")
        return CircleMovementResult(
            x=weighted_mean[0],
            y=weighted_mean[1],
            r=0,
            inlier_idxs=inlier_idxs,
            outlier_idxs=outlier_idxs,
            final_predictions=final_predictions
        )
        
    def get_new_point(self, model: CircleMovementModel, query_point_start):
        delta_x = model.delta_translation[0].item()
        delta_y = model.delta_translation[1].item()
        rotation = model.rotation_angle.item()
        new_center = (query_point_start[0] + delta_x, query_point_start[1] + delta_y)
        print(f"New center position: {new_center}")
        print(f"Circle rotated by {rotation} (degrees or radians idk which)")
        return new_center, rotation
    
    