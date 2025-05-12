import numpy as np
from .point_cloud_generator import PointCloudGenerator
from ..models.circle_movement_predictor import CircleMovementPredictor
from .point_cloud import PointCloud
from typing import List

class CircularPointCloudGenerator(PointCloudGenerator):
    def __init__(self):
        super().__init__()
        self.circle_movement_predictor = CircleMovementPredictor()

    def format_new_query_point(self, query_point_array, previous_query_point):
        return {
            "x": float(query_point_array[0]),
            "y": float(query_point_array[1]),
            "color": previous_query_point["color"],
            "radius": previous_query_point["radius"],
        }
    

    def reconstruct_all_clouds_from_vectors(self, query_points, rotations, point_clouds: List[PointCloud]):
        # TODO: Redraw all OUTLIERS and use weights to lerp between reconstructed point and final position
        # Therefore good weights mean the point is doing well so we keep it,
        # bad weights mean the point is not doing well so we reconstruct
        return [
            self.reconstruct_with_center_rotation(qp, r, point_cloud) for qp, r, point_cloud in zip(query_points, rotations, point_clouds)
        ]


    def reconstruct_with_center_rotation(self, query_point, rotation, point_cloud):
        query_point_array = np.array([query_point["x"], query_point["y"]], dtype=np.float32)
        # Create rotation matrix
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)

        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Rotate the offset vectors directly
        rotated_vectors = np.matmul(point_cloud.vectors_qp_to_cp, rotation_matrix.T)

        # Calculate final positions
        reconstructed_points = query_point_array + rotated_vectors

        return PointCloud(
            query_point=self.format_new_query_point(
                query_point_array=query_point_array, 
                previous_query_point=query_point),
            cloud_points=reconstructed_points,
            radius=query_point["radius"],
            rotation=rotation,
            weights=point_cloud.weights,
            vectors_qp_to_cp=point_cloud.vectors_qp_to_cp,
        )


    def generate_initial_point_clouds(self, query_points):
        """Generate circular point clouds around each query point"""

        if not query_points or not isinstance(query_points, list):
            self.log("process_error: Invalid query points data")
            return []

        # Number of points to generate around each circle
        n_points_per_circle = 12

        # For each query point, generate a circle of points around it
        return [self._generate_point_cloud(point, n_points_per_circle) for point in query_points]


    def _generate_point_cloud(self, center_point, n_points_per_perimeter):
        """Helper method to generate points that fill a circle around a center point"""
        center_x = float(center_point["x"])
        center_y = float(center_point["y"])
        radius = float(center_point["radius"])
        
        # Create a filled circle of points
        circle_points = []
        
        # Determine how dense to make the grid based on perimeter points
        # We'll calculate the grid size to achieve approximately the same density
        grid_step = (2 * radius) / np.sqrt(n_points_per_perimeter * 3)
        
        # Create a grid of points within a square around the circle
        x_min, x_max = center_x - radius, center_x + radius
        y_min, y_max = center_y - radius, center_y + radius
        
        # Generate grid points
        x_coords = np.arange(x_min, x_max, grid_step)
        y_coords = np.arange(y_min, y_max, grid_step)
        
        # Add center point first
        circle_points.append((center_x, center_y))
        
        # Add all points within the radius
        for x in x_coords:
            for y in y_coords:
                # Calculate distance from center
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Only include points within the radius
                if distance <= radius:
                    circle_points.append((x, y))
        
        cloud_points = np.array(circle_points, dtype=np.float32)
        weights = np.array([1 / len(cloud_points)] * len(cloud_points), dtype=np.float32)
        return PointCloud(
            query_point=center_point, 
            cloud_points=cloud_points, 
            radius=center_point["radius"],
            rotation=0.0, 
            weights=weights
        )