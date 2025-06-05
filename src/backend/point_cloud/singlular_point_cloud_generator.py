import numpy as np
from .point_cloud_generator import PointCloudGenerator
from .point_cloud import PointCloud


class SingularPointCloudGenerator(PointCloudGenerator):
    def __init__(self):
        super().__init__()

    def generate_initial_point_clouds(self, query_points):
        """Generate circular point clouds around each query point"""

        if not query_points or not isinstance(query_points, list):
            self.log("process_error: Invalid query points data")
            return []

        # For each query point, generate a circle of points around it
        return [self._generate_point_cloud(point) for point in query_points]

    def _generate_point_cloud(self, center_point):
        """Helper method to generate points that fill a circle around a center point"""
        center_x = float(center_point["x"])
        center_y = float(center_point["y"])

        # Create a filled circle of points
        circle_points = []

        # Add center point first
        circle_points.append((center_x, center_y))

        cloud_points = np.array(circle_points, dtype=np.float32)
        weights = np.array([1 / len(cloud_points)] * len(cloud_points), dtype=np.float32)

        vectors_qp_to_cp = []
        center_point_x_y = np.array([center_x, center_y], dtype=np.float32)
        for point in cloud_points:
            vectors_qp_to_cp.append(point - center_point_x_y)
        vectors_qp_to_cp = np.array(vectors_qp_to_cp, dtype=np.float32)

        return PointCloud(
            query_point=center_point,
            cloud_points=cloud_points,
            radius=center_point["radius"],
            rotation=0.0,
            weights=weights,
            log_fn=self.log,
        )
