import numpy as np
from .point_cloud import PointCloud


class CircularPointCloud(PointCloud):
    # Points ends up being of format:
    # [
    #     {"x": np.float32(293.44388), "y": np.float32(397.61832), "color": "red"},
    #     {"x": np.float32(336.84387), "y": np.float32(280.31833), "color": "green"},
    #     {"x": np.float32(329.84387), "y": np.float32(342.51834), "color": "blue"},
    #     {"x": np.float32(286.44388), "y": np.float32(324.91833), "color": "purple"},
    # ]

    def __init__(self, init_points, point_data_store, session_id, radius=10):
        super().__init__(init_points, point_data_store, session_id)
        self.radius = radius  # Default radius for the circles

    def set_radius(self, radius):
        """Set the radius for circular point interpolation"""
        self.radius = radius

    def generate_cloud_points(self, query_frame=None, height_ratio=None, width_ratio=None):
        """Generate circular point clouds around each query point"""
        assert query_frame is not None
        assert height_ratio is not None
        assert width_ratio is not None

        points = self.get_query_points()

        if not points or not isinstance(points, list):
            self.log("process_error: Invalid query points data")
            return []

        # Number of points to generate around each circle
        n_points_per_circle = 12

        # Initialize an empty array to hold all interpolated points
        all_interpolated_points = []

        # For each query point, generate a circle of points around it
        for point in points:
            circle_points = self._generate_circle_points(point, n_points_per_circle)

            # Convert to the expected format with the query frame and ratios applied
            for cp in circle_points:
                interpolated_point = np.array(
                    [
                        query_frame,  # frame
                        cp[1] * height_ratio,  # y-coordinate
                        cp[0] * width_ratio,  # x-coordinate
                    ],
                    dtype=np.float32,
                )

                all_interpolated_points.append(interpolated_point)

        return np.array(all_interpolated_points, dtype=np.float32)

    def _generate_circle_points(self, center_point, n_points):
        """Helper method to generate points in a circle around a center point"""
        center_x = float(center_point["x"])
        center_y = float(center_point["y"])

        circle_points = []
        for i in range(n_points):
            # Calculate points around a circle
            angle = 2 * np.pi * i / n_points
            x = center_x + self.radius * np.cos(angle)
            y = center_y + self.radius * np.sin(angle)
            circle_points.append((x, y))

        # Add the center point as well
        circle_points.append((center_x, center_y))

        return circle_points

    def recalculate_query_points(self, point_cloud_slice, query_frame, height_ratio, width_ratio, previous_trajectory):
        """Recalculate query points based on the point cloud slice"""
        # Get new midpoint from the point cloud slice
        midpoint = point_cloud_slice.get_final_mean()

        # Get new trajectory
        trajectory = point_cloud_slice.get_trajectory(previous_trajectory)

        # Normalize trajectory if it's not a zero vector
        if np.linalg.norm(trajectory) > 0:
            trajectory = trajectory / np.linalg.norm(trajectory)

        self.log(f"Midpoint: {midpoint}, Trajectory: {trajectory}")

        # Get the current query points
        current_points = self.get_query_points()

        # Calculate the centroid of the current points
        points_array = np.array([(float(point["x"]), float(point["y"])) for point in current_points], dtype=np.float32)
        current_centroid = np.mean(points_array, axis=0)

        # Calculate the offset from the current centroid to the new midpoint
        offset = np.array([midpoint[0] - current_centroid[0], midpoint[1] - current_centroid[1]], dtype=np.float32)

        # Update each point with the offset
        new_points = []
        for point in current_points:
            new_point = {
                "x": float(point["x"]) + offset[0],
                "y": float(point["y"]) + offset[1],
                "color": point["color"],
            }
            new_points.append(new_point)

        # Update query points
        self.set_query_points(new_points)

        # Generate new cloud points
        query_points = self.generate_cloud_points(
            query_frame=query_frame, height_ratio=height_ratio, width_ratio=width_ratio
        )

        return query_points, trajectory

    def initial_trajectory(self):
        """Calculate initial trajectory based on query points"""
        points = self.get_query_points()

        if not points or len(points) < 2:
            # Default trajectory pointing right if insufficient points
            return np.array([1.0, 0.0], dtype=np.float32)

        # Calculate centroid
        points_array = np.array([(float(point["x"]), float(point["y"])) for point in points], dtype=np.float32)
        centroid = np.mean(points_array, axis=0)

        # Find the point furthest from the centroid to use for trajectory
        max_dist = -1
        furthest_point = None

        for point in points_array:
            dist = np.linalg.norm(point - centroid)
            if dist > max_dist:
                max_dist = dist
                furthest_point = point

        if furthest_point is not None:
            # Calculate initial trajectory as vector from centroid to furthest point
            trajectory = furthest_point - centroid
            # Normalize
            if np.linalg.norm(trajectory) > 0:
                trajectory = trajectory / np.linalg.norm(trajectory)
            return trajectory

        # Default trajectory if calculation fails
        return np.array([1.0, 0.0], dtype=np.float32)
