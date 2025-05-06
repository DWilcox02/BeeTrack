import numpy as np
from ..backend.point_cloud.point_cloud_generator import PointCloudGenerator
from ..backend.point_cloud.bee_skeleton import BeeSkeleton

class RhombusPointCloudGenerator(PointCloudGenerator):
    # Points ends up being of format:
    # [
    #     {"x": 292.514404296875, "y": 425.1623229980469, "color": "red"},
    #     {"x": 331.7908020019531, "y": 306.4179992675781, "color": "green"},
    #     {"x": 326.9676208496094, "y": 368.82452392578125, "color": "blue"},
    #     {"x": 282.9793701171875, "y": 352.75115966796875, "color": "purple"},
    # ]

    def __init__(self, init_points, point_data_store, session_id):
        super().__init__(init_points, point_data_store, session_id)
        self.bee_skeleton = BeeSkeleton(init_points)

    # Methods to generate cloud
    def generate_cloud_points(self, query_frame, height_ratio, width_ratio):
        assert(query_frame is not None)
        assert(height_ratio is not None)
        assert(width_ratio is not None)
        return self.interpolate_points(query_frame, height_ratio, width_ratio)

    def interpolate_points(self, query_frame, height_ratio, width_ratio):
        points = self.get_query_points()

        if not points or not isinstance(points, list) or len(points) < 4:
            self.log("process_error: Invalid rhombus points data")
            return
        
        points_array = np.array([(point["x"], point["y"]) for point in points], dtype=np.float32)

        # For quadrilateral interpolation, we need the 4 corners in a specific order
        # Assuming points form a quadrilateral with 4 points
        if len(points_array) != 4:
            raise ValueError("Expected exactly 4 points for area interpolation")

        # Define the corners - this assumes the points are a quadrilateral
        # We need a consistent ordering for the bilinear interpolation
        # Sort points by their position (e.g., top-left, top-right, bottom-right, bottom-left)
        # This is a simple approach - for complex shapes, more sophisticated ordering might be needed
        center = np.mean(points_array, axis=0)
        angles = np.arctan2(points_array[:, 1] - center[1], points_array[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        quad_points = points_array[sorted_indices]

        # Number of subdivisions along each dimension
        n_subdivs = 5  # This will create approximately 25 points across the area

        # Generate a grid of points across the entire area
        area_points = []

        for i in range(n_subdivs):
            for j in range(n_subdivs):
                # Parameters for bilinear interpolation
                u = i / (n_subdivs - 1)
                v = j / (n_subdivs - 1)

                # Bilinear interpolation formula
                # P(u,v) = (1-u)(1-v)P00 + u(1-v)P10 + (1-u)vP01 + uvP11
                point = (
                    (1 - u) * (1 - v) * quad_points[0]
                    + u * (1 - v) * quad_points[1]
                    + u * v * quad_points[2]
                    + (1 - u) * v * quad_points[3]
                )

                area_points.append(point)

        # Convert interpolated points to numpy array
        area_points = np.array(area_points, dtype=np.float32)

        # Create the query points
        interpolated_points = np.zeros(shape=(len(area_points), 3), dtype=np.float32)
        interpolated_points[:, 0] = query_frame
        interpolated_points[:, 1] = area_points[:, 1] * height_ratio  # y
        interpolated_points[:, 2] = area_points[:, 0] * width_ratio  # x

        return interpolated_points
    

    # Methods to recalculate query points
    def update_weights(
        self,
        point_cloud_slice,
        query_frame,
        height_ratio,
        width_ratio,
        previous_trajectory,
    ):
        self.bee_skeleton = BeeSkeleton(self.get_query_points())
        # Get new midpoint and trajectory
        midpoint = point_cloud_slice.get_final_mean()
        trajectory = point_cloud_slice.get_trajectory(previous_trajectory)
        trajectory = trajectory / np.linalg.norm(trajectory)

        self.log(f"Midpoint: {midpoint}, Trajectory: {trajectory}")

        # Use BeeSkeleton to calculate new positions based on midpoint and trajectory
        new_positions = self.bee_skeleton.calculate_new_positions(midpoint, trajectory)

        # Format points for conversion
        points = [
            {"x": new_positions["head"]["x"], "y": new_positions["head"]["y"], "color": "red"},
            {"x": new_positions["butt"]["x"], "y": new_positions["butt"]["y"], "color": "green"},
            {"x": new_positions["left"]["x"], "y": new_positions["left"]["y"], "color": "blue"},
            {"x": new_positions["right"]["x"], "y": new_positions["right"]["y"], "color": "purple"},
        ]
        session_points = [{"x": float(point["x"]), "y": float(point["y"]), "color": point["color"]} for point in points]
        self.set_query_points(session_points)

        # self.log(f"Recalculated points: {points}")

        # Convert to query points
        query_points = self.generate_cloud_points(
            query_frame=query_frame, height_ratio=height_ratio, width_ratio=width_ratio
        )

        return query_points, trajectory
    
    def initial_trajectory(self):
        return self.bee_skeleton.initial_trajectory