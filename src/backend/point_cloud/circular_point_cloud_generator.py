import numpy as np
from .point_cloud_generator import PointCloudGenerator


class CircularPointCloudGenerator(PointCloudGenerator):
    def __init__(self, init_points, point_data_store, session_id):
        super().__init__(init_points, point_data_store, session_id)

    def generate_cloud_points(self):
        """Generate circular point clouds around each query point"""

        points = self.get_query_points()

        if not points or not isinstance(points, list):
            self.log("process_error: Invalid query points data")
            return []

        # Number of points to generate around each circle
        n_points_per_circle = 12

        # Initialize an empty array to hold all interpolated points
        all_interpolated_points = []
        print(points)
        # For each query point, generate a circle of points around it
        for point in points:
            circle_points = self._generate_circle_points(point, n_points_per_circle)

            # Convert to the expected format with the query frame and ratios applied
            for cp in circle_points:
                interpolated_point = np.array([
                        cp[0], # x-coordinate
                        cp[1]  # y-coordinate
                    ],
                    dtype=np.float32,
                )

                all_interpolated_points.append(interpolated_point)

        return np.array(all_interpolated_points, dtype=np.float32)

    def _generate_circle_points(self, center_point, n_points_per_perimeter):
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
        
        return circle_points

    def update_weights(self, initial_positions, final_positions):
        # Both arrays N x 2 of (x, y)
        # print("Initial Points:")
        # print(initial_positions)
        # print("Final points:")
        # print(final_positions)
        # print("Some calculations to update the weights based on the initial and final points and the existing weights....")
        pass

    def recalc_query_points(self, final_positions):
        current_query_points = self.get_query_points()
        slice_i = 0

        new_query_points = []
        for qp in range(len(current_query_points)):
            new_point = np.array([0, 0], dtype=np.float32)
            for cp in range(self.cp_per_qp):
                weight = self.weights[slice_i]
                point = final_positions[slice_i]
                new_point += weight * point
                slice_i += 1
            new_query_points.append(new_point)
        
        # Convert new_query_points into the desired format
        formatted_query_points = []
        for i, point in enumerate(new_query_points):
            formatted_query_points.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "color": current_query_points[i]["color"],
            "radius": current_query_points[i]["radius"]
            })
        
        new_query_points = formatted_query_points
        self.set_query_points(new_query_points)





        # midpoint = point_cloud_slice.get_final_mean()

        # # Get new trajectory
        # trajectory = point_cloud_slice.get_trajectory(previous_trajectory)

        # # Normalize trajectory if it's not a zero vector
        # if np.linalg.norm(trajectory) > 0:
        #     trajectory = trajectory / np.linalg.norm(trajectory)

        # self.log(f"Midpoint: {midpoint}, Trajectory: {trajectory}")

        # # Get the current query points
        # current_points = self.get_query_points()

        # # Calculate the centroid of the current points
        # points_array = np.array([(float(point["x"]), float(point["y"])) for point in current_points], dtype=np.float32)
        # current_centroid = np.mean(points_array, axis=0)

        # # Calculate the offset from the current centroid to the new midpoint
        # offset = np.array([midpoint[0] - current_centroid[0], midpoint[1] - current_centroid[1]], dtype=np.float32)

        # # Update each point with the offset
        # new_points = []
        # for point in current_points:
        #     new_point = {
        #         "x": float(point["x"]) + offset[0],
        #         "y": float(point["y"]) + offset[1],
        #         "color": point["color"],
        #         "radius": point["radius"]
        #     }
        #     new_points.append(new_point)

        # # Update query points
        # self.set_query_points(new_points)

        # # Generate new cloud points
        # cloud_points = self.generate_cloud_points(
        #     query_frame=query_frame, height_ratio=height_ratio, width_ratio=width_ratio
        # )

        # return cloud_points, trajectory


    def calculate_confidence(self):
        return 0.0


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
