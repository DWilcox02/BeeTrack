import numpy as np
from .point_cloud_generator import PointCloudGenerator
from .circle_movement_predictor import CircleMovementPredictor

class CircularPointCloudGenerator(PointCloudGenerator):
    def __init__(self, init_points, point_data_store, session_id):
        super().__init__(init_points, point_data_store, session_id)
        self.circle_movement_predictor = CircleMovementPredictor()

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

    # Which points behave correctly?
    def update_weights(self, initial_positions, final_positions):
        pass
        

    def recalc_query_points(self, initial_positions, final_positions):
        # current_query_points = self.get_query_points()

        # new_query_points = []
        # for cloud_weights, cloud_points in zip(self.weights, final_positions):
        #     new_point = np.array([0, 0], dtype=np.float32)
        #     for weight, point in zip(cloud_weights, cloud_points):
        #         new_point += weight * point
        #     new_query_points.append(new_point)
        
        # formatted_query_points = []
        # for i, point in enumerate(new_query_points):
        #     formatted_query_points.append({
        #     "x": float(point[0]),
        #     "y": float(point[1]),
        #     "color": current_query_points[i]["color"],
        #     "radius": current_query_points[i]["radius"]
        #     })
        
        # new_query_points = formatted_query_points
        # self.set_query_points(new_query_points)
        x_y_query_points = [[point["x"], point["y"]] for point in self.get_query_points()]
        new_query_points = []
        for query_point_start, i_p, f_p in zip(x_y_query_points, initial_positions, final_positions):
            new_center = self.circle_movement_predictor.predict_circle_x_y_r(
                query_point_start=np.array(query_point_start, dtype=np.float32),
                initial_positions=np.array(i_p, dtype=np.float32),
                final_positions=np.array(f_p, dtype=np.float32),
            )
            new_query_points.append(new_center)

        # Set new query points
        formatted_query_points = []
        for i, point in enumerate(new_query_points):
            formatted_query_points.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "color": self.get_query_points()[i]["color"],
            "radius": self.get_query_points()[i]["radius"]
            })
        self.set_query_points(formatted_query_points)  


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
