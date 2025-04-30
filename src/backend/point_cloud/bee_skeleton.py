import numpy as np


class BeeSkeleton:
    def __init__(self, predefined_points):
        color_map = {"head": "red", "butt": "green", "left": "blue", "right": "purple"}
        points = {
            key: next((p for p in predefined_points if p["color"] == color), None) for key, color in color_map.items()
        }
        head, butt, left, right = points["head"], points["butt"], points["left"], points["right"]

        # Calculate midpoint of all points
        self.initial_midpoint = np.mean([[point["x"], point["y"]] for point in [head, butt, left, right]], axis=0)

        # Store points in local space (relative to midpoint as origin)
        self.local_points = {
            "head": np.array([head["x"] - self.initial_midpoint[0], head["y"] - self.initial_midpoint[1]]),
            "butt": np.array([butt["x"] - self.initial_midpoint[0], butt["y"] - self.initial_midpoint[1]]),
            "left": np.array([left["x"] - self.initial_midpoint[0], left["y"] - self.initial_midpoint[1]]),
            "right": np.array([right["x"] - self.initial_midpoint[0], right["y"] - self.initial_midpoint[1]]),
        }

        # Store the initial trajectory (unit vector from midpoint to head)
        self.initial_trajectory = self.local_points["head"] / np.linalg.norm(self.local_points["head"])

    def calculate_new_positions(self, new_midpoint, new_trajectory):
        # Normalize the new trajectory
        new_trajectory = new_trajectory / np.linalg.norm(new_trajectory)

        # Compute rotation angle between initial trajectory and new trajectory
        dot = np.dot(self.initial_trajectory, new_trajectory)
        det = self.initial_trajectory[0] * new_trajectory[1] - self.initial_trajectory[1] * new_trajectory[0]
        rotation_angle = np.arctan2(det, dot)

        # Build 2D rotation matrix
        R = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]]
        )

        new_points = {}

        # Step 1: Rotate local points
        # Step 2: Translate to new world space
        for key, local_point in self.local_points.items():
            # Rotate the point in local space
            rotated_local_point = R @ local_point

            # Translate to world space
            new_points[key] = {
                "x": new_midpoint[0] + rotated_local_point[0],
                "y": new_midpoint[1] + rotated_local_point[1],
            }

        return new_points