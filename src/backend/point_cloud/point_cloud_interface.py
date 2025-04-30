from abc import ABC, abstractmethod
from tapnet.utils import viz_utils
import numpy as np

TRAJECTORY_EPSILON = 10

class PointCloudInterface(ABC):
    @abstractmethod
    def process_video_slice(
        self,
        orig_frames,
        width,
        height,
        query_points,
        resize_width=256,
        resize_height=256,
    ):
        """
        Process a video file to generate a point cloud.

        :param path: The directory path where the video is located.
        :param filename: The name of the video file.
        :param fps: The frames per second to process the video.
        """
        pass


class PointCloudSlice(ABC):
    def __init__(self, orig_frames, tracks, visibles, confidence):
        self.orig_frames = orig_frames
        self.tracks = tracks  # shape (num_points, num_frame, 2), floats
        self.visibles = visibles # shape (num_points, num_frame), bool
        self.confidence = confidence  # float

    def get_video(self):
        return viz_utils.plot_tracks_v2(self.orig_frames, self.tracks, 1.0 - self.visibles)

    def get_final_mean(self):
        final_positions = self.tracks[:, -1, :]  # Extract the final (x, y) positions
        return np.mean(final_positions, axis=0)  # Calculate the mean of the final positions

    def get_trajectory(self, prev_trajectory=None):
        point_filter = np.logical_and(self.visibles[:, 0], self.visibles[:, -1])
        start_positions = self.tracks[:, 0, :][point_filter]  # Extract the starting (x, y) positions
        end_positions = self.tracks[:, -1, :][point_filter]  # Extract the ending (x, y) positions
        trajectories = end_positions - start_positions  # Calculate the trajectories for each point
        mean_trajectory = np.mean(trajectories, axis=0)  # Average the trajectories
        if np.linalg.norm(mean_trajectory) < TRAJECTORY_EPSILON and prev_trajectory is not None:
            mean_trajectory = prev_trajectory
        normalized_trajectory = mean_trajectory / np.linalg.norm(mean_trajectory)  # Normalize the trajectory
        return normalized_trajectory


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