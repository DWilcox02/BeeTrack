from abc import ABC, abstractmethod
from tapnet.utils import viz_utils
import numpy as np

TRAJECTORY_EPSILON = 10

class PointCloudInterface(ABC):
    @abstractmethod
    def process_video(
        self, 
        path: str, 
        filename: str, 
        fps: int,
        max_segments=None,
        save_intermediate=True,
        predefined_points=None
    ):
        """
        Process a video file to generate a point cloud.

        :param path: The directory path where the video is located.
        :param filename: The name of the video file.
        :param fps: The frames per second to process the video.
        """
        pass


class PointCloudSlice(ABC):
    def __init__(self, orig_frames, tracks, visibles):
        self.orig_frames = orig_frames
        self.tracks = tracks  # shape (num_points, num_frame, 2), floats
        self.visibles = visibles # shape (num_points, num_frame), bool

    def get_video(self):
        return viz_utils.plot_tracks_v2(self.orig_frames, self.tracks, 1.0 - self.visibles)

    def get_final_mean(self):
        final_positions = self.tracks[:, -1, :]  # Extract the final (x, y) positions
        return np.mean(final_positions, axis=0)  # Calculate the mean of the final positions

    def get_trajectory(self, prev_trajectory=None):
        start_positions = self.tracks[:, 0, :]  # Extract the starting (x, y) positions
        end_positions = self.tracks[:, -1, :]  # Extract the ending (x, y) positions
        trajectories = end_positions - start_positions  # Calculate the trajectories for each point
        mean_trajectory = np.mean(trajectories, axis=0)  # Average the trajectories
        if np.linalg.norm(mean_trajectory) < TRAJECTORY_EPSILON and prev_trajectory is not None:
            mean_trajectory = prev_trajectory
        normalized_trajectory = mean_trajectory / np.linalg.norm(mean_trajectory)  # Normalize the trajectory
        return normalized_trajectory


class BeeSkeleton():
    def __init__(self, predefined_points):
        """
        Initialize the BeeSkeleton with predefined points.

        :param predefined_points: List of dictionaries with predefined points and their colors.
        """
        color_map = {"head": "red", "butt": "green", "left": "blue", "right": "purple"}
        points = {key: next((p for p in predefined_points if p["color"] == color), None) for key, color in color_map.items()}
        head, butt, left, right = points["head"], points["butt"], points["left"], points["right"]

        midpoint = np.mean(
            [[point["x"], point["y"]] for point in [head, butt, left, right]], axis=0
        )

        # Calculate the normalized vectors from the midpoint to head and left
        v_mid_head = np.array([head["x"] - midpoint[0], head["y"] - midpoint[1]])
        v_mid_head /= np.linalg.norm(v_mid_head)
        self.v_mid_head = v_mid_head

        # Calculate the normalized vector from the midpoint to left
        v_mid_left = np.array([left["x"] - midpoint[0], left["y"] - midpoint[1]])
        v_mid_left /= np.linalg.norm(v_mid_left)

        # Calculate the angle alpha between the normalized vectors
        dot_product = np.dot(v_mid_head, v_mid_left)
        alpha = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to handle numerical precision issues

        self.alpha = alpha

        # Calculate the normalized vector from the midpoint to butt
        v_mid_butt = np.array([butt["x"] - midpoint[0], butt["y"] - midpoint[1]])
        v_mid_butt /= np.linalg.norm(v_mid_butt)

        # Calculate the angle beta between the normalized vectors v_mid_head and v_mid_butt
        dot_product_head_butt = np.dot(v_mid_head, v_mid_butt)
        beta = np.arccos(np.clip(dot_product_head_butt, -1.0, 1.0))  # Clip to handle numerical precision issues

        self.beta = beta

        # Calculate the normalized vector from the midpoint to right
        v_mid_right = np.array([right["x"] - midpoint[0], right["y"] - midpoint[1]])
        v_mid_right /= np.linalg.norm(v_mid_right)

        # Calculate the angle gamma between the normalized vectors v_mid_head and v_mid_right
        dot_product_head_right = np.dot(v_mid_head, v_mid_right)
        gamma = np.arccos(np.clip(dot_product_head_right, -1.0, 1.0))  # Clip to handle numerical precision issues

        # Determine the direction of the angle (counter-clockwise)
        cross_product = np.cross(np.append(v_mid_head, 0), np.append(v_mid_right, 0))
        if cross_product[2] < 0:  # If the z-component of the cross product is negative
            gamma = 2 * np.pi - gamma

        self.gamma = gamma

        # Calculate unnormalized distances from the midpoint to each point
        self.d_mid_head = np.linalg.norm([head["x"] - midpoint[0], head["y"] - midpoint[1]])
        self.d_mid_butt = np.linalg.norm([butt["x"] - midpoint[0], butt["y"] - midpoint[1]])
        self.d_mid_left = np.linalg.norm([left["x"] - midpoint[0], left["y"] - midpoint[1]])
        self.d_mid_right = np.linalg.norm([right["x"] - midpoint[0], right["y"] - midpoint[1]])

    def calculate_new_positions(self, new_midpoint, new_trajectory):
        # Normalize the new trajectory
        new_trajectory = new_trajectory / np.linalg.norm(new_trajectory)
        
        # The head is in the direction of the trajectory at the stored distance
        new_head = {
            "x": new_midpoint[0] + new_trajectory[0] * self.d_mid_head,
            "y": new_midpoint[1] + new_trajectory[1] * self.d_mid_head
        }
        
        # Calculate the rotation matrix based on the new trajectory
        # Assuming original trajectory was [1,0] (x-axis)
        theta = np.arctan2(new_trajectory[1], new_trajectory[0])
        
        # For butt (opposite to head, offset by beta)
        butt_angle = theta + np.pi - self.beta
        new_butt = {
            "x": new_midpoint[0] + self.d_mid_butt * np.cos(butt_angle),
            "y": new_midpoint[1] + self.d_mid_butt * np.sin(butt_angle)
        }
        
        # For left (offset by alpha)
        left_angle = theta + self.alpha
        new_left = {
            "x": new_midpoint[0] + self.d_mid_left * np.cos(left_angle),
            "y": new_midpoint[1] + self.d_mid_left * np.sin(left_angle)
        }
        
        # For right (offset by gamma)
        right_angle = theta + self.gamma
        new_right = {
            "x": new_midpoint[0] + self.d_mid_right * np.cos(right_angle),
            "y": new_midpoint[1] + self.d_mid_right * np.sin(right_angle)
        }
        
        return {
            "head": new_head,
            "butt": new_butt,
            "left": new_left,
            "right": new_right
        }