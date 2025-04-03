from abc import ABC, abstractmethod
from tapnet.utils import viz_utils
import numpy as np

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

    def get_trajectory(self):
        start_positions = self.tracks[:, 0, :]  # Extract the starting (x, y) positions
        end_positions = self.tracks[:, -1, :]  # Extract the ending (x, y) positions
        trajectories = end_positions - start_positions  # Calculate the trajectories for each point
        mean_trajectory = np.mean(trajectories, axis=0)  # Average the trajectories
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
        
        self.d_mid_head = np.linalg.norm(midpoint - np.array([head["x"], head["y"]]))
        self.v_mid_head = (np.array([head["x"], head["y"]]) - midpoint) / self.d_mid_head

        self.d_mid_butt = np.linalg.norm(midpoint - np.array([butt["x"], butt["y"]]))
        self.v_mid_butt = (np.array([butt["x"], butt["y"]]) - midpoint) / self.d_mid_butt
        
        self.d_mid_left = np.linalg.norm(midpoint - np.array([left["x"], left["y"]]))
        self.v_mid_left = (np.array([left["x"], left["y"]]) - midpoint) / self.d_mid_left
        
        self.d_mid_right = np.linalg.norm(midpoint - np.array([right["x"], right["y"]]))
        self.v_mid_right = (np.array([right["x"], right["y"]]) - midpoint) / self.d_mid_right
    
    def pretty_print_skeleton(self):
        print(f"Head: {self.d_mid_head}, {self.v_mid_head}")
        print(f"Butt: {self.d_mid_butt}, {self.v_mid_butt}")
        print(f"Left: {self.d_mid_left}, {self.v_mid_left}")
        print(f"Right: {self.d_mid_right}, {self.v_mid_right}")