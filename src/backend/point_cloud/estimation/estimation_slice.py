import numpy as np
from abc import ABC, abstractmethod


TRAJECTORY_EPSILON = 10


class EstimationSlice(ABC):
    def __init__(self, orig_frames, tracks, visibles):
        self.orig_frames = orig_frames
        self.tracks = tracks  # shape (num_points, num_frame, 2), floats
        self.visibles = visibles # shape (num_points, num_frame), bool
        self.confidence = self.calculate_confidence()  # float

    def calculate_confidence(self):
        return 0.0

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

    def get_points_for_frame(self, frame, num_qp, num_cp_per_qp):
        slice_i = 0

        new_query_points = []
        for _ in range(num_qp):
            qp_points = []
            for _ in range(num_cp_per_qp):
                point = self.tracks[:, frame, :][slice_i]
                qp_points.append(point)
                slice_i += 1
            new_query_points.append(qp_points)
        return new_query_points

    @abstractmethod
    def get_video(self):
        pass

    @abstractmethod
    def get_video_for_points(self, points):
        pass