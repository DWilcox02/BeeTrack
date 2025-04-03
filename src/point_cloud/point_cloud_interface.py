from abc import ABC, abstractmethod
from tapnet.utils import viz_utils

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
        """
        Initialize the PointCloudSlice object.
        :param orig_frames: The original frames of the video.
        :param tracks: The tracks of the points in the video.
        :param visibles: The visibility of the points in the video.
        :param point_cloud: The point cloud data.
        """
        self.orig_frames = orig_frames
        self.tracks = tracks
        self.visibles = visibles

    def get_video(self):
        return viz_utils.plot_tracks_v2(self.orig_frames, self.tracks, 1.0 - self.visibles)