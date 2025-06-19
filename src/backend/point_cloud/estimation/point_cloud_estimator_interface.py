from abc import ABC, abstractmethod
from threading import Event

class PointCloudEstimatorInterface(ABC):
    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    @abstractmethod
    def process_video_slice(
        self,
        orig_frames,
        width,
        height,
        query_points,
        stop_event: Event,
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