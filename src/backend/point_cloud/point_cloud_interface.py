from abc import ABC, abstractmethod

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
