import os
import importlib.util
from tapnet.utils import viz_utils


# Get paths
TAPIR_DIR = os.path.dirname(os.path.abspath(__file__))
POINT_CLOUD_DIR = os.path.dirname(TAPIR_DIR)

# Import point cloud interface
spec = importlib.util.spec_from_file_location(
    "point_cloud_slice", os.path.join(POINT_CLOUD_DIR, "point_cloud_slice.py")
)
point_cloud_slice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(point_cloud_slice)


class TapirPointCloudSlice(point_cloud_slice.PointCloudSlice):
    
    def get_video(self):
        return viz_utils.plot_tracks_v2(self.orig_frames, self.tracks, 1.0 - self.visibles)