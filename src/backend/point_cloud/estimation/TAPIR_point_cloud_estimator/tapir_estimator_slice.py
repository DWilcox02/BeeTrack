import jax.numpy as jnp


from src.backend.point_cloud.estimation.estimation_slice import EstimationSlice
from tapnet.utils import viz_utils


class TapirEstimatorSlice(EstimationSlice):
    def get_video(self):
        return viz_utils.plot_tracks_v2(self.orig_frames, self.tracks, 1.0 - self.visibles)

    def get_video_for_points(self, points):
        visibles = jnp.ones((points.shape[0], points.shape[1]), dtype=bool)
        return viz_utils.plot_tracks_v2(self.orig_frames, points, 1.0 - visibles)