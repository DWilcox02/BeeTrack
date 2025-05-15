import jax.numpy as jnp


from src.backend.point_cloud.estimation.estimation_slice import EstimationSlice
from tapnet.utils import viz_utils


class TapirEstimatorSlice(EstimationSlice):
    def get_video(self, inliers=None):
        if inliers is not None:
            # inliers is boolean list of shape 84 x 1
            inliers_expanded = [[b] * 15 for b in inliers]
            inliers_jnp = jnp.array(inliers_expanded, dtype=bool)
            assert(inliers_jnp.shape[0] == self.tracks.shape[0])
            assert(inliers_jnp.shape[1] == self.tracks.shape[1])
            visibles = inliers_jnp
        else:
            visibles = self.visibles
        return viz_utils.plot_tracks_v2(self.orig_frames, self.tracks, 1.0 - visibles)

    def get_video_for_points(self, points):
        visibles = jnp.ones((points.shape[0], points.shape[1]), dtype=bool)
        return viz_utils.plot_tracks_v2(self.orig_frames, points, 1.0 - visibles)