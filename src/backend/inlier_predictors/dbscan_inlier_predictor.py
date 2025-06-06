import numpy as np
from sklearn.cluster import DBSCAN

from src.backend.point_cloud.point_cloud import PointCloud

from .inlier_predictor_base import InlierPredictorBase

class DBSCANInlierPredictor(InlierPredictorBase):

    def predict_inliers(
        self,
        old_point_cloud: PointCloud,
        final_predictions: np.ndarray,
    ) -> np.ndarray[bool]:
        
        if len(final_predictions) > 1:
            best_final_predictions_masked = final_predictions[old_point_cloud.inliers]
            eps = old_point_cloud.radius * self.dbscan_epsilon

            min_samples = len(best_final_predictions_masked) // 2
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(best_final_predictions_masked)
            inlier_idxs = clustering.core_sample_indices_

            inlier_mask = np.zeros_like(clustering.labels_, dtype=bool)
            inlier_mask[inlier_idxs] = True
            inliers = inlier_mask
        else:
            inliers = [True] * len(final_predictions)

        inliers = np.array(inliers, dtype=bool)
        return inliers
