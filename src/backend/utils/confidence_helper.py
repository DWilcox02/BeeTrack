import numpy as np


def cloud_confidence(
        inliers: np.ndarray, 
        deformity: float,
        deformity_delta: float,
        radius: float,
        num_cloud_points: int
    ) -> float:
    deformity_ratio = min(deformity / (np.pow(radius, 4) * deformity_delta), 1.0)
    deformity_confidence = 1.0 - deformity_ratio

    inlier_confidence = np.sum(inliers) / num_cloud_points

    return (inlier_confidence + deformity_confidence) / 2