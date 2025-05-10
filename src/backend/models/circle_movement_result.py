import numpy as np
from typing import List

class CircleMovementResult():
    def __init__(self, x: float, y: float, r: float, inlier_idxs: List[int], outlier_idxs: List[int], final_predictions: np.ndarray):
        self.x = x
        self.y = y
        self.r = r
        self.inlier_idxs = inlier_idxs
        self.outlier_idxs = outlier_idxs
        self.final_predictions = final_predictions


    def confidence(self):
        return 0.0