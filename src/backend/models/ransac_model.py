import numpy as np
import torch
from copy import copy
from numpy.random import default_rng

rng = default_rng()

class RANSAC():
    def __init__(self, n=10, k=50, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n  # `n`: Minimum number of data points to estimate parameters
        self.k = k  # `k`: Maximum iterations allowed
        self.t = t  # `t`: Threshold value to determine if points are fit well
        self.d = d  # `d`: Number of close data points required to assert model fits well
        self.model = model  # `model`: class implementing `fit` and `predict`
        self.loss = loss  # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric  # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.inliers = []
        self.outliers = []

    def fit(self, X, y):
        # for _ in range(self.k):
        #     ids = rng.permutation(X.shape[0])

        #     maybe_inliers = ids[: self.n]
        #     maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

        #     thresholded = self.loss(y[ids][self.n :], maybe_model.predict(X[ids][self.n :])) < self.t

        #     inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

        #     if inlier_ids.size > self.d:
        #         inlier_points = np.hstack([maybe_inliers, inlier_ids])
        #         better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

        #         this_error = self.metric(y[inlier_points], better_model.predict(X[inlier_points]))

        #         if this_error < self.best_error:
        #             self.best_error = this_error
        #             self.best_fit = better_model

        # return self
        maybe_model = copy(self.model)
        maybe_model.fit(X, y)
        return maybe_model

    def predict(self, X):
        return self.best_fit.predict(X)

    def get_new_point(self, model, query_point_start):
        delta_x = model.delta_translation[0].item()
        delta_y = model.delta_translation[1].item()
        rotation = model.rotation_angle.item()
        new_center = (query_point_start[0] + delta_x, query_point_start[1] + delta_y)
        print(f"New center position: {new_center}")
        return new_center, rotation
    
    def get_inliers_outliers(self):
        return [], []
