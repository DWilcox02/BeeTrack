import numpy as np
import torch
from copy import copy
from numpy.random import default_rng

rng = default_rng()

class RANSAC():
    def __init__(self, n=10, k=50, t=0.05, d=10, model=None):
        self.n = n  # `n`: Minimum number of data points to estimate parameters
        self.k = k  # `k`: Maximum iterations allowed
        self.t = t  # `t`: Threshold value to determine if points are fit well
        self.d = d  # `d`: Number of close data points required to assert model fits well
        self.model = model  # `model`: class implementing `fit` and `predict`
        # self.loss = loss  # `loss`: function of `y_true` and `y_pred` that returns a vector
        # self.metric = metric  # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.inliers = []
        self.outliers = []

    def square_error_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def mse_loss(self, y_true, y_pred):
        return torch.sum(self.square_error_loss(y_true, y_pred)) / y_true.shape[0]

    def fit(self, X, y):
        # Start with all points for initial guess
        self.best_fit = copy(self.model).fit(X, y)
        self.best_error = self.mse_loss(y, self.best_fit.predict(X))
        self.inliers = np.arange(len(X))
        print(f"MSE for all inliers (score to beat): {self.best_error}")

        # Decrease number of samples taken
        for num_samples in range(len(X) - 1, self.n, -1):
        # num_samples = self.n
            for _ in range(self.k):
                ids = rng.permutation(X.shape[0])

                maybe_inliers = ids[: num_samples]
                maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])
                this_error = self.mse_loss(y, maybe_model.predict(X))

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = maybe_model
                    self.inliers = maybe_inliers
        return self

    def predict(self, X):
        return self.best_fit.predict(X)
    
    def get_inliers_outliers(self):
        return self.inliers, []
