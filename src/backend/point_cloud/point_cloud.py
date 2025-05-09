import numpy as np

class PointCloud():
    def __init__(self, query_point: np.ndarray, cloud_points: np.ndarray, rotation: float, weights: np.ndarray, offset_vectors=None):
        self.query_point: np.ndarray = query_point
        self.cloud_points: np.ndarray = cloud_points
        self.rotation: np.ndarray = rotation
        self.weights: np.ndarray = weights
        query_point_ndarry = np.array([query_point["x"], query_point["y"]], dtype=np.float32)
        if offset_vectors is None:
            self.offset_vectors = []
            for point in cloud_points:
                self.offset_vectors.append(point - query_point_ndarry)
            self.offset_vectors = np.array(self.offset_vectors, dtype=np.float32)
        else:
            self.offset_vectors = offset_vectors

    def query_point_array(self):
        return np.array([self.query_point["x"], self.query_point["y"]], dtype=np.float32)