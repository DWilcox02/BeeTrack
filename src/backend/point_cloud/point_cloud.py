import numpy as np

class PointCloud():
    def __init__(self, query_point: np.ndarray, cloud_points: np.ndarray, rotation: float, weights: np.ndarray, vectors_qp_to_cp=None):
        self.query_point: np.ndarray = query_point
        self.cloud_points: np.ndarray = cloud_points
        self.rotation: np.ndarray = rotation
        self.weights: np.ndarray = weights
        query_point_ndarry = np.array([query_point["x"], query_point["y"]], dtype=np.float32)
        if vectors_qp_to_cp is None:
            self.vectors_qp_to_cp = []
            for point in cloud_points:
                self.vectors_qp_to_cp.append(point - query_point_ndarry)
            self.vectors_qp_to_cp = np.array(self.vectors_qp_to_cp, dtype=np.float32)
        else:
            self.vectors_qp_to_cp = vectors_qp_to_cp

    def query_point_array(self):
        return np.array([self.query_point["x"], self.query_point["y"]], dtype=np.float32)