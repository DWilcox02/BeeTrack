import numpy as np

class PointCloud():
    def __init__(self, query_point: np.ndarray, cloud_points: np.ndarray, radius: float, rotation: float, weights: np.ndarray, vectors_qp_to_cp=None, inliers=None):
        self.query_point: np.ndarray = query_point
        self.cloud_points: np.ndarray = cloud_points
        self.radius = radius
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
        if inliers is None:
            self.inliers = [i for i in range(len(cloud_points))]
        else:
            self.inliers = inliers

    def query_point_array(self):
        return np.array([self.query_point["x"], self.query_point["y"]], dtype=np.float32)
    
    def format_new_query_point(self, query_point_array):
        return {
            "x": float(query_point_array[0]),
            "y": float(query_point_array[1]),
            "color": self.query_point["color"],
            "radius": self.query_point["radius"],
        }
    
    def confidence(self):
        return 0.0