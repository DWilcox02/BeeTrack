import numpy as np

class PointCloud():
    def __init__(
            self, 
            query_point: dict, 
            cloud_points: np.ndarray, 
            radius: float, 
            rotation: float, 
            weights: np.ndarray, 
            vectors_qp_to_cp=None, 
            inliers=None,
            orig_vectors=None,
            log_fn=None
        ):
        self.query_point: dict = query_point
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
            self.inliers = [True] * len(cloud_points)
        else:
            self.inliers = inliers
        if orig_vectors is None:
            self.orig_vectors = self.vectors_qp_to_cp
        else:
            self.orig_vectors = orig_vectors
        if log_fn is None:
            self.log_fn = print
        else:
            self.log_fn = log_fn

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    def query_point_array(self):
        return np.array([self.query_point["x"], self.query_point["y"]], dtype=np.float32)
    

    def format_new_query_point(self, query_point_array):
        return {
            "x": float(query_point_array[0]),
            "y": float(query_point_array[1]),
            "color": self.query_point["color"],
            "radius": self.query_point["radius"],
        }
    
    def confidence(
            self,
            inliers: np.ndarray, 
            deformity_delta: float
        ):
        deformity = self.deformity()
        if deformity == 0:
            deformity_ratio = 1.0
        else:
            deformity_ratio = min((self.radius * deformity_delta) / deformity, 1.0)

        inlier_ratio = np.sum(inliers) / len(self.cloud_points)

        label = self.query_point["color"]
        # self.log(f"{label} Deformity Ratio: {deformity_ratio}")
        # self.log(f"{label} Inlier Ratio: {inlier_ratio}")
        self.log(f"{label} Confidence: {(inlier_ratio + deformity_ratio) / 2}")

        return (inlier_ratio + deformity_ratio) / 2
    
    def query_point_predictions(
            self, 
            vectors_qp_to_cp: np.ndarray = None,  
            final_positions: np.ndarray = None,
            rotation: float = None
        ) -> np.ndarray:

        if vectors_qp_to_cp is None:
            vectors_qp_to_cp = self.vectors_qp_to_cp
        if final_positions is None:
            final_positions = self.cloud_points
        if rotation is None:
            rotation = self.rotation

        final_predictions = []
        for vec_qp_to_cp, pos in zip(vectors_qp_to_cp, final_positions):
            rotated_vec = self.rotate_vector(vec_qp_to_cp, rotation)
            final_predictions.append(pos - rotated_vec)

        final_predictions = np.array(final_predictions, dtype=np.float32)
        return final_predictions

    def rotate_vector(self, vector: np.ndarray, angle_degrees: int) -> np.ndarray:
        angle_rad = np.radians(angle_degrees)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        return np.dot(rotation_matrix, vector)
    
    def deformity(
            self, 
            mean: np.ndarray = None,
            points: np.ndarray = None
        ):

        if mean is None: 
            mean = self.query_point_array()
        if points is None:
            points = self.query_point_predictions()

        


        return np.sum(np.linalg.norm(mean - points, axis=1))