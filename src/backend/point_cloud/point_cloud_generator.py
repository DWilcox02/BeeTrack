from abc import ABC, abstractmethod
import numpy as np

class PointCloudGenerator(ABC):
    # Points format:
    # [
    #   {'x': Array(1014.8928, dtype=float32), 'y': Array(642.25415, dtype=float32), 'color': 'red'},
    #   {'x': Array(1074.8928, dtype=float32), 'y': Array(692.25415, dtype=float32), 'color': 'green'},
    #   {'x': Array(1041.4928, dtype=float32), 'y': Array(678.9541, dtype=float32), 'color': 'blue'},
    #   {'x': Array(1054.8928, dtype=float32), 'y': Array(663.9541, dtype=float32), 'color': 'purple'}
    # ]

    def __init__(self, init_points, point_data_store, session_id):
        self.query_points = init_points # Initial query points
        self.point_data_store = point_data_store
        self.session_id = session_id
        self.log_fn = print
        current_cloud_points = self.generate_cloud_points()
        self.cp_per_qp = int(len(current_cloud_points) / len(init_points))
        self.weights = np.array([1 / self.cp_per_qp] * len(current_cloud_points), dtype=np.float32)
        # print(f"Initialised generator with {len(self.weights)} weights")
        # print(f"{self.cp_per_qp} cloud points per query point")


    def get_query_points(self):
        assert(self.session_id in self.point_data_store)
        assert(self.query_points == self.point_data_store[self.session_id]["points"])
        return self.point_data_store[self.session_id]["points"]
    
    def set_query_points(self, points):
        self.query_points = points
        self.export_to_point_data_store()
    
    def export_to_point_data_store(self):
        self.point_data_store[self.session_id]["points"] = self.query_points

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    @abstractmethod
    def generate_cloud_points(self):
        pass

    @abstractmethod
    def update_weights(self, initial_positions, final_positions):
        pass

    @abstractmethod
    def initial_trajectory(self):
        pass

    @abstractmethod
    def recalc_query_points(self, final_positions):
        pass

    @abstractmethod
    def calculate_confidence(self):
        pass