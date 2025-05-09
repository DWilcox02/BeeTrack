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

    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    @abstractmethod
    def generate_initial_point_clouds(self):
        pass

    @abstractmethod
    def update_weights(self, initial_positions, final_positions):
        pass

    @abstractmethod
    def recalc_query_points_rotations(self, final_positions):
        pass

    @abstractmethod
    def calculate_confidence(self):
        pass