import numpy as np
import torch

from src.backend.models.circle_movement_model import CircleMovementModel
from src.backend.models.circle_movement_result import CircleMovementResult
from src.backend.models.ransac_model import RANSAC


class CircleMovementPredictor:
    def predict_circle_x_y_r(self, query_point_start, initial_positions, final_positions) -> CircleMovementResult:
        # Model setup
        query_point_start_tensor = torch.tensor(query_point_start, dtype=torch.float32)
        initial_guess = np.mean(initial_positions, axis=0)  # Mean across all points
        initial_guess_tensor = torch.tensor(initial_guess, dtype=torch.float32)
        circle_movement_model = CircleMovementModel(
            original_center=query_point_start_tensor, initial_guess=initial_guess_tensor
        )
        print(f"Initial guess for center: {initial_guess}")

        # RANSAC setup
        loss = torch.nn.MSELoss()
        ransac_model = RANSAC(
            model=circle_movement_model,
            loss=loss
        )

        # Input/output setup
        initial_distances_directions = []
        for point in initial_positions:
            difference = point - query_point_start
            distance = np.linalg.norm(difference)
            initial_distances_directions.append([distance, difference[0] / distance, difference[1] / distance])
        initial_distances_directions = np.array(initial_distances_directions, dtype=np.float32)

        # Convert to tensors
        X = torch.tensor(initial_distances_directions, dtype=torch.float32)
        y = torch.tensor(final_positions, dtype=torch.float32)

        # Fit with RANSAC
        maybe_model: CircleMovementModel = ransac_model.fit(X, y)

        # Get results
        new_center, rotation = ransac_model.get_new_point(maybe_model, query_point_start)
        inliers, outliers = ransac_model.get_inliers_outliers()

        
        return CircleMovementResult(
            x=new_center[0], 
            y=new_center[1],
            r=rotation,
            inlier_idxs=inliers,
            outlier_idxs=outliers
        )