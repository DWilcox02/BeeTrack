import numpy as np
import torch

from src.backend.models.circle_movement_model import CircleMovementModel

ITERATIONS = 100

class RANSAC():
    def ransac(self, query_point_start, initial_positions, final_positions):
        print(f"Starting prediction with {len(initial_positions)} points")
        print(f"Query point start: {query_point_start}")

        # Convert numpy arrays to torch tensors
        query_point_start_tensor = torch.tensor(query_point_start, dtype=torch.float32)
        initial_guess = np.mean(initial_positions, axis=0)  # Mean across all points
        initial_guess_tensor = torch.tensor(initial_guess, dtype=torch.float32)

        print(f"Initial guess for center: {initial_guess}")

        # Calculate initial distances and directions
        initial_distances_directions = []
        for point in initial_positions:
            difference = point - query_point_start
            distance = np.linalg.norm(difference)
            initial_distances_directions.append([distance, difference[0] / distance, difference[1] / distance])
        initial_distances_directions = np.array(initial_distances_directions, dtype=np.float32)

        print(f"Calculated {len(initial_distances_directions)} distance/direction vectors")

        # Convert to tensors
        X = torch.tensor(initial_distances_directions, dtype=torch.float32)
        y = torch.tensor(final_positions, dtype=torch.float32)

        print(f"Input tensor shape: {X.shape}, Output tensor shape: {y.shape}")

        # Setup model with tensor inputs
        circle_movement_model = CircleMovementModel(
            original_center=query_point_start_tensor, initial_guess=initial_guess_tensor
        )

        print("Model initialized, starting optimization")
        print(f"Training for {ITERATIONS} iterations")

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(circle_movement_model.parameters(), lr=1)

        for t in range(ITERATIONS):
            y_pred = circle_movement_model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        delta_x = circle_movement_model.delta_translation[0].item()
        delta_y = circle_movement_model.delta_translation[1].item()
        rotation = circle_movement_model.rotation_angle.item()

        print(f"Optimization complete")
        print(f"Calculated delta: ({delta_x:.4f}, {delta_y:.4f})")

        new_center = (query_point_start[0] + delta_x, query_point_start[1] + delta_y)
        print(f"New center position: {new_center}")

        return new_center, rotation
