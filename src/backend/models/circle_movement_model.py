import torch
import torch.nn as nn


class CircleMovementModel(nn.Module):
    def __init__(self, original_center, initial_guess):
        super(CircleMovementModel, self).__init__()

        # Original center as a tensor
        self.original_center = original_center

        # Parameters to learn
        self.delta_translation = nn.Parameter(initial_guess - original_center)  # [delta_x, delta_y]
        self.rotation_angle = nn.Parameter(torch.zeros(1))  # Single rotation angle

    def forward(self, initial_distances_directions):
        """
        initial_distances_directions: tensor of shape N x 3
                                      where each row is [distance, dir_x, dir_y]
        """
        # Calculate new center
        new_center = self.original_center + self.delta_translation

        # Extract components
        distances = initial_distances_directions[:, 0].unsqueeze(1)
        directions = initial_distances_directions[:, 1:]  # dir_x and dir_y

        # Create rotation matrix
        cos_theta = torch.cos(self.rotation_angle)
        sin_theta = torch.sin(self.rotation_angle)

        rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Rotate directions
        rotated_directions = torch.matmul(directions, rotation_matrix.T)

        # Scale by distances
        vectors = rotated_directions * distances

        # Calculate final positions
        predicted_points = new_center.unsqueeze(0) + vectors

        return predicted_points
