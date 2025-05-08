import torch
import torch.nn as nn

MODEL_ITERATIONS = 100

class CircleMovementModel(nn.Module):
    def __init__(self, original_center, initial_guess):
        super(CircleMovementModel, self).__init__()

        # Original center as a tensor
        self.original_center = original_center

        # Parameters to learn
        self.delta_translation = nn.Parameter(initial_guess - original_center)  # [delta_x, delta_y]
        self.rotation_angle = nn.Parameter(torch.zeros(1))  # Single rotation angle

    def forward(self, initial_vectors):
        """
        initial_vectors: tensor of shape N x 2
                        where each row is [x_offset, y_offset]
        """
        # Calculate new center
        new_center = self.original_center + self.delta_translation
        
        # Create rotation matrix
        cos_theta = torch.cos(self.rotation_angle)
        sin_theta = torch.sin(self.rotation_angle)
        
        rotation_matrix = torch.tensor([[cos_theta, -sin_theta], 
                                        [sin_theta, cos_theta]])
        
        # Rotate the offset vectors directly
        rotated_vectors = torch.matmul(initial_vectors, rotation_matrix.T)
        
        # Calculate final positions
        predicted_points = new_center.unsqueeze(0) + rotated_vectors
        
        return predicted_points


    def fit(self, X, y):
        # print(f"Training for {MODEL_ITERATIONS} iterations")

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1)

        for t in range(MODEL_ITERATIONS):
            y_pred = self(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return self

    def predict(self, X):
        with torch.no_grad():
            return self(X)