import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backend.point_cloud.point_cloud import PointCloud
from .center_predictor_model import CenterPredictorModel


class QueryPointPredictionModel():

    def __init__(self):
        self.model = CenterPredictorModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.batches = []

    def predict(
            self,
            old_point_cloud: PointCloud,
            final_positions: np.ndarray,
            inliers_rotation: tuple[np.ndarray, float],
        ):
        self.model.eval()

        initial_positions = old_point_cloud.cloud_points
        inputs, mask = self.prepare_prediction(
            final_positions, 
            initial_positions, 
            inliers_rotation
        )

        with torch.no_grad():
            pred = self.model(inputs, mask)

        return pred.squeeze(0).numpy()


    def incremental_fit(
        self, 
        predicted_point_cloud: PointCloud,
        inliers_rotation: tuple[np.ndarray, float],
        true_query_point: np.ndarray,
        initial_positions: np.ndarray,
        num_steps: int = 100
    ):
        self.model.train()
        
        final_positions = predicted_point_cloud.cloud_points
        self.prepare_sample(
            final_positions, 
            initial_positions, 
            inliers_rotation, 
            true_query_point
        )
        
        total_loss = 0
        for _ in range(num_steps):
            for batch in self.batches:
                inputs = torch.tensor(batch["input_feats"], dtype=torch.float32).unsqueeze(0)
                masks = torch.tensor(batch["inliers_mask"], dtype=torch.bool).unsqueeze(0)
                targets = torch.tensor(batch["target"], dtype=torch.float32).unsqueeze(0)

                pred = self.model(inputs, masks)

                loss = F.mse_loss(pred, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        return total_loss / (num_steps * len(self.batches))
    
    
    def prepare_prediction(
            self, 
            final_positions, 
            initial_positions, 
            inliers_rotation, 
    ):
        if not isinstance(final_positions, np.ndarray):
            final_positions = np.array(final_positions, dtype=np.float32)
        if not isinstance(initial_positions, np.ndarray):
            initial_positions = np.array(initial_positions, dtype=np.float32)
        final = final_positions  # N x 2
        init = initial_positions  # N x 2
        inliers_mask, _ = inliers_rotation  # N x 1
        if not isinstance(inliers_mask, np.ndarray):
            inliers_mask = np.array(inliers_mask, dtype=bool)


        motion = final - init  # N x 2

        # Feature per point: [x0, y0, dx, dy]
        input_feats = np.concatenate([init, motion], axis=1)  # N x 4

        return (
            torch.tensor(input_feats, dtype=torch.float32).unsqueeze(0),       # 1 x N x 4
            torch.tensor(inliers_mask.squeeze(), dtype=torch.bool).unsqueeze(0)  # 1 x N
        )


    def prepare_sample(
            self, 
            final_positions, 
            initial_positions, 
            inliers_rotation, 
            true_query_point
        ):
        if not isinstance(final_positions, np.ndarray):
            final_positions = np.array(final_positions, dtype=np.float32)
        if not isinstance(initial_positions, np.ndarray):
            initial_positions = np.array(initial_positions, dtype=np.float32)
        final = final_positions  # N x 2
        init = initial_positions  # N x 2
        inliers_mask, _ = inliers_rotation  # N x 1
        if not isinstance(inliers_mask, np.ndarray):
            inliers_mask = np.array(inliers_mask, dtype=bool)

        motion = final - init  # N x 2

        # Feature per point: [x0, y0, dx, dy]
        input_feats = np.concatenate([init, motion], axis=1)  # N x 4

        current_batch = {
            "input_feats": input_feats,
            "inliers_mask": inliers_mask.squeeze(),
            "target": true_query_point
        }

        self.batches.append(current_batch)