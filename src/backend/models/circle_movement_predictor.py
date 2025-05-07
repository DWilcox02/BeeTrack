
from src.backend.models.circle_movement_result import CircleMovementResult
from src.backend.models.ransac_model import RANSAC


class CircleMovementPredictor:
    def predict_circle_x_y_r(self, query_point_start, initial_positions, final_positions) -> CircleMovementResult:
        ransac_model = RANSAC()
        new_center, rotation = ransac_model.ransac(
            query_point_start=query_point_start,
            initial_positions=initial_positions,
            final_positions=final_positions
        )
        return CircleMovementResult(
            x=new_center[0], 
            y=new_center[1],
            r=rotation,
            inlier_idxs=[],
            outlier_idxs=[]
        )