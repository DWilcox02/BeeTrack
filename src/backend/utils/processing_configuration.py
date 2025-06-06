from dataclasses import dataclass

from src.backend.point_cloud.point_cloud_generator import PointCloudGenerator
from src.backend.point_cloud.estimation.point_cloud_estimator_interface import PointCloudEstimatorInterface
from src.backend.inlier_predictors.inlier_predictor_base import InlierPredictorBase
from src.backend.query_point_predictors.query_point_reconstructor_base import QueryPointReconstructorBase
from src.backend.point_cloud_reconstructors.point_cloud_reconstructor_base import PointCloudReconstructorBase
from src.backend.weight_calculators.weight_calculator_distances_base import WeightCalculatorDistancesBase
from src.backend.weight_calculators.weight_calculator_outliers_base import WeightCalculatorOutliersBase



@dataclass(frozen=True)
class ProcessingConfiguration:
    smoothing_alpha: float
    dbscan_epsilon: float
    deformity_delta: float
    processing_seconds: int
    point_cloud_generator: PointCloudGenerator
    point_cloud_estimator: PointCloudEstimatorInterface
    inlier_predictor: InlierPredictorBase
    query_point_reconstructor: QueryPointReconstructorBase
    point_cloud_non_validated_reconstructor: PointCloudReconstructorBase
    point_cloud_validated_reconstructor: PointCloudReconstructorBase
    weight_calculator_outliers: WeightCalculatorOutliersBase
    weight_calculator_distances: WeightCalculatorDistancesBase