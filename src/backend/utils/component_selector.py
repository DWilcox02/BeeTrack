from enum import Enum
from typing import Optional

# Point Cloud Estimators
from src.backend.point_cloud.estimation.TAPIR_point_cloud_estimator.tapir_estimator import TapirEstimator

# Point Cloud Generators
from src.backend.point_cloud.singlular_point_cloud_generator import SingularPointCloudGenerator
from src.backend.point_cloud.circular_point_cloud_generator import CircularPointCloudGenerator

# Import Inlier Predictors
from src.backend.inlier_predictors.inlier_predictor_base import InlierPredictorBase
from src.backend.inlier_predictors.dbscan_inlier_predictor import DBSCANInlierPredictor
from src.backend.inlier_predictors.hdbscan_inlier_predictor import HDBSCANInlierPredictor

# Import Query Point Predictors
from src.backend.query_point_predictors.query_point_reconstructor_base import QueryPointReconstructorBase
from src.backend.query_point_predictors.inlier_weighted_avg_reconstructor import InlierWeightedAvgReconstructor
from src.backend.query_point_predictors.incremental_nn_reconstructor import IncrementalNNReconstructor

# Import Point Cloud Reconstructors
from src.backend.point_cloud_reconstructors.point_cloud_reconstructor_base import PointCloudReconstructorBase
from src.backend.point_cloud_reconstructors.point_cloud_recons_inliers import PointCloudReconsInliers
from src.backend.point_cloud_reconstructors.point_cloud_redraw_outliers import PointCloudRedrawOutliers
from src.backend.point_cloud_reconstructors.point_cloud_redraw_outliers_random import PointCloudRedrawOutliersRandom
from src.backend.point_cloud_reconstructors.point_cloud_cluster_recovery import PointCloudClusterRecovery

# Import Weight Calculators
from src.backend.weight_calculators.weight_calculator_distances_ewma import WeightCalculatorDistancesEWMA
from src.backend.weight_calculators.weight_calculator_outliers_penalty import WeightCalculatorOutliersPenalty
from src.backend.weight_calculators.incremental_nn_weight_updater import IncrementalNNWeightUpdater


class PointCloudEstimatorSelector(Enum):
    tapir_estimator = TapirEstimator


class PointCloudGeneratorSelector(Enum):
    singular_point_cloud_generator = SingularPointCloudGenerator
    circular_point_cloud_generator = CircularPointCloudGenerator


class InlierPredictorSelector(Enum):
    inlier_predictor_base = InlierPredictorBase
    dbscan_inlier_predictor = DBSCANInlierPredictor
    hdbscan_inlier_predictor = HDBSCANInlierPredictor


class QueryPointReconstructorSelector(Enum):
    query_point_reconstructor_base = QueryPointReconstructorBase
    inlier_weighted_avg_reconstructor = InlierWeightedAvgReconstructor
    incremental_nn_reconstructor = IncrementalNNReconstructor


class PointCloudReconstructorSelector(Enum):
    point_cloud_reconstructor_base = PointCloudReconstructorBase
    point_cloud_recons_inliers = PointCloudReconsInliers
    point_cloud_redraw_outliers = PointCloudRedrawOutliers
    point_cloud_redraw_outliers_random = PointCloudRedrawOutliersRandom
    point_cloud_cluster_recovery = PointCloudClusterRecovery


class WeightCalculatorDistancesSelector(Enum):
    weight_calculator_distances_ewma = WeightCalculatorDistancesEWMA
    incremental_nn_weight_updater = IncrementalNNWeightUpdater


class WeightCalculatorOutliersSelector(Enum):
    weight_calculator_outliers_penalty = WeightCalculatorOutliersPenalty


def create_component(selector_enum_member, dbscan_epsilon: Optional[float] = None):
    component_class = selector_enum_member.value
    if dbscan_epsilon is not None:
        return component_class(dbscan_epsilon)
    else:
        return component_class()

