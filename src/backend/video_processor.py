import os
import json
import time
import mediapy as media
import tempfile
import numpy as np
import gc
import cv2
import base64

from typing import List

from .point_cloud.estimation.point_cloud_estimator_interface import PointCloudEstimatorInterface
from .server.utils.video_utils import extract_frame
from .point_cloud.circular_point_cloud_generator import CircularPointCloudGenerator
from .point_cloud.estimation.estimation_slice import EstimationSlice
from .point_cloud.point_cloud import PointCloud
from src.backend.models.circle_movement_result import CircleMovementResult

# Import Inlier Predictors
from src.backend.inlier_predictors.inlier_predictor_base import InlierPredictorBase
from src.backend.inlier_predictors.dbscan_inlier_predictor import DBSCANInlierPredictor
from src.backend.inlier_predictors.hdbscan_inlier_predictor import HDBSCANInlierPredictor

# Import Inter-Cloud Alignments
from src.backend.inter_cloud_alignment_predictors.inter_cloud_alignment_base import InterCloudAlignmentBase

# Import Point Cloud Reconstructors
from src.backend.point_cloud_reconstructors.point_cloud_reconstructor_base import PointCloudReconstructorBase
from src.backend.point_cloud_reconstructors.point_cloud_recons_inliers import PointCloudReconsInliers
from src.backend.point_cloud_reconstructors.point_cloud_redraw_outliers import PointCloudRedrawOutliers
from src.backend.point_cloud_reconstructors.point_cloud_redraw_outliers_random import PointCloudRedrawOutliersRandom

# Import Query Point Predictors
from src.backend.query_point_predictors.query_point_reconstructor_base import QueryPointReconstructorBase
from src.backend.query_point_predictors.inlier_weighted_avg_reconstructor import InlierWeightedAvgReconstructor
from src.backend.query_point_predictors.incremental_nn_reconstructor import IncrementalNNReconstructor

# Import Weight Calculators
from src.backend.weight_calculators.weight_calculator_base import WeightCalculatorBase
from src.backend.weight_calculators.weight_calculator_distance import WeightCalculatorDistance
from src.backend.weight_calculators.weight_calculator_outliers import WeightCalculatorOutliers
from src.backend.weight_calculators.incremental_nn_weight_updater import IncrementalNNWeightUpdater


# Get paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BACKEND_DIR)
POINT_CLOUD_DIR = os.path.join(BACKEND_DIR, "point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))


NUM_SLICES = 1
CONFIDENCE_THRESHOLD = 0.7


class VideoProcessor():
    def __init__(
        self,
        session_id,
        point_data_store,
        point_cloud_estimator: PointCloudEstimatorInterface,
        send_frame_data_callback, 
        request_validation_callback,
        send_timeline_frame_callback,
        video,
        job_id,
    ):
        self.session_id = session_id
        self.point_cloud_estimator = point_cloud_estimator
        self.send_frame_data_callback = send_frame_data_callback
        self.request_validation_callback = request_validation_callback
        self.send_timeline_frame_callback = send_timeline_frame_callback
        self.video = video
        self.job_id = job_id
        self.log_message = None
        self.point_data_store = point_data_store

        # Shape of initial point clouds
        self.point_cloud_generator = CircularPointCloudGenerator()

        # Choices for experimentation
        self.inlier_predictor = DBSCANInlierPredictor()
        self.inter_cloud_alignment_predictor = InterCloudAlignmentBase()
        self.point_cloud_reconstructor = PointCloudRedrawOutliersRandom()

        # num_points = len(point_data_store[session_id]["points"])
        # self.query_point_reconstructor = IncrementalNNReconstructor(num_point_clouds=num_points)
        # self.weight_distance_calculator = IncrementalNNWeightUpdater(self.query_point_reconstructor.get_prediction_models())
        self.query_point_reconstructor = InlierWeightedAvgReconstructor()
        self.weight_distance_calculator = WeightCalculatorDistance()
        self.weight_outlier_calculator = WeightCalculatorOutliers()


    def set_log_message_function(self, fn):
        self.log_message = fn
        self.point_cloud_estimator.set_logger(self.log)
        self.point_cloud_generator.set_logger(self.log)
        self.inlier_predictor.set_logger(self.log)
        self.inter_cloud_alignment_predictor.set_logger(self.log)
        self.point_cloud_reconstructor.set_logger(self.log)
        self.query_point_reconstructor.set_logger(self.log)
        self.weight_distance_calculator.set_logger(self.log)
        self.weight_outlier_calculator.set_logger(self.log)



    def log(self, message):
        if self.job_id and self.log_message:
            self.log_message(self.job_id, message)
        print(message)
    
    
    def export_to_point_data_store(self, points):
        self.point_data_store[self.session_id]["points"] = points


    def send_current_frame_data(self, query_points, video_path, frame, confidence, request_validation):
        frame_base64, error, width, height = extract_frame(video_path, frame)
        frameData = {"frame": frame_base64, "width": width, "height": height, "frame_idx": frame}
        self.send_frame_data_callback(frameData, query_points, confidence, request_validation)


    def request_validation(self):
        return self.request_validation_callback(self.job_id)


    def send_timeline_frames(self, video_segment):
        # video_segment: [num_frames, height, width, 3], np.uint8, [0, 255]
        # base64_frames = []

        for i, frame in enumerate(video_segment):
            try:
                # Convert RGB to BGR for OpenCV JPEG encoding
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Encode to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                success, buffer = cv2.imencode(".jpg", frame_bgr, encode_param)
                if not success:
                    self.log(f"Failed to encode frame {i}")
                    continue

                # Convert to base64
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                self.send_timeline_frame_callback(frame_base64, i)
            except Exception as e:
                self.log(f"Error encoding frame {i}: {e}")

        # return base64_frames

    def combine_and_write_video(self, save_intermediate, segment_paths, processed_segments, final_output_path, fps, temp_dir, start_time):
        if save_intermediate and segment_paths:
            self.log("Combining segments and writing final video...")

            # Load all segments and concatenate
            all_frames = []
            for segment_path in segment_paths:
                self.log(f"Loading segment: {os.path.basename(segment_path)}")
                segment = np.load(segment_path)
                all_frames.append(segment)
                # Remove the file after loading
                os.remove(segment_path)

            # Concatenate and write
            self.log(f"Concatenating {len(all_frames)} segments...")
            full_video = np.concatenate(all_frames, axis=0)
            self.log(f"Writing final video ({len(full_video)} frames)...")
            media.write_video(final_output_path, full_video, fps=fps)

            # Clean up temp dir
            os.rmdir(temp_dir)

            self.log(f"Saved output video to: {final_output_path}")
            elapsed_time = time.time() - start_time
            self.log(f"Processing completed in {elapsed_time:.2f} seconds")

            return None, fps  # Return None since we wrote directly to disk

        elif not save_intermediate and processed_segments:
            self.log("Concatenating all processed segments...")
            full_video = np.concatenate(processed_segments, axis=0)

            self.log(f"Saving output video to: {final_output_path}")
            media.write_video(final_output_path, full_video, fps=fps)

            elapsed_time = time.time() - start_time
            self.log(f"\nProcessing completed in {elapsed_time:.2f} seconds")

            return full_video, fps

        else:
            self.log("No video segments were processed")
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            return None, fps


    def convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_serializable(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj


    def combine_and_write_tracks(self, tracks, final_tracks_output_path):
        try:
            tracks_list = self.convert_to_serializable(tracks)
            
            os.makedirs(os.path.dirname(final_tracks_output_path), exist_ok=True)
            
            tracks_data = {
                "num_points": len(tracks_list),
                "num_frames": len(tracks_list[0]) if tracks_list else 0,
                "tracks": tracks_list
            }
            
            # Write to JSON file
            with open(final_tracks_output_path, 'w') as f:
                json.dump(tracks_data, f, indent=2)
                
            self.log(f"Tracks successfully written to {final_tracks_output_path}")
            
        except Exception as e:
            self.log(f"Error writing tracks to file: {e}")
            raise


    def resize_points_add_frame(self, cloud_points, query_frame, height_ratio, width_ratio):
        cloud_points = np.array(
            [[query_frame, point[1] * height_ratio, point[0] * width_ratio] for point in cloud_points],
            dtype=np.float32
        )
        return cloud_points


    def flatten_point_clouds(self, point_clouds: List[PointCloud]):
        cloud_point_lists = [cloud.cloud_points for cloud in point_clouds]
        return [point for cloud_points in cloud_point_lists for point in cloud_points]


    def validate_and_update_weights(
            self, 
            predicted_point_clouds: List[PointCloud], 
            true_point_clouds: List[PointCloud], 
            inliers_rotations: List[tuple[np.ndarray, float]],
            path, 
            filename, 
            end_frame, 
            i, 
            segments_to_process
        ):
        predicted_query_points = [p.query_point for p in predicted_point_clouds]
        true_query_points = [cloud.query_point for cloud in true_point_clouds]
        initial_positions = [cloud.cloud_points for cloud in true_point_clouds]

        confidence = min([p.confidence(inliers) for p, (inliers, _) in zip(predicted_point_clouds, inliers_rotations)])
        request_validation = confidence < CONFIDENCE_THRESHOLD

        self.export_to_point_data_store(predicted_query_points)
        self.send_current_frame_data(
            query_points=predicted_query_points,
            video_path=DATA_DIR + path + filename, 
            frame=end_frame - 1, 
            confidence=confidence, 
            request_validation=request_validation
        )

        weights = [ppc.weights for ppc in predicted_point_clouds]

        if request_validation:
            self.request_validation()  # Query points will have been updateds
            true_query_points = self.point_data_store[self.session_id]["points"]
            true_query_points_xy = np.array([[p["x"], p["y"]] for p in true_query_points], dtype=np.float32)
            weights = self.weight_distance_calculator.calculate_distance_weights(
                predicted_point_clouds=predicted_point_clouds,
                inliers_rotations=inliers_rotations,
                true_query_points=true_query_points_xy,
                initial_positions=initial_positions
            )
            # print("Weights after validation")
            # print(weights)
            final_positions = [ pc.cloud_points for pc in predicted_point_clouds]
            true_point_clouds = self.point_cloud_reconstructor.reconstruct_point_clouds(
                old_point_clouds=predicted_point_clouds,
                final_positions=final_positions,
                inliers_rotations=inliers_rotations,
                query_point_reconstructions=true_query_points_xy,
                weights=weights,
            )
        else:
            true_query_points = predicted_query_points
            true_point_clouds = predicted_point_clouds

        self.export_to_point_data_store(true_query_points)

        # Reconstruct after new query points calculated (retains weights)
        
        return true_query_points, true_point_clouds


    def smooth_points(
            self, 
            start_frame, 
            end_frame, 
            start_query_points, 
            end_query_points, 
            slice_result, 
            point_clouds
        ):

        diff = end_frame - start_frame
        num_point_clouds = len(start_query_points)

        interpolated_points = [[] for _ in range(num_point_clouds)]
        mean_points = [[] for _ in range(num_point_clouds)]

        for f in range(diff):
            a = f / diff
            
            for k, (start_qp, end_qp) in enumerate(zip(start_query_points, end_query_points)):
                interpolated_points[k].append((1 - a) * start_qp + a * end_qp)
            
            points_at_frame = slice_result.get_points_for_frame(
                frame=f, num_qp=num_point_clouds, num_cp_per_qp=len(point_clouds[0].cloud_points)
            )
            for k, points in enumerate(points_at_frame):
                mean_points[k].append(np.mean(points, axis=0))

        interpolated_points = np.array(interpolated_points)
        mean_points = np.array(mean_points)

        # Calculate smoothed points
        smoothed_points = []
        for j, (interpolated_tracks, raw_mean_tracks) in enumerate(zip(interpolated_points, mean_points)):
            half = diff // 2
            smoothed_tracks = []
            
            for k in range(diff):
                if k < half:
                    interpolated_weight = -1/(diff - 1)*k + 1
                else:
                    interpolated_weight =  1/(diff - 1)*k
                
                raw_weight = 1 - interpolated_weight
                smoothed_tracks.append(interpolated_weight * interpolated_tracks[k] + raw_weight * raw_mean_tracks[k])
            smoothed_points.append(smoothed_tracks)
        smoothed_points = np.array(smoothed_points)
        return smoothed_points


    def calc_errors(self, predicted_point_clouds: List[PointCloud], true_point_clouds: List[PointCloud], frame: int):
        intra_cloud_metrics = []
        
        cloud_index = 0
        # Intra-Cloud Metrics
        for pred, cloud in zip(predicted_point_clouds, true_point_clouds):
            p = pred.query_point_array()  # Predicted query point
            v = cloud.query_point_array()   # Validated query point
            
            c_p = np.array(pred.cloud_points)  # Predicted cloud points
            c_v = np.array(cloud.cloud_points)      # Validated cloud points
            
            # 1. Query Point Distance: Euclidean distance between prediction and validation
            euclidean_distance = np.linalg.norm(p - v)
            
            # 2. Query Point Vector: Direction of prediction error
            vector = v - p
            
            # 3. Cloud Point Distances: Distance for each predicted vs validated cloud point
            cloud_point_distances = np.linalg.norm(c_v - c_p, axis=1)
            
            # 4. Deformity: Sum of all cloud point distances
            deformity = np.sum(cloud_point_distances)
            
            # 5. Deformity Median
            deformity_median = np.median(cloud_point_distances)
            
            # 6. Inliers and Outliers (using a radius parameter)
            radius = cloud.radius  # This can be made configurable
            distances_from_validated = np.linalg.norm(c_p - v, axis=1)
            inliers = np.sum(distances_from_validated < radius)
            outliers = len(distances_from_validated) - inliers
            
            # Store individual cloud results
            intra_cloud_metrics.append(
                {
                    "cloud_index": cloud_index,
                    "euclidean_distance": euclidean_distance,
                    "vector_true_to_pred": vector.tolist(),
                    "cloud_point_distances": cloud_point_distances.tolist(),
                    "deformity": deformity,
                    "deformity_median": deformity_median,
                    "inliers": inliers,
                    "outliers": outliers,
                    "inlier_ratio": inliers / len(distances_from_validated) if len(distances_from_validated) > 0 else 0,
                }
            )
            cloud_index += 1
        
        # Inter-Cloud Metrics
        inter_cloud_metrics = {}
        if len(predicted_point_clouds) > 1:
            p_points = np.array([pred.query_point_array() for pred in predicted_point_clouds])
            v_points = np.array([cloud.query_point_array() for cloud in true_point_clouds])
            
            d_p = np.zeros((len(p_points), len(p_points)))
            d_v = np.zeros((len(v_points), len(v_points)))
            
            # Calc normalized direction vectors
            e_p = {}
            e_v = {}
            
            for i in range(len(p_points)):
                for j in range(i+1, len(p_points)):
                    # Distances
                    d_p[i, j] = d_p[j, i] = np.linalg.norm(p_points[i] - p_points[j])
                    d_v[i, j] = d_v[j, i] = np.linalg.norm(v_points[i] - v_points[j])
                    
                    # Direction vectors (normalized)
                    p_dir = p_points[i] - p_points[j]
                    p_norm = np.linalg.norm(p_dir)
                    e_p[(i, j)] = p_dir / p_norm if p_norm > 0 else np.zeros(2)
                    
                    v_dir = v_points[i] - v_points[j]
                    v_norm = np.linalg.norm(v_dir)
                    e_v[(i, j)] = v_dir / v_norm if v_norm > 0 else np.zeros(2)
            
            # 1. Change in distances
            delta_distances = {}
            for i in range(len(p_points)):
                for j in range(i+1, len(p_points)):
                    delta_distances[f"{i}-{j}"] = float(abs(d_v[i, j] - d_p[i, j]))
            
            # 2. Changes in directions (angle between vectors)
            theta_changes = {}
            for (i, j) in e_p.keys():
                # Calculate dot product for angle
                dot_product = np.dot(e_v[(i, j)], e_p[(i, j)])
                # Clip to handle floating point errors
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = float(np.arccos(dot_product))
                theta_changes[f"{i}-{j}"] = angle
                
            inter_cloud_metrics = {
                "delta_distances": delta_distances,
                "theta_changes": theta_changes,
                "mean_delta_distance": float(np.mean(list(delta_distances.values()))) if delta_distances else 0,
                "mean_theta_change": float(np.mean(list(theta_changes.values()))) if theta_changes else 0
            }
        
        return {
            "frame": frame,
            "intra_cloud_metrics": intra_cloud_metrics,
            "inter_cloud_metrics": inter_cloud_metrics
        }


    def write_errors(self, errors, final_errors_output_path):
        os.makedirs(os.path.dirname(os.path.abspath(final_errors_output_path)), exist_ok=True)
        
        def serialize_error(error):
            result = {}
            
            result["frame"] = int(error["frame"]) if isinstance(error["frame"], np.integer) else error["frame"]
            
            result["intra_cloud_metrics"] = []
            for metric in error["intra_cloud_metrics"]:
                processed_metric = {}
                for key, value in metric.items():
                    if isinstance(value, np.ndarray):
                        processed_metric[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        processed_metric[key] = float(value) if isinstance(value, np.floating) else int(value)
                    else:
                        processed_metric[key] = value
                result["intra_cloud_metrics"].append(processed_metric)
            
            result["inter_cloud_metrics"] = {}
            for key, value in error["inter_cloud_metrics"].items():
                if isinstance(value, dict):
                    processed_dict = {}
                    for k, v in value.items():
                        processed_dict[k] = float(v) if isinstance(v, np.floating) else v
                    result["inter_cloud_metrics"][key] = processed_dict
                elif isinstance(value, (np.integer, np.floating)):
                    result["inter_cloud_metrics"][key] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    result["inter_cloud_metrics"][key] = value
            
            return result
        
        serializable_errors = [serialize_error(error) for error in errors]
        
        with open(final_errors_output_path, 'w') as f:
            json.dump(serializable_errors, f, indent=2)

        self.log(f"Errors successfully written to {final_errors_output_path}")


    def process_video(self, query_points):
        # Determine FPS from our lookup, default to 30 if not found
        filename = self.video["filename"]
        path = self.video["path"]
        fps = self.video.get("fps", 30)

        self.log(f"Processing video: {filename} at {fps} FPS with predefined points {query_points}")
        self.log(f"Path: {path}")

        try:

            max_segments = None
            save_intermediate=True

            self.log(f"Processing video from: {path}")
            start_time = time.time()

            self.log("Reading video frames...")
            orig_frames = media.read_video(DATA_DIR + path + filename)
            height, width = orig_frames.shape[1:3]
            total_frames = len(orig_frames)
            
            # Normalize FPS to a maximum of 15
            # normalized_fps = min(fps, 15)
            # if normalized_fps < fps:
            #     self.log(f"Normalizing FPS from {fps} to {normalized_fps}")
            #     # Calculate the frame sampling interval
            #     sampling_interval = fps // normalized_fps
            #     # Sample frames at the calculated interval
            #     orig_frames = orig_frames[::sampling_interval]
            #     total_frames = len(orig_frames)
            #     # Update fps to the normalized value
            #     fps = normalized_fps
            
            self.log(f"Video loaded: {total_frames} frames at {fps} FPS, resolution: {width}x{height}")

            # Calculate how many segments we need to process
            total_segments = (total_frames + fps - 1) // fps  # Ceiling division
            if max_segments is not None:
                segments_to_process = min(total_segments, max_segments)
            else:
                segments_to_process = total_segments
            # TODO: Remove
            segments_to_process = NUM_SLICES

            self.log(f"Video will be processed in {segments_to_process} segments")

            # Create temp directory for intermediate results if needed
            temp_dir = None
            segment_paths = []

            processed_segments = []  # If not saving intermediate results, store in memory
            if save_intermediate:
                temp_dir = tempfile.mkdtemp(prefix="video_segments_")
                self.log(f"Using temporary directory for intermediate results: {temp_dir}")                

            
            # Define query points for slice
            resize_height = 256
            resize_width = 256
            query_frame = 0
            # stride = 8

            height_ratio = resize_height / height
            width_ratio = resize_width / width
            
            point_clouds: List[PointCloud] = self.point_cloud_generator.generate_initial_point_clouds(query_points) # N x 2 (for N points)
            inter_point_cloud_matrix = np.array([])
            all_tracks = [[] for _ in range(len(point_clouds))]
            all_errors = []
            # inliers = [True] * (len(point_clouds) * len(point_clouds[0].cloud_points))

            # Process each segment
            for i in range(segments_to_process):
                start_frame = i * fps
                end_frame = min((i + 1) * fps, total_frames)

                self.log(f"Processing segment {i + 1}/{segments_to_process} (frames {start_frame} to {end_frame})...")
                orig_frames_slice = orig_frames[start_frame:end_frame]

                # Process the slice
                try:
                    start_query_points = np.array([pc.query_point_array() for pc in point_clouds], dtype=np.float32)
                    for pc_i, pc in enumerate(point_clouds):
                        self.log(f"Cloud {pc_i} deformity at beginning: {pc.deformity()}")
                    # Flatten point cloud for processing
                    flattened_points = self.flatten_point_clouds(point_clouds)
                    resized_points = self.resize_points_add_frame(
                        cloud_points=flattened_points,
                        query_frame=query_frame,
                        height_ratio=height_ratio,
                        width_ratio=width_ratio,
                    )

                    # Estimate points to generate slice result  
                    slice_result: EstimationSlice = self.point_cloud_estimator.process_video_slice(
                        orig_frames_slice,
                        width,
                        height,
                        resized_points,
                        resize_width=resize_width,
                        resize_height=resize_height,
                    )
                    # slice_result: EstimationSlice = self.point_cloud_estimator.process_cached_dance_15(
                    #     orig_frames_slice
                    # )

                    final_positions = slice_result.get_points_for_frame(
                        frame=-1, 
                        num_qp=len(point_clouds), 
                        num_cp_per_qp=len(point_clouds[0].cloud_points)
                    )
                    
                    # Experimental Predictions
                    inliers_rotations: List[tuple[np.ndarray, float]] = self.inlier_predictor.predict_inliers_rotations(
                        old_point_clouds=point_clouds, 
                        final_positions=final_positions)
                    query_point_reconstructions: List[np.ndarray] = self.query_point_reconstructor.reconstruct_query_points(
                        old_point_clouds=point_clouds,
                        final_positions=final_positions,
                        inliers_rotations=inliers_rotations
                    )
                    aligned_query_point_reconstructions: List[np.ndarray] = self.inter_cloud_alignment_predictor.align_query_points(
                        query_point_reconstructions=query_point_reconstructions,
                        inter_point_cloud_matrix=inter_point_cloud_matrix
                    )
                    weights = self.weight_outlier_calculator.calculate_outlier_weights(
                        old_point_clouds=point_clouds,
                        inliers_rotations=inliers_rotations
                    )
                    # self.log("Weights after updating with outliers:")
                    # self.log(weights)
                    predicted_point_clouds: List[PointCloud] = self.point_cloud_reconstructor.reconstruct_point_clouds(
                        old_point_clouds=point_clouds,
                        final_positions=final_positions,
                        inliers_rotations=inliers_rotations,
                        query_point_reconstructions=aligned_query_point_reconstructions,
                        weights=weights
                    )
                    for pc_i, pc in enumerate(predicted_point_clouds):
                        self.log(f"Predicted cloud {pc_i} deformity pre-validation: {pc.deformity()}")

                    # Update weights and potentially request validation
                    query_points, point_clouds = self.validate_and_update_weights(
                        predicted_point_clouds=predicted_point_clouds, 
                        true_point_clouds=point_clouds,
                        inliers_rotations=inliers_rotations,
                        path=path, 
                        filename=filename,
                        end_frame=end_frame,
                        i=i,
                        segments_to_process=segments_to_process
                    )

                    # Get query points for interpolation
                    end_query_points = np.array([pc.query_point_array() for pc in point_clouds], dtype=np.float32)
                    
                    smoothed_points = self.smooth_points(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_query_points=start_query_points,
                        end_query_points=end_query_points,
                        slice_result=slice_result,
                        point_clouds=point_clouds
                    )

                    slice_errors = self.calc_errors(
                        predicted_point_clouds=predicted_point_clouds, 
                        true_point_clouds=point_clouds, 
                        frame=end_frame
                    )
                    all_errors.append(slice_errors)                    

                    video_segment = slice_result.get_video()
                    # inliers_masks = [b for mask, _ in inliers_rotations for b in mask]
                    # inliers = [a and b for a, b in zip(inliers, inliers_masks)]
                    # video_segment = slice_result.get_video_for_points(interpolated_points)
                    # video_segment = slice_result.get_video_for_points(mean_points)
                    # video_segment = slice_result.get_video_for_points(mean_and_interpolated)
                    smoothed_video_segment = slice_result.get_video_for_points(smoothed_points)
                    for j, sps in enumerate(smoothed_points):
                        all_tracks[j].extend(sps)

                    self.send_timeline_frames(smoothed_video_segment)

                    if save_intermediate:
                        # Save segment to disk
                        segment_path = os.path.join(temp_dir, f"segment_{i:04d}.npy")
                        np.save(segment_path, video_segment)
                        segment_paths.append(segment_path)
                        # Free memory
                        del video_segment
                        gc.collect()
                    else:
                        # Store in memory
                        processed_segments.append(video_segment)

                    self.log(f"Successfully processed segment {i + 1}")
                except Exception as e:
                    self.log(f"Error processing segment {i + 1}: {str(e)}")
                    import traceback

                    self.log(traceback.format_exc())

            # Prepare final video
            final_video_output_path = OUTPUT_DIR + "POINT_CLOUD_" + filename
            name, _ = filename.split(".")
            final_tracks_output_path = OUTPUT_DIR + "TRACKS_" + name + ".txt"
            final_errors_output_path = OUTPUT_DIR + "ERRORS_" + name + ".json"

            self.combine_and_write_video(
                save_intermediate=save_intermediate,
                segment_paths=segment_paths,
                processed_segments=processed_segments,
                final_output_path=final_video_output_path,
                fps=fps,
                temp_dir=temp_dir,
                start_time=start_time
            )

            self.combine_and_write_tracks(
                tracks=all_tracks,
                final_tracks_output_path=final_tracks_output_path
            )

            self.write_errors(
                errors=all_errors, 
                final_errors_output_path=final_errors_output_path
            )

            self.log(f"Processing completed successfully for {filename}")

            return {"success": True, "output_filename": final_video_output_path, "fps": fps}

        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            error_message = f"Error processing video: {str(e)}"
            self.log(error_message)
            self.log(stack_trace)
            return {"success": False, "error": error_message}

