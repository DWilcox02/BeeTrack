import os
import json
import time
import mediapy as media
import tempfile
import numpy as np
import gc
import cv2
import base64
import copy

from threading import Event
from typing import List

from .server.utils.video_utils import extract_frame

from .point_cloud.estimation.estimation_slice import EstimationSlice
from .point_cloud.point_cloud import PointCloud
from src.backend.utils.frontend_communicator import FrontendCommunicator
from src.backend.utils.processing_configuration import ProcessingConfiguration
from src.backend.utils.rotation_calculator import calculate_rotation_deformity_predictions
from src.backend.utils.confidence_helper import cloud_confidence
from src.backend.utils.json_writer import JSONWriter

# Component Abstractions
from src.backend.point_cloud.point_cloud_generator import PointCloudGenerator
from src.backend.point_cloud.estimation.point_cloud_estimator_interface import PointCloudEstimatorInterface
from src.backend.inlier_predictors.inlier_predictor_base import InlierPredictorBase
from src.backend.query_point_predictors.query_point_reconstructor_base import QueryPointReconstructorBase
from src.backend.point_cloud_reconstructors.point_cloud_reconstructor_base import PointCloudReconstructorBase
from src.backend.weight_calculators.weight_calculator_outliers_base import WeightCalculatorOutliersBase
from src.backend.weight_calculators.weight_calculator_distances_base import WeightCalculatorDistancesBase


# Get paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BACKEND_DIR)
POINT_CLOUD_DIR = os.path.join(BACKEND_DIR, "point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))


class VideoProcessor():
    def __init__(
        self,
        session_id,
        point_data_store,
        frontend_communicator: FrontendCommunicator,
        processing_configuration: ProcessingConfiguration, 
        video,
        job_id,
        stop_event: Event
    ):
        self.session_id = session_id
        self.frontend_communicator = frontend_communicator
        self.video = video
        self.job_id = job_id
        self.log_message = None
        self.point_data_store = point_data_store
        self.confidence_threshold = processing_configuration.confidence_threshold
        self.smoothing_alpha = processing_configuration.smoothing_alpha
        self.deformity_delta = processing_configuration.deformity_delta
        self.num_slices = processing_configuration.processing_seconds
        self.stop_event = stop_event

        self.point_cloud_estimator: PointCloudEstimatorInterface = processing_configuration.point_cloud_estimator
        self.point_cloud_generator: PointCloudGenerator = processing_configuration.point_cloud_generator
        self.inlier_predictor: InlierPredictorBase = processing_configuration.inlier_predictor
        self.query_point_reconstructor: QueryPointReconstructorBase = processing_configuration.query_point_reconstructor
        self.weight_calculator_outliers: WeightCalculatorOutliersBase = (
            processing_configuration.weight_calculator_outliers
        )
        self.point_cloud_non_validated_reconstructor: PointCloudReconstructorBase = (
            processing_configuration.point_cloud_non_validated_reconstructor
        )
        self.point_cloud_validated_reconstructor: PointCloudReconstructorBase = (
            processing_configuration.point_cloud_validated_reconstructor
        )        
        self.weight_calculator_distance: WeightCalculatorDistancesBase = (
            processing_configuration.weight_calculator_distances
        )

        self.json_writer = JSONWriter()


    def set_log_message_function(self, fn):
        self.log_message = fn
        self.point_cloud_estimator.set_logger(self.log)
        self.point_cloud_generator.set_logger(self.log)
        self.inlier_predictor.set_logger(self.log)
        self.point_cloud_non_validated_reconstructor.set_logger(self.log)
        self.query_point_reconstructor.set_logger(self.log)
        self.weight_calculator_distance.set_logger(self.log)
        self.weight_calculator_outliers.set_logger(self.log)
        self.json_writer.set_logger(self.log)


    def log(self, message):
        if self.job_id and self.log_message:
            self.log_message(self.job_id, message)
        print(message)
    
    
    def export_to_point_data_store(self, points):
        self.point_data_store[self.session_id]["points"] = copy.deepcopy(points)


    def send_current_frame_data(self, query_points, video_path, frame, confidence, request_validation):
        frame_base64, error, width, height = extract_frame(video_path, frame)
        frameData = {"frame": frame_base64, "width": width, "height": height, "frame_idx": frame}
        self.frontend_communicator.send_frame_data_callback(frameData, query_points, confidence, request_validation)


    def request_validation(self):
        return self.frontend_communicator.request_validation_callback(self.job_id)


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
                self.frontend_communicator.send_timeline_frame_callback(frame_base64, i)
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
            current_point_clouds: List[PointCloud], 
            predicted_point_clouds: List[PointCloud], 
            inliers: List[np.ndarray[bool]],
            deformities: List[float],
            path, 
            filename, 
            end_frame, 
        ):
        predicted_query_points = [copy.deepcopy(p.query_point) for p in predicted_point_clouds]
        current_query_points = [cloud.query_point for cloud in current_point_clouds]

        confidences = [
            cloud_confidence(
                inliers=ils,
                deformity=d,
                deformity_delta=self.deformity_delta,
                radius=ppc.radius,
                num_cloud_points=len(ppc.cloud_points),
            )
            for ppc, ils, d in zip(predicted_point_clouds, inliers, deformities)
        ]
        confidence = min(confidences)
        for ppc, c in zip(predicted_point_clouds, confidences):
            self.log(f"{ppc.get_cloud_label()} confidence: {round(c, 2)}")

        request_validation = confidence <= self.confidence_threshold

        self.export_to_point_data_store(predicted_query_points)
        self.send_current_frame_data(
            query_points=predicted_query_points,
            video_path=DATA_DIR + path + filename, 
            frame=end_frame - 1, 
            confidence=confidence, 
            request_validation=request_validation
        )

        # User validation
        if request_validation:
            # Threshold reached, re-construct based on validation
            self.request_validation()  # Query points will have been updated
            current_query_points = self.point_data_store[self.session_id]["points"]
            current_query_points_xy = np.array([[p["x"], p["y"]] for p in current_query_points], dtype=np.float32)
            
            new_point_clouds: List[PointCloud] = []
            for predicted_point_cloud, true_query_point in zip(predicted_point_clouds, current_query_points_xy):
                updated_weights_distances = self.weight_calculator_distance.calculate_distance_weights(
                    predicted_point_cloud=predicted_point_cloud,
                    true_query_point=true_query_point
                )
                cloud_inliers = np.array([True] * len(predicted_point_cloud.cloud_points), dtype=bool)
                new_point_cloud = self.point_cloud_validated_reconstructor.reconstruct_point_cloud(
                    old_point_cloud=predicted_point_cloud,
                    final_positions=predicted_point_cloud.cloud_points,
                    inliers=cloud_inliers,
                    rotation=predicted_point_cloud.rotation,
                    query_point_reconstruction=true_query_point,
                    weights=updated_weights_distances,
                )
                new_point_clouds.append(new_point_cloud)
            current_point_clouds = new_point_clouds
        else:
            # Sufficiently confidence, accept current predictions
            current_query_points = predicted_query_points
            current_point_clouds = predicted_point_clouds

        self.export_to_point_data_store(current_query_points)
        self.add_validation()
        
        return current_query_points, current_point_clouds


    def should_stop(self):
        """Check if processing should stop."""
        return self.stop_event is not None and self.stop_event.is_set()


    def generate_segment_tracks(
        self,
        start_frame: int,
        end_frame: int,
        start_cloud_points: np.ndarray[np.float32],  # query points, not cloud points
        end_cloud_points: np.ndarray[np.float32],  # query points, not cloud points
        slice_result: EstimationSlice,
        point_clouds: List[PointCloud],
        inliers: List[np.ndarray[bool]],
    ):
        diff = end_frame - start_frame
        num_point_clouds = len(start_cloud_points)

        interpolated_points = [[] for _ in range(num_point_clouds)]
        mean_points = [[] for _ in range(num_point_clouds)]

        for f in range(diff):
            a = f / diff

            for k, (start_qp, end_qp) in enumerate(zip(start_cloud_points, end_cloud_points)):
                interpolated_points[k].append((1 - a) * start_qp + a * end_qp)

            points_at_frame = slice_result.get_points_for_frame(
                frame=f, lengths=[len(pc.cloud_points) for pc in point_clouds]
            )

            # Calculate weighted mean using inliers and weights
            for k, points in enumerate(points_at_frame):
                # Ensure points is a numpy array
                points = np.array(points)
                current_inliers = inliers[k]
                current_weights = point_clouds[k].weights

                # Make sure we have matching lengths
                if len(current_inliers) > len(points):
                    current_inliers = current_inliers[: len(points)]
                elif len(current_inliers) < len(points):
                    # Pad with False if needed
                    current_inliers = np.concatenate(
                        [current_inliers, np.array([False] * (len(points) - len(current_inliers)))]
                    )

                # Filter to only include inliers
                if len([x for x in current_inliers if x]) < 2:
                    # If very few inliers, just take weighted average of all
                    current_inliers = np.array([True] * len(points), dtype=bool)

                inlier_points = points[current_inliers]
                inlier_weights = current_weights[current_inliers]

                weight_sum = np.sum(inlier_weights)
                normalized_weights = inlier_weights / weight_sum

                weighted_avg = np.array([0.0, 0.0])
                for point, weight in zip(inlier_points, normalized_weights):
                    weighted_avg += weight * point

                mean_points[k].append(weighted_avg)

        interpolated_points = np.array(interpolated_points)
        mean_points = np.array(mean_points)

        # Calculate smoothed points
        smoothed_points = []
        for j, (interpolated_tracks, raw_mean_tracks) in enumerate(zip(interpolated_points, mean_points)):
            half = diff // 2
            smoothed_tracks = []

            for k in range(diff):
                if k < half:
                    interpolated_weight = (-1 / (diff - 1) * k + 1) ** self.smoothing_alpha
                else:
                    interpolated_weight = (1 / (diff - 1) * k) ** self.smoothing_alpha

                raw_weight = 1 - interpolated_weight
                smoothed_tracks.append(interpolated_weight * interpolated_tracks[k] + raw_weight * raw_mean_tracks[k])
            smoothed_points.append(smoothed_tracks)
        smoothed_points = np.array(smoothed_points)
        return interpolated_points, mean_points, smoothed_points


    def calc_errors(
            self, 
            predicted_point_clouds: List[PointCloud], 
            true_point_clouds: List[PointCloud], 
            frame: int,
            inliers: List[np.ndarray[bool]],
            deformities: List[float],
            initial_positions: List[np.ndarray[np.float32]],
            final_positions: List[np.ndarray[np.float32]],
            vectors_qp_to_cp: List[np.ndarray[np.float32]]
    ):
        intra_cloud_metrics = []
        
        for cloud_index in range(len(predicted_point_clouds)):
            pred = predicted_point_clouds[cloud_index]
            cloud = true_point_clouds[cloud_index]
            ils = inliers[cloud_index]
            d = deformities[cloud_index]
            init_pos = initial_positions[cloud_index]
            final_pos = final_positions[cloud_index]
            vec_qp_to_cp = vectors_qp_to_cp[cloud_index]
            recons_pos = cloud.cloud_points

            p = pred.query_point_array()  # Predicted query point
            v = cloud.query_point_array()   # Validated query point
                        
            # Euclidean Distance
            euclidean_distance = np.linalg.norm(p - v)
            
            # Query Point Vector: Direction of prediction error
            vector = v - p
                                                            
            # Store individual cloud results
            intra_cloud_metrics.append(
                {
                    "cloud_index": cloud_index,
                    "predicted_query_point": p,
                    "validated_query_point": v,
                    "euclidean_distance": euclidean_distance,
                    "vector_true_to_pred": vector.tolist(),
                    "initial_positions": init_pos,
                    "final_positions": final_pos,
                    "reconstructed_positions": recons_pos,
                    "vectors_qp_to_cp": vec_qp_to_cp,
                    "deformity": d,
                    "inliers": ils,
                    "inlier_ratio": len([x for x in ils if x]) / len(init_pos),
                }
            )
        
        return {
            "frame": frame,
            "intra_cloud_metrics": intra_cloud_metrics,
        }


    def add_tracks(
        self, 
        start_frame: int,
        interpolated_tracks: np.ndarray, 
        raw_mean_tracks: np.ndarray, 
        smoothed_tracks: np.ndarray
    ):
        new_tracks = []
        
        num_qp, segment_size, _ = interpolated_tracks.shape
        
        track_data = [
            (interpolated_tracks, "interpolated"),
            (raw_mean_tracks, "raw_mean"), 
            (smoothed_tracks, "smoothed")
        ]
        
        for tracks, track_type in track_data:
            for point_idx in range(num_qp):
                for frame_idx in range(segment_size):
                    x, y = tracks[point_idx, frame_idx]
                    
                    track_dict = {
                        "frame": start_frame + frame_idx,
                        "x": float(x),
                        "y": float(y),
                        "bodypart": f"point_{point_idx}_{track_type}"
                    }
                    new_tracks.append(track_dict)
        
        self.frontend_communicator.add_tracks_callback(new_tracks)


    def add_validation(self):
        validated_points = self.point_data_store[self.session_id]["points"]
        self.frontend_communicator.add_validation_callback(validation=validated_points)


    def process_video(self, query_points):
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
                        
            self.log(f"Video loaded: {total_frames} frames at {fps} FPS, resolution: {width}x{height}")

            # Calculate how many segments to process
            total_segments = (total_frames + fps - 1) // fps 
            if max_segments is not None:
                segments_to_process = min(total_segments, max_segments)
            else:
                segments_to_process = total_segments

            segments_to_process = self.num_slices

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

            height_ratio = resize_height / height
            width_ratio = resize_width / width
            
            point_clouds: List[PointCloud] = self.point_cloud_generator.generate_initial_point_clouds(query_points) # N x 2 (for N points)
            all_tracks = [[] for _ in range(len(point_clouds))]
            all_errors = []
            self.add_validation()

            # Process each segment
            for i in range(segments_to_process):
                if self.should_stop():
                    self.log("Processing stopped by user request")
                    return {"success": False, "stopped": True, "message": "Processing stopped by user"}

                start_frame = i * fps
                end_frame = min((i + 1) * fps, total_frames)

                self.log(f"Processing segment {i + 1}/{segments_to_process} (frames {start_frame} to {end_frame})...")
                orig_frames_slice = orig_frames[start_frame:end_frame]

                # Process the slice
                try:
                    if self.should_stop():
                        self.log("Processing stopped by user request")
                        return {"success": False, "stopped": True, "message": "Processing stopped by user"}

                    start_query_points = np.array([pc.query_point_array() for pc in point_clouds], dtype=np.float32)
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
                        stop_event=self.stop_event,
                        resize_width=resize_width,
                        resize_height=resize_height,
                    )

                    if self.should_stop():
                        self.log("Processing stopped by user request")
                        return {"success": False, "stopped": True, "message": "Processing stopped by user"}
                    
                    final_positions = slice_result.get_points_for_frame(
                        frame=-1, lengths=[len(pc.cloud_points) for pc in point_clouds]
                    )

                    # Perform predictions and calculations
                    predicted_point_clouds: List[PointCloud] = []
                    deformities: List[float] = []
                    inliers: List[np.ndarray[bool]] = []
                    for old_point_cloud, final_cloud_points in zip(point_clouds, final_positions):
                        r_d_fps: tuple[float, float, np.ndarray] = calculate_rotation_deformity_predictions(
                            old_point_cloud=old_point_cloud,
                            final_positions=final_cloud_points
                        )
                        rotation: float = r_d_fps[0]
                        deformity: float = r_d_fps[1]
                        final_predictions: np.ndarray = r_d_fps[2]
                        cloud_inliers = self.inlier_predictor.predict_inliers(
                            old_point_cloud=old_point_cloud,
                            final_predictions=final_predictions
                        )
                        query_point_reconstruction: np.ndarray = self.query_point_reconstructor.reconstruct_query_point(
                            point_cloud=old_point_cloud,
                            final_predictions=final_predictions,
                            inliers=cloud_inliers
                        )
                        updated_weights_outliers = self.weight_calculator_outliers.calculate_outlier_weights(
                            old_point_cloud=old_point_cloud,
                            inliers=cloud_inliers
                        )
                        predicted_point_cloud: PointCloud = (
                            self.point_cloud_non_validated_reconstructor.reconstruct_point_cloud(
                                old_point_cloud=old_point_cloud,
                                final_positions=final_cloud_points,
                                inliers=cloud_inliers,
                                rotation=rotation,
                                query_point_reconstruction=query_point_reconstruction,
                                weights=updated_weights_outliers,
                            )
                        )
                        predicted_point_clouds.append(predicted_point_cloud)
                        deformities.append(deformity)
                        inliers.append(cloud_inliers)

                    # Save info for errors
                    all_initial_positions = slice_result.get_points_for_frame(
                        frame=0, lengths=[len(pc.cloud_points) for pc in point_clouds]
                    )
                    all_vectors_qp_to_cp = [ppc.vectors_qp_to_cp for ppc in predicted_point_clouds]

                    # Update weights and potentially request validation
                    query_points, point_clouds = self.validate_and_update_weights(
                        current_point_clouds=point_clouds,
                        predicted_point_clouds=predicted_point_clouds, 
                        inliers=inliers,
                        deformities=deformities,
                        path=path, 
                        filename=filename,
                        end_frame=end_frame,
                    )

                    # Get query points for interpolation
                    end_query_points = np.array([pc.query_point_array() for pc in point_clouds], dtype=np.float32)
                    
                    interpolated_tracks, raw_mean_tracks, smoothed_tracks = self.generate_segment_tracks(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_cloud_points=start_query_points,
                        end_cloud_points=end_query_points,
                        slice_result=slice_result,
                        point_clouds=point_clouds,
                        inliers=inliers
                    )

                    slice_errors = self.calc_errors(
                        predicted_point_clouds=predicted_point_clouds,
                        true_point_clouds=point_clouds,
                        frame=end_frame,
                        inliers=inliers,
                        deformities=deformities,
                        initial_positions=all_initial_positions,
                        final_positions=final_positions,
                        vectors_qp_to_cp=all_vectors_qp_to_cp,
                    )
                    all_errors.append(slice_errors)                    

                    video_segment = slice_result.get_video()
                    # inliers_masks = [b for mask, _ in inliers_rotations for b in mask]
                    # inliers = [a and b for a, b in zip(inliers, inliers_masks)]
                    # video_segment = slice_result.get_video_for_points(interpolated_points)
                    # video_segment = slice_result.get_video_for_points(mean_points)
                    # video_segment = slice_result.get_video_for_points(mean_and_interpolated)
                    smoothed_video_segment = slice_result.get_video_for_points(smoothed_tracks)
                    for j, sps in enumerate(smoothed_tracks):
                        all_tracks[j].extend(sps)

                    self.add_tracks(
                        start_frame=start_frame,
                        interpolated_tracks=interpolated_tracks,
                        raw_mean_tracks=raw_mean_tracks, 
                        smoothed_tracks=smoothed_tracks
                    )
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

            if self.should_stop():
                self.log("Processing stopped by user request")
                return {"success": False, "stopped": True, "message": "Processing stopped by user"}

            # Prepare final video
            filename_output_path = "POINT_CLOUD_" + filename
            final_video_output_path = OUTPUT_DIR + filename_output_path
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

            self.json_writer.combine_and_write_tracks(
                tracks=all_tracks, final_tracks_output_path=final_tracks_output_path
            )

            self.json_writer.write_errors(
                errors=all_errors, 
                final_errors_output_path=final_errors_output_path
            )

            self.log(f"Processing completed successfully for {filename}")

            return {"success": True, "output_filename": filename_output_path, "fps": fps}

        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            error_message = f"Error processing video: {str(e)}"
            self.log(error_message)
            self.log(stack_trace)
            return {"success": False, "error": error_message}

