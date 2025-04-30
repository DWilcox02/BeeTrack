import os
import importlib.util
import json
import time
import mediapy as media
import tempfile
import numpy as np
import gc
from .point_cloud_interface import PointCloudInterface, BeeSkeleton


# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.dirname(BACKEND_DIR)
POINT_CLOUD_DIR = os.path.join(BACKEND_DIR, "point_cloud/")
TAPIR_DIR = os.path.join(POINT_CLOUD_DIR, "TAPIR_point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Import tapir_point_cloud module
spec = importlib.util.spec_from_file_location("tapir_point_cloud", os.path.join(TAPIR_DIR, "tapir_point_cloud.py"))
tapir_point_cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tapir_point_cloud)

# Import video_utils
spec = importlib.util.spec_from_file_location("video_utils", os.path.join(BACKEND_DIR, "server/utils/video_utils.py"))
video_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_utils)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))


NUM_SLICES = 3
CONFIDENCE_THRESHOLD = 0.8

class VideoProcessor():
    def __init__(
        self,
        session_id: str,
        point_data_store: dict,
        point_cloud: PointCloudInterface,
        send_frame_data_callback, 
        request_validation_callback,
        job_id=None,
    ):
        self.session_id = session_id
        self.point_data_store = point_data_store
        self.point_cloud = point_cloud
        self.send_frame_data_callback = send_frame_data_callback
        self.request_validation_callback = request_validation_callback
        self.job_id = job_id
        self.log_message = None



    def set_log_message_function(self, fn):
        self.log_message = fn


    def log(self, message):
        if self.job_id and self.log_message:
            self.log_message(self.job_id, message)
        print(message)


    def get_points(self):
        return self.point_data_store[self.session_id]["points"]


    def set_points(self, new_points):
        self.point_data_store[self.session_id]["points"] = new_points


    def send_current_frame_data(self, video_path, frame, confidence, request_validation):
        frame_base64, error, width, height = video_utils.extract_frame(video_path, frame)
        frameData = {"frame": frame_base64, "width": width, "height": height, "frame_idx": frame}
        self.send_frame_data_callback(frameData, self.get_points(), confidence, request_validation)


    def request_validation(self):
        return self.request_validation_callback()


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
            self.log(f"\nProcessing completed in {elapsed_time:.2f} seconds")

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


    def convert_select_points_to_query_points(self, query_frame, points, height_ratio, width_ratio):
        """Convert select points to query points with linear interpolation.

        Args:
        points (json): [{'x': _, 'y': _}, ...]

        Returns:
        query_points: [num_points, 3], in [t, y, x]
        """
        points_array = np.array([(point["x"], point["y"]) for point in points], dtype=np.float32)

        # For quadrilateral interpolation, we need the 4 corners in a specific order
        # Assuming points form a quadrilateral with 4 points
        if len(points_array) != 4:
            raise ValueError("Expected exactly 4 points for area interpolation")

        # Define the corners - this assumes the points are a quadrilateral
        # We need a consistent ordering for the bilinear interpolation
        # Sort points by their position (e.g., top-left, top-right, bottom-right, bottom-left)
        # This is a simple approach - for complex shapes, more sophisticated ordering might be needed
        center = np.mean(points_array, axis=0)
        angles = np.arctan2(points_array[:, 1] - center[1], points_array[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        quad_points = points_array[sorted_indices]

        # Number of subdivisions along each dimension
        n_subdivs = 5  # This will create approximately 25 points across the area

        # Generate a grid of points across the entire area
        area_points = []

        for i in range(n_subdivs):
            for j in range(n_subdivs):
                # Parameters for bilinear interpolation
                u = i / (n_subdivs - 1)
                v = j / (n_subdivs - 1)

                # Bilinear interpolation formula
                # P(u,v) = (1-u)(1-v)P00 + u(1-v)P10 + (1-u)vP01 + uvP11
                point = (
                    (1 - u) * (1 - v) * quad_points[0]
                    + u * (1 - v) * quad_points[1]
                    + u * v * quad_points[2]
                    + (1 - u) * v * quad_points[3]
                )

                area_points.append(point)

        # Convert interpolated points to numpy array
        area_points = np.array(area_points, dtype=np.float32)

        # Create the query points
        query_points = np.zeros(shape=(len(area_points), 3), dtype=np.float32)
        query_points[:, 0] = query_frame
        query_points[:, 1] = area_points[:, 1] * height_ratio  # y
        query_points[:, 2] = area_points[:, 0] * width_ratio  # x

        return query_points


    def recalculate_query_points(
        self,
        point_cloud_slice,
        bee_skeleton,
        query_frame,
        height_ratio,
        width_ratio,
        previous_trajectory,
    ):
        # Get new midpoint and trajectory
        midpoint = point_cloud_slice.get_final_mean()
        trajectory = point_cloud_slice.get_trajectory(previous_trajectory)
        trajectory = trajectory / np.linalg.norm(trajectory)

        self.log(f"Midpoint: {midpoint}, Trajectory: {trajectory}")

        # Use BeeSkeleton to calculate new positions based on midpoint and trajectory
        new_positions = bee_skeleton.calculate_new_positions(midpoint, trajectory)

        # Format points for conversion
        points = [
            {"x": new_positions["head"]["x"], "y": new_positions["head"]["y"], "color": "red"},
            {"x": new_positions["butt"]["x"], "y": new_positions["butt"]["y"], "color": "green"},
            {"x": new_positions["left"]["x"], "y": new_positions["left"]["y"], "color": "blue"},
            {"x": new_positions["right"]["x"], "y": new_positions["right"]["y"], "color": "purple"},
        ]
        session_points = [{"x": float(point["x"]), "y": float(point["y"]), "color": point["color"]} for point in points]
        self.set_points(session_points)

        # self.log(f"Recalculated points: {points}")

        # Convert to query points
        query_points = self.convert_select_points_to_query_points(
            query_frame=query_frame, points=points, height_ratio=height_ratio, width_ratio=width_ratio
        )

        return query_points, trajectory


    def process_video(self, video):
        # Determine FPS from our lookup, default to 30 if not found
        filename = video["filename"]
        path = video["path"]
        fps = video.get("fps", 30)

        assert(self.session_id in self.point_data_store)
        points = self.point_data_store[self.session_id].get("points")

        if not points or not isinstance(points, list) or len(points) < 4:
            self.log("process_error: Invalid points data")
            return

        self.log(f"Processing video: {filename} at {fps} FPS with predefined points {points}")
        self.log(f"Path: {path}")

        try:
            # Set the logger if we have a self.job_id
            if self.job_id and self.log_message:
                # Create a custom logger function to send messages to the frontend
                def custom_logger(message):
                    self.log_message(self.job_id, message)
                    print(message)  # Still print to console for debugging

                self.point_cloud.set_logger(custom_logger)

            max_segments = None
            save_intermediate=True

            self.log(f"\nProcessing video from: {path}")
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
            stride = 8

            height_ratio = resize_height / height
            width_ratio = resize_width / width
            bee_skeleton = BeeSkeleton(self.get_points())
            query_points = self.convert_select_points_to_query_points(query_frame=query_frame, points=self.get_points(), height_ratio=height_ratio, width_ratio=width_ratio)
            current_trajectory = bee_skeleton.initial_trajectory
            # print(query_points)

            # Process each segment
            for i in range(segments_to_process):
                start_frame = i * fps
                end_frame = min((i + 1) * fps, total_frames)

                self.log(f"Processing segment {i + 1}/{segments_to_process} (frames {start_frame} to {end_frame})...")
                orig_frames_slice = orig_frames[start_frame:end_frame]

                # Process the slice
                try:              
                    # Use saved points  
                    slice_result = self.point_cloud.process_video_slice(
                        orig_frames_slice, 
                        width, 
                        height, 
                        query_points, 
                        resize_width=resize_width, 
                        resize_height=resize_height
                    )
                    
                    # Calculate points for next slice
                    query_points, current_trajectory = self.recalculate_query_points(
                        slice_result, 
                        bee_skeleton, 
                        query_frame, 
                        height_ratio,
                        width_ratio,
                        current_trajectory
                    )
                    
                    
                    # Points format:
                    # [
                    #   {'x': Array(1014.8928, dtype=float32), 'y': Array(642.25415, dtype=float32), 'color': 'red'}, 
                    #   {'x': Array(1074.8928, dtype=float32), 'y': Array(692.25415, dtype=float32), 'color': 'green'}, 
                    #   {'x': Array(1041.4928, dtype=float32), 'y': Array(678.9541, dtype=float32), 'color': 'blue'}, 
                    #   {'x': Array(1054.8928, dtype=float32), 'y': Array(663.9541, dtype=float32), 'color': 'purple'}
                    # ]

                
                    request_validation = slice_result.confidence < CONFIDENCE_THRESHOLD
                    self.send_current_frame_data(
                        video_path=DATA_DIR + path + filename, 
                        frame=end_frame - 1, 
                        confidence=slice_result.confidence, 
                        request_validation=request_validation
                    )

                    if request_validation:
                        response = self.request_validation()

                        if response:
                            # Validation received, re-calculate query points
                            bee_skeleton = BeeSkeleton(self.get_points())
                            query_points = self.convert_select_points_to_query_points(
                                query_frame=query_frame, points=self.get_points(), height_ratio=height_ratio, width_ratio=width_ratio
                            )
                            current_trajectory = bee_skeleton.initial_trajectory
                        else:
                            # Handle timeout or validation failure
                            self.log("Validation failed or timed out, continuing with default behavior")


                    video_segment = slice_result.get_video()

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
            final_output_path = OUTPUT_DIR + "POINT_CLOUD_" + filename

            self.combine_and_write_video(
                save_intermediate=save_intermediate,
                segment_paths=segment_paths,
                processed_segments=processed_segments,
                final_output_path=final_output_path,
                fps=fps,
                temp_dir=temp_dir,
                start_time=start_time
            )

            self.log(f"Processing completed successfully for {filename}")

            return {"success": True, "output_filename": final_output_path, "fps": fps}

        except Exception as e:
            stack_trace = traceback.format_exc()
            error_message = f"Error processing video: {str(e)}"
            self.log(error_message)
            self.log(stack_trace)
            return {"success": False, "error": error_message}



