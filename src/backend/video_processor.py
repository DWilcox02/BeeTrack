import os
import json
import time
import mediapy as media
import tempfile
import numpy as np
import gc

from .point_cloud.estimation.point_cloud_estimator_interface import PointCloudEstimatorInterface
from .server.utils.video_utils import extract_frame
from .point_cloud.circular_point_cloud_generator import CircularPointCloudGenerator
from .point_cloud.estimation.estimation_slice import EstimationSlice

# Get paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BACKEND_DIR)
POINT_CLOUD_DIR = os.path.join(BACKEND_DIR, "point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))


NUM_SLICES = 3
CONFIDENCE_THRESHOLD = 0.8

class VideoProcessor():
    def __init__(
        self,
        session_id,
        point_data_store,
        point_cloud_estimator: PointCloudEstimatorInterface,
        send_frame_data_callback, 
        request_validation_callback,
        video,
        job_id=None,
    ):
        self.session_id = session_id
        self.point_cloud_estimator = point_cloud_estimator
        self.send_frame_data_callback = send_frame_data_callback
        self.request_validation_callback = request_validation_callback
        self.video = video
        self.job_id = job_id
        self.log_message = None
        self.point_cloud = CircularPointCloudGenerator(
            init_points=point_data_store[session_id]["points"], 
            point_data_store=point_data_store, 
            session_id=session_id)
        # self.point_cloud = RhombusPointCloud(
        #     init_points=point_data_store[session_id]["points"],
        #     point_data_store=point_data_store,
        #     session_id=session_id
        # )


    def set_log_message_function(self, fn):
        self.log_message = fn


    def log(self, message):
        if self.job_id and self.log_message:
            self.log_message(self.job_id, message)
        print(message)


    def send_current_frame_data(self, video_path, frame, confidence, request_validation):
        frame_base64, error, width, height = extract_frame(video_path, frame)
        frameData = {"frame": frame_base64, "width": width, "height": height, "frame_idx": frame}
        self.send_frame_data_callback(frameData, self.point_cloud.get_query_points(), confidence, request_validation)


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

    def resize_points_add_frame(self, cloud_points, query_frame, height_ratio, width_ratio):
        cloud_points = np.array(
            [[query_frame, point[1] * height_ratio, point[0] * width_ratio] for point in cloud_points],
            dtype=np.float32
        )
        return cloud_points


    def process_video(self):
        # Determine FPS from our lookup, default to 30 if not found
        filename = self.video["filename"]
        path = self.video["path"]
        fps = self.video.get("fps", 30)

        self.log(f"Processing video: {filename} at {fps} FPS with predefined points {self.point_cloud.get_query_points()}")
        self.log(f"Path: {path}")

        try:
            # Set the logger if we have a self.job_id
            if self.job_id and self.log_message:
                # Create a custom logger function to send messages to the frontend
                def custom_logger(message):
                    self.log_message(self.job_id, message)
                    print(message)  # Still print to console for debugging

                self.point_cloud_estimator.set_logger(custom_logger)
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
            # stride = 8

            height_ratio = resize_height / height
            width_ratio = resize_width / width
            

            # NEXT: point_cloud_generator = CircularPointCloud(blah blah blah), initialise with uniform weights
            # Process each segment
            for i in range(segments_to_process):
                start_frame = i * fps
                end_frame = min((i + 1) * fps, total_frames)

                self.log(f"Processing segment {i + 1}/{segments_to_process} (frames {start_frame} to {end_frame})...")
                orig_frames_slice = orig_frames[start_frame:end_frame]

                # Process the slice
                try:
                    cloud_points = self.point_cloud.generate_cloud_points() # N x 2 (for N points)
                    resized_points = self.resize_points_add_frame(
                        cloud_points=cloud_points, 
                        query_frame=query_frame, 
                        height_ratio=height_ratio, 
                        width_ratio=width_ratio
                    )
                    # Use saved points  
                    slice_result: EstimationSlice = self.point_cloud_estimator.process_video_slice(
                        orig_frames_slice,
                        width,
                        height,
                        resized_points,
                        resize_width=resize_width,
                        resize_height=resize_height,
                    )

                    initial_positions = slice_result.get_final_points_for_frame(
                        frame=0, 
                        num_qp=self.point_cloud.num_qp,
                        num_cp_per_qp=self.point_cloud.num_cp_per_qp
                    )
                    final_positions = slice_result.get_final_points_for_frame(
                        frame=-1, 
                        num_qp=self.point_cloud.num_qp, 
                        num_cp_per_qp=self.point_cloud.num_cp_per_qp
                    )
                    
                    # Calculate points for next slice
                    self.point_cloud.update_weights(
                        initial_positions=initial_positions,
                        final_positions=final_positions
                    )
                    self.point_cloud.recalc_query_points(final_positions=final_positions)
                
                    confidence = self.point_cloud.calculate_confidence()
                    request_validation = confidence < CONFIDENCE_THRESHOLD
                    self.send_current_frame_data(
                        video_path=DATA_DIR + path + filename, 
                        frame=end_frame - 1, 
                        confidence=confidence, 
                        request_validation=request_validation
                    )

                    if i < segments_to_process - 1 and request_validation:
                        self.request_validation() # Query points will have been updated


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

