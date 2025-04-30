import os
import sys
import importlib.util
import json
import traceback
from .point_cloud_type import PointCloudType

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.dirname(BACKEND_DIR)
POINT_CLOUD_DIR = os.path.join(BACKEND_DIR, "point_cloud/")
TAPIR_DIR = os.path.join(POINT_CLOUD_DIR, "TAPIR_point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import tapir_point_cloud module
spec = importlib.util.spec_from_file_location("tapir_point_cloud", os.path.join(TAPIR_DIR, "tapir_point_cloud.py"))
tapir_point_cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tapir_point_cloud)

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))

# Optional log_message function - will be set by the Flask app


class VideoProcessor():
    def __init__(self, session_id: str, point_data_store: dict, point_cloud_type: PointCloudType, job_id=None, socketio=None, validation_events=None):
        self.session_id = session_id
        self.point_data_store = point_data_store
        self.point_cloud_type = point_cloud_type
        self.job_id = job_id
        self.socketio = socketio
        self.validation_events = validation_events
        self.log_message = None


    def log(self, message):
        if self.job_id and self.log_message:
            self.log_message(self.job_id, message)
        print(message)


    def get_point_cloud(self):
        if self.point_cloud_type == PointCloudType.TAPIR:
            return tapir_point_cloud.TapirPointCloud(
                self.socketio, self.session_id, self.point_data_store, self.validation_events
            )
        self.log(f"Error: no matching point cloud type for {self.point_cloud_type}")


    # def process_video_wrapper(self, video):
    #     # Extract directory path and filename

    #     path_parts = video["path"].split("/")
    #     filename = video["filename"]
    #     folder_path = "/".join(path_parts[:-1]) + "/" if len(path_parts) > 1 else ""

    #     # Determine FPS from our lookup, default to 30 if not found
    #     fps = video.get("fps", 30)

    #     self.log(f"Processing video: {filename} at {fps} FPS")
    #     self.log(f"Path: {folder_path}")

    #     try:
    #         # Create an instance of TapirPointCloud
    #         point_cloud = self.get_point_cloud()

    #         # Set the logger if we have a self.job_id
    #         if self.job_id and self.log_message:
    #             # Create a custom logger function to send messages to the frontend
    #             def custom_logger(message):
    #                 self.log_message(self.job_id, message)
    #                 print(message)  # Still print to console for debugging

    #             point_cloud.set_logger(custom_logger)

    #         # Call the process_video method with appropriate parameters
    #         result = point_cloud.process_video(folder_path, filename, fps)

    #         # Generate relative URL path to output file
    #         output_filename = "SEMI_DENSE_" + filename
    #         output_rel_path = os.path.join("output", output_filename)

    #         self.log(f"Processing completed successfully for {filename}")

    #         return {"success": True, "output_filename": output_filename, "output_path": output_rel_path, "fps": fps}

    #     except Exception as e:
    #         stack_trace = traceback.format_exc()
    #         error_message = f"Error processing video: {str(e)}"
    #         self.log(error_message)
    #         self.log(stack_trace)
    #         return {"success": False, "error": error_message}


    def process_video_wrapper_with_points(self, video):
        # Determine FPS from our lookup, default to 30 if not found
        filename = video["filename"]
        folder_path = video["path"]
        fps = video.get("fps", 30)

        assert(self.session_id in self.point_data_store)
        points = self.point_data_store[self.session_id].get("points")

        if not points or not isinstance(points, list) or len(points) < 4:
            self.socketio.emit("process_error", {"error": "Invalid points data"})
            return

        self.log(f"Processing video: {filename} at {fps} FPS with predefined points {points}")
        self.log(f"Path: {folder_path}")

        try:
            # Create an instance of TapirPointCloud
            point_cloud = self.get_point_cloud()

            # Set the logger if we have a self.job_id
            if self.job_id and self.log_message:
                # Create a custom logger function to send messages to the frontend
                def custom_logger(message):
                    self.log_message(self.job_id, message)
                    print(message)  # Still print to console for debugging

                point_cloud.set_logger(custom_logger)

            # TODO: move "run" process here
            # Call the process_video method with appropriate parameters
            result = point_cloud.process_video(folder_path, filename, fps, predefined_points=True)

            # Generate relative URL path to output file
            output_filename = "SEMI_DENSE_" + filename
            output_rel_path = os.path.join("output", output_filename)

            self.log(f"Processing completed successfully for {filename}")

            return {"success": True, "output_filename": output_filename, "output_path": output_rel_path, "fps": fps}

        except Exception as e:
            stack_trace = traceback.format_exc()
            error_message = f"Error processing video: {str(e)}"
            self.log(error_message)
            self.log(stack_trace)
            return {"success": False, "error": error_message}


    def set_log_message_function(self, fn):
        self.log_message = fn
