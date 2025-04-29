import os
import sys
import importlib.util
import json
import traceback

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.dirname(BACKEND_DIR)
POINT_CLOUD_DIR = os.path.join(BACKEND_DIR, "point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import tapir_point_cloud module
spec = importlib.util.spec_from_file_location(
    "tapir_point_cloud", os.path.join(POINT_CLOUD_DIR, "tapir_point_cloud.py")
)
tapir_point_cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tapir_point_cloud)

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))

# Optional log_message function - will be set by the Flask app
log_message = None


def process_video_wrapper(video, job_id=None, socketio=None):
    """
    Adapter function to call tapir_point_cloud.process_video with the right parameters

    Args:
        video_name: name of video from JSON
        job_id: ID for logging to frontend (optional)

    Returns:
        Dictionary with processing results
    """

    # Create a logging function that either uses the global log_message or falls back to print
    def log(message):
        if job_id and log_message:
            log_message(job_id, message)
        print(message)

    # Extract directory path and filename

    path_parts = video["path"].split("/")
    filename = video["filename"]
    folder_path = "/".join(path_parts[:-1]) + "/" if len(path_parts) > 1 else ""

    # Determine FPS from our lookup, default to 30 if not found
    fps = video.get("fps", 30)

    log(f"Processing video: {filename} at {fps} FPS")
    log(f"Path: {folder_path}")

    try:
        # Create an instance of TapirPointCloud
        point_cloud = tapir_point_cloud.TapirPointCloud(socketio)

        # Set the logger if we have a job_id
        if job_id and log_message:
            # Create a custom logger function to send messages to the frontend
            def custom_logger(message):
                log_message(job_id, message)
                print(message)  # Still print to console for debugging

            point_cloud.set_logger(custom_logger)

        # Call the process_video method with appropriate parameters
        result = point_cloud.process_video(folder_path, filename, fps)

        # Generate relative URL path to output file
        output_filename = "SEMI_DENSE_" + filename
        output_rel_path = os.path.join("output", output_filename)

        log(f"Processing completed successfully for {filename}")

        return {"success": True, "output_filename": output_filename, "output_path": output_rel_path, "fps": fps}

    except Exception as e:
        stack_trace = traceback.format_exc()
        error_message = f"Error processing video: {str(e)}"
        log(error_message)
        log(stack_trace)
        return {"success": False, "error": error_message}


def process_video_wrapper_with_points(
        video, 
        session_id,
        point_data_store, 
        job_id=None, 
        socketio=None,
        validation_events=None
    ):
    """
    Adapter function to call tapir_point_cloud.process_video with the right parameters and predefined points

    Args:
        video_name: name of video from JSON
        points: list of point coordinates
        job_id: ID for logging to frontend (optional)

    Returns:
        Dictionary with processing results
    """

    # Create a logging function that either uses the global log_message or falls back to print
    def log(message):
        if job_id and log_message:
            log_message(job_id, message)
        print(message)

    # Determine FPS from our lookup, default to 30 if not found
    filename = video["filename"]
    folder_path = video["path"]
    fps = video.get("fps", 30)

    assert(session_id in point_data_store)
    points = point_data_store[session_id].get("points")

    if not points or not isinstance(points, list) or len(points) < 4:
        socketio.emit("process_error", {"error": "Invalid points data"})
        return

    log(f"Processing video: {filename} at {fps} FPS with predefined points {points}")
    log(f"Path: {folder_path}")

    try:
        # Create an instance of TapirPointCloud
        point_cloud = tapir_point_cloud.TapirPointCloud(socketio, session_id, point_data_store, validation_events)

        # Set the logger if we have a job_id
        if job_id and log_message:
            # Create a custom logger function to send messages to the frontend
            def custom_logger(message):
                log_message(job_id, message)
                print(message)  # Still print to console for debugging

            point_cloud.set_logger(custom_logger)

        # Call the process_video method with appropriate parameters
        result = point_cloud.process_video(folder_path, filename, fps, predefined_points=True)

        # Generate relative URL path to output file
        output_filename = "SEMI_DENSE_" + filename
        output_rel_path = os.path.join("output", output_filename)

        log(f"Processing completed successfully for {filename}")

        return {"success": True, "output_filename": output_filename, "output_path": output_rel_path, "fps": fps}

    except Exception as e:
        stack_trace = traceback.format_exc()
        error_message = f"Error processing video: {str(e)}"
        log(error_message)
        log(stack_trace)
        return {"success": False, "error": error_message}


# Function to set the log_message function from outside (e.g., from Flask app)
def set_log_message_function(fn):
    global log_message
    log_message = fn
