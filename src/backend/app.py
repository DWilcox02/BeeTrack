import os
import sys
import uuid
import json
import queue
import threading
import time
import cv2
import base64
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)  # Go up one level to src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # Go up another level to project root

# Add src directory to python path to enable imports
sys.path.append(SRC_DIR)

# Data and output directories
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# File types and settings
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Try to import the point_cloud_adapter
try:
    # First, try to import from current directory
    try:
        from server.point_cloud_adapter import (
            process_video_wrapper,
            process_video_wrapper_with_points,
            set_log_message_function,
        )

        POINT_CLOUD_AVAILABLE = True
    except ImportError:
        # If not in current directory, try to import from original location
        sys.path.append(CURRENT_DIR)
        from server.point_cloud_adapter import (
            process_video_wrapper,
            process_video_wrapper_with_points,
            set_log_message_function,
        )

        POINT_CLOUD_AVAILABLE = True
except ImportError:
    print(f"Warning: point_cloud_adapter module could not be imported")
    print(f"Looked in: {CURRENT_DIR}")
    POINT_CLOUD_AVAILABLE = False

# Global dictionaries to store processing logs, locks, and point data
processing_logs = {}
processing_locks = {}
point_data_store = {}

# Load video metadata
try:
    videos = json.load(open(os.path.join(DATA_FOLDER, "video_meta.json")))
except:
    print(f"Error loading video metadata from {os.path.join(DATA_FOLDER, 'video_meta.json')}")
    videos = {}

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------


def log_message(job_id, message):
    """Log a message to be sent to the frontend."""
    if job_id in processing_logs:
        processing_logs[job_id].put(message)
        # Emit the message to the client via Socket.IO
        socketio.emit(f"log_message_{job_id}", {"message": message})


def cleanup_job(job_id, timeout=300):
    """Clean up job resources after a timeout."""
    time.sleep(timeout)  # Keep logs for 5 minutes
    if job_id in processing_logs:
        del processing_logs[job_id]
    if job_id in processing_locks:
        del processing_locks[job_id]


def init_job_logging(job_id):
    """Initialize logging for a job."""
    processing_logs[job_id] = queue.Queue()
    processing_locks[job_id] = threading.Lock()
    return processing_logs[job_id], processing_locks[job_id]


def allowed_file(filename):
    """Check if file extension is in allowed extensions."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_processed_videos():
    """Get list of processed videos in the output directory."""
    processed_videos = []
    if os.path.exists(OUTPUT_FOLDER):
        for file in os.listdir(OUTPUT_FOLDER):
            if allowed_file(file):
                processed_videos.append(file)
    return sorted(processed_videos)


def extract_first_frame(video_path):
    """Extract the first frame from a video and return it as a base64-encoded image."""
    try:
        # Extract first frame using OpenCV
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, "Failed to extract frame from video", None, None

        # Convert BGR to RGB for proper display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get dimensions
        h, w = frame_rgb.shape[:2]

        # Convert to JPEG, then to base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode(".jpg", frame_rgb, encode_param)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        return frame_base64, None, w, h
    except Exception as e:
        return None, str(e), None, None


# Set log message function for point cloud adapter if available
if POINT_CLOUD_AVAILABLE:
    set_log_message_function(log_message)

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------


@app.route("/api/videos")
def get_videos():
    """Return the list of available videos."""
    return jsonify(videos)


@app.route("/api/processed_videos")
def get_processed_videos_api():
    """Return the list of processed videos."""
    return jsonify(get_processed_videos())


@app.route("/api/video_info/<path:filename>")
def get_video_info(filename):
    """Get information about a video."""
    if filename not in videos:
        return jsonify({"error": "Video not found"}), 404

    video = videos[filename]
    processed_filename = "SEMI_DENSE_" + os.path.basename(filename)
    processed_exists = os.path.exists(os.path.join(OUTPUT_FOLDER, processed_filename))

    return jsonify(
        {
            "filename": filename,
            "processed_filename": processed_filename if processed_exists else None,
            "video_info": video,
            "point_cloud_available": POINT_CLOUD_AVAILABLE,
        }
    )


@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve the original video file from the data directory."""
    video = videos.get(filename)
    if not video:
        return "Video not found", 404

    directory = os.path.join(DATA_FOLDER, video["path"])
    video_filename = video["filename"]
    return send_from_directory(directory, video_filename)


@app.route("/output/<filename>")
def serve_processed_video(filename):
    """Serve processed video files from the output directory."""
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/api/extract_first_frame/<path:filename>")
def extract_first_frame_api(filename):
    """Extract the first frame from a video and return it as a base64-encoded image."""
    try:
        # Get the video path
        video = videos.get(filename)
        if not video:
            return jsonify({"error": f"Video '{filename}' not found in video_meta.json"}), 404

        directory = os.path.join(DATA_FOLDER, video["path"])
        video_path = os.path.join(directory, video["filename"])

        # Extract the first frame
        frame_base64, error, width, height = extract_first_frame(video_path)

        if error:
            return jsonify({"error": error}), 500

        # Return image dimensions and base64 data
        return jsonify({"image": f"data:image/jpeg;base64,{frame_base64}", "width": width, "height": height})

    except Exception as e:
        app.logger.error(f"Error extracting frame: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/job_status/<job_id>", methods=["GET"])
def job_status(job_id):
    """Get the current status and result of a processing job."""
    if job_id not in processing_logs:
        return jsonify({"error": "Job not found"}), 404

    # Check if job has a result message
    with processing_locks[job_id]:
        # Make a copy of all messages
        messages = list(processing_logs[job_id].queue)
        result = None
        status = "processing"

        for msg in messages:
            if msg.startswith("RESULT:"):
                # Extract the JSON result
                result = json.loads(msg[7:])
                status = "completed"
                break
            elif msg.startswith("ERROR:"):
                status = "error"
                break

    return jsonify({"job_id": job_id, "status": status, "result": result})


# -------------------------------------------------------------------------
# Socket.IO Events
# -------------------------------------------------------------------------


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on("process_video")
def handle_process_video(data):
    """Process video with TAPIR point cloud."""
    if not POINT_CLOUD_AVAILABLE:
        emit("process_error", {"error": "Point cloud processing is not available"})
        return

    video_path = data.get("video_path")
    if not video_path:
        emit("process_error", {"error": "No video path provided"})
        return

    # Create a unique job ID
    job_id = str(uuid.uuid4())

    # Initialize logging for this job
    init_job_logging(job_id)

    # Function to run the processing in a background thread
    def run_processing():
        try:
            # Process the video
            result = process_video_wrapper(video_path, job_id)

            if result.get("success", False):
                # Create URL for the processed video
                output_url = f"/output/{result['output_filename']}"
                result["output_url"] = output_url

                # Signal completion
                log_message(job_id, f"DONE: Processing completed successfully.")

                # Store result for later retrieval
                log_message(job_id, f"RESULT:{json.dumps(result)}")

                # Emit the completion event
                socketio.emit(f"process_complete_{job_id}", result)
            else:
                error_msg = result.get("error", "Unknown error during processing")
                log_message(job_id, f"ERROR: {error_msg}")
                socketio.emit(f"process_error_{job_id}", {"error": error_msg})

        except Exception as e:
            stack_trace = traceback.format_exc()
            error_msg = f"Error processing video: {str(e)}"
            app.logger.error(f"{error_msg}\n{stack_trace}")
            log_message(job_id, f"ERROR: {error_msg}")
            socketio.emit(f"process_error_{job_id}", {"error": error_msg})

        # Clean up eventually
        cleanup_thread = threading.Thread(target=cleanup_job, args=(job_id,))
        cleanup_thread.daemon = True
        cleanup_thread.start()

    # Start processing in a background thread
    processing_thread = threading.Thread(target=run_processing)
    processing_thread.daemon = True
    processing_thread.start()

    # Return immediately with the job ID
    emit("process_started", {"job_id": job_id})


@socketio.on("process_video_with_points")
def handle_process_video_with_points(data):
    """Process video with predefined points."""
    if not POINT_CLOUD_AVAILABLE:
        emit("process_error", {"error": "Point cloud processing is not available"})
        return

    video_path = data.get("video_path")
    points = data.get("points")

    if not video_path:
        emit("process_error", {"error": "No video path provided"})
        return

    if not points or not isinstance(points, list) or len(points) < 4:
        emit("process_error", {"error": "Invalid points data"})
        return

    # Create a unique job ID
    job_id = str(uuid.uuid4())

    # Initialize logging for this job
    init_job_logging(job_id)

    # Function to run the processing in a background thread
    def run_processing():
        try:
            # Process the video
            result = process_video_wrapper_with_points(video_path, points, job_id)

            if result.get("success", False):
                # Create URL for the processed video
                output_url = f"/output/{result['output_filename']}"
                result["output_url"] = output_url

                # Signal completion
                log_message(job_id, f"DONE: Processing completed successfully.")

                # Store result for later retrieval
                log_message(job_id, f"RESULT:{json.dumps(result)}")

                # Emit the completion event
                socketio.emit(f"process_complete_{job_id}", result)
            else:
                error_msg = result.get("error", "Unknown error during processing")
                log_message(job_id, f"ERROR: {error_msg}")
                socketio.emit(f"process_error_{job_id}", {"error": error_msg})

        except Exception as e:
            stack_trace = traceback.format_exc()
            error_msg = f"Error processing video: {str(e)}"
            app.logger.error(f"{error_msg}\n{stack_trace}")
            log_message(job_id, f"ERROR: {error_msg}")
            socketio.emit(f"process_error_{job_id}", {"error": error_msg})

        # Clean up eventually
        cleanup_thread = threading.Thread(target=cleanup_job, args=(job_id,))
        cleanup_thread.daemon = True
        cleanup_thread.start()

    # Start processing in a background thread
    processing_thread = threading.Thread(target=run_processing)
    processing_thread.daemon = True
    processing_thread.start()

    # Return immediately with the job ID
    emit("process_started", {"job_id": job_id})


@socketio.on("save_points")
def handle_save_points(data):
    """Save point positions."""
    points = data.get("points")
    if not points:
        emit("save_points_error", {"error": "No points provided"})
        return

    try:
        # Log the points
        message = "Points positions:\n"
        for point in points:
            point_msg = f"{point['color']}: ({point['x']:.2f}, {point['y']:.2f})"
            message += point_msg + "\n"
            app.logger.info(point_msg)

        # Emit success event
        emit("points_saved", {"success": True, "message": message, "points": points})
    except Exception as e:
        app.logger.error(f"Error saving points: {str(e)}")
        emit("save_points_error", {"error": str(e)})


@socketio.on("subscribe_to_job")
def handle_subscribe_to_job(data):
    """Subscribe to job updates."""
    job_id = data.get("job_id")
    if not job_id or job_id not in processing_logs:
        emit("subscription_error", {"error": "Invalid job ID"})
        return

    # Send all existing messages
    with processing_locks[job_id]:
        messages = list(processing_logs[job_id].queue)
        for msg in messages:
            emit(f"log_message_{job_id}", {"message": msg})

    emit("subscription_success", {"job_id": job_id})


# Run the application
if __name__ == "__main__":
    print(f"Using data directory: {DATA_FOLDER}")
    print(f"Using output directory: {OUTPUT_FOLDER}")
    print(f"Point Cloud processing available: {POINT_CLOUD_AVAILABLE}")
    socketio.run(app, host="127.0.0.1", port=5001, debug=True, allow_unsafe_werkzeug=True)
