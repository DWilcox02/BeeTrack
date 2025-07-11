import os
import sys
import json
import queue
import threading
import time
import traceback
from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from .server.utils.video_utils import extract_frame
from .video_processor import VideoProcessor
from .utils.frontend_communicator import FrontendCommunicator
from .utils.processing_configuration import ProcessingConfiguration
from .utils.component_selector import (
    PointCloudEstimatorSelector,
    PointCloudGeneratorSelector,
    InlierPredictorSelector,
    QueryPointReconstructorSelector,
    PointCloudReconstructorSelector,
    WeightCalculatorDistancesSelector,
    WeightCalculatorOutliersSelector,
    create_component
)
from src.backend.point_cloud.point_cloud_generator import PointCloudGenerator
from src.backend.point_cloud.estimation.point_cloud_estimator_interface import PointCloudEstimatorInterface
from src.backend.inlier_predictors.inlier_predictor_base import InlierPredictorBase
from src.backend.query_point_predictors.query_point_reconstructor_base import QueryPointReconstructorBase
from src.backend.point_cloud_reconstructors.point_cloud_reconstructor_base import PointCloudReconstructorBase
from src.backend.weight_calculators.weight_calculator_distances_base import WeightCalculatorDistancesBase
from src.backend.weight_calculators.weight_calculator_outliers_base import WeightCalculatorOutliersBase


POINT_CLOUD_AVAILABLE = True
POINT_CLOUD_TYPE = "TAPIR"

# Path configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BACKEND_DIR)  # Go up one level to src/
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
CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Global dictionaries to store processing logs, locks, and point data
processing_logs = {}
processing_locks = {}
point_data_store = {}
validation_events = {}

# Job control
running_jobs = {}  # job_id -> {"thread": thread_obj, "stop_event": threading.Event()}
job_stop_events = {}  # job_id -> threading.Event()

# Load video metadata
try:
    videos = json.load(open(os.path.join(DATA_FOLDER, "video_meta.json")))
except Exception as _:
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
        socketio.emit("job_log", {
            "job_id": job_id,
            "message": message
        })


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


def get_video_info(filename):
    [path, video_name] = filename.split("/")
    path = path + "/"
    video = next((
        video for video in videos 
        if video["filename"] == video_name and
           video["path"] == path
    ), None)
    return video


def parse_points(points: str):
    """Parse points from a string to a list of dictionaries."""
    try:
        points = json.loads(points)
        parsed_points = []
        for point in points:
            parsed_point = {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "color": point.get("color", "#FFFFFF"),
                "radius": float(point["radius"]),
            }
            parsed_points.append(parsed_point)
        return parsed_points
    except Exception as e:
        app.logger.error(f"Error parsing points: {str(e)}")
        return []


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------


@app.route("/api/videos")
def get_videos():
    """Return the list of available videos."""
    # Convert json to list
    return jsonify(videos)


@app.route("/api/processed_videos")
def get_processed_videos_api():
    """Return the list of processed videos."""
    print(get_processed_videos())
    return jsonify(get_processed_videos())


@app.route("/api/video/<path:filename>")
def serve_video(filename):
    print("CALLED HERE")
    """Serve the original video file from the data directory."""
    # video = videos.get(filename)
    video = filename
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
        # print(filename)  # Dance_1_min/dance_15_secs_700x700_50fps.mp4
        # Get the video path
        # video = videos.get(filename)
        video = filename
        if not video:
            return jsonify({"error": f"Video '{filename}' not found in video_meta.json"}), 404

        video_path = os.path.join(DATA_FOLDER, video)
        # video_path = os.path.join(directory, video)

        # Extract the first frame
        frame_base64, error, width, height = extract_frame(video_path)
        # print(frame_base64)

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


@socketio.on("process_video_with_points")
def handle_process_video_with_points(data):
    """Process video with predefined points."""
    if not POINT_CLOUD_AVAILABLE:
        emit("process_error", {"error": "Point cloud processing is not available"})
        return

    session_id = data.get("session_id")
    job_id = data.get("job_id")
    confidence_threshold = float(data.get("confidence_threshold"))
    smoothing_alpha = float(data.get("smoothing_alpha"))
    if smoothing_alpha < 0:
        smoothing_alpha = 0.0
    dbscan_epsilon = float(data.get("dbscan_epsilon"))
    deformity_delta = float(data.get("deformity_delta"))
    processing_seconds = int(data.get("processing_seconds"))
    point_cloud_estimator_label = str(data.get("point_cloud_estimator"))
    point_cloud_generator_label = str(data.get("point_cloud_generator"))
    inlier_predictor_label = str(data.get("inlier_predictor"))
    query_point_reconstructor_label = str(data.get("query_point_reconstructor"))
    point_cloud_non_validated_reconstructor_label = str(data.get("point_cloud_non_validated_reconstructor"))
    point_cloud_validated_reconstructor_label = str(data.get("point_cloud_validated_reconstructor"))
    weight_calculator_outliers_label = str(data.get("weight_calculator_outliers"))
    weight_calculator_distances_label = str(data.get("weight_calculator_distances"))

    if session_id not in point_data_store:
        print(f"Session ID {session_id} not found in point_data_store")
        emit('update_point_error', {"error": "Session not found"})
        return

    session_data = point_data_store[session_id]
    video_path = session_data.get("video_path")

    # Initialize logging for this job
    init_job_logging(job_id)

    # Initialize stop event
    stop_event = threading.Event()
    job_stop_events[job_id] = stop_event

    video = get_video_info(video_path)

    # Choose point cloud estimator
    point_cloud_estimator: PointCloudEstimatorInterface = create_component(PointCloudEstimatorSelector[point_cloud_estimator_label])
    point_cloud_generator: PointCloudGenerator = create_component(
        PointCloudGeneratorSelector[point_cloud_generator_label]
    )
    inlier_predictor: InlierPredictorBase = create_component(
        InlierPredictorSelector[inlier_predictor_label], dbscan_epsilon
    )
    query_point_reconstructor: QueryPointReconstructorBase = create_component(
        QueryPointReconstructorSelector[query_point_reconstructor_label]
    )
    point_cloud_non_validated_reconstructor: PointCloudReconstructorBase = create_component(
        PointCloudReconstructorSelector[point_cloud_non_validated_reconstructor_label]
    )
    point_cloud_validated_reconstructor: PointCloudReconstructorBase = create_component(
        PointCloudReconstructorSelector[point_cloud_validated_reconstructor_label]
    )
    weight_calculator_outliers: WeightCalculatorOutliersBase = create_component(
        WeightCalculatorOutliersSelector[weight_calculator_outliers_label]
    )
    weight_calculator_distances: WeightCalculatorDistancesBase = create_component(
        WeightCalculatorDistancesSelector[weight_calculator_distances_label]
    )

    # Function to run the processing in a background thread
    def run_processing():
        try:
            processing_configuration = ProcessingConfiguration(
                confidence_threshold=confidence_threshold,
                smoothing_alpha=smoothing_alpha,
                dbscan_epsilon=dbscan_epsilon,
                deformity_delta=deformity_delta,
                processing_seconds=processing_seconds,
                point_cloud_generator=point_cloud_generator,
                point_cloud_estimator=point_cloud_estimator,
                inlier_predictor=inlier_predictor,
                query_point_reconstructor=query_point_reconstructor,
                point_cloud_non_validated_reconstructor=point_cloud_non_validated_reconstructor,
                point_cloud_validated_reconstructor=point_cloud_validated_reconstructor,
                weight_calculator_outliers=weight_calculator_outliers,
                weight_calculator_distances=weight_calculator_distances,
            )
            frontend_communicator = FrontendCommunicator(
                socketio=socketio,
                session_id=session_id,
                log_message=log_message,
                validation_events=validation_events
            )
            video_processor = VideoProcessor(
                session_id=session_id, 
                point_data_store=point_data_store,
                frontend_communicator=frontend_communicator,
                processing_configuration=processing_configuration,
                video=video,
                job_id=job_id,
                stop_event=stop_event
            )

            video_processor.set_log_message_function(log_message)
            # Process the video
            init_query_points = point_data_store[session_id]["points"]
            result: dict = video_processor.process_video(init_query_points)  # {"success": True, "output_filename": final_video_output_path, "fps": fps}

            if stop_event.is_set():
                log_message(job_id, "STOPPED: Processing was stopped by user")
                socketio.emit(f"process_stopped_{job_id}", {"message": "Processing stopped by user"})
                return

            if result.get("success", False):

                # Signal completion
                log_message(job_id, "DONE: Processing completed successfully.")

                # Store result for later retrieval
                log_message(job_id, f"RESULT:{json.dumps(result)}")

                result["job_id"] = job_id

                # Emit the completion event
                socketio.emit("process_complete", result)
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
        
        finally:
            if job_id in job_stop_events:
                del job_stop_events[job_id]
            if job_id in running_jobs:
                del running_jobs[job_id]

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


@socketio.on("start_new_session")
def handle_start_new_session(data):
    """Start a new session for point data."""
    session_id = data.get("session_id")
    parsed_points = parse_points(data.get("points", []))
    point_data_store[session_id] = {
        "points": parsed_points,
        "video_path": data.get("video_path"),
        "frame_width": int(data.get("frame_width")),
        "frame_height": int(data.get("frame_height")),
    }
    emit("session_started", {"session_id": session_id})


@socketio.on('update_point')
def handle_update_point(data):
    """Update a point's position and emit updated plot data through socket."""
    try:
        session_id = data.get("session_id")
        point_index = int(data.get("point_index"))
        x = float(data.get("x"))
        y = float(data.get("y"))
        radius = float(data.get("radius"))

        if session_id not in point_data_store:
            print(f"Session ID {session_id} not found in point_data_store")
            emit('update_point_error', {"error": "Session not found"})
            return

        session_data = point_data_store[session_id]

        if point_index < 0 or point_index >= len(session_data["points"]):
            print(f"Invalid point index {point_index} for session {session_id}")
            emit('update_point_error', {"error": "Invalid point index"})
            return

        # Update the point position
        session_data["points"][point_index]["x"] = x
        session_data["points"][point_index]["y"] = y
        session_data["points"][point_index]["radius"] = radius

        session_points = session_data["points"]
        
        points_json = []
        for point in session_points:
            points_json.append(
                {
                    "x": json.dumps(float(point["x"])),
                    "y": json.dumps(float(point["y"])),
                    "color": point["color"],
                    "radius": json.dumps(float(point["radius"])),
                }
            )
    
        # print(f"Updating, session data points: {session_points}")

        emit('update_point_response', {
            "success": True, 
            "points": points_json
        })
    except Exception as e:
        app.logger.error(f"Error updating point: {str(e)}")
        app.logger.error(traceback.format_exc())
        emit("update_point_error", {"error": str(e)})


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


@socketio.on("validation_response")
def handle_validation_response(data):
    request_id = data.get("request_id")

    if request_id and request_id in validation_events:
        # Store the response and set the event
        validation_events[request_id]["response"] = data
        validation_events[request_id]["event"].set()


@socketio.on("update_all_points")
def handle_update_all_points(data):
    """Update all point positions at once and emit updated plot data through socket."""
    try:
        session_id = data.get("session_id")
        points = data.get("points")

        if not session_id or not isinstance(points, list):
            emit("update_all_points_response", {"success": False, "error": "Missing session_id or invalid points data"})
            return

        if session_id not in point_data_store:
            print(f"Session ID {session_id} not found in point_data_store")
            emit("update_all_points_response", {"success": False, "error": "Session not found"})
            return

        session_data = point_data_store[session_id]

        # Validate the number of points matches the existing session
        if len(points) != len(session_data["points"]):
            print(
                f"Invalid number of points for session {session_id}: expected {len(session_data['points'])}, got {len(points)}"
            )
            emit("update_all_points_response", {"success": False, "error": "Invalid number of points"})
            return

        # Validate each point has required fields
        for i, point in enumerate(points):
            if not all(key in point for key in ["x", "y", "color", "radius"]):
                print(f"Invalid point data at index {i}: missing required fields")
                emit("update_all_points_response", {"success": False, "error": f"Invalid point data at index {i}"})
                return

        # Update all points
        for i, point in enumerate(points):
            session_data["points"][i]["x"] = float(point["x"])
            session_data["points"][i]["y"] = float(point["y"])
            session_data["points"][i]["color"] = point["color"]
            session_data["points"][i]["radius"] = float(point["radius"])

        # Format points for JSON response
        points_json = []
        for point in session_data["points"]:
            points_json.append(
                {
                    "x": json.dumps(float(point["x"])),
                    "y": json.dumps(float(point["y"])),
                    "color": point["color"],
                    "radius": json.dumps(float(point["radius"])),
                }
            )

        emit("update_all_points_response", {"success": True, "points": points_json})
    except Exception as e:
        app.logger.error(f"Error updating all points: {str(e)}")
        app.logger.error(traceback.format_exc())
        emit("update_all_points_response", {"success": False, "error": str(e)})
        

@socketio.on("stop_job")
def handle_stop_job(data):
    """Handle stop job request via socket."""
    job_id = data.get("job_id")

    if not job_id:
        emit("stop_job_error", {"error": "No job ID provided"})
        return

    if job_id not in job_stop_events:
        emit("stop_job_error", {"error": "Job not found or already completed"})
        return

    # Signal the job to stop
    job_stop_events[job_id].set()
    log_message(job_id, "STOP: Job stop requested by user")

    emit("stop_job_success", {"job_id": job_id, "message": "Stop signal sent"})


# Run the application
if __name__ == "__main__":
    print(f"Using data directory: {DATA_FOLDER}")
    print(f"Using output directory: {OUTPUT_FOLDER}")
    print(f"Point Cloud processing available: {POINT_CLOUD_AVAILABLE}")
    socketio.run(app, host="127.0.0.1", port=5001, debug=True, allow_unsafe_werkzeug=True)
