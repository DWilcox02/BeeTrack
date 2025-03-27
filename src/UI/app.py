import os
import sys
import uuid
import queue
import threading
import json
from flask import Flask, render_template, send_from_directory, request, jsonify, url_for, Response

# Add the src directory to the Python path so we can import from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)  # Go up one level to src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # Go up another level to project root
sys.path.append(SRC_DIR)  # Add src/ to Python path

try:
    from point_cloud_adapter import process_video_wrapper, set_log_message_function

    POINT_CLOUD_AVAILABLE = True
except ImportError:
    print("Warning: point_cloud_adapter module could not be imported")
    POINT_CLOUD_AVAILABLE = False

app = Flask(__name__)

# Calculate absolute path to the data directory
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

videos = json.load(open(os.path.join(DATA_FOLDER, "video_meta.json")))

app.config["DATA_FOLDER"] = DATA_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create a dictionary to store processing logs keyed by job ID
processing_logs = {}
processing_locks = {}


# Function to log messages to the job queue
def log_message(job_id, message):
    """Log a message to be sent to the frontend."""
    if job_id in processing_logs:
        processing_logs[job_id].put(message)


# Set the log_message function in the adapter module
if POINT_CLOUD_AVAILABLE:
    set_log_message_function(log_message)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_processed_videos():
    """Get list of processed videos in the output directory"""
    processed_videos = []

    if os.path.exists(OUTPUT_FOLDER):
        for file in os.listdir(OUTPUT_FOLDER):
            if allowed_file(file):
                processed_videos.append(file)

    return sorted(processed_videos)


@app.route("/")
def index():
    # Get all video files from the data directory and its subdirectories
    processed_videos = get_processed_videos()
    return render_template(
        "index.html", videos=videos, processed_videos=processed_videos, point_cloud_available=POINT_CLOUD_AVAILABLE
    )


@app.route("/play/<path:filename>")
def play_video(filename):
    # Check if there's a processed version of this video
    processed_filename = "SEMI_DENSE_" + os.path.basename(filename)
    processed_exists = os.path.exists(os.path.join(OUTPUT_FOLDER, processed_filename))

    # The filename includes the relative path from the data directory
    return render_template(
        "player.html",
        filename=filename,
        processed_filename=processed_filename if processed_exists else None,
        point_cloud_available=POINT_CLOUD_AVAILABLE,
    )


@app.route("/videos/<path:filename>")
def serve_video(filename):
    # This will serve the video file from the data directory
    video = videos[filename]
    directory = os.path.join(DATA_FOLDER, video["path"])
    filename = video["filename"]
    return send_from_directory(directory, filename)


@app.route("/output/<filename>")
def serve_processed_video(filename):
    # This will serve processed video files from the output directory
    return send_from_directory(OUTPUT_FOLDER, filename)


# API endpoint to stream logs for a specific job
@app.route("/api/logs/<job_id>", methods=["GET"])
def get_logs(job_id):
    """Stream logs for a specific processing job."""

    def generate():
        if job_id not in processing_logs:
            # Create a new queue for this job if it doesn't exist
            processing_logs[job_id] = queue.Queue()
            processing_locks[job_id] = threading.Lock()

        log_queue = processing_logs[job_id]

        # Send initial message
        yield f"data: Processing started for job {job_id}\n\n"

        # Keep the connection open for at most 30 minutes
        end_time = time.time() + 30 * 60

        while time.time() < end_time:
            try:
                # Try to get a message with a timeout
                message = log_queue.get(timeout=1)

                # Send the message to the client
                yield f"data: {message}\n\n"

                # Mark the message as processed
                log_queue.task_done()

                # Check if this is the final message
                if message.startswith("DONE:") or message.startswith("ERROR:"):
                    break

            except queue.Empty:
                # Send a keep-alive message every second
                yield "data: \n\n"  # Empty message to keep connection alive

    return Response(generate(), mimetype="text/event-stream")


# API endpoint to check status of a processing job
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


# API endpoint to run point cloud processing on a video
@app.route("/api/process_video", methods=["POST"])
def process_video():
    if not POINT_CLOUD_AVAILABLE:
        return jsonify({"error": "Point cloud processing is not available"}), 500

    data = request.json
    video_path = data.get("video_path")

    if not video_path:
        return jsonify({"error": "No video path provided"}), 400

    # Create a unique job ID
    job_id = str(uuid.uuid4())

    # Create a queue for this job's logs
    processing_logs[job_id] = queue.Queue()

    # Create a lock for this job
    processing_locks[job_id] = threading.Lock()

    # Function to run the processing in a background thread
    def run_processing():
        try:
            # Process the video
            result = process_video_wrapper(video_path, job_id)

            if result.get("success", False):
                # Create URL for the processed video
                output_url = url_for("serve_processed_video", filename=result["output_filename"])
                result["output_url"] = output_url

                # Signal completion
                log_message(job_id, f"DONE: Processing completed successfully.")

                # Store result for later retrieval
                log_message(job_id, f"RESULT:{json.dumps(result)}")
            else:
                error_msg = result.get("error", "Unknown error during processing")
                log_message(job_id, f"ERROR: {error_msg}")

        except Exception as e:
            import traceback

            stack_trace = traceback.format_exc()
            error_msg = f"Error processing video: {str(e)}"
            app.logger.error(f"{error_msg}\n{stack_trace}")
            log_message(job_id, f"ERROR: {error_msg}")

        # Clean up eventually
        def cleanup_job():
            import time

            time.sleep(300)  # Keep logs for 5 minutes
            if job_id in processing_logs:
                del processing_logs[job_id]
            if job_id in processing_locks:
                del processing_locks[job_id]

        cleanup_thread = threading.Thread(target=cleanup_job)
        cleanup_thread.daemon = True
        cleanup_thread.start()

    # Start processing in a background thread
    processing_thread = threading.Thread(target=run_processing)
    processing_thread.daemon = True
    processing_thread.start()

    # Return immediately with the job ID
    return jsonify({"job_id": job_id})


if __name__ == "__main__":
    # Import time here to avoid circular imports
    import time

    # Print the data directory path on startup for verification
    print(f"Using data directory: {DATA_FOLDER}")
    print(f"Using output directory: {OUTPUT_FOLDER}")
    print(f"Point Cloud processing available: {POINT_CLOUD_AVAILABLE}")
    app.run(debug=True)
