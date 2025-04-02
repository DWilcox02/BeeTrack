import os
import sys
import uuid
import queue
import threading
import json
from flask import Flask, render_template, send_from_directory, request, jsonify, url_for, Response, session
import cv2
import base64
import plotly.graph_objects as go
import plotly.io as pio


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

# Plotly points
point_data_store = {}


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

@app.route("/api/extract_first_frame/<path:filename>")
def extract_first_frame(filename):
    """Extract the first frame from a video and return it as a base64-encoded image."""
    try:
        # Get the video path
        video = videos.get(filename)
        if not video:
            return jsonify({"error": f"Video '{filename}' not found in video_meta.json"}), 404

        directory = os.path.join(DATA_FOLDER, video["path"])
        video_path = os.path.join(directory, video["filename"])

        # Extract first frame using OpenCV
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"error": "Failed to extract frame from video"}), 500

        # Convert BGR to RGB for proper display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if the image is too large (optional, for better performance)
        max_dim = 1024
        h, w = frame_rgb.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            height, width = new_h, new_w

        # Convert to JPEG to reduce size, then to base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode(".jpg", frame_rgb, encode_param)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # Return image dimensions and base64 data
        height, width = frame_rgb.shape[:2]
        return jsonify({"image": f"data:image/jpeg;base64,{frame_base64}", "width": width, "height": height})

    except Exception as e:
        app.logger.error(f"Error extracting frame: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Add these new routes to app.py
@app.route("/play/<path:filename>/frame_analysis")
def frame_analysis(filename):
    """Render the analysis page for the first frame of the video."""
    try:
        # Check if there's a processed version of this video
        processed_filename = "SEMI_DENSE_" + os.path.basename(filename)
        processed_exists = os.path.exists(os.path.join(OUTPUT_FOLDER, processed_filename))

        # Get the video metadata
        video = videos.get(filename)
        if not video:
            return "Video not found", 404

        # Extract the first frame
        directory = os.path.join(DATA_FOLDER, video["path"])
        video_path = os.path.join(directory, video["filename"])

        # Extract first frame using OpenCV
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Failed to extract frame from video", 500

        # Convert BGR to RGB for proper display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if the image is too large
        max_dim = 1024
        h, w = frame_rgb.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            height, width = new_h, new_w
        else:
            height, width = h, w

        # Convert to base64 for embedding in plotly
        _, buffer = cv2.imencode(".jpg", frame_rgb)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # Create session ID for this analysis session
        session_id = str(uuid.uuid4())

        # Create initial points in a rectangle shape
        points = [
            {"x": width * 0.25, "y": height * 0.25, "color": "red"},
            {"x": width * 0.75, "y": height * 0.25, "color": "green"},
            {"x": width * 0.75, "y": height * 0.75, "color": "blue"},
            {"x": width * 0.25, "y": height * 0.75, "color": "purple"},
        ]

        # Store the points and image size for this session
        point_data_store[session_id] = {"points": points, "width": width, "height": height, "filename": filename}

        # Create a plotly figure
        fig = go.Figure()

        # Add the image as a layout image
        fig.add_layout_image(
            dict(
                source=f"data:image/jpeg;base64,{frame_base64}",
                x=0,
                y=0,
                sizex=width,
                sizey=height,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
        )

        # Add each point as a scatter trace
        for point in points:
            fig.add_trace(
                go.Scatter(
                    x=[point["x"]],
                    y=[point["y"]],
                    mode="markers",
                    marker=dict(size=15, color=point["color"]),
                    name=f"Point ({point['color']})",
                )
            )

        # Configure the layout
        fig.update_layout(
            xaxis=dict(range=[0, width], title="X", fixedrange=True),
            yaxis=dict(
                range=[height, 0],  # Invert y-axis for image coordinates
                title="Y",
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
            ),
            showlegend=True,
            dragmode=False,
            height=min(600, height + 100),
            width=min(800, width + 100),
            margin=dict(l=50, r=50, b=50, t=50),
            title="First Frame Analysis",
        )

        # Configure for no toolbar and no zoom/pan
        fig.update_layout(
            modebar=dict(remove=["zoom", "pan", "select", "lasso", "zoomIn", "zoomOut", "autoScale", "resetScale"])
        )

        # Convert the figure to HTML
        plot_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=True,
            config={
                "displayModeBar": False,
                "staticPlot": True,  # Make it completely static
            },
        )

        # Return the analysis template with the plotly figure
        return render_template(
            "frame_analysis.html",
            filename=filename,
            plot_html=plot_html,
            session_id=session_id,
            processed_filename=processed_filename if processed_exists else None,
            point_cloud_available=POINT_CLOUD_AVAILABLE,
        )

    except Exception as e:
        app.logger.error(f"Error in frame analysis: {str(e)}")
        return f"Error: {str(e)}", 500


@app.route("/api/update_point", methods=["POST"])
def update_point():
    """Update a point's position."""
    try:
        data = request.json
        session_id = data.get("session_id")
        point_index = data.get("point_index")
        x = data.get("x")
        y = data.get("y")

        if session_id not in point_data_store:
            return jsonify({"error": "Session not found"}), 404

        session_data = point_data_store[session_id]

        if point_index < 0 or point_index >= len(session_data["points"]):
            return jsonify({"error": "Invalid point index"}), 400

        # Update the point position
        session_data["points"][point_index]["x"] = x
        session_data["points"][point_index]["y"] = y

        # Generate updated plot
        width = session_data["width"]
        height = session_data["height"]

        fig = go.Figure()

        # Add each point as a scatter trace
        for i, point in enumerate(session_data["points"]):
            fig.add_trace(
                go.Scatter(
                    x=[point["x"]],
                    y=[point["y"]],
                    mode="markers",
                    marker=dict(size=15 if i != point_index else 20, color=point["color"]),
                    name=f"Point ({point['color']})",
                )
            )

        # Configure the layout
        fig.update_layout(
            xaxis=dict(range=[0, width], title="X", fixedrange=True),
            yaxis=dict(range=[height, 0], title="Y", scaleanchor="x", scaleratio=1, fixedrange=True),
            showlegend=True,
            dragmode=False,
            height=min(600, height + 100),
            width=min(800, width + 100),
            margin=dict(l=50, r=50, b=50, t=50),
        )

        # Convert to JSON for update
        plot_json = fig.to_json()

        return jsonify({"success": True, "plot_data": plot_json, "point": session_data["points"][point_index]})

    except Exception as e:
        app.logger.error(f"Error updating point: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/save_points", methods=["POST"])
def save_points():
    """Save and log the positions of the 4 points."""
    try:
        data = request.json
        session_id = data.get("session_id")

        if session_id not in point_data_store:
            return jsonify({"error": "Session not found"}), 404

        session_data = point_data_store[session_id]
        points = session_data["points"]

        # Log the points
        message = "Points positions:\n"
        for point in points:
            point_msg = f"{point['color']}: ({point['x']:.2f}, {point['y']:.2f})"
            message += point_msg + "\n"
            app.logger.info(point_msg)

        # Add to processing logs if a job_id is provided
        job_id = data.get("job_id")
        if job_id and job_id in processing_logs:
            log_message(job_id, message)

        return jsonify({"success": True, "message": message, "points": points})

    except Exception as e:
        app.logger.error(f"Error saving points: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Import time here to avoid circular imports
    import time

    # Print the data directory path on startup for verification
    print(f"Using data directory: {DATA_FOLDER}")
    print(f"Using output directory: {OUTPUT_FOLDER}")
    print(f"Point Cloud processing available: {POINT_CLOUD_AVAILABLE}")
    app.run(debug=True)
