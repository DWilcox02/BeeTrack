import os
import sys
from flask import Flask, render_template, send_from_directory, request, jsonify, url_for
import json

# Add the src directory to the Python path so we can import from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)  # Go up one level to src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # Go up another level to project root
sys.path.append(SRC_DIR)  # Add src/ to Python path

try:
    from point_cloud_adapter import process_video_wrapper

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


@app.route("/api/process_video", methods=["POST"])
def process_video():
    if not POINT_CLOUD_AVAILABLE:
        return jsonify({"error": "Point cloud processing is not available"}), 500

    data = request.json
    video_path = data.get("video_path")

    if not video_path:
        return jsonify({"error": "No video path provided"}), 400

    try:
        # Call our adapter function to process the video
        result = process_video_wrapper(video_path)

        if result.get("success", False):
            # Create URL for the processed video
            output_url = url_for("serve_processed_video", filename=result["output_filename"])
            result["output_url"] = output_url

            return jsonify(result)
        else:
            app.logger.error(f"Processing error: {result.get('error', 'Unknown error')}")
            return jsonify({"error": result.get("error", "Unknown error during processing")}), 500

    except Exception as e:
        app.logger.exception("Exception in video processing")
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

if __name__ == "__main__":
    # Print the data directory path on startup for verification
    print(f"Using data directory: {DATA_FOLDER}")
    print(f"Using output directory: {OUTPUT_FOLDER}")
    print(f"Point Cloud processing available: {POINT_CLOUD_AVAILABLE}")
    app.run(debug=True)
