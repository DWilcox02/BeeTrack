import os
from flask import Flask, render_template, send_from_directory
import os.path

app = Flask(__name__)

# Calculate absolute path to the data directory from the app.py location
# app.py is in /src/UI while data is at project root, so go up two levels
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

app.config["DATA_FOLDER"] = DATA_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_video_files(directory):
    """Recursively get all video files from a directory and its subdirectories"""
    video_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if allowed_file(file):
                # Get the relative path from the data directory
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                video_files.append(rel_path)

    return sorted(video_files)


@app.route("/")
def index():
    # Get all video files from the data directory and its subdirectories
    videos = get_video_files(app.config["DATA_FOLDER"])
    return render_template("index.html", videos=videos)


@app.route("/play/<path:filename>")
def play_video(filename):
    # The filename includes the relative path from the data directory
    return render_template("player.html", filename=filename)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    # This will serve the video file from the data directory
    # The filename parameter includes any subdirectories within data/
    video_path = os.path.join(DATA_FOLDER, filename)
    directory = os.path.dirname(video_path)
    file = os.path.basename(filename)

    # For debugging
    print(f"Serving video: {video_path}")
    print(f"Directory: {directory}")
    print(f"File: {file}")

    return send_from_directory(directory, file)


if __name__ == "__main__":
    # Print the data directory path on startup for verification
    print(f"Using data directory: {DATA_FOLDER}")
    app.run(debug=True)
