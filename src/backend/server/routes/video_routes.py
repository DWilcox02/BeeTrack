import os
from flask import Blueprint, render_template, send_from_directory, jsonify, current_app
from ..utils.video_utils import extract_first_frame
from ..config import DATA_FOLDER, OUTPUT_FOLDER, POINT_CLOUD_AVAILABLE

video_bp = Blueprint("video", __name__)


@video_bp.route("/play/<path:filename>")
def play_video(filename):
    """Render the video player page."""
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


@video_bp.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve the original video file from the data directory."""
    # Get videos variable from the Flask app
    videos = video_bp.videos

    video = videos[filename]
    directory = os.path.join(DATA_FOLDER, video["path"])
    filename = video["filename"]
    return send_from_directory(directory, filename)


@video_bp.route("/output/<filename>")
def serve_processed_video(filename):
    """Serve processed video files from the output directory."""
    return send_from_directory(OUTPUT_FOLDER, filename)


@video_bp.route("/api/extract_first_frame/<path:filename>")
def extract_first_frame_api(filename):
    """Extract the first frame from a video and return it as a base64-encoded image."""
    try:
        # Get the video path
        videos = video_bp.videos
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
        current_app.logger.error(f"Error extracting frame: {str(e)}")
        return jsonify({"error": str(e)}), 500
