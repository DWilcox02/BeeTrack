import os
import cv2
import base64
from ..config import OUTPUT_FOLDER, ALLOWED_EXTENSIONS


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

        # Resize if the image is too large (optional, for better performance)
        max_dim = 1024
        h, w = frame_rgb.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            height, width = new_h, new_w
        else:
            height, width = h, w

        # Convert to JPEG to reduce size, then to base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode(".jpg", frame_rgb, encode_param)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        return frame_base64, None, width, height
    except Exception as e:
        return None, str(e), None, None
