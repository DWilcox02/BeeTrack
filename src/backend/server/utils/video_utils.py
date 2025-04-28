import os
import cv2
import base64
from src.backend.server.config import OUTPUT_FOLDER, ALLOWED_EXTENSIONS


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


def extract_frame(video_path, i=0):
    """Extract the frame with index i from a video and return it as a base64-encoded image."""
    try:
        print(f"Extracting frame {i} from video: {video_path}")
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            return None, "Failed to open video file", None, None

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate frame index
        if i < 0 or i >= total_frames:
            cap.release()
            return None, f"Invalid frame index: {i}. Video has {total_frames} frames.", None, None

        # Set position to the requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read the frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, f"Failed to extract frame {i} from video", None, None

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