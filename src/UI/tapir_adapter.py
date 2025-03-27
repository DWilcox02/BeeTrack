# tapir_adapter.py
# Place this file in the same directory as app.py (src/UI/)

import os
import sys
import importlib.util
import json

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import tapir_bulk.py
spec = importlib.util.spec_from_file_location("tapir_bulk", os.path.join(SRC_DIR, "tapir_bulk.py"))
tapir_bulk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tapir_bulk)

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))

def process_video_wrapper(video_name):
    """
    Adapter function to call tapir_bulk.process_video with the right parameters

    Args:
        video_name: name of video from JSON

    Returns:
        Dictionary with processing results
    """
    # Extract directory path and filename
    video = videos.get(video_name, None)
    path_parts = video["path"].split("/")
    filename = video["filename"]
    folder_path = "/".join(path_parts[:-1]) + "/" if len(path_parts) > 1 else ""

    # Determine FPS from our lookup, default to 30 if not found
    fps = video.get("fps", 30)

    # Set up to match the global variables that tapir_bulk.process_video expects
    # We'll patch the function's environment temporarily
    original_videos = getattr(tapir_bulk, "videos", None)
    original_video_number = getattr(tapir_bulk, "video_number", None)

    # Create a temporary mock for the global variables
    mock_video = {"path": folder_path, "filename": filename, "fps": fps}
    setattr(tapir_bulk, "videos", [mock_video])
    setattr(tapir_bulk, "video_number", 0)

    try:
        # Call the original process_video function
        result = tapir_bulk.process_video(folder_path, filename, fps)

        # Generate relative URL path to output file
        output_filename = "SEMI_DENSE_" + filename
        output_rel_path = os.path.join("output", output_filename)

        return {"success": True, "output_filename": output_filename, "output_path": output_rel_path, "fps": fps}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Restore original globals
        if original_videos is not None:
            setattr(tapir_bulk, "videos", original_videos)
        if original_video_number is not None:
            setattr(tapir_bulk, "video_number", original_video_number)
