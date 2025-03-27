import os
import sys
import importlib.util
import json

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
POINT_CLOUD_DIR = os.path.join(SRC_DIR, "point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

spec = importlib.util.spec_from_file_location("tapir_point_cloud", os.path.join(POINT_CLOUD_DIR, "tapir_point_cloud.py"))
tapir_point_cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tapir_point_cloud)

# Video FPS lookup dictionary based on your existing data
videos = json.load(open(os.path.join(DATA_DIR, "video_meta.json")))

def process_video_wrapper(video_name):
    """
    Adapter function to call tapir_point_cloud.process_video with the right parameters

    Args:
        video_name: name of video from JSON

    Returns:
        Dictionary with processing results
    """
    # Extract directory path and filename
    video = videos.get(video_name, None)

    if not video:
        return {"success": False, "error": f"Video '{video_name}' not found in video_meta.json"}

    path_parts = video["path"].split("/")
    filename = video["filename"]
    folder_path = "/".join(path_parts[:-1]) + "/" if len(path_parts) > 1 else ""

    # Determine FPS from our lookup, default to 30 if not found
    fps = video.get("fps", 30)

    try:
        # Call the original process_video function
        point_cloud = tapir_point_cloud.TapirPointCloud()
        result = point_cloud.process_video(folder_path, filename, fps)

        # Generate relative URL path to output file
        output_filename = "SEMI_DENSE_" + filename
        output_rel_path = os.path.join("output", output_filename)

        return {"success": True, "output_filename": output_filename, "output_path": output_rel_path, "fps": fps}
    except Exception as e:
        import traceback

        stack_trace = traceback.format_exc()
        print(f"Error processing video: {str(e)}\n{stack_trace}")
        return {"success": False, "error": str(e)}