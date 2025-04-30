import os
import sys

# Paths configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
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

# # Point cloud availability - try to import the adapter
# # Use a more specific import path that matches the project structure
# try:
#     # First, try to import from current directory (in case it was moved there)
#     try:
#         from ..point_cloud.point_cloud_adapter import process_video_wrapper, set_log_message_function

#         POINT_CLOUD_AVAILABLE = True
#     except ImportError:
#         # If not in current directory, try to import from original location
#         sys.path.append(CURRENT_DIR)  # Make sure the UI directory is in path
#         from src.backend.point_cloud.point_cloud_adapter import process_video_wrapper, set_log_message_function

#         POINT_CLOUD_AVAILABLE = True
# except ImportError:
#     print(f"Warning: point_cloud_adapter module could not be imported")
#     print(f"Looked in: {CURRENT_DIR}")
#     POINT_CLOUD_AVAILABLE = False
