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
