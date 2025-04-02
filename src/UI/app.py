import os
import json
from flask import Flask

from .config import DATA_FOLDER, OUTPUT_FOLDER, POINT_CLOUD_AVAILABLE
from .routes.main_routes import main_bp
from .routes.video_routes import video_bp
from .routes.processing_routes import processing_bp
from .routes.analysis_routes import analysis_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Configure app
    app.config["DATA_FOLDER"] = DATA_FOLDER
    app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

    # Load video metadata
    videos = json.load(open(os.path.join(DATA_FOLDER, "video_meta.json")))

    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(processing_bp)
    app.register_blueprint(analysis_bp)

    # Attach videos data to each blueprint for access
    main_bp.videos = videos
    video_bp.videos = videos
    analysis_bp.videos = videos

    return app


# Create the Flask application
app = create_app()

if __name__ == "__main__":
    # Print the data directory path on startup for verification
    print(f"Using data directory: {DATA_FOLDER}")
    print(f"Using output directory: {OUTPUT_FOLDER}")
    print(f"Point Cloud processing available: {POINT_CLOUD_AVAILABLE}")
    app.run(host="127.0.0.1", port=5001, debug=True)
