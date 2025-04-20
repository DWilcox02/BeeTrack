from flask import Blueprint, render_template
from ..utils.video_utils import get_processed_videos
from ..config import POINT_CLOUD_AVAILABLE

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Render the main index page with the list of videos."""
    # Get all processed videos from the output directory
    processed_videos = get_processed_videos()

    # Get videos variable from the Flask app
    videos = main_bp.videos

    return render_template(
        "index.html", videos=videos, processed_videos=processed_videos, point_cloud_available=POINT_CLOUD_AVAILABLE
    )
