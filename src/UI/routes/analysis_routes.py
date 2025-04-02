import os
import uuid
import cv2
import base64
import plotly.graph_objects as go
import plotly.io as pio
from flask import Blueprint, render_template, jsonify, current_app, request
from ..utils.logging_utils import log_message, point_data_store, processing_logs
from ..config import DATA_FOLDER, OUTPUT_FOLDER, POINT_CLOUD_AVAILABLE

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/play/<path:filename>/frame_analysis")
def frame_analysis(filename):
    """Render the analysis page for the first frame of the video."""
    try:
        # Check if there's a processed version of this video
        processed_filename = "SEMI_DENSE_" + os.path.basename(filename)
        processed_exists = os.path.exists(os.path.join(OUTPUT_FOLDER, processed_filename))

        # Get the video metadata
        videos = analysis_bp.videos
        video = videos.get(filename)
        if not video:
            return "Video not found", 404

        # Extract the first frame
        directory = os.path.join(DATA_FOLDER, video["path"])
        video_path = os.path.join(directory, video["filename"])

        # Extract first frame using OpenCV
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Failed to extract frame from video", 500

        # Convert BGR to RGB for proper display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if the image is too large
        max_dim = 1024
        h, w = frame_rgb.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            height, width = new_h, new_w
        else:
            height, width = h, w

        # Convert to base64 for embedding in plotly
        _, buffer = cv2.imencode(".jpg", frame_rgb)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # Create session ID for this analysis session
        session_id = str(uuid.uuid4())

        # Create initial points in a rectangle shape
        points = [
            {"x": width * 0.25, "y": height * 0.25, "color": "red"},
            {"x": width * 0.75, "y": height * 0.25, "color": "green"},
            {"x": width * 0.75, "y": height * 0.75, "color": "blue"},
            {"x": width * 0.25, "y": height * 0.75, "color": "purple"},
        ]

        # Store the points and image size for this session
        point_data_store[session_id] = {"points": points, "width": width, "height": height, "filename": filename}

        # Create a plotly figure
        fig = go.Figure()

        # Add the image as a layout image with correct positioning
        fig.add_layout_image(
            dict(
                source=f"data:image/jpeg;base64,{frame_base64}",
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=width,
                sizey=height,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
        )

        # Add each point as a scatter trace
        for point in points:
            fig.add_trace(
                go.Scatter(
                    x=[point["x"]],
                    y=[point["y"]],
                    mode="markers",
                    marker=dict(size=15, color=point["color"]),
                    name=f"Point ({point['color']})",
                )
            )

        # Configure the layout
        fig.update_layout(
            xaxis=dict(
                range=[0, width],
                title="X",
                fixedrange=True,
                showgrid=False,  # Hide grid for cleaner appearance with image
            ),
            yaxis=dict(
                range=[height, 0],  # Invert y-axis for image coordinates
                title="Y",
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
                showgrid=False,  # Hide grid for cleaner appearance with image
            ),
            showlegend=True,
            dragmode="pan",  # Allow panning but disable other interactions
            height=min(600, height + 100),
            width=min(800, width + 100),
            margin=dict(l=50, r=50, b=50, t=50),
            title="First Frame Analysis",
            template="plotly_white",  # Use a cleaner template
        )

        # Configure modebar with necessary tools only
        fig.update_layout(modebar=dict(remove=["select", "lasso", "autoScale", "resetScale"]))

        # Convert the figure to HTML with proper config for image display
        plot_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=True,
            config={
                "displayModeBar": True,  # Show the mode bar
                "staticPlot": False,  # Allow basic interactivity - IMPORTANT for images
                "scrollZoom": False,  # Disable scroll zooming
                "displaylogo": False,  # Hide the Plotly logo
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "resetScale2d"],
            },
        )

        # Return the analysis template with the plotly figure
        return render_template(
            "frame_analysis.html",
            filename=filename,
            plot_html=plot_html,
            session_id=session_id,
            processed_filename=processed_filename if processed_exists else None,
            point_cloud_available=POINT_CLOUD_AVAILABLE,
        )

    except Exception as e:
        current_app.logger.error(f"Error in frame analysis: {str(e)}")
        return f"Error: {str(e)}", 500


@analysis_bp.route("/api/update_point", methods=["POST"])
def update_point():
    """Update a point's position."""
    try:
        data = request.json
        session_id = data.get("session_id")
        point_index = data.get("point_index")
        x = data.get("x")
        y = data.get("y")

        if session_id not in point_data_store:
            return jsonify({"error": "Session not found"}), 404

        session_data = point_data_store[session_id]

        if point_index < 0 or point_index >= len(session_data["points"]):
            return jsonify({"error": "Invalid point index"}), 400

        # Update the point position
        session_data["points"][point_index]["x"] = x
        session_data["points"][point_index]["y"] = y

        # Generate updated plot
        width = session_data["width"]
        height = session_data["height"]

        fig = go.Figure()

        # Add each point as a scatter trace
        for i, point in enumerate(session_data["points"]):
            fig.add_trace(
                go.Scatter(
                    x=[point["x"]],
                    y=[point["y"]],
                    mode="markers",
                    marker=dict(size=15 if i != point_index else 20, color=point["color"]),
                    name=f"Point ({point['color']})",
                )
            )

        # Configure the layout
        fig.update_layout(
            xaxis=dict(range=[0, width], title="X", fixedrange=True),
            yaxis=dict(range=[height, 0], title="Y", scaleanchor="x", scaleratio=1, fixedrange=True),
            showlegend=True,
            dragmode="pan",
            height=min(600, height + 100),
            width=min(800, width + 100),
            margin=dict(l=50, r=50, b=50, t=50),
        )

        # Convert to JSON for update
        plot_json = fig.to_json()

        return jsonify({"success": True, "plot_data": plot_json, "point": session_data["points"][point_index]})

    except Exception as e:
        current_app.logger.error(f"Error updating point: {str(e)}")
        return jsonify({"error": str(e)}), 500


@analysis_bp.route("/api/save_points", methods=["POST"])
def save_points():
    """Save and log the positions of the 4 points."""
    try:
        data = request.json
        session_id = data.get("session_id")

        if session_id not in point_data_store:
            return jsonify({"error": "Session not found"}), 404

        session_data = point_data_store[session_id]
        points = session_data["points"]

        # Log the points
        message = "Points positions:\n"
        for point in points:
            point_msg = f"{point['color']}: ({point['x']:.2f}, {point['y']:.2f})"
            message += point_msg + "\n"
            current_app.logger.info(point_msg)

        # Add to processing logs if a job_id is provided
        job_id = data.get("job_id")
        if job_id and job_id in processing_logs:
            log_message(job_id, message)

        return jsonify({"success": True, "message": message, "points": points})

    except Exception as e:
        current_app.logger.error(f"Error saving points: {str(e)}")
        return jsonify({"error": str(e)}), 500
