import uuid
import json
import threading
import time
import queue
from flask import Blueprint, request, jsonify, url_for, Response, current_app
from ..utils.logging_utils import log_message, init_job_logging, cleanup_job, processing_logs, processing_locks, point_data_store
from ..config import POINT_CLOUD_AVAILABLE
from ..point_cloud_adapter import process_video_wrapper, process_video_wrapper_with_points, set_log_message_function

processing_bp = Blueprint("processing", __name__)
set_log_message_function(log_message)


@processing_bp.route("/api/logs/<job_id>", methods=["GET"])
def get_logs(job_id):
    """Stream logs for a specific processing job."""

    def generate():
        if job_id not in processing_logs:
            # Create a new queue for this job if it doesn't exist
            init_job_logging(job_id)

        log_queue = processing_logs[job_id]

        # Send initial message
        yield f"data: Processing started for job {job_id}\n\n"

        # Keep the connection open for at most 30 minutes
        end_time = time.time() + 30 * 60

        while time.time() < end_time:
            try:
                # Try to get a message with a timeout
                message = log_queue.get(timeout=1)

                # Send the message to the client
                yield f"data: {message}\n\n"

                # Mark the message as processed
                log_queue.task_done()

                # Check if this is the final message
                if message.startswith("DONE:") or message.startswith("ERROR:"):
                    break

            except queue.Empty:
                # Send a keep-alive message every second
                yield "data: \n\n"  # Empty message to keep connection alive

    return Response(generate(), mimetype="text/event-stream")


@processing_bp.route("/api/job_status/<job_id>", methods=["GET"])
def job_status(job_id):
    """Get the current status and result of a processing job."""
    if job_id not in processing_logs:
        return jsonify({"error": "Job not found"}), 404

    # Check if job has a result message
    with processing_locks[job_id]:
        # Make a copy of all messages
        messages = list(processing_logs[job_id].queue)
        result = None
        status = "processing"

        for msg in messages:
            if msg.startswith("RESULT:"):
                # Extract the JSON result
                result = json.loads(msg[7:])
                status = "completed"
                break
            elif msg.startswith("ERROR:"):
                status = "error"
                break

    return jsonify({"job_id": job_id, "status": status, "result": result})

@processing_bp.route("/api/process_video_with_points", methods=["POST"])
def process_video_with_points():
    """API endpoint to run point cloud processing on a video."""
    if not POINT_CLOUD_AVAILABLE:
        return jsonify({"error": "Point cloud processing is not available"}), 500

    data = request.json
    video_path = data.get("video_path")
    session_id = data.get("session_id")

    if not video_path:
        return jsonify({"error": "No video path provided"}), 400
    if not session_id:
        return jsonify({"error": "No session id provided"}), 400

    session_data = point_data_store[session_id]
    points = session_data["points"]

    # Create a unique job ID
    job_id = str(uuid.uuid4())

    # Initialize logging for this job
    init_job_logging(job_id)

    # Function to run the processing in a background thread
    def run_processing():
        try:
            # Process the video
            result = process_video_wrapper_with_points(video_path, points, job_id)

            if result.get("success", False):
                # Create URL for the processed video
                output_url = url_for("video.serve_processed_video", filename=result["output_filename"])
                result["output_url"] = output_url

                # Signal completion
                log_message(job_id, f"DONE: Processing completed successfully.")

                # Store result for later retrieval
                log_message(job_id, f"RESULT:{json.dumps(result)}")
            else:
                error_msg = result.get("error", "Unknown error during processing")
                log_message(job_id, f"ERROR: {error_msg}")

        except Exception as e:
            import traceback

            stack_trace = traceback.format_exc()
            error_msg = f"Error processing video: {str(e)}"
            current_app.logger.error(f"{error_msg}\n{stack_trace}")
            log_message(job_id, f"ERROR: {error_msg}")

        # Clean up eventually
        cleanup_thread = threading.Thread(target=cleanup_job, args=(job_id,))
        cleanup_thread.daemon = True
        cleanup_thread.start()

    # Start processing in a background thread
    processing_thread = threading.Thread(target=run_processing)
    processing_thread.daemon = True
    processing_thread.start()

    # Return immediately with the job ID
    return jsonify({"job_id": job_id})
