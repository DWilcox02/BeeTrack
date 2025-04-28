import queue
import threading
import time

# Global dictionaries to store processing logs and locks
processing_logs = {}
processing_locks = {}

# Dictionary to store point data for frame analysis
point_data_store = {}


def log_message(job_id, message):
    """Log a message to be sent to the frontend."""
    if job_id in processing_logs:
        processing_logs[job_id].put(message)


def cleanup_job(job_id, timeout=300):
    """Clean up job resources after a timeout."""
    time.sleep(timeout)  # Keep logs for 5 minutes
    if job_id in processing_logs:
        del processing_logs[job_id]
    if job_id in processing_locks:
        del processing_locks[job_id]


def init_job_logging(job_id):
    """Initialize logging for a job."""
    processing_logs[job_id] = queue.Queue()
    processing_locks[job_id] = threading.Lock()
    return processing_logs[job_id], processing_locks[job_id]
