import json
import uuid
from threading import Event
from flask_socketio import SocketIO, emit
from collections.abc import Callable


class FrontendCommunicator():
    
    def __init__(
        self,
        socketio: SocketIO,
        session_id: str,
        log_message: Callable,
        validation_events: dict,
    ):
        self.socketio = socketio
        self.session_id = session_id
        self.log_message = log_message
        self.validation_events = validation_events
    

    def send_frame_data_callback(self, frameData, points, confidence, request_validation):
        # Serialize points to JSON
        points_json = []
        for point in points:
            points_json.append(
                {
                    "x": json.dumps(float(point["x"])),
                    "y": json.dumps(float(point["y"])),
                    "color": point["color"],
                    "radius": json.dumps(float(point["radius"])),
                }
            )

        self.socketio.emit(
            "update_points_with_frame",
            {
                "success": True,
                "points": points_json,
                "frameData": frameData,
                "confidence": json.dumps(float(confidence)),
                "request_validation": json.dumps(str(request_validation)),
            },
        )
        return


    def request_validation_callback(self, job_id):
        # Generate a unique ID for validation request
        request_id = str(uuid.uuid4())

        # Create an event to wait on
        validation_event = Event()
        self.validation_events[request_id] = {"event": validation_event, "response": None}

        # Emit the event with the request ID
        self.socketio.emit("validation_request", {"message": "Please validate this data", "request_id": request_id})

        self.log_message(job_id=job_id, message=f"Requesting frontend validation (ID: {request_id})")

        # Wait for the validation to be completed (with timeout)
        if validation_event.wait(timeout=300):
            response = self.validation_events[request_id]["response"]
            # Clean up
            del self.validation_events[request_id]
            self.log_message(job_id=job_id, message="Validation received, recalc query points")
            return response
        else:
            # Timeout occurred
            del self.validation_events[request_id]
            self.log_message(job_id=job_id, message="Validation request timed out")
            return None
        

    def send_timeline_frame_callback(self, frame, frameIndex):
        self.socketio.emit("add_timeline_frame", {"frame": frame, "frame_index": frameIndex})


    def add_validation_callback(self, validation):
        self.socketio.emit("add_validation", {"validation_point": validation})


    def add_tracks_callback(self, new_tracks):
        # new_tracks: [
        #   {
        #       "frame": int,
        #       "x": float,
        #       "y": float,
        #       "bodypart": string
        #   }
        # ]

        self.socketio.emit("add_tracks", {"new_tracks": new_tracks})
