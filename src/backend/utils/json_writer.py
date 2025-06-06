import numpy as np
import json
import os

class JSONWriter():

    def __init__(self):
        self.log_fn = print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "to_pylist"):  # Handle ArrayImpl objects
            return obj.to_pylist()
        elif hasattr(obj, "tolist"):  # Handle other array-like objects
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_serializable(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
        

    def combine_and_write_tracks(self, tracks, final_tracks_output_path):
        try:
            tracks_list = self.convert_to_serializable(tracks)

            os.makedirs(os.path.dirname(final_tracks_output_path), exist_ok=True)

            tracks_data = {
                "num_points": len(tracks_list),
                "num_frames": len(tracks_list[0]) if tracks_list else 0,
                "tracks": tracks_list,
            }

            # Write to JSON file
            with open(final_tracks_output_path, "w") as f:
                json.dump(tracks_data, f, indent=2)

            self.log(f"Tracks successfully written to {final_tracks_output_path}")

        except Exception as e:
            self.log(f"Error writing tracks to file: {e}")
            raise

    def write_errors(self, errors, final_errors_output_path):
        os.makedirs(os.path.dirname(os.path.abspath(final_errors_output_path)), exist_ok=True)

        serializable_errors = self.convert_to_serializable(errors)

        with open(final_errors_output_path, "w") as f:
            json.dump(serializable_errors, f, indent=2)

        self.log(f"Errors successfully written to {final_errors_output_path}")
