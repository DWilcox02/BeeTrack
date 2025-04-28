import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import haiku as hk  # noqa: F401
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from functools import partial  # noqa: F401
import importlib.util
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tqdm import tqdm
import time
import torch
import os
import gc
import tempfile


NUM_SLICES = 3


# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

POINT_CLOUD_DIR = os.path.join(BACKEND_DIR, "point_cloud/")
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "point_cloud_interface", os.path.join(POINT_CLOUD_DIR, "point_cloud_interface.py")
)
point_cloud_interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(point_cloud_interface)

DATA_DIR = "data/"
OUTPUT_DIR = "output/"

torch.cuda.empty_cache()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

print("Starting video processing pipeline...")

MODEL_TYPE = "tapir"

if MODEL_TYPE == "tapir":
    checkpoint_path = "tapir/tapir_checkpoint_panning.npy"
else:
    checkpoint_path = "bootstapir/bootstapir_checkpoint_v2.npy"
checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_path)
print(f"Loading checkpoint from: {checkpoint_path}")
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state["params"], ckpt_state["state"]

kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
if MODEL_TYPE == "bootstapir":
    kwargs.update(dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0))
print("Initializing TAPIR model...")
tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)


class TapirPointCloud(point_cloud_interface.PointCloudInterface):
    def __init__(self):
        self.log_fn = print  # Default to standard print

    def set_logger(self, log_fn):
        self.log_fn = log_fn

    def log(self, message):
        self.log_fn(message)

    def sample_grid_points(self, frame_idx, height, width, stride=1):
        points = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
        points = points.transpose(1, 2, 0)
        out_height, out_width = points.shape[0:2]
        frame_idx = np.ones((out_height, out_width, 1)) * frame_idx
        points = np.concatenate((frame_idx, points), axis=-1).astype(np.int32)
        points = points.reshape(-1, 3)  # [out_height*out_width, 3]
        return points

    def convert_select_points_to_query_points(self, query_frame, points, height_ratio, width_ratio):
        """Convert select points to query points with linear interpolation.

        Args:
        points (json): [{'x': _, 'y': _}, ...]

        Returns:
        query_points: [num_points, 3], in [t, y, x]
        """
        points_array = np.array([(point['x'], point['y']) for point in points], dtype=np.float32)
    
        # For quadrilateral interpolation, we need the 4 corners in a specific order
        # Assuming points form a quadrilateral with 4 points
        if len(points_array) != 4:
            raise ValueError("Expected exactly 4 points for area interpolation")
        
        # Define the corners - this assumes the points are a quadrilateral
        # We need a consistent ordering for the bilinear interpolation
        # Sort points by their position (e.g., top-left, top-right, bottom-right, bottom-left)
        # This is a simple approach - for complex shapes, more sophisticated ordering might be needed
        center = np.mean(points_array, axis=0)
        angles = np.arctan2(points_array[:, 1] - center[1], points_array[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        quad_points = points_array[sorted_indices]
        
        # Number of subdivisions along each dimension
        n_subdivs = 5  # This will create approximately 25 points across the area
        
        # Generate a grid of points across the entire area
        area_points = []
        
        for i in range(n_subdivs):
            for j in range(n_subdivs):
                # Parameters for bilinear interpolation
                u = i / (n_subdivs - 1)
                v = j / (n_subdivs - 1)
                
                # Bilinear interpolation formula
                # P(u,v) = (1-u)(1-v)P00 + u(1-v)P10 + (1-u)vP01 + uvP11
                point = (1 - u) * (1 - v) * quad_points[0] + \
                        u * (1 - v) * quad_points[1] + \
                        u * v * quad_points[2] + \
                        (1 - u) * v * quad_points[3]
                        
                area_points.append(point)
        
        # Convert interpolated points to numpy array
        area_points = np.array(area_points, dtype=np.float32)
        
        # Create the query points
        query_points = np.zeros(shape=(len(area_points), 3), dtype=np.float32)
        query_points[:, 0] = query_frame
        query_points[:, 1] = area_points[:, 1] * height_ratio  # y
        query_points[:, 2] = area_points[:, 0] * width_ratio   # x

        return query_points

    def recalculate_query_points(
        self,
        point_cloud_slice,
        bee_skeleton,
        query_frame,
        height,
        width,
        resize_height,
        resize_width,
        previous_trajectory,
    ):
        # Get new midpoint and trajectory
        midpoint = point_cloud_slice.get_final_mean()
        trajectory = point_cloud_slice.get_trajectory(previous_trajectory)
        trajectory = trajectory / np.linalg.norm(trajectory)

        self.log(f"Midpoint: {midpoint}, Trajectory: {trajectory}")

        # Use BeeSkeleton to calculate new positions based on midpoint and trajectory
        new_positions = bee_skeleton.calculate_new_positions(midpoint, trajectory)

        # Format points for conversion
        points = [
            {"x": new_positions["head"]["x"], "y": new_positions["head"]["y"], "color": "red"},
            {"x": new_positions["butt"]["x"], "y": new_positions["butt"]["y"], "color": "green"},
            {"x": new_positions["left"]["x"], "y": new_positions["left"]["y"], "color": "blue"},
            {"x": new_positions["right"]["x"], "y": new_positions["right"]["y"], "color": "purple"},
        ]

        self.log(f"Recalculated points: {points}")

        # Convert to query points
        height_ratio = resize_height / height
        width_ratio = resize_width / width
        query_points = self.convert_select_points_to_query_points(
            query_frame=query_frame, points=points, height_ratio=height_ratio, width_ratio=width_ratio
        )

        return query_points, trajectory


    def process_video(
        self,
        path: str,
        filename: str,
        fps: int,
        max_segments=None,
        save_intermediate=True,
        predefined_points=None,
    ):
        self.log(f"\nProcessing video from: {path}")
        start_time = time.time()

        self.log("Reading video frames...")
        orig_frames = media.read_video(DATA_DIR + path + filename)
        height, width = orig_frames.shape[1:3]
        total_frames = len(orig_frames)
        
        # Normalize FPS to a maximum of 15
        normalized_fps = min(fps, 15)
        if normalized_fps < fps:
            self.log(f"Normalizing FPS from {fps} to {normalized_fps}")
            # Calculate the frame sampling interval
            sampling_interval = fps // normalized_fps
            # Sample frames at the calculated interval
            orig_frames = orig_frames[::sampling_interval]
            total_frames = len(orig_frames)
            # Update fps to the normalized value
            fps = normalized_fps
        
        self.log(f"Video loaded: {total_frames} frames at {fps} FPS, resolution: {width}x{height}")

        # Calculate how many segments we need to process
        total_segments = (total_frames + fps - 1) // fps  # Ceiling division
        if max_segments is not None:
            segments_to_process = min(total_segments, max_segments)
        else:
            segments_to_process = total_segments
        # TODO: Remove
        segments_to_process = NUM_SLICES

        self.log(f"Video will be processed in {segments_to_process} segments")

        # Create temp directory for intermediate results if needed
        temp_dir = None
        segment_paths = []

        if save_intermediate:
            temp_dir = tempfile.mkdtemp(prefix="video_segments_")
            self.log(f"Using temporary directory for intermediate results: {temp_dir}")
        else:
            # If not saving intermediate results, store in memory
            processed_segments = []

        
        # Define query points for slice
        resize_height = 256
        resize_width = 256
        query_frame = 0
        stride = 8
        # predefined_points = None # TODO: Remove
        if predefined_points is None:
            query_points = self.sample_grid_points(query_frame, resize_height, resize_width, stride)
        else:
            bee_skeleton = point_cloud_interface.BeeSkeleton(predefined_points)
            height_ratio = resize_height / height
            width_ratio = resize_width / width
            query_points = self.convert_select_points_to_query_points(query_frame=query_frame, points=predefined_points, height_ratio=height_ratio, width_ratio=width_ratio)
            current_trajectory = bee_skeleton.initial_trajectory
        # print(query_points)

        # Process each segment
        for i in range(segments_to_process):
            start_frame = i * fps
            end_frame = min((i + 1) * fps, total_frames)

            self.log(f"Processing segment {i + 1}/{segments_to_process} (frames {start_frame} to {end_frame})...")
            orig_frames_slice = orig_frames[start_frame:end_frame]

            # Process the slice
            try:                
                slice_result = self.process_video_slice(orig_frames_slice, width, height, query_points, resize_width=resize_width, resize_height=resize_height)
                if predefined_points is not None:
                    query_points, current_trajectory = self.recalculate_query_points(
                        slice_result, 
                        bee_skeleton, 
                        query_frame, 
                        height, 
                        width, 
                        resize_height, 
                        resize_width, 
                        current_trajectory
                    )
                video_segment = slice_result.get_video()

                if save_intermediate:
                    # Save segment to disk
                    segment_path = os.path.join(temp_dir, f"segment_{i:04d}.npy")
                    np.save(segment_path, video_segment)
                    segment_paths.append(segment_path)
                    # Free memory
                    del video_segment
                    gc.collect()
                else:
                    # Store in memory
                    processed_segments.append(video_segment)

                self.log(f"Successfully processed segment {i + 1}")
            except Exception as e:
                self.log(f"Error processing segment {i + 1}: {str(e)}")
                import traceback

                self.log(traceback.format_exc())

        # Prepare final video
        final_output_path = OUTPUT_DIR + "POINT_CLOUD_" + filename

        if save_intermediate and segment_paths:
            self.log("Combining segments and writing final video...")

            # Load all segments and concatenate
            all_frames = []
            for segment_path in segment_paths:
                self.log(f"Loading segment: {os.path.basename(segment_path)}")
                segment = np.load(segment_path)
                all_frames.append(segment)
                # Remove the file after loading
                os.remove(segment_path)

            # Concatenate and write
            self.log(f"Concatenating {len(all_frames)} segments...")
            full_video = np.concatenate(all_frames, axis=0)
            self.log(f"Writing final video ({len(full_video)} frames)...")
            media.write_video(final_output_path, full_video, fps=fps)

            # Clean up temp dir
            os.rmdir(temp_dir)

            self.log(f"Saved output video to: {final_output_path}")
            elapsed_time = time.time() - start_time
            self.log(f"\nProcessing completed in {elapsed_time:.2f} seconds")

            return None, fps  # Return None since we wrote directly to disk

        elif not save_intermediate and processed_segments:
            self.log("Concatenating all processed segments...")
            full_video = np.concatenate(processed_segments, axis=0)

            self.log(f"Saving output video to: {final_output_path}")
            media.write_video(final_output_path, full_video, fps=fps)

            elapsed_time = time.time() - start_time
            self.log(f"\nProcessing completed in {elapsed_time:.2f} seconds")

            return full_video, fps

        else:
            self.log("No video segments were processed")
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            return None, fps

    def process_video_slice(
        self,
        orig_frames,
        width,
        height,
        query_points,
        resize_width = 256,
        resize_height = 256,
    ):
        """Process a slice of video frames and return the processed segment."""

        self.log("Preprocessing frames...")
        frames = media.resize_video(orig_frames, (resize_height, resize_width))
        frames = model_utils.preprocess_frames(frames[None])
        self.log("Generating feature grids...")
        feature_grids = tapir.get_feature_grids(frames, is_training=False)
        chunk_size = 64

        def chunk_inference(query_points):
            query_points = query_points.astype(np.float32)[None]

            outputs = tapir(
                video=frames,
                query_points=query_points,
                is_training=False,
                query_chunk_size=chunk_size,
                feature_grids=feature_grids,
            )
            tracks, occlusions, expected_dist = (
                outputs["tracks"],
                outputs["occlusion"],
                outputs["expected_dist"],
            )

            visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
            return tracks[0], visibles[0]

        self.log("JIT compiling inference function...")
        chunk_inference = jax.jit(chunk_inference)

        self.log("Processing track points...")
        total_chunks = (query_points.shape[0] + chunk_size - 1) // chunk_size  # noqa: F841
        tracks = []
        visibles = []

        # Create a custom tqdm class that logs progress
        class LoggingTqdm:
            def __init__(self, iterable, desc, total, log_fn):
                self.iterable = iterable
                self.desc = desc
                self.total = total
                self.log_fn = log_fn
                self.current = 0
                self.last_logged = -1

            def __iter__(self):
                for item in self.iterable:
                    self.current += 1
                    # Log progress at 10% intervals
                    progress_pct = int((self.current / self.total) * 10)
                    if progress_pct > self.last_logged:
                        self.last_logged = progress_pct
                        self.log_fn(
                            f"{self.desc}: {self.current}/{self.total} ({self.current / self.total * 100:.1f}%)"
                        )
                    yield item

        # Use regular tqdm for console display, but also log progress at intervals
        for i in LoggingTqdm(
            tqdm(range(0, query_points.shape[0], chunk_size), desc="Processing chunks"),
            "Processing chunks",
            (query_points.shape[0] + chunk_size - 1) // chunk_size,
            self.log,
        ):
            query_points_chunk = query_points[i : i + chunk_size]
            num_extra = chunk_size - query_points_chunk.shape[0]
            if num_extra > 0:
                query_points_chunk = np.concatenate([query_points_chunk, np.zeros([num_extra, 3])], axis=0)
            tracks2, visibles2 = chunk_inference(query_points_chunk)
            if num_extra > 0:
                tracks2 = tracks2[:-num_extra]
                visibles2 = visibles2[:-num_extra]
            tracks.append(tracks2)
            visibles.append(visibles2)

        self.log("Concatenating results...")
        tracks = jnp.concatenate(tracks, axis=0)
        visibles = jnp.concatenate(visibles, axis=0)

        self.log("Converting coordinates and generating visualization...")
        tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
        slice_result = point_cloud_interface.PointCloudSlice(orig_frames, tracks, visibles)
        

        return slice_result
