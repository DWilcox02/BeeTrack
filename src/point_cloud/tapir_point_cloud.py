import matplotlib
matplotlib.use("Agg")  # Use non-interactive backendimport haiku as hk
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from functools import partial
import importlib.util
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils
from tqdm import tqdm
import time
import torch
import os
import gc
import tempfile


NUM_SLICES = 3


# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
POINT_CLOUD_DIR = os.path.join(SRC_DIR, "point_cloud/")
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")

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
    checkpoint_path = "checkpoints/tapir/tapir_checkpoint_panning.npy"
else:
    checkpoint_path = "checkpoints/bootstapir/bootstapir_checkpoint_v2.npy"
print(f"Loading checkpoint from: {checkpoint_path}")
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state["params"], ckpt_state["state"]

kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
if MODEL_TYPE == "bootstapir":
    kwargs.update(dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0))
print("Initializing TAPIR model...")
tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)


class TapirPointCloud(point_cloud_interface.PointCloudInterface):
    
    def sample_grid_points(self, frame_idx, height, width, stride=1):
        """Sample grid points with (time height, width) order."""
        points = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
        points = points.transpose(1, 2, 0)
        out_height, out_width = points.shape[0:2]
        frame_idx = np.ones((out_height, out_width, 1)) * frame_idx
        points = np.concatenate((frame_idx, points), axis=-1).astype(np.int32)
        points = points.reshape(-1, 3)  # [out_height*out_width, 3]
        return points

    def process_video(self, path: str, filename: str, fps: int, max_segments=None, save_intermediate=True):
        """
        Process a video file, splitting it into segments and optionally saving intermediate results.
        
        Args:
            path: Directory path to the video
            filename: Name of the video file
            fps: Frames per second
            max_segments: Maximum number of segments to process (None for all)
            save_intermediate: Whether to save intermediate results to disk to save memory
        
        Returns:
            Tuple of (processed video array, fps) or (None, fps) if no segments processed
        """
        print(f"\nProcessing video from: {path}")
        start_time = time.time()

        print("Reading video frames...")
        orig_frames = media.read_video(DATA_DIR + path + filename)
        height, width = orig_frames.shape[1:3]
        total_frames = len(orig_frames)
        print(f"Video loaded: {total_frames} frames at {fps} FPS, resolution: {width}x{height}")
        
        # Calculate how many segments we need to process
        total_segments = (total_frames + fps - 1) // fps  # Ceiling division
        if max_segments is not None:
            segments_to_process = min(total_segments, max_segments)
        else:
            segments_to_process = total_segments
        segments_to_process = NUM_SLICES
        
        print(f"Video will be processed in {segments_to_process} segments")
        
        # Create temp directory for intermediate results if needed
        temp_dir = None
        segment_paths = []
        
        if save_intermediate:
            temp_dir = tempfile.mkdtemp(prefix="video_segments_")
            print(f"Using temporary directory for intermediate results: {temp_dir}")
        else:
            # If not saving intermediate results, store in memory
            processed_segments = []
        
        # Process each segment
        for i in range(segments_to_process):
            start_frame = i * fps
            end_frame = min((i + 1) * fps, total_frames)
            
            print(f"Processing segment {i+1}/{segments_to_process} (frames {start_frame} to {end_frame})...")
            orig_frames_slice = orig_frames[start_frame:end_frame]
            
            # Process the slice
            try:
                video_segment = self.process_video_slice(orig_frames_slice, width, height)
                
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
                    
                print(f"Successfully processed segment {i+1}")
            except Exception as e:
                print(f"Error processing segment {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Prepare final video
        final_output_path = OUTPUT_DIR + "SEMI_DENSE_" + filename
        
        if save_intermediate and segment_paths:
            print("Combining segments and writing final video...")
            
            # Load all segments and concatenate
            all_frames = []
            for segment_path in segment_paths:
                segment = np.load(segment_path)
                all_frames.append(segment)
                # Remove the file after loading
                os.remove(segment_path)
            
            # Concatenate and write
            full_video = np.concatenate(all_frames, axis=0)
            media.write_video(final_output_path, full_video, fps=fps)
            
            # Clean up temp dir
            os.rmdir(temp_dir)
            
            print(f"Saved output video to: {final_output_path}")
            elapsed_time = time.time() - start_time
            print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
            
            return None, fps  # Return None since we wrote directly to disk
        
        elif not save_intermediate and processed_segments:
            print("Concatenating all processed segments...")
            full_video = np.concatenate(processed_segments, axis=0)
            
            print(f"Saving output video to: {final_output_path}")
            media.write_video(final_output_path, full_video, fps=fps)
            
            elapsed_time = time.time() - start_time
            print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
            
            return full_video, fps
        
        else:
            print("No video segments were processed")
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            return None, fps

    def process_video_slice(self, orig_frames, width, height):
        resize_height = 256  # @param {type: "integer"}
        resize_width = 256  # @param {type: "integer"}
        stride = 8  # @param {type: "integer"}
        query_frame = 0  # @param {type: "integer"}

        print("Preprocessing frames...")
        frames = media.resize_video(orig_frames, (resize_height, resize_width))
        frames = model_utils.preprocess_frames(frames[None])
        print("Generating feature grids...")
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

        print("JIT compiling inference function...")
        chunk_inference = jax.jit(chunk_inference)

        print("Processing track points...")
        query_points = self.sample_grid_points(query_frame, resize_height, resize_width, stride)
        total_chunks = (query_points.shape[0] + chunk_size - 1) // chunk_size
        tracks = []
        visibles = []

        for i in tqdm(range(0, query_points.shape[0], chunk_size), desc="Processing chunks"):
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

        print("Concatenating results...")
        tracks = jnp.concatenate(tracks, axis=0)
        visibles = jnp.concatenate(visibles, axis=0)

        print("Converting coordinates and generating visualization...")
        tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
        video = viz_utils.plot_tracks_v2(orig_frames, tracks, 1.0 - visibles)

        return video