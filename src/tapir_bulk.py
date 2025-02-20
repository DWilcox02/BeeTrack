import matplotlib
import haiku as hk
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import gc
from functools import partial
import psutil
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils
from tqdm import tqdm
import time

MODEL_TYPE = "bootstapir"

if MODEL_TYPE == "tapir":
    checkpoint_path = "checkpoints/tapir/tapir_checkpoint_panning.npy"
else:
    checkpoint_path = "checkpoints/bootstapir/bootstapir_checkpoint_v2.npy"
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state["params"], ckpt_state["state"]

kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
if MODEL_TYPE == "bootstapir":
    kwargs.update(dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0))
tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)


# Enable memory-efficient JAX operations
jax.config.update("jax_enable_x64", False)  # Use float32 instead of float64
jax.config.update("jax_default_matmul_precision", "bfloat16")  # Use lower precision


# Optimize grid point sampling
@partial(jax.jit, static_argnums=(1, 2, 3))
def sample_grid_points(frame_idx, height, width, stride=1):
    points = jnp.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    points = points.transpose(1, 2, 0)
    out_height, out_width = points.shape[0:2]
    frame_idx = jnp.ones((out_height, out_width, 1)) * frame_idx
    points = jnp.concatenate((frame_idx, points), axis=-1).astype(jnp.int32)
    return points.reshape(-1, 3)

def clear_memory():
    """Clear unused memory"""
    print("Clearing memory...")
    gc.collect()
    backend = jax.lib.xla_bridge.get_backend()
    for device in backend.devices():
        device.memory_stats()  # Force garbage collection on GPU
    print("Memory cleared")


def process_video_in_chunks(file_path, chunk_frames=32):
    """Process video in smaller chunks to manage memory"""
    print(f"Reading video in chunks of {chunk_frames} frames...")
    # Read video info first
    video = media.read_video(file_path)
    total_frames = len(video)
    print(f"Total frames to process: {total_frames}")

    frames = []
    for start_idx in tqdm(range(0, total_frames, chunk_frames), desc="Processing video chunks"):
        end_idx = min(start_idx + chunk_frames, total_frames)
        chunk = video[start_idx:end_idx]
        frames.append(chunk)
        clear_memory()

    return np.concatenate(frames, axis=0)


# Load checkpoint more efficiently
def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    start_time = time.time()
    with open(checkpoint_path, "rb") as f:
        ckpt_state = np.load(f, allow_pickle=True).item()
    print(f"Checkpoint loaded in {time.time() - start_time:.2f} seconds")
    return ckpt_state["params"], ckpt_state["state"]


# Process videos efficiently
def process_video(video_info, data_dir, output_dir, resize_dims=(512, 512), stride=8, chunk_size=32):
    start_time = time.time()
    file_path = f"{data_dir}{video_info['path']}{video_info['filename']}"

    print(f"\n{'=' * 50}")
    print(f"Starting processing for: {video_info['filename']}")
    print(f"{'=' * 50}")

    print(f"\n1. Loading video from: {file_path}")
    video = media.read_video(file_path)
    height, width = video.shape[1:3]
    print(f"Video loaded: {width}x{height}, {len(video)} frames")

    # Immediately resize to save memory
    print("\n2. Preprocessing frames...")
    print("- Storing original frames")
    orig_frames = video
    del video
    clear_memory()

    print(f"- Resizing frames to {resize_dims}")
    frames = media.resize_video(orig_frames, resize_dims)
    frames = model_utils.preprocess_frames(frames[None])
    print("Preprocessing complete")

    # Process feature grids in smaller chunks
    print("\n3. Generating feature grids...")
    chunk_frames = 8  # Reduced chunk size
    feature_grids = []
    n_chunks = (frames.shape[1] + chunk_frames - 1) // chunk_frames

    # Process and concatenate in smaller sub-chunks
    for i in tqdm(range(0, frames.shape[1], chunk_frames), desc="Processing feature grids", total=n_chunks):
        chunk = frames[:, i : i + chunk_frames]
        chunk_grids = tapir.get_feature_grids(chunk, is_training=False)

        # If this is not the first chunk, concatenate with previous
        if feature_grids:
            feature_grids = jax.tree.map(
                lambda prev, new: jnp.concatenate([prev, new], axis=1), feature_grids, chunk_grids
            )
        else:
            feature_grids = chunk_grids

        clear_memory()  # Clear memory after each chunk

    print("\n4. Processing query points...")
    query_points = sample_grid_points(0, resize_dims[0], resize_dims[1], stride)
    print(f"Generated {query_points.shape[0]} query points")

    # Reduce chunk size for inference
    chunk_size = 32  # Smaller chunk size for processing

    tracks = []
    visibles = []

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

    print("\n5. JIT compiling chunk_inference...")
    chunk_inference = jax.jit(chunk_inference)
    print("Compilation complete")

    print("\n6. Processing tracking chunks...")
    n_chunks = (query_points.shape[0] + chunk_size - 1) // chunk_size

    for i in tqdm(range(0, query_points.shape[0], chunk_size)):
        clear_memory()  # Clear memory before each chunk
        query_chunk = query_points[i : i + chunk_size].copy()

        try:
            process = psutil.Process()
            print(f"Memory before chunk {i}: {process.memory_info().rss / (1024 * 1024 * 1024):.2f} GB")
            chunk_tracks, chunk_visibles = chunk_inference(query_chunk)
            tracks.append(chunk_tracks)
            visibles.append(chunk_visibles)
            del chunk_tracks, chunk_visibles, query_chunk
            print(f"Memory after chunk {i}: {process.memory_info().rss / (1024 * 1024 * 1024):.2f} GB")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            raise

        clear_memory()  # Clear memory after each chunk

    print("\n7. Finalizing results...")
    # Concatenate results in smaller batches
    final_tracks = []
    final_visibles = []
    batch_size = 4  # Process concatenation in small batches

    for i in range(0, len(tracks), batch_size):
        batch_tracks = tracks[i : i + batch_size]
        batch_visibles = visibles[i : i + batch_size]

        if final_tracks:
            final_tracks = jnp.concatenate([final_tracks, jnp.concatenate(batch_tracks, axis=0)], axis=0)
            final_visibles = jnp.concatenate([final_visibles, jnp.concatenate(batch_visibles, axis=0)], axis=0)
        else:
            final_tracks = jnp.concatenate(batch_tracks, axis=0)
            final_visibles = jnp.concatenate(batch_visibles, axis=0)

        clear_memory()

    # Convert coordinates
    print("Converting coordinates...")
    final_tracks = transforms.convert_grid_coordinates(final_tracks, (resize_dims[1], resize_dims[0]), (width, height))

    # Generate output video
    print("\n8. Generating output video...")
    video = viz_utils.plot_tracks_v2(orig_frames, final_tracks, 1.0 - final_visibles)
    output_path = f"{output_dir}{video_info['path']}SEMI_DENSE_{video_info['filename']}"
    media.write_video(output_path, video, fps=video_info["fps"])

    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"{'=' * 50}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Output saved to: {output_path}")
    print(f"{'=' * 50}\n")

    clear_memory()
    return output_path


# Main execution
if __name__ == "__main__":
    print("Initializing TAPIR tracking pipeline...")
    DATA_DIR = "data/"
    OUTPUT_DIR = "output/"

    videos = [
        {"path": "Dance_1_min/", "filename": "dance_15_secs_700x700_50fps.mp4", "fps": 50},
        {"path": "Full_Hive_43_mins/", "filename": "full_hive_23_secs_4k.mp4", "fps": 30},
        {"path": "Outside_Florea_6_mins/", "filename": "outside_botgard_5_secs_1080_50fps.mp4", "fps": 50},
        {"path": "Outside_Florea_6_mins/", "filename": "outside_botgard_5_secs_1080_15fps.mp4", "fps": 15},
    ]

    video_number = 0  # Process single video for now
    print(f"\nProcessing video {video_number + 1}/{len(videos)}")
    print(f"Video details: {videos[video_number]}")
    
    output_path = process_video(
        videos[video_number], DATA_DIR, OUTPUT_DIR, resize_dims=(512, 512), stride=8, chunk_size=64
    )
    print(f"Pipeline complete! Final output saved to: {output_path}")