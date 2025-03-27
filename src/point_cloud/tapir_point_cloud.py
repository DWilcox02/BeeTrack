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
import sys
import json

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

    def process_video(self, path: str, filename: str, fps: int):
        print(f"\nProcessing video from: {path}")
        start_time = time.time()

        print("Reading video frames...")
        orig_frames = media.read_video(DATA_DIR + path + filename)
        height, width = orig_frames.shape[1:3]
        print(f"Video loaded: {len(orig_frames)} frames at {fps} FPS, resolution: {width}x{height}")

        # Limit to first second of video
        orig_frames = orig_frames[:fps]
        print(f"Limited to first {len(orig_frames)} frames (1 second)")

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

        output_path = OUTPUT_DIR + "SEMI_DENSE_" + filename
        print(f"Saving output video to: {output_path}")
        media.write_video(output_path, video, fps=fps)

        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")

        return video, fps