import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import torch
import haiku as hk  # noqa: F401
import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
from tapnet.models import tapir_model
from tapnet.utils import model_utils, transforms
from tqdm import tqdm
from src.backend.point_cloud.estimation.TAPIR_point_cloud_estimator.tapir_estimator_slice import TapirEstimatorSlice
from src.backend.point_cloud.estimation.point_cloud_estimator_interface import PointCloudEstimatorInterface


TAPIR_DIR = os.path.dirname(os.path.abspath(__file__))
ESTIMATION_DIR = os.path.dirname(TAPIR_DIR)
POINT_CLOUD_DIR = os.path.dirname(ESTIMATION_DIR)
BACKEND_DIR = os.path.dirname(POINT_CLOUD_DIR)
SRC_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/")

MODEL_TYPE = "tapir"


torch.cuda.empty_cache()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


class TapirEstimator(PointCloudEstimatorInterface):
    def __init__(self):
        self.log_fn = print  # Default to standard print

        print("Initializing TAPIR...")


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
        self.tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)

    def process_video_slice(
        self,
        orig_frames,
        width,
        height,
        query_points,
        resize_width=256,
        resize_height=256,
    ):
        """Process a slice of video frames and return the processed segment."""

        self.log("Preprocessing frames...")
        frames = media.resize_video(orig_frames, (resize_height, resize_width))
        frames = model_utils.preprocess_frames(frames[None])
        self.log("Generating feature grids...")
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        chunk_size = 64

        def chunk_inference(query_points):
            query_points = query_points.astype(np.float32)[None]

            outputs = self.tapir(
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
                        self.log_fn(f"{self.desc}: {self.current}/{self.total} ({self.current / self.total * 100:.1f}%)")
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

        slice_result = TapirEstimatorSlice(orig_frames, tracks, visibles)

        return slice_result