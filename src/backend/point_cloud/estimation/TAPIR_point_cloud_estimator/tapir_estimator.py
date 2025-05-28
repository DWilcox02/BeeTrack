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
from threading import Event
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
        stop_event: Event,
        resize_width=256,
        resize_height=256,
    ):
        """Process a slice of video frames and return the processed segment."""

        if stop_event is not None and stop_event.is_set():
            self.log("TAPIR processing stopped by user request")
            return None

        self.log("Preprocessing frames...")
        frames = media.resize_video(orig_frames, (resize_height, resize_width))
        frames = model_utils.preprocess_frames(frames[None])
        self.log("Generating feature grids...")
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        chunk_size = 64

        # Stop check after preprocessing
        if stop_event is not None and stop_event.is_set():
            self.log("TAPIR processing stopped by user request")
            return None

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

                    if stop_event is not None and stop_event.is_set():
                        self.log("TAPIR processing stopped by user request")
                        return None
                    
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
            if stop_event is not None and stop_event.is_set():
                self.log("TAPIR processing stopped by user request")
                return None
            
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

        if stop_event is not None and stop_event.is_set():
            self.log("TAPIR processing stopped by user request")
            return None
        
        self.log("Concatenating results...")
        tracks = jnp.concatenate(tracks, axis=0)
        # visibles = jnp.concatenate(visibles, axis=0)
        visibles = jnp.ones((tracks.shape[0], tracks.shape[1]), dtype=bool)

        self.log("Converting coordinates and generating visualization...")
        tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))

        slice_result = TapirEstimatorSlice(orig_frames, tracks, visibles)

        return slice_result
    
    def process_cached_dance_15(self, orig_frames):
        final_positions_1 = np.array(
            [
                [405.46313, 275.38248],
                [381.6413, 313.76587],
                [390.66608, 300.41714],
                [387.59235, 295.32755],
                [390.50797, 286.26953],
                [395.26138, 300.64355],
                [401.66403, 316.95172],
                [384.75418, 265.34808],
                [394.65866, 273.7934],
                [444.30222, 336.88055],
                [409.3038, 298.57706],
                [426.74088, 316.58762],
                [401.03464, 256.6683],
                [388.9503, 264.30646],
                [399.73303, 266.10452],
                [405.46313, 275.38248],
                [401.4218, 288.46118],
                [416.88834, 293.4696],
                [395.64093, 263.16043],
                [412.37546, 260.35382],
                [411.20096, 270.23154],
                [413.80917, 277.36972],
                [423.64462, 289.82068],
                [426.5883, 260.5347],
                [396.14966, 242.51627],
                [418.23056, 267.11713],
                [426.26782, 269.00394],
                [433.195, 279.62772],
            ]
        )
        final_positions_2 = np.array(
            [
                [455.06262, 285.04],
                [436.16556, 314.25174],
                [423.07205, 288.02032],
                [414.90656, 242.61607],
                [439.1224, 303.39917],
                [452.66943, 312.73285],
                [461.38364, 318.084],
                [434.61755, 280.2224],
                [439.73785, 284.81198],
                [448.29578, 294.47937],
                [457.21936, 302.00302],
                [463.71844, 307.39337],
                [429.40933, 263.32748],
                [422.74567, 254.26103],
                [423.76358, 278.0618],
                [455.06262, 285.04],
                [461.28018, 292.49753],
                [469.94888, 297.92535],
                [445.76923, 265.1062],
                [447.42215, 269.57187],
                [465.18997, 281.74393],
                [468.7299, 283.39752],
                [476.41333, 287.0656],
                [455.07242, 259.26138],
                [453.39584, 265.24167],
                [457.60773, 268.52423],
                [444.09778, 239.69925],
                [485.95218, 280.7014],
            ]
        )
        final_positions_3 = np.array(
            [
                [477.09567, 316.59927],
                [463.3938, 333.95972],
                [469.2752, 323.49704],
                [478.1107, 327.95706],
                [452.1015, 338.22614],
                [492.24216, 335.56848],
                [479.63177, 332.51202],
                [467.63663, 308.9802],
                [478.55652, 314.22238],
                [483.066, 328.90707],
                [505.89572, 334.66052],
                [502.77097, 336.50043],
                [465.41916, 293.8194],
                [472.2407, 297.5977],
                [479.07193, 303.5983],
                [477.09567, 316.59927],
                [479.58975, 350.8451],
                [481.4729, 373.41364],
                [480.977, 286.0829],
                [489.2604, 295.69406],
                [495.84247, 307.35104],
                [494.84537, 349.74948],
                [462.66544, 400.27606],
                [491.9859, 281.10352],
                [494.74686, 294.34662],
                [499.17038, 306.80573],
                [481.31625, 395.6426],
                [477.26422, 410.91675],
            ]
        )
        
        tracks = np.array([final_positions_1, final_positions_2, final_positions_3])
        visibles = jnp.ones((tracks.shape[0], tracks.shape[1]), dtype=bool)
        return TapirEstimatorSlice(orig_frames, tracks, visibles)