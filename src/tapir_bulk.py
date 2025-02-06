import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
import haiku as hk
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
# import tree

MODEL_TYPE = "bootstapir"  # 'tapir' or 'bootstapir'

from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils

# @title Load Checkpoint {form-width: "25%"}
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


def sample_grid_points(frame_idx, height, width, stride=1):
    """Sample grid points with (time height, width) order."""
    points = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    points = points.transpose(1, 2, 0)
    out_height, out_width = points.shape[0:2]
    frame_idx = np.ones((out_height, out_width, 1)) * frame_idx
    points = np.concatenate((frame_idx, points), axis=-1).astype(np.int32)
    points = points.reshape(-1, 3)  # [out_height*out_width, 3]
    return points

DATA_DIR = "data/"

# v0 = {"filename": "dance_15_secs_700x700_50fps.mp4", "fps": 50}

# v1 = {"filename": "full_hive_23_secs_4k.mp4", "fps": 30}

# v2 = {"filename": "outside_botgard_5_secs_1080_50fps.mp4", "fps": 50}

# videos = [v0, v1, v2]
