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

videos = [
    {"path": "Dance_1_min/", "filename": "dance_15_secs_700x700_50fps.mp4", "fps": 50},
    {"path": "Full_Hive_43_mins/", "filename": "full_hive_23_secs_4k.mp4", "fps": 30},
    {"path": "Outside_Florea_6_mins/", "filename": "outside_botgard_5_secs_1080_50fps.mp4", "fps": 50},
    {"path": "Outside_Florea_6_mins/", "filename": "outside_botgard_5_secs_1080_15fps.mp4", "fps": 15},
]


video_number = 3
file_path = DATA_DIR + videos[video_number]["path"] + videos[video_number]["filename"]

orig_frames = media.read_video(file_path)
fps = videos[video_number]["fps"]
height, width = orig_frames.shape[1:3]

# Show video, if needed
media.show_video(orig_frames, fps=fps)

resize_height = 512 
resize_width = 512 
stride = 8 
query_frame = 0 

frames = media.resize_video(orig_frames, (resize_height, resize_width))
frames = model_utils.preprocess_frames(frames[None])
feature_grids = tapir.get_feature_grids(frames, is_training=False)
chunk_size = 64

print("Definintng chunk inference")
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

    # Binarize occlusions
    visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
    return tracks[0], visibles[0]

print("Here 1")
chunk_inference = jax.jit(chunk_inference)


query_points = sample_grid_points(query_frame, resize_height, resize_width, stride)
print("Here 2")
tracks = []
visibles = []
for i in range(0, query_points.shape[0], chunk_size):
    print("Here point ", i)
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
tracks = jnp.concatenate(tracks, axis=0)
visibles = jnp.concatenate(visibles, axis=0)

tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))

# We show the point tracks without rainbows so you can see the input.
video = viz_utils.plot_tracks_v2(orig_frames, tracks, 1.0 - visibles)
# media.show_video(video, fps=fps)

OUTPUT_DIR = "output/"
output_path = OUTPUT_DIR + videos[video_number]["path"] + "SEMI_DENSE_" +  videos[video_number]["filename"]
media.write_video(output_path, video, fps=fps)
