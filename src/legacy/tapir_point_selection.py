# @title Predict Point Tracks for the Selected Points {form-width: "25%"}

resize_height = 256  # @param {type: "integer"}
resize_width = 256  # @param {type: "integer"}


def convert_select_points_to_query_points(frame, points):
    """Convert select points to query points.

    Args:
      points: [num_points, 2], in [x, y]

    Returns:
      query_points: [num_points, 3], in [t, y, x]
    """
    points = np.stack(points)  # noqa: F821
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)  # noqa: F821
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]
    query_points[:, 2] = points[:, 0]
    return query_points


frames = media.resize_video(video, (resize_height, resize_width))  # noqa: F821
query_points = convert_select_points_to_query_points(select_frame, select_points)  # noqa: F821
height, width = video.shape[1:3]  # noqa: F821
query_points = transforms.convert_grid_coordinates(  # noqa: F821
    query_points,
    (1, height, width),
    (1, resize_height, resize_width),
    coordinate_format="tyx",
)
tracks, visibles = inference(frames, query_points)  # noqa: F821

# Visualize sparse point tracks
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))  # noqa: F821
video_viz = viz_utils.paint_point_track(video, tracks, visibles, colormap)  # noqa: F821
media.show_video(video_viz, fps=10)  # noqa: F821
