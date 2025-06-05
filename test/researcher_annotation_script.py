import cv2
import pathlib
import pandas as pd
import os


def overlay_tracking_data(name):
    try:
        video_path = "../data/Recording2023/recording-2023.mp4"
        video_annotations_path = f"annotations/recording-2023_filtered_filtered_{name}.csv"
        annotations_df = pd.read_csv(video_annotations_path)
        cap = cv2.VideoCapture(video_path)
        size = (int(cap.get(3)), int(cap.get(4)))

        # Create output directory if it doesn't exist
        os.makedirs("annotated", exist_ok=True)

        video_annotations = pathlib.Path(video_annotations_path)
        out_name = f"annotated/{name}-{video_annotations.stem}-annotated.mp4"  # Changed to .mp4

        # Use H.264 codec with standard frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 60 or fps < 1:  # Handle unusual frame rates
            fps = 30.0

        out = cv2.VideoWriter(
            out_name,
            cv2.VideoWriter_fourcc(*"avc1"),  # H.264 codec
            fps,
            size,
        )

        ret = True
        frame_n = 0
        while ret:
            ret, frame = cap.read()
            if ret:
                if frame_n in annotations_df["frame"].values:
                    frame_df = annotations_df[annotations_df["frame"] == frame_n]
                    for ind, data in frame_df.iterrows():
                        x_center = int(data["x"])
                        y_center = int(data["y"])
                        cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
                out.write(frame)
            frame_n += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    except Exception:
        print(f"error with {name}")

names = ["n31", "n31_1", "n32", "n33", "n33_1", "n34", "n35", "n49", "n49_1"]

for name in names:
    overlay_tracking_data(name)