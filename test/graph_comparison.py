#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse


def plot_csv(csv_filename, fps):
    """
    Plot frame vs distance from a CSV file.

    Args:
        csv_filename (str): Path to the CSV file
        fps (int): Frames per second - adds vertical lines every fps frames
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filename)

        # Verify required columns exist
        if "frame" not in df.columns or "distance" not in df.columns:
            print("Error: CSV must contain 'frame' and 'distance' columns")
            return

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(df["frame"], df["distance"], marker="o", linewidth=2, markersize=4)

        # Add vertical lines every fps frames
        max_frame = df["frame"].max()
        for frame in range(fps, int(max_frame) + 1, fps):
            plt.axvline(x=frame, color="green", linestyle="-", alpha=0.8, linewidth=1)

        # Customize the plot
        plt.xlabel("Frame")
        plt.ylabel("Distance")
        plt.title("Distance vs Frame")
        plt.grid(True, alpha=0.3)

        # Add some styling
        plt.tight_layout()

        # Show the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found")
    except pd.errors.EmptyDataError:
        print(f"Error: '{csv_filename}' is empty")
    except Exception as e:
        print(f"Error reading CSV file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot frame vs distance from CSV file")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("fps", type=int, help="Frames per second (adds vertical lines every fps frames)")

    args = parser.parse_args()
    plot_csv(args.csv_file, args.fps)


if __name__ == "__main__":
    main()
