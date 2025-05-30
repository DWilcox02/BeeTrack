#!/usr/bin/env python3
"""
Script to visualize deformity values from JSON files containing frame and cloud data.
Creates a grouped bar chart with frames on x-axis and deformity values on y-axis,
grouped by cloud index.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys


def load_json_files(file_paths):
    """Load and parse multiple JSON files."""
    all_data = []

    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {len(data)} frames from {file_path}")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}")
            sys.exit(1)

    return all_data


def extract_deformity_data(data):
    """Extract deformity values organized by frame and cloud index."""
    deformity_data = defaultdict(dict)

    for frame_data in data:
        frame = frame_data["frame"]

        for cloud_metrics in frame_data["intra_cloud_metrics"]:
            cloud_index = cloud_metrics["cloud_index"]
            deformity = cloud_metrics["deformity"]

            deformity_data[frame][cloud_index] = deformity

    return deformity_data


def create_grouped_bar_chart(deformity_data):
    """Create a grouped bar chart of deformity values."""
    # Get all unique frames and cloud indices
    frames = sorted(deformity_data.keys())
    all_cloud_indices = set()

    for frame_data in deformity_data.values():
        all_cloud_indices.update(frame_data.keys())

    cloud_indices = sorted(all_cloud_indices)

    # Prepare data for plotting
    x = np.arange(len(frames))
    width = 0.8 / len(cloud_indices)  # Width of bars

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bars for each cloud index
    colors = plt.cm.Set1(np.linspace(0, 1, len(cloud_indices)))

    for i, cloud_idx in enumerate(cloud_indices):
        deformity_values = []

        for frame in frames:
            # Get deformity value for this frame and cloud index, or 0 if not present
            deformity_values.append(deformity_data[frame].get(cloud_idx, 0))

        bars = ax.bar(x + i * width, deformity_values, width, label=f"Cloud {cloud_idx}", color=colors[i], alpha=0.8)

        # Add value labels on top of bars
        for bar, value in zip(bars, deformity_values):
            if value > 0:  # Only show label if there's a value
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(deformity_values) * 0.01,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Customize the chart
    ax.set_xlabel("Frame", fontsize=12, fontweight="bold")
    ax.set_ylabel("Deformity Value", fontsize=12, fontweight="bold")
    ax.set_title("Deformity Values by Frame and Cloud Index", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(cloud_indices) - 1) / 2)
    ax.set_xticklabels(frames)
    ax.legend(title="Cloud Index", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Create grouped bar chart of deformity values from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python script.py file1.json file2.json file3.json
  python script.py data/*.json
        """,
    )

    parser.add_argument("files", nargs="+", help="JSON files to process")
    parser.add_argument(
        "--output",
        "-o",
        default="deformity_chart.png",
        help="Output filename for the chart (default: deformity_chart.png)",
    )
    parser.add_argument("--show", "-s", action="store_true", help="Show the chart interactively instead of saving")

    args = parser.parse_args()

    # Load data from all files
    print(f"Loading data from {len(args.files)} file(s)...")
    all_data = load_json_files(args.files)

    if not all_data:
        print("No data found in the provided files.")
        sys.exit(1)

    # Extract deformity data
    deformity_data = extract_deformity_data(all_data)

    if not deformity_data:
        print("No deformity data found in the files.")
        sys.exit(1)

    print(f"Found deformity data for {len(deformity_data)} frames")

    # Create the chart
    fig = create_grouped_bar_chart(deformity_data)

    # Save or show the chart
    if args.show:
        plt.show()
    else:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Chart saved as {args.output}")


if __name__ == "__main__":
    main()


# python deformity_chart.py errors/ERRORS_point_cloud_base.json errors/ERRORS_point_cloud_inliers_only.json errors/ERRORS_point_cloud_redraw_outliers_original.json