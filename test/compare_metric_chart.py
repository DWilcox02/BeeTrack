#!/usr/bin/env python3
"""
Script to visualize metric values from JSON files containing frame and cloud data.
Creates visualizations split by JSON file to show differences between files.
Data is organized by: JSON file -> frame -> cloud index
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import os


def load_json_files(file_paths):
    """Load and parse multiple JSON files, keeping them separate."""
    files_data = {}

    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Use just the filename (without path) as the key
                file_key = os.path.basename(file_path)
                files_data[file_key] = data
                print(f"Loaded {len(data)} frames from {file_path}")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}")
            sys.exit(1)

    return files_data


def get_available_metrics(files_data):
    """Get all available metrics from all files."""
    metrics = set()

    for file_data in files_data.values():
        for frame_data in file_data:
            if "intra_cloud_metrics" in frame_data:
                for cloud_metrics in frame_data["intra_cloud_metrics"]:
                    metrics.update(cloud_metrics.keys())

    # Remove non-metric fields
    metrics.discard("cloud_index")
    return sorted(metrics)


def extract_metric_data_by_file(files_data, metric):
    """Extract metric values organized by file, frame, and cloud index."""
    metric_data_by_file = {}

    for file_name, file_data in files_data.items():
        metric_data = defaultdict(dict)

        for frame_data in file_data:
            frame = frame_data["frame"]

            if "intra_cloud_metrics" not in frame_data:
                continue

            for cloud_metrics in frame_data["intra_cloud_metrics"]:
                cloud_index = cloud_metrics["cloud_index"]

                if metric in cloud_metrics:
                    metric_value = cloud_metrics[metric]
                    metric_data[frame][cloud_index] = metric_value

        metric_data_by_file[file_name] = metric_data

    return metric_data_by_file


def create_comparison_chart(metric_data_by_file, metric_name):
    """Create a chart comparing metric values across different JSON files."""

    # Get all unique frames and cloud indices across all files
    all_frames = set()
    all_cloud_indices = set()

    for file_data in metric_data_by_file.values():
        all_frames.update(file_data.keys())
        for frame_data in file_data.values():
            all_cloud_indices.update(frame_data.keys())

    frames = sorted(all_frames)
    cloud_indices = sorted(all_cloud_indices)
    file_names = list(metric_data_by_file.keys())

    if not frames or not cloud_indices or not file_names:
        print(f"No data found for metric '{metric_name}'")
        return None

    # Create subplots - one for each cloud index
    fig, axes = plt.subplots(len(cloud_indices), 1, figsize=(14, 6 * len(cloud_indices)))

    # Handle case where there's only one cloud index
    if len(cloud_indices) == 1:
        axes = [axes]

    # Color map for different files
    colors = plt.cm.Set1(np.linspace(0, 1, len(file_names)))

    for cloud_idx_pos, cloud_idx in enumerate(cloud_indices):
        ax = axes[cloud_idx_pos]

        # Width of bars and positions
        x = np.arange(len(frames))
        width = 0.8 / len(file_names)

        # Create bars for each file
        for file_idx, file_name in enumerate(file_names):
            metric_values = []

            for frame in frames:
                # Get metric value for this frame and cloud index from this file
                file_data = metric_data_by_file[file_name]
                metric_values.append(file_data[frame].get(cloud_idx, 0))

            # Create bars
            bars = ax.bar(
                x + file_idx * width, metric_values, width, label=file_name, color=colors[file_idx], alpha=0.8
            )

            # Add value labels on top of bars
            for bar, value in zip(bars, metric_values):
                if value > 0:  # Only show label if there's a value
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(metric_values) * 0.01,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=45 if len(str(f"{value:.2f}")) > 4 else 0,
                    )

        # Customize each subplot
        ax.set_xlabel("Frame", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{metric_name.replace('_', ' ').title()}", fontsize=10, fontweight="bold")
        ax.set_title(
            f"Cloud Index {cloud_idx} - {metric_name.replace('_', ' ').title()} Comparison",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks(x + width * (len(file_names) - 1) / 2)
        ax.set_xticklabels(frames)
        ax.legend(title="JSON File", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    # Main title for the entire figure
    fig.suptitle(
        f"{metric_name.replace('_', ' ').title()} Values Comparison Across JSON Files",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    return fig


def create_side_by_side_chart(metric_data_by_file, metric_name):
    """Create a side-by-side comparison chart with all files in one view per cloud index."""

    # Get all unique frames and cloud indices across all files
    all_frames = set()
    all_cloud_indices = set()

    for file_data in metric_data_by_file.values():
        all_frames.update(file_data.keys())
        for frame_data in file_data.values():
            all_cloud_indices.update(frame_data.keys())

    frames = sorted(all_frames)
    cloud_indices = sorted(all_cloud_indices)
    file_names = list(metric_data_by_file.keys())

    if not frames or not cloud_indices or not file_names:
        print(f"No data found for metric '{metric_name}'")
        return None

    # Create one large chart with all data
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create grouped positions for each frame-cloud combination
    n_groups = len(frames) * len(cloud_indices)
    x_positions = np.arange(n_groups)
    width = 0.8 / len(file_names)

    # Color map for different files
    colors = plt.cm.Set1(np.linspace(0, 1, len(file_names)))

    # Create bars for each file
    for file_idx, file_name in enumerate(file_names):
        metric_values = []
        group_labels = []

        for frame in frames:
            for cloud_idx in cloud_indices:
                # Get metric value for this frame and cloud index from this file
                file_data = metric_data_by_file[file_name]
                value = file_data[frame].get(cloud_idx, 0)
                metric_values.append(value)

                if file_idx == 0:  # Only create labels once
                    group_labels.append(f"F{frame}-C{cloud_idx}")

        # Create bars
        bars = ax.bar(
            x_positions + file_idx * width, metric_values, width, label=file_name, color=colors[file_idx], alpha=0.8
        )

        # Add value labels on top of bars (only for non-zero values)
        for bar, value in zip(bars, metric_values):
            if value > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(metric_values) * 0.01,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    # Customize the chart
    ax.set_xlabel("Frame-Cloud Index Combinations", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{metric_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{metric_name.replace('_', ' ').title()} Values - All Files Comparison", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x_positions + width * (len(file_names) - 1) / 2)
    ax.set_xticklabels(group_labels, rotation=45, ha="right")
    ax.legend(title="JSON File", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Adjust layout
    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Create comparison charts of metric values from multiple JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python script.py deformity file1.json file2.json file3.json
  python script.py mean_distance data/*.json
  python script.py --list-metrics file1.json  # Show available metrics
  python script.py deformity file1.json file2.json --chart-type side-by-side
        """,
    )

    parser.add_argument("metric", nargs="?", help="Metric to visualize (e.g., 'deformity', 'mean_distance', etc.)")
    parser.add_argument("files", nargs="*", help="JSON files to process")
    parser.add_argument(
        "--output",
        "-o",
        help="Output filename for the chart (default: {metric}_comparison.png)",
    )
    parser.add_argument("--show", "-s", action="store_true", help="Show the chart interactively instead of saving")
    parser.add_argument(
        "--list-metrics", "-l", action="store_true", help="List available metrics in the JSON files and exit"
    )
    parser.add_argument(
        "--chart-type",
        "-t",
        choices=["separated", "side-by-side"],
        default="separated",
        help="Chart type: 'separated' (one subplot per cloud index) or 'side-by-side' (all in one chart)",
    )

    args = parser.parse_args()

    # Handle case where user wants to list available metrics
    if args.list_metrics:
        if not args.files:
            print("Error: Please provide at least one JSON file to analyze.")
            sys.exit(1)

        print(f"Loading data from {len(args.files)} file(s) to analyze available metrics...")
        files_data = load_json_files(args.files)

        if not files_data:
            print("No data found in the provided files.")
            sys.exit(1)

        metrics = get_available_metrics(files_data)
        if metrics:
            print(f"\nAvailable metrics:")
            for metric in metrics:
                print(f"  - {metric}")
        else:
            print("No metrics found in the data.")
        sys.exit(0)

    # Validate arguments for normal operation
    if not args.metric:
        print("Error: Please specify a metric to visualize.")
        print("Use --list-metrics to see available metrics in your files.")
        sys.exit(1)

    if not args.files:
        print("Error: Please provide at least one JSON file to process.")
        sys.exit(1)

    if len(args.files) < 2:
        print("Warning: Only one file provided. Comparison works best with multiple files.")

    # Load data from all files (keeping them separate)
    print(f"Loading data from {len(args.files)} file(s)...")
    files_data = load_json_files(args.files)

    if not files_data:
        print("No data found in the provided files.")
        sys.exit(1)

    # Check if the specified metric exists
    available_metrics = get_available_metrics(files_data)
    if args.metric not in available_metrics:
        print(f"Error: Metric '{args.metric}' not found in the data.")
        print(f"Available metrics: {', '.join(available_metrics)}")
        sys.exit(1)

    # Extract metric data organized by file
    metric_data_by_file = extract_metric_data_by_file(files_data, args.metric)

    if not metric_data_by_file:
        print(f"No {args.metric} data found in the files.")
        sys.exit(1)

    total_frames = sum(len(file_data) for file_data in metric_data_by_file.values())
    print(f"Found {args.metric} data for {total_frames} total frames across {len(files_data)} files")

    # Create the appropriate chart
    if args.chart_type == "side-by-side":
        fig = create_side_by_side_chart(metric_data_by_file, args.metric)
    else:
        fig = create_comparison_chart(metric_data_by_file, args.metric)

    if fig is None:
        sys.exit(1)

    # Determine output filename
    if args.output:
        output_filename = args.output
    else:
        chart_type_suffix = "comparison" if args.chart_type == "separated" else "sidebyside"
        output_filename = f"{args.metric}_{chart_type_suffix}.png"

    # Save or show the chart
    if args.show:
        plt.show()
    else:
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Chart saved as {output_filename}")


if __name__ == "__main__":
    main()


# Example usage:
# python script.py deformity errors/ERRORS_point_cloud_base.json errors/ERRORS_point_cloud_inliers_only.json errors/ERRORS_point_cloud_redraw_outliers_original.json
# python script.py mean_distance errors/ERRORS_point_cloud_base.json errors/ERRORS_point_cloud_inliers_only.json --chart-type side-by-side
# python script.py --list-metrics errors/ERRORS_point_cloud_base.json
