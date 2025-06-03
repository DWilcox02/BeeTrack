import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

_, csv_path = sys.argv

# Load the CSV data
df = pd.read_csv(csv_path)

# ============================================================================
# DYNAMIC FILENAME DETECTION
# ============================================================================

# Get unique filenames
unique_filenames = sorted(df["filename"].unique())
print(f"Found unique filenames: {unique_filenames}")

# Determine which filename corresponds to cluster and no_cluster
# Strategy: look for patterns in the names
cluster_filename = None
no_cluster_filename = None

if len(unique_filenames) == 2:
    # If exactly 2 filenames, try to identify them by patterns
    for filename in unique_filenames:
        if "cluster" in filename.lower() and "no_cluster" not in filename.lower():
            cluster_filename = filename
        elif "no_cluster" in filename.lower():
            no_cluster_filename = filename

    # If pattern matching failed, just use them in order
    if cluster_filename is None or no_cluster_filename is None:
        cluster_filename = unique_filenames[0]
        no_cluster_filename = unique_filenames[1]
        print(f"Warning: Could not identify cluster/no_cluster pattern. Using:")
        print(f"  First filename as 'cluster': {cluster_filename}")
        print(f"  Second filename as 'no_cluster': {no_cluster_filename}")
    else:
        print(f"Identified filenames:")
        print(f"  Cluster type: {cluster_filename}")
        print(f"  No-cluster type: {no_cluster_filename}")

elif len(unique_filenames) > 2:
    print(f"Warning: Found {len(unique_filenames)} unique filenames. Expected 2.")
    print("Please specify which filenames to compare or modify the script.")
    sys.exit(1)
else:
    print(f"Error: Found only {len(unique_filenames)} unique filename(s). Need at least 2 for comparison.")
    sys.exit(1)

# ============================================================================
# FIRST PLOT: Original Bar Chart
# ============================================================================

# Create the figure for the bar chart
fig1, ax1 = plt.subplots(figsize=(14, 8))

# Get unique frames and cloud indices
frames = sorted(df["frame"].unique())
cloud_indices = sorted(df["cloud_index"].unique())

# Set up the bar positions
n_frames = len(frames)
n_clouds = len(cloud_indices)
bar_width = 0.35
group_width = bar_width * 2 + 0.1  # Width for each cloud_index group (2 bars + spacing)
frame_width = group_width * n_clouds + 0.2  # Width for each frame group

# Create positions for each bar
positions = []
labels = []

for i, frame in enumerate(frames):
    frame_start = i * frame_width

    for j, cloud_idx in enumerate(cloud_indices):
        # Filter data for this frame and cloud_index
        frame_cloud_data = df[(df["frame"] == frame) & (df["cloud_index"] == cloud_idx)]

        if len(frame_cloud_data) == 2:  # Should have both cluster and no_cluster data
            cluster_data = frame_cloud_data[frame_cloud_data["filename"] == cluster_filename]
            no_cluster_data = frame_cloud_data[frame_cloud_data["filename"] == no_cluster_filename]

            if not cluster_data.empty and not no_cluster_data.empty:
                group_start = frame_start + j * group_width

                # Positions for cluster and no_cluster bars
                cluster_pos = group_start
                no_cluster_pos = group_start + bar_width

                # Plot the bars
                ax1.bar(
                    cluster_pos,
                    cluster_data["euclidean_distance"].iloc[0],
                    bar_width,
                    label=cluster_filename if i == 0 and j == 0 else "",
                    color="skyblue",
                    alpha=0.8,
                )

                ax1.bar(
                    no_cluster_pos,
                    no_cluster_data["euclidean_distance"].iloc[0],
                    bar_width,
                    label=no_cluster_filename if i == 0 and j == 0 else "",
                    color="lightcoral",
                    alpha=0.8,
                )

                # Store position for x-axis labels (center of the group)
                if j == 0:  # Only add frame label once per frame
                    center_pos = frame_start + (group_width * n_clouds) / 2 - group_width / 2
                    positions.append(center_pos)
                    labels.append(f"Frame {frame}")

# Customize the bar chart
ax1.set_xlabel("Frame", fontsize=12, fontweight="bold")
ax1.set_ylabel("Euclidean Distance", fontsize=12, fontweight="bold")
ax1.set_title(
    "Euclidean Distance Comparison: Cluster vs No-Cluster Reconstructions\nby Frame and Cloud Index",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

# Set x-axis ticks and labels
ax1.set_xticks(positions)
ax1.set_xticklabels(labels)

# Add legend
ax1.legend(loc="upper right")

# Add grid for better readability
ax1.grid(True, alpha=0.3, axis="y")

# Add cloud index labels
for i, frame in enumerate(frames):
    frame_start = i * frame_width
    for j, cloud_idx in enumerate(cloud_indices):
        group_center = frame_start + j * group_width + bar_width / 2
        ax1.text(
            group_center,
            -max(df["euclidean_distance"]) * 0.08,
            f"Cloud {cloud_idx}",
            ha="center",
            va="top",
            fontsize=9,
            rotation=0,
        )

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the bar chart
plt.show()

# ============================================================================
# SECOND PLOT: Box and Whisker Plot
# ============================================================================

# Create the figure for the box plot
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Get data for box plot using dynamic filenames
cluster_distances = df[df["filename"] == cluster_filename]["euclidean_distance"]
no_cluster_distances = df[df["filename"] == no_cluster_filename]["euclidean_distance"]

# Create box plot
box_data = [cluster_distances, no_cluster_distances]
box_labels = [cluster_filename, no_cluster_filename]

bp = ax2.boxplot(
    box_data,
    labels=box_labels,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", alpha=0.7),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="black", linewidth=1.5),
    capprops=dict(color="black", linewidth=1.5),
    flierprops=dict(marker="o", markerfacecolor="red", markersize=5, alpha=0.7),
)

# Color the boxes to match the bar chart
bp["boxes"][0].set_facecolor("skyblue")
bp["boxes"][0].set_alpha(0.8)
bp["boxes"][1].set_facecolor("lightcoral")
bp["boxes"][1].set_alpha(0.8)

# Customize the box plot
ax2.set_xlabel("Reconstruction Type", fontsize=12, fontweight="bold")
ax2.set_ylabel("Euclidean Distance", fontsize=12, fontweight="bold")
ax2.set_title("Box and Whisker Plot:\nEuclidean Distance Distribution", fontsize=14, fontweight="bold", pad=20)
ax2.grid(True, alpha=0.3, axis="y")

# Rotate x-axis labels for better readability
ax2.tick_params(axis="x", rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the box plot
plt.show()

# Optional: Save the plots
# plt.figure(fig1)
# plt.savefig('euclidean_distance_bar_chart.png', dpi=300, bbox_inches='tight')
# plt.figure(fig2)
# plt.savefig('euclidean_distance_boxplot.png', dpi=300, bbox_inches='tight')

# Calculate and print statistical summary
print("\nStatistical Summary:")
print("=" * 60)


# Function to calculate box-and-whisker statistics
def box_whisker_stats(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)
    return {
        "min": data.min(),
        "q1": q1,
        "median": data.median(),
        "q3": q3,
        "max": data.max(),
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
    }


# Calculate statistics for cluster_filename
cluster_stats = box_whisker_stats(cluster_distances)
print(f"\n{cluster_filename}:")
print(f"  Mean: {cluster_distances.mean():.4f}")
print(f"  Median: {cluster_stats['median']:.4f}")
print(f"  Box-and-Whisker values:")
print(f"    Min: {cluster_stats['min']:.4f}")
print(f"    Q1 (25th percentile): {cluster_stats['q1']:.4f}")
print(f"    Median (50th percentile): {cluster_stats['median']:.4f}")
print(f"    Q3 (75th percentile): {cluster_stats['q3']:.4f}")
print(f"    Max: {cluster_stats['max']:.4f}")
print(f"    Lower Whisker: {cluster_stats['lower_whisker']:.4f}")
print(f"    Upper Whisker: {cluster_stats['upper_whisker']:.4f}")

# Calculate statistics for no_cluster_filename
no_cluster_stats = box_whisker_stats(no_cluster_distances)
print(f"\n{no_cluster_filename}:")
print(f"  Mean: {no_cluster_distances.mean():.4f}")
print(f"  Median: {no_cluster_stats['median']:.4f}")
print(f"  Box-and-Whisker values:")
print(f"    Min: {no_cluster_stats['min']:.4f}")
print(f"    Q1 (25th percentile): {no_cluster_stats['q1']:.4f}")
print(f"    Median (50th percentile): {no_cluster_stats['median']:.4f}")
print(f"    Q3 (75th percentile): {no_cluster_stats['q3']:.4f}")
print(f"    Max: {no_cluster_stats['max']:.4f}")
print(f"    Lower Whisker: {no_cluster_stats['lower_whisker']:.4f}")
print(f"    Upper Whisker: {no_cluster_stats['upper_whisker']:.4f}")

# Compare the two distributions
print("\nComparison:")
print(
    f"  Mean difference ({cluster_filename} - {no_cluster_filename}): {cluster_distances.mean() - no_cluster_distances.mean():.4f}"
)
print(
    f"  Median difference ({cluster_filename} - {no_cluster_filename}): {cluster_stats['median'] - no_cluster_stats['median']:.4f}"
)
