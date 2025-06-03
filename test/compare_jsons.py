import json
import csv
import os
import sys

def compare_euclidean_distances(file1_path, file2_path, output_csv_path):
    """
    Compare euclidean distances from two JSON files and write results to CSV.

    Args:
        file1_path (str): Path to first JSON file
        file2_path (str): Path to second JSON file
        output_csv_path (str): Path for output CSV file
    """

    # Read both JSON files
    with open(file1_path, "r") as f1:
        data1 = json.load(f1)

    with open(file2_path, "r") as f2:
        data2 = json.load(f2)

    # Get filenames without extension for the CSV
    filename1 = os.path.splitext(os.path.basename(file1_path))[0]
    filename2 = os.path.splitext(os.path.basename(file2_path))[0]

    # Prepare CSV data
    csv_rows = []

    # Process first file
    for frame_data in data1:
        frame = frame_data["frame"]
        for cloud_metric in frame_data["intra_cloud_metrics"]:
            cloud_index = cloud_metric["cloud_index"]
            euclidean_distance = cloud_metric["euclidean_distance"]

            csv_rows.append(
                {
                    "frame": frame,
                    "cloud_index": cloud_index,
                    "filename": filename1,
                    "euclidean_distance": euclidean_distance,
                }
            )

    # Process second file
    for frame_data in data2:
        frame = frame_data["frame"]
        for cloud_metric in frame_data["intra_cloud_metrics"]:
            cloud_index = cloud_metric["cloud_index"]
            euclidean_distance = cloud_metric["euclidean_distance"]

            csv_rows.append(
                {
                    "frame": frame,
                    "cloud_index": cloud_index,
                    "filename": filename2,
                    "euclidean_distance": euclidean_distance,
                }
            )

    # Sort by frame, then cloud_index, then filename for better organization
    csv_rows.sort(key=lambda x: (x["frame"], x["cloud_index"], x["filename"]))

    # Write to CSV
    with open(output_csv_path, "w", newline="") as csvfile:
        fieldnames = ["frame", "cloud_index", "filename", "euclidean_distance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Comparison complete! Results saved to {output_csv_path}")
    print(f"Total rows written: {len(csv_rows)}")


# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1> <file2>")
    else:
        _, file1_path, file2_path = sys.argv
        
    output_csv_path = "euclidean_distance_comparison.csv"

    compare_euclidean_distances(file1_path, file2_path, output_csv_path)
