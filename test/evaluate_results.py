import json
import numpy as np
import argparse


def load_json_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main():
    parser = argparse.ArgumentParser(description="Compare reference and test point tracking data")
    parser.add_argument("reference_file", help="Path to reference JSON file")
    parser.add_argument("test_file", help="Path to test result JSON file")

    args = parser.parse_args()

    reference_data = load_json_file(args.reference_file)
    test_data = load_json_file(args.test_file)

    ref_num_points = reference_data["num_points"]
    ref_num_frames = reference_data["num_frames"]
    ref_tracks = np.array(reference_data["tracks"])

    test_num_points = test_data["num_points"]
    test_num_frames = test_data["num_frames"]
    test_tracks = np.array(test_data["tracks"])

    assert ref_num_points == test_num_points, (
        f"Number of points mismatch: reference={ref_num_points}, test={test_num_points}"
    )
    assert ref_num_frames == test_num_frames, (
        f"Number of frames mismatch: reference={ref_num_frames}, test={test_num_frames}"
    )
    assert ref_tracks.shape == test_tracks.shape, (
        f"Track array shapes don't match: reference={ref_tracks.shape}, test={test_tracks.shape}"
    )

    print(f"Data dimensions match: {ref_num_points} points across {ref_num_frames} frames")

    distance_array = np.zeros((ref_num_points, ref_num_frames))

    for i in range(ref_num_points):
        for j in range(ref_num_frames):
            distance_array[i, j] = calculate_distance(ref_tracks[i, j], test_tracks[i, j])

    point_means = np.mean(distance_array, axis=1)
    point_variances = np.var(distance_array, axis=1)

    average_mean = np.mean(point_means)
    average_variance = np.mean(point_variances)

    print("\nResults:")
    print(f"Average mean distance: {average_mean:.4f}")
    print(f"Average variance: {average_variance:.4f}")

    print("\nPer-point statistics:")
    for i in range(ref_num_points):
        print(f"Point {i}: Mean={point_means[i]:.4f}, Variance={point_variances[i]:.4f}")


if __name__ == "__main__":
    main()
