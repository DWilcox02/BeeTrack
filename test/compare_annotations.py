import csv
import sys
import math
from collections import defaultdict


def read_filtered_coords(filename, target_bodypart):
    coords = {}
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["bodypart"] == target_bodypart:
                frame = int(row["frame"])
                x = float(row["x"])
                y = float(row["y"])
                coords[frame] = (x, y)
    return coords


def compute_distances(coords1, coords2):
    common_frames = sorted(set(coords1.keys()) & set(coords2.keys()))
    distances = []
    for frame in common_frames:
        x1, y1 = coords1[frame]
        x2, y2 = coords2[frame]
        dist = math.hypot(x2 - x1, y2 - y1)
        distances.append((frame, dist))
    return distances


def write_distances_to_csv(distances, output_file="distance_output.csv"):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "distance"])
        for frame, distance in distances:
            writer.writerow([frame, distance])


def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <file1> <bodypart1> <file2> <bodypart2>")
        sys.exit(1)

    file1, bodypart1, file2, bodypart2 = sys.argv[1:]

    coords1 = read_filtered_coords(file1, bodypart1)
    coords2 = read_filtered_coords(file2, bodypart2)

    distances = compute_distances(coords1, coords2)
    write_distances_to_csv(distances)

    total_distance = sum(d for _, d in distances)
    print(f"Total distance: {total_distance:.4f}")


if __name__ == "__main__":
    main()
