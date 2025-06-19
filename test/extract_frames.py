#!/usr/bin/env python3
import csv
import json
import sys
import argparse


def extract_frames(csv_filename):
    """
    Extract data from specified frames (0, 30, 60, 90, 120, 150, 180) from CSV file
    and output in JSON format.
    """
    target_frames = [0, 30, 60, 90, 120, 150, 180]
    results = []

    try:
        with open(csv_filename, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:
                frame = int(row["frame"])
                if frame in target_frames:
                    result = {"x": int(row["x"]), "y": float(row["y"]), "color": "red", "radius": 32}
                    results.append(result)

        # Sort results by frame order to maintain consistency
        frame_data = {}
        with open(csv_filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                frame = int(row["frame"])
                if frame in target_frames:
                    frame_data[frame] = {"x": int(row["x"]), "y": float(row["y"]), "color": "red", "radius": 32}

        # Output results in frame order
        for frame in target_frames:
            if frame in frame_data:
                print(json.dumps(frame_data[frame]))

    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing column {e} in CSV file.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data format - {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract frame data from CSV file")
    parser.add_argument("csv_file", help="Path to the CSV file")

    args = parser.parse_args()
    extract_frames(args.csv_file)


if __name__ == "__main__":
    main()
