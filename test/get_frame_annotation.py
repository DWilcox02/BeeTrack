import csv
import sys


def find_row(filename, bodypart, frame):
    try:
        frame = int(frame)
    except ValueError:
        print(f"Error: frame '{frame}' is not a valid integer.")
        return

    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["frame"].isdigit() and int(row["frame"]) == frame and row["bodypart"] == bodypart:
                print(row)
                return
        print(f"No matching row found for bodypart '{bodypart}' and frame '{frame}'.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <filename> <bodypart> <frame>")
    else:
        _, filename, bodypart, frame = sys.argv
        find_row(filename, bodypart, frame)
