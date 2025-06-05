import pandas as pd
import sys
import os

NAME = "n50"


def filter_csv(filename):
    try:
        # Read the CSV file
        df = pd.read_csv(filename)

        # Filter rows where 'individual' is not 'dancer'
        filtered_df = df[df["individual"] == NAME]

        # Create output filename
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_{NAME}.csv"

        # Save filtered data to new CSV
        filtered_df.to_csv(output_filename, index=False)

        print(f"Filtered data saved to '{output_filename}'")
        print(f"Original rows: {len(df)}")
        print(f"Filtered rows: {len(filtered_df)}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except KeyError:
        print("Error: 'individual' column not found in the CSV file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_filename>")
        sys.exit(1)

    csv_filename = sys.argv[1]
    filter_csv(csv_filename)
