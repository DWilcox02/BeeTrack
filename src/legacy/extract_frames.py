import json
import numpy as np

# Load the data from file
with open("test/references/leah/TRACKS_LEAH_dance_7_point_5_secs_700x700_15fps.txt", "r") as f:
    data = json.load(f)

# Convert "tracks" to a NumPy array
tracks = np.array(data["tracks"])  # Shape: (3, 90, 2)

# Indices to extract
indices = [0, 15, 30, 45, 60, 75]

# Extract the desired frames for each point
# Output shape: (6, 3, 2)
result = np.array([tracks[:, i, :] for i in indices])

print("Result shape:", result.shape)
print(result)
