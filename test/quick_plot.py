import matplotlib.pyplot as plt
import numpy as np


def create_scatter_plot(points_data, title="Scatter Plot", xlabel="X", ylabel="Y"):
    """
    Create a scatter plot from points data.

    Args:
        points_data: List of [x, y] coordinates or numpy array
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
    """
    # Convert to numpy array if it isn't already
    points = np.array(points_data)

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, alpha=0.7, s=50, c="blue", edgecolors="black", linewidth=0.5)

    # Customize the plot
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add some padding to the axes
    x_margin = (max(x) - min(x)) * 0.05
    y_margin = (max(y) - min(y)) * 0.05
    plt.xlim(min(x) - x_margin, max(x) + x_margin)
    plt.ylim(min(y) - y_margin, max(y) + y_margin)

    plt.tight_layout()
    plt.show()


# Your sample data
sample_points = [[451.70832348, 283.97800922],
 [468.51944791, 256.74960319],
 [481.06802446, 277.03315151],
 [471.99189777, 271.42944871],
 [462.91574511, 265.82572988],
 [453.83961842, 260.22202708],
 [444.76349173, 254.61832428],
 [475.46432166, 286.1092782 ],
 [466.38819497, 280.5055754 ],
 [457.31204231, 274.90185657],
 [448.23591562, 269.29815377],
 [439.15978893, 263.69445097],
 [478.93672952, 300.78913366],
 [469.86060283, 295.18543086],
 [460.78447614, 289.58172806],
 [451.70832348, 283.97800922],
 [442.63219679 ,278.37430642],
 [433.5560701  ,272.77060362],
 [464.25690003 ,304.26155755],
 [455.18077334, 298.65785475],
 [446.10462068, 293.05413591],
 [437.02849399, 287.45043311],
 [427.9523673 , 281.84673031],
 [458.65319723, 313.33768424],
 [449.57707054, 307.73398144],
 [440.50091788, 302.1302626 ],
 [431.42479119, 296.5265598 ],
 [422.3486645 , 290.922857  ]]

# Create the scatter plot
create_scatter_plot(sample_points, title="Sample Data Scatter Plot", xlabel="X Coordinate", ylabel="Y Coordinate")

# To use with new data, simply replace sample_points with your new data:
# new_points = [[your, data], [goes, here]]
# create_scatter_plot(new_points, title="My New Plot")
