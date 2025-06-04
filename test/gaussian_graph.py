import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde


plt.rc("font", **{"family": "serif", "serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"]})


positions_1 = [
    [405.46313, 275.38248],
    [381.6413, 313.76587],
    [390.66608, 300.41714],
    [387.59235, 295.32755],
    [390.50797, 286.26953],
    [395.26138, 300.64355],
    [401.66403, 316.95172],
    [384.75418, 265.34808],
    [394.65866, 273.7934],
    [444.30222, 336.88055],
    [409.3038, 298.57706],
    [426.74088, 316.58762],
    [401.03464, 256.6683],
    [388.9503, 264.30646],
    [399.73303, 266.10452],
    [405.46313, 275.38248],
    [401.4218, 288.46118],
    [416.88834, 293.4696],
    [395.64093, 263.16043],
    [412.37546, 260.35382],
    [411.20096, 270.23154],
    [413.80917, 277.36972],
    [423.64462, 289.82068],
    [426.5883, 260.5347],
    [396.14966, 242.51627],
    [418.23056, 267.11713],
    [426.26782, 269.00394],
    [433.195, 279.62772],
]


positions_2 = [
    [455.06262, 285.04],
    [436.16556, 314.25174],
    [423.07205, 288.02032],
    [414.90656, 242.61607],
    [439.1224, 303.39917],
    [452.66943, 312.73285],
    [461.38364, 318.084],
    [434.61755, 280.2224],
    [439.73785, 284.81198],
    [448.29578, 294.47937],
    [457.21936, 302.00302],
    [463.71844, 307.39337],
    [429.40933, 263.32748],
    [422.74567, 254.26103],
    [423.76358, 278.0618],
    [455.06262, 285.04],
    [461.28018, 292.49753],
    [469.94888, 297.92535],
    [445.76923, 265.1062],
    [447.42215, 269.57187],
    [465.18997, 281.74393],
    [468.7299, 283.39752],
    [476.41333, 287.0656],
    [455.07242, 259.26138],
    [453.39584, 265.24167],
    [457.60773, 268.52423],
    [444.09778, 239.69925],
    [485.95218, 280.7014],
]


positions_3 = [
    [477.09567, 316.59927],
    [463.3938, 333.95972],
    [469.2752, 323.49704],
    [478.1107, 327.95706],
    [452.1015, 338.22614],
    [492.24216, 335.56848],
    [479.63177, 332.51202],
    [467.63663, 308.9802],
    [478.55652, 314.22238],
    [483.066, 328.90707],
    [505.89572, 334.66052],
    [502.77097, 336.50043],
    [465.41916, 293.8194],
    [472.2407, 297.5977],
    [479.07193, 303.5983],
    [477.09567, 316.59927],
    [479.58975, 350.8451],
    [481.4729, 373.41364],
    [480.977, 286.0829],
    [489.2604, 295.69406],
    [495.84247, 307.35104],
    [494.84537, 349.74948],
    [462.66544, 400.27606],
    [491.9859, 281.10352],
    [494.74686, 294.34662],
    [499.17038, 306.80573],
    [481.31625, 395.6426],
    [477.26422, 410.91675],
]

# First set of predictions
predictions_1 = [
    [405.46313, 275.38248],
    [406.15472, 293.19666],
    [420.72116, 303.04663],
    [410.79105, 289.7859],
    [406.85025, 272.55673],
    [404.74728, 278.7596],
    [404.29352, 286.89664],
    [406.63815, 274.83395],
    [409.68622, 275.10812],
    [452.47336, 330.02414],
    [410.61856, 283.5495],
    [421.19925, 293.38895],
    [421.60385, 281.18173],
    [402.6631, 280.64874],
    [406.58945, 274.27567],
    [405.46313, 275.38248],
    [394.56543, 280.29004],
    [403.17554, 277.12735],
    [401.18262, 286.35913],
    [411.06073, 275.38138],
    [403.02982, 277.08792],
    [398.78165, 276.055],
    [401.7607, 280.3348],
    [423.95883, 290.58978],
    [386.66382, 264.4002],
    [401.8883, 280.82993],
    [403.06915, 274.5456],
    [403.13995, 276.99823],
]

# First set predictions - 20deg rotation
predictions_1_20deg = [
    [405.46313, 275.38248],
    [397.6413, 286.05307],
    [419.80795, 292.6086],
    [407.49664, 282.18567],
    [401.17462, 267.7943],
    [396.69046, 276.83502],
    [393.8555, 287.80984],
    [408.56274, 266.77713],
    [409.2296, 269.88913],
    [449.63556, 327.64294],
    [405.39954, 284.0061],
    [413.59903, 296.68335],
    [428.74744, 272.6683],
    [407.4255, 274.97314],
    [408.97064, 271.43787],
    [405.46313, 275.38248],
    [392.1842, 283.12787],
    [398.41315, 282.80295],
    [408.7828, 283.0647],
    [416.27975, 274.92477],
    [405.8676, 279.46915],
    [399.23825, 281.274],
    [399.8361, 288.39163],
    [434.39685, 289.67657],
    [394.7206, 266.3248],
    [407.5639, 285.59232],
    [406.3636, 282.1458],
    [404.05316, 287.43625],
]

# Second set of predictions
predictions_2 = [
    [455.06262, 285.04],
    [456.73477, 289.7383],
    [453.12714, 285.39084],
    [436.79053, 233.1302],
    [452.8352, 287.0569],
    [458.21112, 289.53418],
    [458.75418, 288.02893],
    [457.81625, 285.76407],
    [454.7654, 283.49725],
    [455.1522, 286.30823],
    [455.90463, 286.97546],
    [454.2326, 285.50943],
    [453.92276, 283.8967],
    [439.08795, 267.97385],
    [431.93472, 284.9182],
    [455.06262, 285.04],
    [453.10904, 285.64114],
    [453.60663, 284.21255],
    [455.25513, 286.99014],
    [448.7369, 284.59943],
    [458.3336, 289.91507],
    [453.70236, 284.71225],
    [453.21466, 281.52396],
    [457.7019, 289.31647],
    [447.85422, 288.44034],
    [443.89493, 284.8665],
    [422.21387, 249.18512],
    [455.89713, 283.33087],
]

# Second set predictions - 20deg rotation
predictions_2_20deg = [
    [455.06262, 285.04],
    [447.1102, 284.18158],
    [450.41528, 275.26996],
    [432.22638, 226.21751],
    [446.41885, 283.3524],
    [449.94247, 289.03784],
    [448.6333, 290.7408],
    [458.31256, 277.49545],
    [453.4095, 278.4368],
    [451.944, 284.45596],
    [450.8442, 288.3314],
    [447.31992, 290.07355],
    [459.4795, 274.27213],
    [442.79245, 261.55746],
    [433.787, 281.71002],
    [455.06262, 285.04],
    [451.2568, 288.8493],
    [449.90213, 290.62894],
    [462.1678, 282.42603],
    [453.79733, 283.24347],
    [461.54175, 291.7673],
    [455.0583, 289.77267],
    [452.71835, 289.79257],
    [467.82278, 286.60458],
    [456.12283, 288.93665],
    [450.3113, 288.57098],
    [426.77798, 256.0978],
    [458.609, 293.45172],
]

# Third set of predictions
predictions_3 = [
    [477.09567, 316.59927],
    [487.90723, 313.3905],
    [499.3303, 326.12653],
    [501.3094, 322.4154],
    [468.4438, 324.51334],
    [501.72806, 313.68454],
    [482.26126, 302.45694],
    [489.5206, 318.46606],
    [493.58408, 315.5371],
    [491.23715, 322.05066],
    [507.21048, 319.63297],
    [497.22934, 313.30176],
    [485.98837, 318.33282],
    [485.9535, 313.93997],
    [485.92834, 311.76944],
    [477.09567, 316.59927],
    [472.73337, 342.67395],
    [467.7601, 357.07138],
    [486.51868, 309.28156],
    [487.94568, 310.72162],
    [487.67133, 314.20743],
    [479.81784, 348.43475],
    [440.78152, 390.7902],
    [489.35645, 311.1586],
    [485.26102, 316.23056],
    [482.82812, 320.51852],
    [458.11758, 401.18427],
    [447.20917, 408.28726],
]

# Third set predictions - 20deg rotation
predictions_3_20deg = [
    [477.09567, 316.59927],
    [479.3938, 306.24692],
    [498.41708, 315.6885],
    [498.01498, 314.8152],
    [462.7682, 319.75092],
    [493.67123, 311.75995],
    [471.82324, 303.37015],
    [491.4452, 310.40924],
    [493.12747, 310.3181],
    [488.39935, 319.66946],
    [501.99146, 320.08957],
    [489.62912, 316.59616],
    [493.13196, 309.8194],
    [490.7159, 308.26434],
    [488.30954, 308.93164],
    [477.09567, 316.59927],
    [470.35214, 345.51178],
    [462.9977, 362.74698],
    [494.11887, 305.98715],
    [493.1647, 310.265],
    [490.50916, 316.58865],
    [480.27444, 353.65375],
    [438.8569, 398.84702],
    [499.79446, 310.2454],
    [493.3178, 318.15515],
    [488.50372, 325.2809],
    [461.412, 408.7845],
    [448.12238, 418.72528],
]

# Error and rotation information
results_info = {
    "set_1": {"error": 6451.446009190472, "rotation": 320, "inliers": "27/28"},
    "set_2": {"error": 5623.428654860267, "rotation": 310, "inliers": "26/28"},
    "set_3": {"error": 64531.34156238173, "rotation": 320, "inliers": "22/28"},
}


def deformity(points: np.ndarray) -> float:
    points_mean = np.mean(points, axis=0)
    centered_points = points - points_mean
    cov_matrix = np.cov(centered_points.T)
    variance = np.linalg.det(cov_matrix)

    return variance

def graph_predictions(positions, predictions):
    plt.figure(figsize=(10, 8))

    # Plot the points
    pos_x, pos_y = zip(*positions)
    pred_x, pred_y = zip(*predictions)

    # Plot positions as blue circles
    plt.scatter(pos_x, pos_y, color="blue", s=10, label="Positions", zorder=3)

    # Plot predictions as red squares
    plt.scatter(pred_x, pred_y, color="red", s=10, marker="s", label="Predictions", zorder=3)

    # Draw arrows connecting corresponding points
    for i in range(len(positions)):
        dx = predictions[i][0] - positions[i][0]
        dy = predictions[i][1] - positions[i][1]
        plt.arrow(
            positions[i][0],
            positions[i][1],
            dx,
            dy,
            head_width=0.2,
            head_length=0.15,
            fc="gray",
            ec="gray",
            alpha=0.7,
            zorder=2,
        )

    # Customize the plot
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Lines Connecting Positions to Predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")  # Equal aspect ratio

    # Show the plot
    plt.tight_layout()
    plt.show()



def create_heatmap_comparison(coordinates, bins=50, figsize=(15, 5)):
    """
    Create a 2D heatmap from coordinates and compare to fitted Gaussian distribution.

    Parameters:
    coordinates: array-like of shape (n, 2) - list of [x, y] coordinates
    bins: int - number of bins for histogram
    figsize: tuple - figure size
    """

    # Convert to numpy array
    coords = np.array(coordinates)
    x, y = coords[:, 0], coords[:, 1]

    # Calculate sample statistics
    mean = np.mean(coords, axis=0)
    cov = np.cov(coords.T)

    print(f"Sample mean: [{mean[0]:.3f}, {mean[1]:.3f}]")
    print(f"Sample covariance matrix:")
    print(f"[[{cov[0, 0]:.3f}, {cov[0, 1]:.3f}],")
    print(f" [{cov[1, 0]:.3f}, {cov[1, 1]:.3f}]]")

    # Create grid for evaluation
    x_range = np.linspace(x.min() - 1, x.max() + 1, bins)
    y_range = np.linspace(y.min() - 1, y.max() + 1, bins)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))

    # Create theoretical Gaussian distribution
    rv = multivariate_normal(mean, cov)
    gaussian_pdf = rv.pdf(pos)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Original data histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    im1 = axes[0].imshow(
        hist.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis", aspect="auto"
    )
    axes[0].set_title("Data Histogram")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0], label="Count")

    # 2. Theoretical Gaussian
    im2 = axes[1].imshow(
        gaussian_pdf,
        origin="lower",
        extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
        cmap="viridis",
        aspect="auto",
    )
    axes[1].set_title("Theoretical Gaussian")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1], label="Probability Density")

    # 3. Difference (normalized)
    # Normalize histogram to probability density
    hist_normalized = hist / (len(coordinates) * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))

    # Interpolate Gaussian to histogram grid
    gaussian_interp = rv.pdf(np.dstack((np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2))))

    difference = hist_normalized.T - gaussian_interp
    im3 = axes[2].imshow(
        difference, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="RdBu_r", aspect="auto"
    )
    axes[2].set_title("Difference (Data - Gaussian)")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[2], label="Density Difference")

    plt.tight_layout()
    plt.show()

    # Calculate goodness of fit metrics
    mse = np.mean((hist_normalized.T - gaussian_interp) ** 2)
    print(f"\nMean Squared Error: {mse:.6f}")

    return mean, cov, mse


def create_kde_comparison(coordinates, bins=50, figsize=(15, 5)):
    """
    Alternative approach using Kernel Density Estimation for smoother visualization.
    """
    coords = np.array(coordinates)
    x, y = coords[:, 0], coords[:, 1]

    # Calculate statistics
    mean = np.mean(coords, axis=0)
    cov = np.cov(coords.T)

    # Create KDE
    kde = gaussian_kde(coords.T)

    # Create grid
    x_range = np.linspace(x.min() - 1, x.max() + 1, bins)
    y_range = np.linspace(y.min() - 1, y.max() + 1, bins)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate KDE and Gaussian
    kde_pdf = kde(pos).reshape(X.shape)
    rv = multivariate_normal(mean, cov)
    gaussian_pdf = rv.pdf(np.dstack((X, Y)))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # KDE
    im1 = axes[0].imshow(
        kde_pdf,
        origin="lower",
        extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
        cmap="viridis",
        aspect="auto",
    )
    axes[0].set_title("KDE of Data")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0], label="Density")

    # Gaussian
    im2 = axes[1].imshow(
        gaussian_pdf,
        origin="lower",
        extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
        cmap="viridis",
        aspect="auto",
    )
    axes[1].set_title("Theoretical Gaussian")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1], label="Density")

    # Difference
    difference = kde_pdf - gaussian_pdf
    im3 = axes[2].imshow(
        difference,
        origin="lower",
        extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
        cmap="RdBu_r",
        aspect="auto",
    )
    axes[2].set_title("Difference (KDE - Gaussian)")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[2], label="Density Difference")

    plt.tight_layout()
    plt.show()

    return kde_pdf, gaussian_pdf, difference



def combined_analysis_plot(positions, predictions, coordinates_to_map, bins=50, figsize=(18, 6)):
    # Calculate statistics for the non-Gaussian data
    coords = np.array(coordinates_to_map)
    x, y = coords[:, 0], coords[:, 1]
    mean = np.mean(coords, axis=0)
    cov = np.cov(coords.T)

    print(f"Non-Gaussian data statistics:")
    print(f"Sample mean: [{mean[0]:.3f}, {mean[1]:.3f}]")
    print(f"Sample covariance matrix:")
    print(f"[[{cov[0, 0]:.3f}, {cov[0, 1]:.3f}],")
    print(f" [{cov[1, 0]:.3f}, {cov[1, 1]:.3f}]]")

    # Determine common axis limits from all data
    all_x = list(x) + [pos[0] for pos in positions] + [pred[0] for pred in predictions]
    all_y = list(y) + [pos[1] for pos in positions] + [pred[1] for pred in predictions]

    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Subplot 1: Graph predictions
    ax1 = axes[0]
    pos_x, pos_y = zip(*positions)
    pred_x, pred_y = zip(*predictions)

    # Plot positions as blue circles
    ax1.scatter(pos_x, pos_y, color="blue", s=10, label="Final Positions", zorder=3)

    # Plot predictions as red squares
    ax1.scatter(pred_x, pred_y, color="red", s=10, marker="s", label="Query Point Predictions", zorder=3)

    # Draw arrows connecting corresponding points
    for i in range(len(positions)):
        dx = predictions[i][0] - positions[i][0]
        dy = predictions[i][1] - positions[i][1]
        ax1.arrow(
            positions[i][0],
            positions[i][1],
            dx,
            dy,
            head_width=0.2,
            head_length=0.15,
            fc="gray",
            ec="gray",
            alpha=0.7,
            zorder=2,
        )

    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_title("Final Positions vs Query Point Predictions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal")

    # Subplot 2: Theoretical Gaussian heatmap
    ax2 = axes[1]

    # Create grid for evaluation using common limits
    x_range = np.linspace(x_min, x_max, bins)
    y_range = np.linspace(y_min, y_max, bins)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))

    # Create theoretical Gaussian distribution
    rv = multivariate_normal(mean, cov)
    gaussian_pdf = rv.pdf(pos)

    im2 = ax2.imshow(
        gaussian_pdf,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="viridis",
        aspect="auto",
    )
    ax2.set_title("Theoretical Gaussian of Query Point Predictions")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    plt.colorbar(im2, ax=ax2, label="Probability Density")

    # Subplot 3: Data histogram of non-Gaussian data
    ax3 = axes[2]

    # Create histogram with common limits
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    im3 = ax3.imshow(hist.T, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="viridis", aspect="auto")
    ax3.set_title("Query Point Prediction Histogram")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y (pixels)")
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    plt.colorbar(im3, ax=ax3, label="Count")


    fig.suptitle(f"Deformity for Rotation = 310 degrees. Variance = {round(np.linalg.det(cov), 2)}", fontsize=16)

    # Adjust subplot spacing to shift middle subplot to the right
    plt.subplots_adjust(wspace=0.4)  # Increase horizontal spacing between subplots

    plt.tight_layout()
    plt.show()

    # Calculate and return some metrics
    hist_normalized = hist / (len(coordinates_to_map) * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))
    gaussian_interp = rv.pdf(np.dstack((np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2))))
    mse = np.mean((hist_normalized.T - gaussian_interp) ** 2)
    print(f"\nMean Squared Error between data and Gaussian: {mse:.6f}")
    
    plt.figure(fig)
    plt.savefig('good_variance_1.svg', dpi=300, bbox_inches='tight')

    return mean, cov, mse


# Example usage:
# Assuming you have your positions, predictions, and coordinates_to_map variables defined
mean, cov, mse = combined_analysis_plot(positions_1, predictions_1, predictions_1)