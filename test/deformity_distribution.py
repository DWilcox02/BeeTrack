import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy import stats

parameter = {"x": 10, "y": 10, "radius": 20}


def _generate_point_cloud(center_point, n_points_per_perimeter):
    """Helper method to generate points that fill a circle around a center point"""
    center_x = float(center_point["x"])
    center_y = float(center_point["y"])
    radius = float(center_point["radius"])

    # Create a filled circle of points
    circle_points = []

    # Determine how dense to make the grid based on perimeter points
    # We'll calculate the grid size to achieve approximately the same density
    grid_step = (2 * radius) / np.sqrt(n_points_per_perimeter * 3)

    # Create a grid of points within a square around the circle
    x_min, x_max = center_x - radius, center_x + radius
    y_min, y_max = center_y - radius, center_y + radius

    # Generate grid points
    x_coords = np.arange(x_min, x_max, grid_step)
    y_coords = np.arange(y_min, y_max, grid_step)

    # Add center point first
    circle_points.append((center_x, center_y))

    # Add all points within the radius
    for x in x_coords:
        for y in y_coords:
            # Calculate distance from center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Only include points within the radius
            if distance <= radius:
                circle_points.append((x, y))

    cloud_points = np.array(circle_points, dtype=np.float32)

    return cloud_points


# Generate the point cloud
points = _generate_point_cloud(parameter, n_points_per_perimeter=12)

# Calculate pairwise distances
distances = pdist(points)

print(f"Number of points: {len(points)}")
print(f"Number of pairwise distances: {len(distances)}")
print(f"Distance range: {distances.min():.2f} to {distances.max():.2f}")

# Fit to normal distribution
mu, sigma = stats.norm.fit(distances)
print(f"\nNormal Distribution Fit:")
print(f"Mean (μ): {mu:.2f}")
print(f"Standard deviation (σ): {sigma:.2f}")

# Fit to many more distributions
distributions_to_test = [
    ("Normal", stats.norm),
    ("Gamma", stats.gamma),
    ("Beta", stats.beta),
    ("Log-normal", stats.lognorm),
    ("Weibull", stats.weibull_min),
    ("Exponential", stats.expon),
    ("Chi-squared", stats.chi2),
    ("Rayleigh", stats.rayleigh),
    ("Uniform", stats.uniform),
    ("Pareto", stats.pareto),
    ("Gumbel", stats.gumbel_r),
    ("Laplace", stats.laplace),
    ("Logistic", stats.logistic),
    ("Student-t", stats.t),
    ("F-distribution", stats.f),
    ("Inverse Gaussian", stats.invgauss),
    ("Burr", stats.burr),
    ("Generalized Extreme Value", stats.genextreme),
    ("Skew Normal", stats.skewnorm),
    ("Rice", stats.rice),
]

# Store results
fit_results = []

print("Fitting multiple distributions...")
for name, distribution in distributions_to_test:
    try:
        # Fit parameters
        if name == "F-distribution":
            # F-distribution needs special handling
            params = distribution.fit(distances, f0=1)
        else:
            params = distribution.fit(distances)

        # Calculate log-likelihood and AIC
        log_likelihood = np.sum(distribution.logpdf(distances, *params))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood

        # KS test
        ks_stat, ks_p = stats.kstest(distances, lambda x: distribution.cdf(x, *params))

        fit_results.append(
            {
                "name": name,
                "distribution": distribution,
                "params": params,
                "aic": aic,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "log_likelihood": log_likelihood,
            }
        )

    except Exception as e:
        print(f"Could not fit {name}: {e}")

# Sort by AIC (best fit first)
fit_results.sort(key=lambda x: x["aic"])

print(f"\nTop 10 Distribution Fits (by AIC):")
print(f"{'Rank':<4} {'Distribution':<25} {'AIC':<10} {'KS p-value':<12} {'Parameters'}")
print("-" * 80)
for i, result in enumerate(fit_results[:10]):
    params_str = ", ".join([f"{p:.3f}" for p in result["params"]])
    print(f"{i + 1:<4} {result['name']:<25} {result['aic']:<10.2f} {result['ks_p']:<12.6f} {params_str}")

# Store top distributions for plotting
top_distributions = fit_results[:5]

# Create histogram with top fitted distributions
plt.figure(figsize=(14, 10))

# Plot histogram
n, bins, patches = plt.hist(
    distances, bins=50, alpha=0.7, color="skyblue", edgecolor="black", density=True, label="Data"
)

# Generate x values for smooth curves
x = np.linspace(distances.min(), distances.max(), 1000)

# Plot top 5 distributions
colors = ["red", "green", "orange", "purple", "brown"]
for i, result in enumerate(top_distributions):
    try:
        pdf = result["distribution"].pdf(x, *result["params"])
        plt.plot(x, pdf, color=colors[i], linewidth=2, label=f"{result['name']} (AIC: {result['aic']:.1f})")
    except:
        pass

plt.xlabel("Pairwise Distance")
plt.ylabel("Probability Density")
plt.title("Histogram with Top 5 Fitted Distributions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Alternative fitting methods using Maximum Likelihood Estimation
print(f"\n" + "=" * 60)
print("ALTERNATIVE FITTING METHODS")
print("=" * 60)

# 1. Method of Moments for specific distributions
print("\n1. METHOD OF MOMENTS:")
sample_mean = np.mean(distances)
sample_var = np.var(distances)
sample_skew = stats.skew(distances)
sample_kurt = stats.kurtosis(distances)

print(f"Sample moments:")
print(f"  Mean: {sample_mean:.3f}")
print(f"  Variance: {sample_var:.3f}")
print(f"  Skewness: {sample_skew:.3f}")
print(f"  Kurtosis: {sample_kurt:.3f}")

# Method of moments for Gamma distribution
alpha_mom = sample_mean**2 / sample_var
beta_mom = sample_var / sample_mean
print(f"Gamma (Method of Moments): α={alpha_mom:.3f}, β={beta_mom:.3f}")

# 2. Kernel Density Estimation (non-parametric)
print("\n2. KERNEL DENSITY ESTIMATION (Non-parametric):")
from scipy.stats import gaussian_kde

kde = gaussian_kde(distances)
kde_x = np.linspace(distances.min(), distances.max(), 1000)
kde_pdf = kde(kde_x)

plt.figure(figsize=(10, 6))
plt.hist(distances, bins=50, alpha=0.7, density=True, label="Data")
plt.plot(kde_x, kde_pdf, "r-", linewidth=2, label="KDE")
plt.xlabel("Pairwise Distance")
plt.ylabel("Probability Density")
plt.title("Kernel Density Estimation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. Mixture Models
print("\n3. GAUSSIAN MIXTURE MODEL:")
from sklearn.mixture import GaussianMixture

# Try different numbers of components
n_components_range = range(1, 6)
gmm_results = []

for n_comp in n_components_range:
    gmm = GaussianMixture(n_components=n_comp, random_state=42)
    gmm.fit(distances.reshape(-1, 1))

    # Calculate BIC and AIC
    bic = gmm.bic(distances.reshape(-1, 1))
    aic = gmm.aic(distances.reshape(-1, 1))

    gmm_results.append({"n_components": n_comp, "gmm": gmm, "bic": bic, "aic": aic})

print("Components  AIC      BIC")
print("-" * 25)
for result in gmm_results:
    print(f"{result['n_components']:<11} {result['aic']:<8.1f} {result['bic']:<8.1f}")

# Plot best GMM
best_gmm = min(gmm_results, key=lambda x: x["bic"])
print(f"\nBest GMM: {best_gmm['n_components']} components (BIC: {best_gmm['bic']:.1f})")

# Plot GMM
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=50, alpha=0.7, density=True, label="Data")

x_gmm = np.linspace(distances.min(), distances.max(), 1000).reshape(-1, 1)
gmm_pdf = np.exp(best_gmm["gmm"].score_samples(x_gmm))
plt.plot(x_gmm, gmm_pdf, "r-", linewidth=2, label=f"GMM ({best_gmm['n_components']} components)")

plt.xlabel("Pairwise Distance")
plt.ylabel("Probability Density")
plt.title("Gaussian Mixture Model Fit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. Empirical Distribution Function
print("\n4. EMPIRICAL DISTRIBUTION FUNCTION:")
from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(distances)
x_ecdf = np.linspace(distances.min(), distances.max(), 1000)
y_ecdf = ecdf(x_ecdf)

plt.figure(figsize=(10, 6))
plt.plot(x_ecdf, y_ecdf, "b-", linewidth=2, label="Empirical CDF")

# Compare with best parametric fit
if fit_results:
    best_dist = fit_results[0]
    theoretical_cdf = best_dist["distribution"].cdf(x_ecdf, *best_dist["params"])
    plt.plot(x_ecdf, theoretical_cdf, "r--", linewidth=2, label=f"{best_dist['name']} CDF")

plt.xlabel("Pairwise Distance")
plt.ylabel("Cumulative Probability")
plt.title("Empirical vs Theoretical CDF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print basic statistics
print(f"\nDistance Statistics:")
print(f"Mean: {distances.mean():.2f}")
print(f"Median: {np.median(distances):.2f}")
print(f"Standard deviation: {distances.std():.2f}")

# Q-Q plot for best distribution
if fit_results:
    plt.figure(figsize=(12, 4))

    # Q-Q plot for normal distribution
    plt.subplot(1, 3, 1)
    stats.probplot(distances, dist="norm", plot=plt)
    plt.title("Q-Q Plot: Normal Distribution")
    plt.grid(True)

    # Q-Q plot for best distribution
    plt.subplot(1, 3, 2)
    best_dist = fit_results[0]
    if best_dist["name"] != "Normal":
        try:
            stats.probplot(distances, dist=best_dist["distribution"], sparams=best_dist["params"], plot=plt)
            plt.title(f"Q-Q Plot: {best_dist['name']}")
            plt.grid(True)
        except:
            plt.text(
                0.5,
                0.5,
                f"Q-Q plot not available\nfor {best_dist['name']}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

    # Residuals plot
    plt.subplot(1, 3, 3)
    sorted_distances = np.sort(distances)
    theoretical_quantiles = best_dist["distribution"].ppf(np.linspace(0.01, 0.99, len(distances)), *best_dist["params"])
    residuals = sorted_distances - np.sort(theoretical_quantiles)
    plt.scatter(np.sort(theoretical_quantiles), residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Residuals")
    plt.title(f"Residuals: {best_dist['name']}")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Additional diagnostic: Anderson-Darling test for normality
print(f"\n5. ANDERSON-DARLING TEST FOR NORMALITY:")
ad_stat, critical_values, significance_levels = stats.anderson(distances, dist="norm")
print(f"Anderson-Darling statistic: {ad_stat:.4f}")
for i, (cv, sl) in enumerate(zip(critical_values, significance_levels)):
    result = "Reject" if ad_stat > cv else "Fail to reject"
    print(f"  {sl}% significance level: {cv:.4f} - {result} normality")

# Shapiro-Wilk test (if sample size allows)
if len(distances) <= 5000:  # Shapiro-Wilk has limitations on sample size
    print(f"\n6. SHAPIRO-WILK TEST FOR NORMALITY:")
    sw_stat, sw_p = stats.shapiro(distances)
    print(f"Shapiro-Wilk statistic: {sw_stat:.4f}, p-value: {sw_p:.6f}")
    print(f"Result: {'Reject' if sw_p < 0.05 else 'Fail to reject'} normality hypothesis")
