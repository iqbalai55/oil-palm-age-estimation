import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial.distance import pdist

from age_estimation import (
    estimate_age,
    compute_crown_diameter_points_from_polygon,
    age_estimation_using_diameter,
    age_estimation_using_cpa,
    compute_cpa,
)
from lowner_jon_ellipse import welzl


# =====================================
# PARAMETERS
# =====================================
GSD = 1
THETA = np.linspace(0, 2 * np.pi, 120)
TARGET_AGE_LIMIT = 25


# =====================================
# FIND MAX DIAMETER FOR TARGET AGE
# =====================================
print(f"Computing maximum crown diameter for {TARGET_AGE_LIMIT} years...")

diameter_search = np.linspace(1, 30, 2000)
age_search = np.array([age_estimation_using_diameter(d) for d in diameter_search])

valid_idx = np.where(age_search >= TARGET_AGE_LIMIT)[0][0]
max_diameter = diameter_search[valid_idx]
print(f"  → Maximum diameter for {TARGET_AGE_LIMIT} years ≈ {max_diameter:.2f} m")

diameter_range = np.linspace(1, max_diameter, 50)


# =====================================
# VISUALISE EXAMPLE POLYGON DATA
# =====================================
polygon_data = [
    "0 0.432571 0.455729 0.388858 0.476190 0.362816 0.471540 0.367467 0.460379 0.396298 0.435268 "
    "0.409319 0.412946 0.424200 0.401786 0.378627 0.401786 0.341425 0.395275 0.343285 0.380394 "
    "0.371187 0.371094 0.401879 0.372954 0.383278 0.355283 0.358166 0.333891 0.349795 0.315290 "
    "0.376767 0.324591 0.408389 0.338542 0.438151 0.345052 0.446522 0.343192 0.440011 0.320871 "
    "0.435361 0.296689 0.441871 0.269717 0.455822 0.284598 0.470703 0.310640 0.482794 0.326451 "
    "0.503255 0.315290 0.523717 0.299479 0.540458 0.279948 0.545108 0.303199 0.536737 0.333891 "
    "0.530227 0.347842 0.552548 0.333891 0.572080 0.332031 0.594401 0.332031 0.603702 0.340402 "
    "0.622303 0.366443 0.599981 0.371094 0.563709 0.378534 0.579520 0.389695 0.605562 0.396205 "
    "0.628813 0.408296 0.629743 0.424107 0.597191 0.425037 0.618583 0.436198 0.605562 0.444568 "
    "0.576730 0.445499 0.573940 0.458519 0.566499 0.462240 0.571150 0.488281 0.586961 0.528274 "
    "0.562779 0.511533 0.537667 0.481771 0.537667 0.505022 0.535807 0.517113 0.519066 0.494792 "
    "0.509766 0.520833 0.493955 0.506882 0.470703 0.516183 0.455822 0.497582 0.453962 0.475260 "
    "0.454892 0.450149",

    "0 0.237699 0.852726 0.206948 0.862699 0.185339 0.867686 0.199468 0.840259 0.248504 0.803690 "
    "0.212766 0.775432 0.204455 0.750499 0.226064 0.751330 0.260140 0.767952 0.255153 0.756316 "
    "0.252660 0.733876 0.265126 0.727227 0.262633 0.694814 0.270113 0.674867 0.282580 0.689827 "
    "0.291722 0.741356 0.300033 0.747174 0.314993 0.719747 0.338265 0.697307 0.344914 0.707281 "
    "0.331616 0.758810 0.374003 0.711436 0.388132 0.718085 0.378989 0.738032 0.369847 0.749668 "
    "0.378158 0.755485 0.378158 0.759641 0.409741 0.741356 0.415559 0.767121 0.386469 0.779588 "
    "0.425532 0.785406 0.431350 0.797872 0.405585 0.812832 0.424701 0.835273 0.404754 0.846077 "
    "0.369847 0.848570 0.341589 0.840259 0.355718 0.860206 0.374834 0.880984 0.388132 0.898438 "
    "0.383976 0.920878 0.359874 0.910904 0.350731 0.897606 0.357380 0.934176 0.345745 0.944980 "
    "0.329122 0.922540 0.316656 0.901762 0.312500 0.880153 0.307513 0.911735 0.297540 0.913398 "
    "0.288398 0.893451 0.256815 0.920047 0.252660 0.903424 0.237699 0.925864 0.217753 0.930851 "
    "0.219415 0.915060 0.238531 0.862699",
]

print(f"\nVisualising {len(polygon_data)} example polygon(s)...")
fig, axes = plt.subplots(1, len(polygon_data), figsize=(12, 6))
axes = np.atleast_1d(axes)

for i, poly_str in enumerate(polygon_data):
    tokens = poly_str.strip().split()
    coords = np.array(list(map(float, tokens[1:])))  # skip class label

    if len(coords) % 2 != 0:
        print(f"  Warning: polygon {i + 1} has odd coordinate count — dropping last value.")
        coords = coords[:-1]

    points = coords.reshape(-1, 2)

    ellipse = welzl(points)
    center = ellipse[0]
    semi_major = float(ellipse[1])
    semi_minor = float(ellipse[2])
    angle_deg = np.degrees(float(ellipse[3]))

    p1, p2 = compute_crown_diameter_points_from_polygon(points)
    diameter = np.linalg.norm(p1 - p2)

    ax = axes[i]
    ax.add_patch(Polygon(points, closed=True, facecolor="green", alpha=0.3, label="Polygon Mask"))
    ax.scatter(points[:, 0], points[:, 1], c="darkgreen", s=10, alpha=0.7, label="Polygon Points")
    ax.add_patch(Ellipse(
        xy=center,
        width=2 * semi_major,
        height=2 * semi_minor,
        angle=angle_deg,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        label="Fitted Ellipse",
    ))
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]],
        color="blue", linestyle="--", linewidth=2,
        label=f"Crown Diameter: {diameter:.2f}",
    )
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color="orange", s=30, label="Farthest Points")
    ax.set_aspect("equal")
    ax.set_title(f"Oil Palm {i + 1}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =====================================
# MAIN SIMULATION — SYNTHETIC CROWNS
# =====================================
print("\nRunning full modelling simulation over diameter range...")

computed_diameters = []
ages_pipeline = []
ages_linear = []
ages_exp = []

plt.figure(figsize=(8, 8))

for D in diameter_range:
    n_fronds = np.random.randint(14, 18)
    frond_amplitude = 0.10
    roughness = 0.03
    ellipse_ratio = 1 + 0.12 * np.random.randn()
    angle_rotation = np.random.uniform(0, 2 * np.pi)

    radial_wave = 1 + frond_amplitude * np.sin(n_fronds * THETA)
    r = (D / 2) * radial_wave * (1 + roughness * np.random.randn(len(THETA)))

    x = r * np.cos(THETA) * ellipse_ratio
    y = r * np.sin(THETA)

    x_rot = x * np.cos(angle_rotation) - y * np.sin(angle_rotation)
    y_rot = x * np.sin(angle_rotation) + y * np.cos(angle_rotation)
    points = np.column_stack([x_rot, y_rot])

    ellipse = welzl(points)
    cpa = compute_cpa(ellipse, GSD)

    computed_D = np.max(pdist(points)) * GSD

    age_linear = age_estimation_using_cpa(cpa)
    age_exp = age_estimation_using_diameter(computed_D)
    age_pipeline = estimate_age(points, GSD)

    computed_diameters.append(computed_D)
    ages_linear.append(age_linear)
    ages_exp.append(age_exp)
    ages_pipeline.append(age_pipeline)

    plt.plot(x_rot, y_rot, color="green", alpha=0.3)

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Synthetic Crown Shapes for All Diameters")
plt.grid(alpha=0.3)
plt.show()


# =====================================
# SORT RESULTS BY CROWN DIAMETER
# =====================================
computed_diameters = np.array(computed_diameters)
ages_linear = np.array(ages_linear)
ages_exp = np.array(ages_exp)
ages_pipeline = np.array(ages_pipeline)

idx = np.argsort(computed_diameters)
cd_sorted = computed_diameters[idx]
al_sorted = ages_linear[idx]
ae_sorted = ages_exp[idx]
ap_sorted = ages_pipeline[idx]


# =====================================
# MODEL COMPARISON PLOT (ALL AGES)
# =====================================
plt.figure(figsize=(12, 8))

plt.plot(cd_sorted, ap_sorted, color="black", linewidth=4, label="Piecewise Model")
plt.plot(cd_sorted, al_sorted, color="#2ca02c", linewidth=2.5, alpha=0.9, label="Linear CPA Model (≤13 yrs)")
plt.plot(cd_sorted, ae_sorted, color="#ff7f0e", linewidth=2.5, alpha=0.9, label="Exponential Diameter Model")

plt.axhline(y=13, color="red", linewidth=2, linestyle="--", alpha=0.8)
plt.text(1, 13.5, "CPA Validity Limit (13 years)", color="red",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.xlabel("Crown Diameter (m)")
plt.ylabel("Age (years)")
plt.title(f"Oil Palm Age Estimation — Model Comparison (≤ {TARGET_AGE_LIMIT} Years)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yticks(np.arange(0, TARGET_AGE_LIMIT + 1, 1))
plt.ylim(0, TARGET_AGE_LIMIT)
plt.xlim(0, 16)
plt.tight_layout()
plt.show()


# =====================================
# FILTER TO AGE ≤ 13 (CPA VALID RANGE)
# =====================================
mask = ap_sorted <= 13

diam_f = cd_sorted[mask]
al_f = al_sorted[mask]
ae_f = ae_sorted[mask]

sort_f = np.argsort(diam_f)
diam_f = diam_f[sort_f]
al_f = al_f[sort_f]
ae_f = ae_f[sort_f]


# =====================================
# LINEAR vs EXPONENTIAL COMPARISON PLOT
# =====================================
plt.figure(figsize=(12, 6))

plt.plot(diam_f, al_f, label="Linear CPA", color="#2ca02c", linewidth=2.5)
plt.plot(diam_f, ae_f, label="Exponential Diameter", color="#ff7f0e", linewidth=2.5)
plt.fill_between(diam_f, al_f, ae_f, color="purple", alpha=0.2, label="Difference Area")

plt.xlabel("Crown Diameter (m)")
plt.ylabel("Age (years)")
plt.title("Oil Palm Age Estimation: Linear CPA vs Exponential Diameter")
plt.legend()
plt.grid(alpha=0.3)
plt.yticks(np.arange(0, 19, 1))
plt.ylim(0, 18)
plt.tight_layout()
plt.show()


# =====================================
# ABSOLUTE ERROR PLOT
# =====================================
error = al_f - ae_f
error_abs = np.abs(error)

plt.figure(figsize=(12, 6))
plt.plot(diam_f, error_abs, label="|Linear − Exponential|", color="purple", linewidth=2)
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.xlabel("Crown Diameter (m)")
plt.ylabel("Absolute Age Difference (years)")
plt.title("Absolute Error Between Linear and Exponential Models")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================
# ERROR STATISTICS
# =====================================
mae = np.mean(error_abs)
rmse = np.sqrt(np.mean(error ** 2))
std_err = np.std(error)
max_err = np.max(error_abs)
min_err = np.min(error_abs)
diam_at_max_err = diam_f[np.argmax(error_abs)]
diam_at_min_err = diam_f[np.argmin(error_abs)]

print("\n--- Error Statistics (Linear vs Exponential) ---")
print(f"  Mean Absolute Error (MAE) : {mae:.2f} years")
print(f"  RMSE                      : {rmse:.2f} years")
print(f"  Std Dev of Error          : {std_err:.2f} years")
print(f"  Max Absolute Error        : {max_err:.2f} years  @ diameter = {diam_at_max_err:.2f} m")
print(f"  Min Absolute Error        : {min_err:.2f} years  @ diameter = {diam_at_min_err:.2f} m")
print("\nSimulation complete.")