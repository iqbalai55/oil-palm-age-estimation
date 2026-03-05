from lowner_jon_ellipse import welzl
import numpy as np
from scipy.spatial import ConvexHull

CPA_THRESHOLD = 86  # m2 (area)


def compute_cpa(points, gsd=1.0, use_hull=True):
    """
    Compute the crown projection area (CPA) of a set of points.

    Parameters:
        points: np.array of shape (n,2), crown polygon points
        gsd: ground sampling distance
        use_hull: if True, fit ellipse only on convex hull for speed

    Returns:
        cpa: crown projection area in same units as gsd**2
    """

    if use_hull:
        # use only convex hull points for speed
        hull = ConvexHull(points)
        points_fit = points[hull.vertices]
    else:
        points_fit = points

    ellipse = welzl(points_fit)
    if ellipse is None:
        # fallback if welzl fails (e.g., degenerate points)
        x_span = np.max(points[:, 0]) - np.min(points[:, 0])
        y_span = np.max(points[:, 1]) - np.min(points[:, 1])
        return np.pi * (x_span / 2) * (y_span / 2) * (gsd ** 2)

    _, a, b, _ = ellipse
    return np.pi * a * b * (gsd ** 2)


# Crown diameter could be derived from CPA assuming a circular crown,
# but that introduces geometric approximation.
# Since crown diameter is defined as the maximum horizontal distance 
# across the crown, it is more accurate to compute it directly from 
# the segmented polygon by finding the maximum pairwise distance.
def compute_crown_diameter_from_polygon(points, gsd=1.0):
    """
    Compute the crown diameter using rotating calipers on the convex hull.
    Returns the diameter in the same units as gsd.
    """
    points = np.asarray(points)
    
    # compute convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # use rotating calipers on hull points
    max_dist, (pt1, pt2) = rotating_calipers_diameter(hull_points)
    return max_dist * gsd


def compute_crown_diameter_points_from_polygon(points, gsd=1.0):
    """
    Returns the two points on the convex hull that define the true crown diameter.
    """
    points = np.asarray(points)
    
    # compute convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # get points of max distance
    max_dist, (pt1, pt2) = rotating_calipers_diameter(hull_points)
    return max_dist, np.array(pt1) * gsd, np.array(pt2) * gsd


# The linear CPA-based age model is valid only for young palms 
# (approximately ≤ 13 years, corresponding to CPA ≤ 86 m²).
# For larger crowns, canopy growth becomes nonlinear and the CPA
# model loses reliability. Therefore, a diameter-based nonlinear
# model is used for mature palms.def estimate_age(points, gsd):
def estimate_age(points, gsd):
    
    cpa = compute_cpa(points, gsd)

    if cpa <= CPA_THRESHOLD:
        return age_estimation_using_cpa(cpa)

    diameter = compute_crown_diameter_from_polygon(points, gsd)
    return age_estimation_using_diameter(diameter)


# based on https://www.sciencedirect.com/science/article/abs/pii/S0924271614001968
def age_estimation_using_cpa(cpa):
    """
    Linear model valid for young palms (≤ 13 years)
    Age = 0.59 + 0.15 * CPA
    
    Where CPA is crown projection area in square meters.
    """
    return 0.59 + 0.15 * cpa


# based on https://www.researchgate.net/publication/301823103_Relationships_between_Crown_Size_and_Aboveground_Biomass_of_Oil_Palms_An_Evaluation_of_Allometric_Models
def age_estimation_using_diameter(diameter):
    """
    Exponential age model for mature oil palms.
    
    Age = 0.7344 * exp(0.2733 * CD)
    
    where CD is crown diameter in meters.
    """
    if diameter is None:
        return None
    
    return 0.7344 * np.exp(0.2733 * diameter)


def rotating_calipers_diameter(points):
    points = np.asarray(points)
    n = len(points)
    if n < 2:
        return 0, (None, None)

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    max_dist = 0
    pt1 = pt2 = points[0]
    j = 1

    for i in range(n):
        next_i = (i + 1) % n
        while True:
            next_j = (j + 1) % n
            if cross(points[i], points[next_i], points[next_j]) > cross(points[i], points[next_i], points[j]):
                j = next_j
            else:
                break

        d1 = np.linalg.norm(points[i] - points[j])
        d2 = np.linalg.norm(points[i] - points[(j+1)%n])

        if d1 > max_dist:
            max_dist = d1
            pt1, pt2 = points[i], points[j]
        if d2 > max_dist:
            max_dist = d2
            pt1, pt2 = points[i], points[(j+1)%n]

    return max_dist, (pt1, pt2)
