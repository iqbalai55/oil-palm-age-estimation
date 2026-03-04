from lowner_jon_ellipse import welzl
import numpy as np

CPA_THRESHOLD = 86  # m2 (area)


def compute_cpa(ellipse, gsd):
    if ellipse is None:
        return None

    _, a, b, _ = ellipse
    return np.pi * a * b * (gsd ** 2)


# Crown diameter could be derived from CPA assuming a circular crown,
# but that introduces geometric approximation.
# Since crown diameter is defined as the maximum horizontal distance 
# across the crown, it is more accurate to compute it directly from 
# the segmented polygon by finding the maximum pairwise distance.
def compute_crown_diameter_from_polygon(points, gsd):
    max_dist = 0.0
    n = len(points)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist

    return max_dist * gsd

# jus for viz
def compute_crown_diameter_points_from_polygon(points, gsd=1.0):
    """
    Returns the two points that define the polygon's true crown diameter.
    """
    max_dist = 0
    pt1, pt2 = None, None

    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
                pt1, pt2 = points[i], points[j]

    return pt1, pt2


# The linear CPA-based age model is valid only for young palms 
# (approximately ≤ 13 years, corresponding to CPA ≤ 86 m²).
# For larger crowns, canopy growth becomes nonlinear and the CPA
# model loses reliability. Therefore, a diameter-based nonlinear
# model is used for mature palms.def estimate_age(points, gsd):
def estimate_age(points, gsd):
    ellipse = welzl(points)
    if ellipse is None:
        return None

    cpa = compute_cpa(ellipse, gsd)

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
