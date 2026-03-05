import time
import numpy as np
from modelling import generate_synthetic_data
from age_estimation import (
    compute_cpa,
    compute_crown_diameter_from_polygon,
    estimate_age,
)

# Example parameters
NUM_SAMPLES = 50
DIAMETER_RANGE = np.linspace(1, 15, NUM_SAMPLES)
GSD = 1


def time_function(func, *args, n_repeat=1, **kwargs):
    """
    Measure the average execution time of a function over n_repeat runs
    and automatically print the result with the function's name.
    
    Returns:
        result: last function return value
    """
    start = time.time()
    result = None
    for _ in range(n_repeat):
        result = func(*args, **kwargs)
    end = time.time()
    avg_time = (end - start) / n_repeat
    print(f"  → {func.__name__} average runtime over {n_repeat} run(s): {avg_time:.6f} s")
    return result


if __name__ == "__main__":
    print(f"Timing synthetic crown generation for {NUM_SAMPLES} samples...")
    synthetic_crowns = time_function(generate_synthetic_data, DIAMETER_RANGE, visualize=False)

    print("\nTiming individual CPA, diameter, and age computations on one crown...")
    sample_points = synthetic_crowns[0]
    
    cpa_val = time_function(compute_cpa, sample_points, GSD, n_repeat=10)
    diam_val = time_function(compute_crown_diameter_from_polygon, sample_points, GSD, n_repeat=10)
    age_val = time_function(estimate_age, sample_points, GSD, n_repeat=10)