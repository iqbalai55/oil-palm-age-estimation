# Oil Palm Age Estimation from Crown Polygons

This project provides a **Python-based approach** to estimate the **age of oil palm trees** using **crown geometry** derived from segmented polygons. The method considers both **young and mature palms**, using different models depending on the crown size.


## Workflow Overview

1. **Compute Minimum Enclosing Ellipse**
   Using the `welzl` algorithm from `lowner_jon_ellipse`, the **minimum enclosing ellipse** of the crown polygon is computed.

2. **Crown Projection Area (CPA)**
   The area of the ellipse, adjusted by **ground sampling distance (GSD)**, gives the **Crown Projection Area (CPA)**.
   This is used to estimate the age for **young palms** (CPA ≤ 86 m²) with a **linear model**.

3. **Crown Diameter for Mature Palms**
   For larger palms (CPA > 86 m²), the **maximum horizontal distance** across the polygon is computed as the **crown diameter**.
   This is used in a **nonlinear exponential model** for **mature palms**.

4. **Age Estimation Models**

   * **Young palms (Linear CPA model)**

     ```
     Age = 0.59 + 0.15 * CPA
     ```
   * **Mature palms (Exponential Diameter model)**

     ```
     Age = 0.7344 * exp(0.2733 * CD)
     ```

   > CD = crown diameter in meters


## Usage

```python
from age_estimation import estimate_age

# points: Nx2 array of crown polygon coordinates
# gsd: ground sampling distance (m per pixel)
age = estimate_age(points, gsd)
```

The function automatically chooses the appropriate model based on crown size.



