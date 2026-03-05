# Oil Palm Age Estimation from Crown Polygons

This project provides a **Python-based approach** to estimate the **age of oil palm trees** using **crown geometry** derived from segmented polygons. The method considers both **young and mature palms**, using different models depending on the crown size.


The age estimation of oil palms begins by computing the minimum enclosing ellipse of the crown polygon using the welzl algorithm from lowner_jon_ellipse. The area of this ellipse, adjusted by the ground sampling distance (GSD), provides the Crown Projection Area (CPA), which serves as a basis for estimating the age of young palms. For palms with a CPA of 86 m² or less, a simple linear model is used:

Age = 0.59 + 0.15 × CPA

For larger or mature palms (CPA > 86 m²), the age estimation relies on the crown diameter, defined as the maximum horizontal distance across the crown polygon. This is calculated using the rotating calipers algorithm applied to the convex hull of the crown. The resulting crown diameter (CD) feeds into a nonlinear exponential model to estimate the age of mature palms:

Age = 0.7344 × exp(0.2733 × CD)

By combining these two approaches, the workflow provides accurate age estimation for both young and mature oil palms, taking into account the geometric characteristics of the crown.
## Usage

```python
from age_estimation import estimate_age

# points: Nx2 array of crown polygon coordinates
# gsd: ground sampling distance (m per pixel)
age = estimate_age(points, gsd)
```

The function automatically chooses the appropriate model based on crown size.



