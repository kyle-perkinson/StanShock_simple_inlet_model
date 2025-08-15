"""
PROCEDURE

1. Load in geometry (two bodies) and initial flow conditions.
2. Run geometry.py to obtain upper and lower walls and inflections
3. Determine x0 = minimum of geometry.py plus some buffer
4. Define initial bounding streamlines (both at theta_flow, consisting of upper_wall y/lower wall y.
5. Generate first slice being composed of entirely this region.
6. Identify the next point, which is the minimum of:
    a. 
"""