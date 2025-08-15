import numpy as np
import matplotlib.pyplot as plt
from utils.geometry import Geometry, Inflection
# Example polygons (as Nx2 arrays)
poly1 = np.array([
    [0, 2], [1, 3], [3, 3], [4, 2], [3, 1], [1, 1]
])

poly2 = np.array([
    [0, 0], [1, 0.5], [3, 0.5], [4, 0], [3, -1], [1, -1]
])

# Assume Geometry class from previous code is already defined
geom = Geometry(poly1, poly2)

# Plot
plt.figure(figsize=(8, 4))

# Original polygons
plt.plot(poly1[:, 0], poly1[:, 1], 'o--', label='Polygon 1')
plt.plot(poly2[:, 0], poly2[:, 1], 'o--', label='Polygon 2')

# Internal walls after processing
upper = geom.upper_wall
lower = geom.lower_wall
plt.plot(upper[:, 0], upper[:, 1], 'r-', linewidth=2, label='Upper Wall')
plt.plot(lower[:, 0], lower[:, 1], 'b-', linewidth=2, label='Lower Wall')

plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Polygons and Internal Flow Walls')
plt.legend()
plt.grid(True)
plt.show()

print('blah')