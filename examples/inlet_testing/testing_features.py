import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from src.moc_inlet.utils.geometry import Geometry, DomainSlice, Region
from src.moc_inlet.utils.flowstate import FlowState
from src.moc_inlet.utils.geo_reader import geo_reader


body1, body2 = geo_reader("sample_inlet.csv")
geom = Geometry(body1, body2)

plt.figure(figsize=(12, 4))
plt.fill(body1[:,0], body1[:,1], edgecolor='k', facecolor='lightgray')
plt.fill(body2[:,0], body2[:,1], edgecolor='k', facecolor='lightgray')
upper = geom.upper_wall
lower = geom.lower_wall
x_start = min(geom.x) * 0.90
y_fs_upper = upper.y[0]
y_fs_lower = lower.y[0]






x_end = max(geom.x)
plt.plot(upper[:, 0], upper[:, 1], 'r-', linewidth=2, label='Upper Wall')
plt.plot(lower[:, 0], lower[:, 1], 'b-', linewidth=2, label='Lower Wall')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Original Geometry and Internal Flow Walls')
plt.legend(frameon=False)
plt.show()

freestream = FlowState("air")
freestream.set_state(T=70, P=8729, M=4.03, theta=0.00)

inflow_freestream = DomainSlice(x_start)
inflow_freestream.add_region(y_fs_lower, y_fs_upper,freestream)
pass











