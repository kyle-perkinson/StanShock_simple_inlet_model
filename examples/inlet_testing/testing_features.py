import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from moc_inlet.moc_inlet import moc_inlet


solution = moc_inlet("examples/inlet_testing/sample_inlet.csv", T=70, P=8729, M=4.03, theta=0.00, fluid="air",N_wave=10)


# j_inds = np.lexsort((y, sigmas))
# sigmas = sigmas[j_inds]
# y = y[j_inds]
# x_next = []
# for j in range(len(j_inds) - 1):
#     x_next_j =  x_i + (y[j+1] - y[j]) / (np.tan(sigmas[j]) - np.tan(sigmas[j+1]))
#     x_next.append(x_next_j)

# plt.figure()





# plt.figure(figsize=(12, 4))
# plt.fill(body1[:,0], body1[:,1], edgecolor='k', facecolor='lightgray')
# plt.fill(body2[:,0], body2[:,1], edgecolor='k', facecolor='lightgray')

# plt.plot(upper[:, 0], upper[:, 1], 'r-', linewidth=2, label='Upper Wall')
# plt.plot(lower[:, 0], lower[:, 1], 'b-', linewidth=2, label='Lower Wall')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.title('Original Geometry and Internal Flow Walls')
# plt.legend(frameon=False)
# plt.show()










