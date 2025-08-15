import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# Emami Scramjet Parameterized Geometry
# Origin at ramp tip
# Inputs are AR and Liso_Hth
AR = 4.15
Liso_Hth = 12.7  # ratio of isolator length to throat height
H_th = 0.01016  # throat height

# ~~~~~RELEVANT DIMENSIONS~~~~~~
# ~~Lengths~~
lx = 0.15
L_rp = 0.248158  # Ramp length
L_bt = 0.102671  # Bottom ramp length
L_iso = Liso_Hth * H_th  # Isolator length
L_dd = 0.129032  # Lower diffuser length
L_cd = 0.272  # Lower combustor length
L_fp = 0.150  # Total flap length

L_fd = 0.080431  # flat section before diffuser
L_du = 0.048601  # upper diffuser length
L_cu = 0.267637  # upper comb chamber wall length
L_ovu1 = 0.007886 * Liso_Hth  # length from noz tip to upper outer first vertice
L_nu = 0.076583  # upper nozzle narrowing section length
L_fu = 0.078055  # upper nozzle flat section length
L_ovu2 = 0.035137 * Liso_Hth
L_ovu3 = 0.003711 * Liso_Hth
L_ovu4 = 0.005963 * Liso_Hth
L_tot = L_rp + L_iso + L_dd + L_cd + L_fp

# ~~Heights~~
H_rp = 0.048237
H_bt = 0.012392  # bottom ramp height
H_cc = 0.067106  # combustor total height
H_noz = 0.009435  # upper nozzle height
H_exit = AR * H_th
h_wall = 0.005
h_cowl = 0.013
hy_half = 0.5

x_cle = 0.184658  # cowl leading edge
y_cle = 0.055958  # cowl leading edge

# ~~~~~Calculate flap angle/coordinates~~~~~
H_exit = H_th * AR  # nozzle exit area
H_fp = H_cc - H_exit - H_noz  # flap height

L_fp_int = L_fp / 1.163  # chamfer location
h_fp_min = 0.0066  # flap min thickness
h_fp_max = 0.0103  # flap max thickness (connection to ext. cc)

theta_fp = np.arcsin(H_fp / L_fp)

xf1 = np.cos(theta_fp) * L_fp_int
yf1 = np.sin(theta_fp) * L_fp_int

xf2 = np.cos(theta_fp) * L_fp
yf2 = np.sin(theta_fp) * L_fp - np.cos(theta_fp) * h_fp_min

# DEFINE BOUNDING BOX:
xd1 = 0
xd2 = L_rp
xd3 = xd2 + L_iso
xd4 = xd3 + L_dd
xd5 = xd4 + L_cd
xd6 = xd5 + xf1
xd7 = xd5 + xf2
xd8 = xd5
xd9 = L_bt

yd1 = 0
yd2 = H_rp
yd3 = H_rp
yd4 = 0
yd5 = 0
yd6 = yf1
yd7 = yf2
yd8 = -h_fp_max
yd9 = -H_bt

# Define UPPER WALL:
xu1 = x_cle
xu2 = L_rp
xu3 = xu2 + L_iso + L_fd
xu4 = xu3 + L_du
xu5 = xu4 + L_cu
xu6 = xu5 + L_nu
xu7 = xu6 + L_fu
xu8 = xu7
xu9 = xu7 - L_ovu1
xu10 = xu9 - L_ovu2
xu11 = xu10 - L_ovu3
xu12 = xu11 - L_ovu4

yu1 = y_cle
yu2 = H_rp + H_th
yu3 = yu2
yu4 = H_cc
yu5 = H_cc
yu6 = H_cc - H_noz
yu7 = yu6
yu8 = yu7 + h_wall
yu9 = yu6 + 3.4 * h_wall
yu10 = yu9
yu11 = yu10 - 1.7 * h_wall
yu12 = yu11

x_d = np.array([xd1, xd2, xd3, xd4, xd5, xd6, xd7, xd8, xd9])
y_d = np.array([yd1, yd2, yd3, yd4, yd5, yd6, yd7, yd8, yd9])
x_u = np.array([xu1, xu2, xu3, xu4, xu5, xu6, xu7, xu8, xu9, xu10, xu11, xu12])
y_u = np.array([yu1, yu2, yu3, yu4, yu5, yu6, yu7, yu8, yu9, yu10, yu11, yu12])


store_path = '../02_raw_data'
df_up = pd.DataFrame({
    'xu': x_u,
    'yu': y_u,
})
df_up.to_csv(os.path.join(store_path, 'gu.csv'), index=False)
df_down = pd.DataFrame({
    'xd': x_d,
    'yd': y_d,
})

df_down.to_csv(os.path.join(store_path, 'gd.csv'), index=False)

print('done')
