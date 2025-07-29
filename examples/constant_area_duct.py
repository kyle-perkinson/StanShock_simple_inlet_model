from __future__ import annotations

import time
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.plot import SnapshotDiagram, XTDiagram

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0

XSMALL_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=XSMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Plotting utilities
scale = 1e3

# Data
datadir = Path("./data")
datadir.mkdir(exist_ok=True)
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
(figdir / "anim").mkdir(exist_ok=True)

# Chemistry
mech = "data/mechanisms/Nitrogen.yaml"

gas = ct.Solution(mech)
"""
GEOMETRY INPUTS
    from ("Experimental Investigation of Inlet-Combustor Isolators for a Dual-Mode Scramjet...")
        Note that the inlet conditions represent post shock properties (assuming 3 oblique shocks)
"""
H_th = 0.01016  # throat height, m
L_H_th = 12.7  # isolator to throat height ratio
L_iso = L_H_th * H_th  # isolator length, m

W = 0.0508  # Constant Scramjet Width (m)


"""
AMBIENT CONDITIONS
"""
gas1 = ct.Solution(mech)
gas2 = ct.Solution(mech)
# ISOLATOR INLET CONDITIONS
M1 = 2.1993  # isolator inlet Mach number
T1 = 152.48  # isolator inlet static temp, K
p1 = 81741.125  # isolator inlet static pressure, Pa
gas1.TP = T1, p1  # isolator inlet solution/flow initialization
u1 = M1 * gas1.sound_speed
state1 = gas1, u1  # isolator inlet velocity, m/s
# POST-SHOCK ISOLATOR CONDITIONS
gas2.TP = T1 * 1.770, p1 * 1.519
gas2.TP = 260, 2.55e5
# u2 = (M1 * 0.542) * gas2.sound_speed
u2 = 311
state2 = gas2, u2
# NOZZLE EXIT CONDITIONS
p2 = 8278.763  # nozzle exit static pressure, Pa

# Time parameters
t_stab = 0.0025
t_close = 0.8  # duration of closing nozzle
tFinal = 1.25
AR_i = 4.15
AR_f = 1

physics_model = ThermoTable(gas1)

# Define the grid
N_x = 500
xShock = 0.5 * L_iso


def D_H(t, x):
    return (2 * W * H(t, x)) / (W + H(t, x))


def H(t, x):
    x = np.asarray(x)  # ensure x is an array
    return np.ones_like(x, dtype=float) * H_th


def dHdx(t, x):
    x = np.asarray(x)  # ensure x is an array
    return np.zeros_like(x, dtype=float)


def dHdt(t, x):
    x = np.asarray(x)
    return np.zeros_like(x)


x = np.linspace(0, L_iso, N_x)


def A(t, x):
    return H(t, x) * W


def dAdx(t, x):
    return W * dHdx(t, x)


def dAdt(t, x):
    return W * dHdt(t, x)


def dlnAdx(t, x):
    return dAdx(t, x) / A(t, x)


def dlnAdt(t, x):
    return dAdt(t, x) / A(t, x)


# Define the boundary conditions
BC_inlet = gas1.density, u1, gas1.P, None
BC_outlet = None, u2, gas2.P, None

BCs = (BC_inlet, BC_outlet)

# plt.figure()
# plt.plot(x, A(x,0))
# plt.show()
# YOU COMMENTED OUT RHS!!!
try:
    # Initialize and run the simulation
    ss = Combustor(
        n=N_x,
        x=x,
        dlnA_dx=dlnAdx,
        dlnA_dt=dlnAdt,
        d_outer=D_H,
        wall_temperature=330.0,
        include_boundary_layer=True,
        include_pseudoshock=True,
        initialization=("riemann", state1, state2, xShock),
        boundary_conditions=BCs,
        physics=physics_model,
        cfl=1.0,
        include_diffusion=True,
        output_every=100,
        plot_state_interval=50,
    )

    import traceback

    t0 = time.perf_counter()
    plot_variables = [
        "density",
        "velocity",
        "pressure",
        "temperature",
        "mach",
    ]
    ss.xt_diagrams = [
        XTDiagram(ss, variable, skipSteps=10) for variable in plot_variables
    ]

    ss.snapshot_diagrams = [
        SnapshotDiagram(ss, variable, skipSteps=10) for variable in plot_variables
    ]
    ss.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
except Exception as e:
    print("An error occurred:", e)
    print("Full traceback:")
    traceback.print_exc()
finally:
    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=figdir)
    # code.interact(local=locals())
