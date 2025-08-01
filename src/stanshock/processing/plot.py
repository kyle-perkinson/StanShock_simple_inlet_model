from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class XTDiagram:
    """
    This class is used to store the relevant data for the XT diagram
        inputs:
            domain=component to be plotted
            variable=string of the variable
            skipSteps=number of iterations between updates
            x=mesh for plotting
            limits = tuple of maximum and minimum for the pcolor (vMin,vMax)
    """

    def __init__(self, domain, variable, skipSteps=0, x=None, limits=None):
        self.name = None
        self.skipSteps = 0
        self.limits = limits

        self.name = variable.lower()
        self.skipSteps = skipSteps  # number of timesteps to skip
        # check interpolation grid
        geometry = domain.geometry
        if x is None:
            self.x = geometry.x
        elif (x[-1] > geometry.x[-1]) or (x[0] < geometry.x[0]):
            msg = "Invalid Interpolation Grid"
            raise Exception(msg)
        else:
            self.x = x

        self.variable = []  # list of numpy arrays of the variable w.r.t x
        self.t = []  # list of times

        self.update(domain)

    def update(self, domain):
        """
        This method updates the XT diagram.
            inputs:
                XTDiagram: the XTDiagram object
        """
        variable = self.name
        state = domain.state
        geometry = domain.geometry
        idx_cells = domain.idx_cells

        if variable in ["density", "r", "rho"]:
            self.variable.append(
                np.interp(self.x, geometry.x, state.density[idx_cells])
            )
        elif variable in ["velocity", "u"]:
            self.variable.append(
                np.interp(self.x, geometry.x, state.velocity[idx_cells])
            )
        elif variable in ["pressure", "p"]:
            self.variable.append(
                np.interp(self.x, geometry.x, state.pressure[idx_cells])
            )
        elif variable in ["temperature", "t"]:
            T = domain.physics.get_temperature(state)
            self.variable.append(np.interp(self.x, geometry.x, T[idx_cells]))
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            self.variable.append(np.interp(self.x, geometry.x, state.gamma[idx_cells]))
        elif variable in domain.physics.scalar_names:
            scalarIndex = domain.physics.scalar_names.index(variable)
            self.variable.append(
                np.interp(self.x, geometry.x, state.composition[idx_cells, scalarIndex])
            )
        elif variable in ["mach", "m"]:
            M = np.abs(state.velocity) / domain.physics.get_sound_speed(state)
            self.variable.append(np.interp(self.x, self.x, M[idx_cells]))
        else:
            msg = f"Invalid Variable Name: {variable}"
            raise Exception(msg)
        self.t.append(domain.t)

    def save_to_csv(self, output_dir="xt_output"):
        """
        Save the variable data to a timestamped CSV file.
        Each row is one time snapshot, with the first column being time (ms) and
        the rest being the values at each x location.
        """
        if not Path.exists():
            Path.mkdir(parents=True)
        filename = f"{self.name}.csv"
        filepath = Path(output_dir / filename)

        # First row is the header: Time, x0, x1, ..., xN
        header = [0] + [f"{xi:.6f}" for xi in self.x]

        with Path.open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t_val, profile in zip(self.t, self.variable, strict=False):
                row = [f"{t_val * 1e3:.6f}"] + [f"{v:.6e}" for v in profile]
                writer.writerow(row)

        print(f"Saved {self.name} data to: {filepath}")

    def plot(self, figdir="."):
        """
        This method creates a contour plot of the XTDiagram data
            inputs:
                figdir = directory in which to save the plot
        """
        plt.figure()
        t = [t * 1000.0 for t in self.t]
        X, T = np.meshgrid(self.x, t)
        variableMatrix = np.zeros(X.shape)
        for k, variablek in enumerate(self.variable):
            variableMatrix[k, :] = variablek
        variable = self.name
        if variable in ["density", "r", "rho"]:
            plt.title(r"$\rho~[\mathrm{kg/m^3}]$")
        elif variable in ["velocity", "u"]:
            plt.title(r"$u~[\mathrm{m/s}]$")
        elif variable in ["pressure", "p"]:
            variableMatrix /= 1.0e5  # convert to bar
            plt.title(r"$p~[\mathrm{bar}]$")
        elif variable in ["temperature", "t"]:
            plt.title(r"$T~[\mathrm{K}]$")
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            plt.title(r"$\gamma~[\mathrm{-}]$")
        elif variable in ["mixture fraction"]:
            plt.title(r"$Z~[\mathrm{-}]$")
        elif variable in ["progress variable"]:
            plt.title(r"$C~[\mathrm{-}]$")
        elif variable in ["mach", "m"]:
            plt.title(r"$M~[\mathrm{-}]$")
        else:
            plt.title(r"$\mathrm{" + variable + "}$")

        if self.limits is None:
            plt.pcolormesh(X, T, variableMatrix, cmap="jet")
        else:
            plt.pcolormesh(
                X,
                T,
                variableMatrix,
                cmap="jet",
                vmin=self.limits[0],
                vmax=self.limits[1],
            )
        plt.xlabel(r"$x~[\mathrm{m}]$")
        plt.ylabel(r"$t~[\mathrm{ms}]$")
        plt.axis([min(self.x), max(self.x), min(t), max(t)])
        plt.colorbar()
        plt.savefig(Path(figdir) / f"{variable}.png", bbox_inches="tight", dpi=300)


def add_h_plot(domain, ax, scale=1.0):
    ax1 = ax.twinx()
    ax1.set_zorder(-np.inf)
    ax.patch.set_visible(False)

    geometry = domain.geometry
    x = geometry.x
    h = geometry.h if geometry.h is not None else geometry.d_outer(x)
    ax1.plot(x * scale, h * scale, color="0.8", linestyle="--")
    ax1.axhline(0, color="0.8", linestyle="--")
    ax1.set_aspect("equal")
    ax1.set_ylabel("h [mm]")
    return ax1


def get_state_plot_bounds(domain):
    rlims = [0, 0]
    ulims = [0, 0]
    plims = [0, 0]
    Tlims = [0, 0]
    Mlims = [0, 0]

    g = 1.4

    for ibc in [0, 1]:
        if type(domain.boundary_conditions[ibc]) is not str:
            if domain.boundary_conditions[ibc][0] is not None:
                rlims[ibc] = domain.boundary_conditions[ibc][0]
            if domain.boundary_conditions[ibc][1] is not None:
                ulims[ibc] = domain.boundary_conditions[ibc][1]
            if domain.boundary_conditions[ibc][2] is not None:
                plims[ibc] = domain.boundary_conditions[ibc][2]

    p1 = plims[0]
    r1 = rlims[0]

    rlims = np.max(rlims)
    ulims = np.max(ulims)
    plims = np.max(plims)

    a = domain.physics.get_sound_speed(domain.state)
    T = domain.physics.get_temperature(domain.state)
    T1 = T[0]
    cp = domain.physics.get_cp(domain.state)
    cp_max = max([cp[0], cp[-1]])

    a_arr = [a[0], a[-1]]
    Mlims = np.max(ulims / a_arr)

    p_rat = 1 + ((2 * g) / (g + 1)) * Mlims**2 - 1
    r_rat = (1 + ((g + 1) / (g - 1)) * p_rat) / (((g + 1) / (g - 1)) + p_rat)
    T_rat = p_rat / r_rat

    u2 = np.sqrt(2 * cp_max * T1 * T_rat)
    k = 2.0
    rlims = [0, r_rat * k]
    ulims = [0, np.ceil(u2)]
    plims = [0, np.ceil(p_rat)]
    Tlims = [0, np.ceil(T_rat)]
    Mlims = [0, 5]
    return T1, p1, r1, Mlims, plims, Tlims, ulims, rlims


def plot_state(domain, filename, limits):
    T1, p1, r1, Mlims, plims, Tlims, ulims, rlims = (
        limits[0],
        limits[1],
        limits[2],
        limits[3],
        limits[4],
        limits[5],
        limits[6],
        limits[7],
    )
    xscale = 1.0e3
    physics = domain.physics
    state = domain.state
    geometry = domain.geometry
    idx_cells = domain.idx_cells
    T = physics.get_temperature(state)

    subtitle_str = []
    if plims != [0, 0]:
        subtitle_str.append(f"$P_1 = {p1:.0f}$ Pa")
    if Tlims != [0, 0]:
        subtitle_str.append(f"$T_1 = {T1:.0f}$ K")
    if rlims != [0, 0]:
        subtitle_str.append(f"$\\rho_1 = {r1:.2f}$ [kg/m$^3$]")
    subtitle = ", ".join(subtitle_str)

    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(6, 9))
    ax[0].plot(geometry.x * xscale, state.density[idx_cells])
    ax[0].set_ymargin(0.1)
    ax[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    if geometry.h is not None:
        add_h_plot(domain, ax[0], scale=xscale)
    if rlims == [0, 0]:
        ax[0].set_ylim([0, 5])
    else:
        ax[0].set_ylim(rlims)

    ax[1].plot(geometry.x * xscale, state.velocity[idx_cells])
    ax[1].set_ymargin(0.1)
    if ulims == [0, 0]:
        ax[1].set_ylim([0, 800])
    else:
        ax[1].set_ylim(ulims)
    ax[1].set_xlabel("x [cm]")
    ax[1].set_ylabel(r"$u$ [m/s]")
    if geometry.h is not None:
        add_h_plot(domain, ax[1], scale=xscale)

    ax[2].set_ymargin(0.1)
    if plims == [0, 0]:
        ax[2].plot(geometry.x * xscale, state.pressure[idx_cells])
        ax[2].set_ylim([0, 250000])
        ax[2].set_ylabel(r"$P$ [Pa]")
    else:
        ax[2].plot(geometry.x * xscale, state.pressure[idx_cells] / p1)
        ax[2].set_ylim(plims)
        ax[2].set_ylabel(r"$p / p_1$")
    if geometry.h is not None:
        add_h_plot(domain, ax[2], scale=xscale)

    ax[3].plot(geometry.x * xscale, T[idx_cells])
    ax[3].set_ymargin(0.1)
    ax[3].set_ylabel(r"$T$ [K]")
    if geometry.h is not None:
        add_h_plot(domain, ax[3], scale=xscale)

    M = np.abs(state.velocity) / physics.get_sound_speed(state)
    ax[4].plot(geometry.x * xscale, M[idx_cells])
    ax[4].axhline(1.0, color="r", linestyle="--")
    ax[4].set_ymargin(0.1)
    ax[4].set_ylabel(r"$M$ [-]")
    if Mlims == [0, 0]:
        ax[4].set_ylim([0, 5])
    else:
        ax[4].set_ylim(Mlims)
    if geometry.h is not None:
        add_h_plot(domain, ax[4], scale=xscale)

    if physics.is_flamelet:
        state = physics.set_state(state)
        Y_H2 = physics.lookup("H2", state)[idx_cells]
        Y_OH = physics.lookup("OH", state)[idx_cells]
        Y_H2O = physics.lookup("H2O", state)[idx_cells]
    else:
        Y = state.mass_fractions[idx_cells]
        Y_H2 = Y[:, physics.gas.species_index("H2")]
        Y_OH = Y[:, physics.gas.species_index("OH")]
        Y_H2O = Y[:, physics.gas.species_index("H2O")]
    ax[5].plot(geometry.x * xscale, Y_H2, label=r"$\mathrm{H}_2$")
    ax[5].plot(geometry.x * xscale, Y_OH, label=r"$\mathrm{OH}$")
    ax[5].plot(geometry.x * xscale, Y_H2O, label=r"$\mathrm{H}_2\mathrm{O}$")
    if Y_H2.max() < 1e-6:
        ax[5].set_ylim(-1e-3, 1e-3)
    else:
        ax[5].set_ymargin(0.1)
    ax[5].set_ylabel(r"$Y_k$ [-]")
    ax[5].legend(loc="upper right")
    if geometry.h is not None:
        add_h_plot(domain, ax[5], scale=xscale)

    ax[6].scatter(
        domain.injector.fluid_tips[:, 0] * xscale,
        domain.injector.fluid_tips[:, 1] * 1e3 * domain.injector.n_inj,
        s=1,
    )
    ax[6].set_ymargin(0.1)
    ax[6].set_ylabel(r"$\dot{m}_f$ [g/s]")
    if geometry.h is not None:
        add_h_plot(domain, ax[6], scale=xscale)

    ax[6].set_xlabel("x [mm]")
    full_title = (
        r"$t = %.4f$ ms" % (domain.t * 1e3) + "\n" + r"\footnotesize{" + subtitle + "}"
    )
    fig.suptitle(full_title)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


class SnapshotDiagram:
    """
    This class stores and plots snapshots of a given variable over the spatial domain at selected time intervals.

    Inputs:
        domain: simulation domain with geometry and state
        variable: string specifying the variable to plot (e.g., "pressure", "temperature")
        skipSteps: number of iterations between updates
        x: mesh for interpolation (defaults to domain mesh)
    """

    def __init__(self, domain, variable, skipSteps=0, x=None):
        self.name = variable.lower()
        self.skipSteps = skipSteps
        self.x = x if x is not None else domain.geometry.x

        self.snapshots = []  # List of (time, interpolated variable array)
        self.counter = 0

        self.update(domain)

    def update(self, domain):
        if self.counter % self.skipSteps != 0:
            self.counter += 1
            return

        state = domain.state
        geometry = domain.geometry
        physics = domain.physics
        x = self.x
        variable = self.name

        if variable in ["density", "r", "rho"]:
            y = np.interp(x, geometry.x, state.density)
        elif variable in ["velocity", "u"]:
            y = np.interp(x, geometry.x, state.velocity)
        elif variable in ["pressure", "p"]:
            y = np.interp(x, geometry.x, state.pressure / 1e5)  # Convert to bar
        elif variable in ["temperature", "t"]:
            T = physics.get_temperature(state)
            y = np.interp(x, geometry.x, T)
        elif variable in ["gamma", "g"]:
            y = np.interp(x, geometry.x, state.gamma)
        elif variable in ["mach", "m"]:
            M = np.abs(state.velocity) / physics.get_sound_speed(state)
            y = np.interp(x, geometry.x, M)
        elif variable in physics.scalar_names:
            i = physics.scalar_names.index(variable)
            y = np.interp(x, geometry.x, state.composition[:, i])
        else:
            msg: str = f"Invalid Variable Name: {variable}"
            raise Exception(msg)

        self.snapshots.append((domain.t, y))
        self.counter += 1

    def plot(self, figdir="."):
        """
        Plot snapshots over the domain at each sampled time
        """
        plt.figure()
        for t, y in self.snapshots:
            plt.plot(self.x, y, label=f"{t * 1e3:.2f} ms")

        plt.xlabel("x [m]")
        var_label = {
            "pressure": "p [bar]",
            "temperature": "T [K]",
            "density": r"$\rho$ [kg/m³]",
            "velocity": "u [m/s]",
            "gamma": r"$\gamma$",
            "mach": "M",
        }.get(self.name, self.name)
        plt.ylabel(var_label)
        plt.title(f"{var_label} snapshots")
        plt.legend(loc="best", fontsize="small", ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(figdir) / f"{self.name}_snapshots.png", dpi=300)
        plt.close()
