from __future__ import annotations

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


def plot_state(domain, filename):
    xscale = 1.0e3
    physics = domain.physics
    state = domain.state
    geometry = domain.geometry
    idx_cells = domain.idx_cells
    T = physics.get_temperature(state)

    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(6, 9))
    ax[0].plot(geometry.x * xscale, state.density[idx_cells])
    ax[0].set_ymargin(0.1)
    ax[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    if geometry.h is not None:
        add_h_plot(domain, ax[0], scale=xscale)

    ax[1].plot(geometry.x * xscale, state.velocity[idx_cells])
    ax[1].set_ymargin(0.1)
    ax[1].set_ylabel(r"$u$ [m/s]")
    if geometry.h is not None:
        add_h_plot(domain, ax[1], scale=xscale)

    ax[2].plot(geometry.x * xscale, state.pressure[idx_cells])
    ax[2].set_ymargin(0.1)
    ax[2].set_ylabel(r"$p$ [Pa]")
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

    fig.suptitle(rf"$t = {domain.t * 1.0e3:.4f}$ ms")

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
