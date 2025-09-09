import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.collections import PolyCollection

from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.geometry import Geometry
from moc_inlet.utils.domain import Domain, Region, SolutionSlice

@dataclass(frozen=True)
class PlotVariable:
    name: str
    ylabel: str
    ylims: tuple[float, float]
    yticks: np.ndarray
    color: str

class PlotSettings:
    p = PlotVariable(
        name="p",
        ylabel="P [Pa]",
        ylims=(0, 500000),
        yticks=np.arange(0, 500001, 1e5),
        color="black"
    )
    T = PlotVariable(
        name="T",
        ylabel="T [K]",
        ylims=(0, 500),
        yticks=np.arange(0, 501, 1e2),
        color="red"
    )
    r = PlotVariable(
        name="r",
        ylabel=r"$\rho$ [kg/m^3]",
        ylims=(0, 5),
        yticks=np.arange(0, 5.01, 1),
        color="green"
    )
    M = PlotVariable(
        name="M",
        ylabel="Mach",
        ylims=(0, 5),
        yticks=np.arange(0, 5.01, 1),
        color="blue"
    )
    pn = PlotVariable(
        name="pn",
        ylabel=r"$P / P_{amb}$",
        ylims=(0, 1),
        yticks=np.arange(0, 1.01, 0.2),
        color="black"
    )
    Tn = PlotVariable(
        name="Tn",
        ylabel=r"$T / T_{amb}$",
        ylims=(0, 1),
        yticks=np.arange(0, 1.01, 0.2),
        color="red"
    )
    rn = PlotVariable(
        name="rn",
        ylabel=r"$\rho / \rho_{amb}$",
        ylims=(0, 1),
        yticks=np.arange(0, 1.01, 0.2),
        color="green"
    )

    registry = {v.name: v for v in [p, T, r, M, pn, Tn, rn]}

    @classmethod
    def get(cls, key: str) -> PlotVariable:
        return cls.registry[key]

def plot_solution(domain: Domain, geom: Geometry, variable, freestream: FlowState, fig=None,
                  normalize=True):
    cmap = 'Blues'
    settings = PlotSettings.get(variable)
    polygons = []
    values = []
    points = []



    for i in range(len(domain.slices) - 1):
        slice_i = domain.slices[i]
        slice_ip1 = domain.slices[i+1]
        for region in slice_i:
            poly = region_polygon(region, slice_i, slice_ip1)
            polygons.append(poly)
            points.extend(poly)
            if region.flowstate is None:
                val = np.nan
            else:
                fs = region.flowstate
                if not hasattr(fs, variable):
                    raise AttributeError(f"FlowState has no '{variable}'")
                val = getattr(fs, variable)
                if normalize and variable != "M":
                    val /= getattr(freestream, variable)
            values.append(val)

    if fig is None:
        fig, (ax, cax) = plt.subplots(
            ncols=2, figsize=(12, 3),
            gridspec_kw={"width_ratios": [5, 0.05]}
        )
    else:
        ax, cax = fig.axes

    coll = PolyCollection(polygons, array=np.array(values), cmap=cmap)
    coll.set_clim(*settings.ylims)
    ax.add_collection(coll)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.fill(geom.body1.points[:,0],geom.body1.points[:,1],facecolor='lightgray',edgecolor='k',linewidth=1.5)
    ax.fill(geom.body2.points[:,0],geom.body2.points[:,1],facecolor='lightgray',edgecolor='k',linewidth=1.5)
    ax.set_xlim([geom.x0, max(geom.body1.points[:,0])])
    # ax.set_ylim([-0.02, 0.07])


    cbar = plt.colorbar(coll, cax=cax, ticks=settings.yticks)
    cbar.set_label(settings.ylabel)
    plt.tight_layout()
    return fig

def region_polygon(region: Region, slice_i: SolutionSlice, slice_ip1: SolutionSlice):

    x_i, x_ip1 = slice_i.x, slice_ip1.x

    y_low_i = region.lower.y_at(x_i)
    y_high_i = region.upper.y_at(x_i)

    reg_ip1 = None
    for r in slice_ip1:
        if r.lower is region.lower and r.upper is region.upper:
            reg_ip1 = r
            break

    if reg_ip1 is None:
        y_low_ip1 = region.lower.y_at(x_ip1)
        y_high_ip1 = region.upper.y_at(x_ip1)
    else:
        y_low_ip1 = reg_ip1.lower.y_at(x_ip1)
        y_high_ip1 = reg_ip1.upper.y_at(x_ip1)

    if y_low_i > y_high_i:
        y_low_i, y_high_i = y_high_i, y_low_i
    if y_low_ip1 > y_high_ip1:
        y_low_ip1, y_high_ip1 = y_high_ip1, y_low_ip1

    coords = np.array([
        [x_i,   y_low_i],
        [x_ip1, y_low_ip1],
        [x_ip1, y_high_ip1],
        [x_i,   y_high_i],
    ])
    coords = sort_coordinates(coords)
    return coords


def sort_coordinates(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(-angles)
    return list_of_xy_coords[indices]