import numpy as np

from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.geo_reader import geo_reader
from moc_inlet.utils.geometry import Geometry, Inflections
from moc_inlet.utils.domain import Domain, Region, SolutionSlice, build_regions
from moc_inlet.utils.segment import SegmentList, Farfield, WallSegment, Wave
from moc_inlet.utils.flow_solvers import incident_wave, reflected_wave


def moc_inlet(geometry_filepath: str, T: float, P: float, M: float, theta: float, fluid: str, N_wave: int):
    body1, body2 = geo_reader(geometry_filepath)
    geom = Geometry(body1, body2)
    freestream = FlowState(fluid)
    freestream.set_state(T=T, P=P, M=M, theta=theta)

    x0, x_end, domain = initialize(geom, freestream)
    x_i = x0
    while x_i <= x_end:
        x_next, events, segments_x_i, slice_x_i = find_x_next(domain, geom, x_i)
        slice_x_next = moc_solver(x_next, events, segments_x_i, slice_x_i, geom, N_wave)
        domain.add_slice(slice_x_next)
        x_i = x_next


def find_x_next(domain: Domain, geom: Geometry, x_i: float, tol: float = 1e-10):
    slice_x_i = domain.get_slice(x_i)
    segments_x_i = slice_x_i.as_segmentlist()
    plot_segments(segments_x_i)
    crossings = segments_x_i.get_crossings(x_i)
    y = crossings.y
    sigmas = crossings.sigma
    paths = crossings.segments

    next_inflections = geom.inflections.x[geom.inflections.x > x_i + tol]
    x_infl = None
    infl_objects = []
    if len(next_inflections) > 0:
        x_infl = np.min(next_inflections)
        infl_indices = np.where(np.isclose(geom.inflections.x, x_infl, atol=tol))[0]
        infl_objects = [geom.inflections[i] for i in infl_indices]

    x_intersections = []
    intersection_pairs = []

    for j in range(len(crossings) - 1):
            seg1, seg2 = paths[j], paths[j + 1]
            y1, y2 = y[j], y[j + 1]

            # skip segments at same y (e.g., PM fan)
            if np.isclose(y1, y2, atol=tol):
                continue

            tan1 = np.tan(sigmas[j]) if not np.isclose(np.cos(sigmas[j]), 0.0) else np.inf
            tan2 = np.tan(sigmas[j + 1]) if not np.isclose(np.cos(sigmas[j + 1]), 0.0) else np.inf

            denom = tan1 - tan2
            if np.isclose(denom, 0.0):
                continue  # parallel or vertical segments

            x_next_j = x_i + (y2 - y1) / denom
            if x_next_j > x_i + tol:
                x_intersections.append(x_next_j)
                intersection_pairs.append((seg1, seg2))


    x_candidates = []
    if x_infl is not None:
        x_candidates.append(x_infl)
    x_candidates.extend(x_intersections)

    if not x_candidates:
        return None, None, segments_x_i, slice_x_i

    x_next = min(x_candidates)
    results = []

    if x_infl is not None and np.isclose(x_next, x_infl, atol=tol):
        results.extend(infl_objects)

    for x_j, pair in zip(x_intersections, intersection_pairs):
        if np.isclose(x_j, x_next, atol=tol):
            results.append(pair)

    return x_next, results, segments_x_i, slice_x_i

def initialize(geom: Geometry, freestream: FlowState):
    domain = Domain()
    x0 = geom.x0
    x_end = geom.x_end
    inflow = Region(geom.upper_bbox, geom.lower_bbox, freestream)
    start_slice = SolutionSlice(x0, inflow)
    domain.add_slice(start_slice)
    return x0, x_end, domain

def moc_solver(x_next: float, events: list, segments_x_i, slice_x_i: SolutionSlice, geom: Geometry, N_wave: int):
    """
    Classify events at x_i and redirect them to the correct solver.
    """
    out_segments = []
    if not events:
        return slice_x_i
    
    for event in events:
        if isinstance(event, Inflections):
            inflow = slice_x_i.get_flowstate(x_next, event.y)
            incident_waves = incident_wave(x_next, event, inflow, N_wave)
            if incident_waves:
                out_segments.extend(incident_waves)
            #Need to switch segments (wall segment should still be defined if there's no inflection... should be retained. 
            #If it's the first inflection, 

        # Case 2: Wave + Wall interaction
        elif isinstance(event, tuple) and len(event) == 2:
            seg1, seg2 = event
            y_next = seg1.y_at(x_next)


            if isinstance(seg1, Wave) and isinstance(seg2, WallSegment):
                reflected_waves = reflected_wave(x_next, seg1, seg2)
            elif isinstance(seg2, Wave) and isinstance(seg1, WallSegment):
                reflected_waves = reflected_wave(x_next, seg2, seg1)
            if reflected_waves:
                out_segments.extend(reflected_waves)

            # elif isinstance(seg1, Wave) and isinstance(seg2, Wave):
            #     RiemannProblem(x_i, event, current)

            # Case 4: Wall + Wall interaction (error)
            elif isinstance(seg1, WallSegment) and isinstance(seg2, WallSegment):
                raise RuntimeError("Self-intersecting geometry not permitted. Please revise your input geometry.")
    for seg in segments_x_i.segments:
        if seg not in [s for e in events for s in (e if isinstance(e, tuple) else [])]:
            if not isinstance(seg, WallSegment):  # walls are handled separately
                out_segments.append(seg)
    
    active_inflections = [e for e in events if isinstance(e, Inflections)]
    wall_segments = geom.get_walls(x_next, active_inflections=active_inflections)
    out_segments.extend(wall_segments)
    segments_x_next = SegmentList(x_next, out_segments)
    return build_regions(segments_x_next, x_next, geom)



import matplotlib.pyplot as plt

def plot_segments(seglist, ax=None, xlim=None, ylim=None):
    """
    Plot all segments in a SegmentList.
    Handles infinite endpoints by clipping to plot bounds.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # auto domain if not provided
    if xlim is None:
        xs = [s.x_start for s in seglist.segments if np.isfinite(s.x_start)]
        xs += [s.x_end for s in seglist.segments if np.isfinite(s.x_end)]
        if xs:
            xmin, xmax = min(xs), max(xs)
        else:
            xmin, xmax = 0, 1
        xlim = (xmin, xmax)

    if ylim is None:
        ys = [s.y_start for s in seglist.segments if np.isfinite(s.y_start)]
        ys += [s.y_end for s in seglist.segments if np.isfinite(s.y_end)]
        if ys:
            ymin, ymax = min(ys), max(ys)
        else:
            ymin, ymax = 0, 1
        ylim = (ymin, ymax)

    for seg in seglist.segments:
        x0, y0 = seg.x_start, seg.y_start
        x1, y1 = seg.x_end, seg.y_end

        # handle infinities by clipping to plotting window
        if not np.isfinite(x1) or not np.isfinite(y1):
            # direction vector
            dx = np.cos(seg.sigma)
            dy = np.sin(seg.sigma)

            # extend to right boundary of xlim
            x1 = xlim[1]
            y1 = y0 + (x1 - x0) * (dy / dx if dx != 0 else np.sign(dy) * 1e6)
        if hasattr(seg, "wave_type"):
            if seg.wave_type == "shock":
                color = "red"
            elif seg.wave_type == "expansion":
                color = "blue"
            else:
                color = "black"
        elif seg.__class__.__name__.lower().startswith("wall"):
            color = "black"
        else:
            color = "black"

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1)


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    return ax