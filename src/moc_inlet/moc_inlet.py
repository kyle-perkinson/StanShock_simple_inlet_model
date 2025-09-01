import numpy as np

from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.geo_reader import geo_reader
from moc_inlet.utils.geometry import Geometry, Inflections
from moc_inlet.utils.domain import Domain, Region, SolutionSlice, build_regions
from moc_inlet.utils.segment import Segment, SegmentList, Farfield, WallSegment, Wave
from moc_inlet.utils.flow_solvers import incident_wave, reflected_wave, riemann_problem
import matplotlib.pyplot as plt

def moc_inlet(geometry_filepath: str, T: float, P: float, M: float, theta: float, fluid: str, N_wave: int):
    body1, body2 = geo_reader(geometry_filepath)
    geom = Geometry(body1, body2)
    freestream = FlowState(fluid)
    freestream.set_state(T=T, P=P, M=M, theta=theta)

    x0, x_end, domain = initialize(geom, freestream)
    x_i = x0
    while x_i <= x_end:
        x_next, events = find_x_next(domain, geom, x_i)
        domain = moc_solver(x_next, x_i, events, domain, geom, N_wave)
        x_i = x_next


def find_x_next(domain: Domain, geom: Geometry, x_i: float, tol=1e-12):
    slice_x_i = domain.get_slice(x_i)
    segments_x_i = slice_x_i.as_segmentlist()
    crossings = segments_x_i.get_crossings(x_i)

    x_intersections, pairs = [], []
    for i, seg1 in enumerate(crossings.segments):
        for j, seg2 in enumerate(crossings.segments):
            if j <= i:
                continue
            pt = get_intersection(seg1, seg2, tol)
            if pt is None:
                continue
            xj, yj = pt
            if xj > x_i + tol:
                x_intersections.append(xj)
                pairs.append((seg1, seg2))

    # --- inflections ---
    next_infl = geom.inflections.x[geom.inflections.x > x_i + tol]
    x_infl = None
    infl_objects = []
    if len(next_infl) > 0:
        x_infl = np.min(next_infl)
        infl_indices = np.where(np.isclose(geom.inflections.x, x_infl, atol=tol))[0]
        infl_objects = [geom.inflections[i] for i in infl_indices]

    # --- candidates ---
    candidates = []
    if x_infl is not None:
        candidates.append(x_infl)
    candidates.extend(x_intersections)

    if not candidates:
        return None, []

    x_next = min(candidates)

    results = []
    if x_infl is not None and np.isclose(x_next, x_infl, atol=tol):
        results.extend(infl_objects)
    for xj, pair in zip(x_intersections, pairs):
        if np.isclose(xj, x_next, atol=tol):
            results.append(pair)
    return x_next, results






def initialize(geom: Geometry, freestream: FlowState):
    domain = Domain()
    x0 = geom.x0
    x_end = geom.x_end
    geom.upper_bbox.state = freestream
    geom.lower_bbox.state = freestream
    inflow = Region(geom.upper_bbox, geom.lower_bbox, freestream)
    start_slice = SolutionSlice(x0, inflow)
    domain.add_slice(start_slice)
    return x0, x_end, domain

def moc_solver(x_next: float, x_i: float, events: list, domain: Domain, geom: Geometry, N_wave: int):
    """
    Classify events at x_i and redirect them to the correct solver.
    """
    slice_x_i = domain.get_slice(x_i)
    segments_x_i = slice_x_i.as_segmentlist()

    out_segments = []
    if not events:
        return slice_x_i
    
    for event in events:
        if isinstance(event, Inflections):
            inflow = slice_x_i.get_flowstate(x_next, event.y)
            incident_waves = incident_wave(x_next, event, inflow, N_wave)
            if incident_waves:
                out_segments.extend(incident_waves)
        # Case 2: Wave + Wall interaction
        elif isinstance(event, tuple) and len(event) == 2:
            seg1, seg2 = event
            if isinstance(seg1, Wave) and isinstance(seg2, WallSegment):
                reflected_waves = reflected_wave(x_next, seg1, seg2)
            elif isinstance(seg2, Wave) and isinstance(seg1, WallSegment):
                reflected_waves = reflected_wave(x_next, seg2, seg1)
            if reflected_waves:
                out_segments.extend(reflected_waves)

            elif isinstance(seg1, Wave) and isinstance(seg2, Wave):
                refracted_waves = riemann_problem(x_next, seg1, seg2)
                if refracted_waves:
                    out_segments.extend(refracted_waves)

            elif isinstance(seg1, WallSegment) and isinstance(seg2, WallSegment):
                raise RuntimeError("Self-intersecting geometry not permitted. Please revise your input geometry.")
            

    for seg in segments_x_i.segments:
        event_segs = [s for e in events for s in (e if isinstance(e, tuple) else [])]
        if seg not in event_segs:
            if not isinstance(seg, WallSegment):
                out_segments.append(seg)
        else:
            if isinstance(seg, Wave) and seg.x_end == np.inf:
                propagate_termination(domain, seg, x_next)

    
    active_inflections = [e for e in events if isinstance(e, Inflections)]
    wall_segments = geom.get_walls(x_next, active_inflections=active_inflections)
    out_segments.extend(wall_segments)
    segments_x_next = SegmentList(x_next, out_segments)
    plot_segments(segments_x_next)
    slice_x_next = build_regions(segments_x_next, x_next, geom)
    domain.add_slice(slice_x_next)
    return domain





def plot_segments(seglist, ax=None, xlim=None, ylim=None):
    """
    Plot all segments in a SegmentList.
    Handles infinite endpoints by clipping to plot bounds.
    """
    if ax is None:
        fig, ax = plt.subplots()

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




def get_intersection(seg1: Segment, seg2: Segment, tol=1e-12):
    (x1,y1),(dx1,dy1) = seg1._get_line_params()
    (x2,y2),(dx2,dy2) = seg2._get_line_params()

    denom = dx1*dy2 - dy1*dx2
    if np.isclose(denom, 0.0, atol=tol):
        return None  # parallel or coincident

    t1 = ((x2-x1)*dy2 - (y2-y1)*dx2) / denom
    t2 = ((x2-x1)*dy1 - (y2-y1)*dx1) / denom

    px, py = x1 + t1*dx1, y1 + t1*dy1

    def valid(seg, t):
        if np.isinf(seg.x_end):
            return t >= -tol  # ray
        else:
            return -tol <= t <= 1+tol

    if not valid(seg1, t1) or not valid(seg2, t2):
        return None
    return px, py


def propagate_termination(domain: Domain, wave: Wave, x_end: float):
    """
    Ensures that all SolutionSlices referencing this wave reflect termination.
    """
    for slice in domain.slices:
        slice_segments = slice.as_segmentlist().segments
        if wave in slice_segments:
            wave.terminate(x_end)