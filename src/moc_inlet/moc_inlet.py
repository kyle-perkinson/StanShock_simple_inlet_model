import numpy as np

from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.geo_reader import geo_reader
from moc_inlet.utils.geometry import Geometry, Inflection
from moc_inlet.utils.domain import Domain, Region, SolutionSlice, build_regions
from moc_inlet.utils.segment import Segment, SegmentList,WallSegment, Wave
from moc_inlet.utils.flow_solvers import incident_wave, reflected_wave, riemann_problem
import matplotlib.pyplot as plt
from moc_inlet.utils.solution_colorbar import plot_solution


def moc_inlet(geometry_filepath: str, T: float, P: float, M: float, theta: float, fluid: str, wave_res: float):
    # wave_res = np.radians(wave_res)
    body1, body2 = geo_reader(geometry_filepath)
    geom = Geometry(body1, body2)
    freestream = FlowState(fluid)
    freestream.set_state(T=T, P=P, M=M, theta=theta)

    x0, x_end, domain = initialize(geom, freestream)
    x_i = x0

    while x_i <= x_end:        
        x_next, events = find_x_next(domain, geom, x_i)



        domain = moc_solver(x_next, x_i, events, domain, geom, wave_res)

        fig = plot_solution(domain, geom, 'M', freestream, None, True)
        fig = plot_segments(domain, geom, fig)
        ax, cax = fig.axes
        ax.axvline(x_next, color='k', linestyle=':', linewidth=2)
        if x_i < x_end:
            plt.close(fig)
        x_i = x_next


def find_x_next(domain: Domain, geom: Geometry, x_i: float, tol=1e-12):
    slice_x_i = domain.get_slice(x_i)
    segments_x_i = slice_x_i.as_segmentlist()
    downstream_geo = geom.get_downstream_geo(x_i)

    for seg in downstream_geo:
        if seg not in segments_x_i.segments:
            segments_x_i.segments.append(seg)
    crossings = segments_x_i.get_crossings(x_i)

    candidates = []

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
    candidates.extend(x_intersections)

    x_infl, inflections = geom.get_next_inflection(x_i, tol)

    
    if x_infl is not None:
        candidates.append(x_infl)
    
    if not candidates:
        return None, []
    x_next = min(candidates)
    results = []
    if x_infl is not None and np.isclose(x_next, x_infl, atol=tol):
        results.extend(inflections)
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



def moc_solver(x_next: float, x_i: float, events: list, domain: Domain, geom: Geometry, wave_res: int):
    """
    Classify events at x_i and redirect them to the correct solver.
    """
    slice_x_i = domain.get_slice(x_i)
    segments_x_i = slice_x_i.as_segmentlist()

    out_segments = []
    if not events:
        print("No events were inputted!")
        return slice_x_i
    
    for event in events:
        if isinstance(event, Inflection):
            seg = event.segment
            y_next = seg.y_at(x_next)
            inflow = slice_x_i.get_flowstate(x_next, y_next)
            incident_waves = incident_wave(x_next, seg, inflow, wave_res)
            if incident_waves:
                out_segments.extend(incident_waves)
        # Case 2: Wave + Wall interaction
        elif isinstance(event, tuple) and len(event) == 2:
            seg1, seg2 = event
            if isinstance(seg1, Wave) and isinstance(seg2, WallSegment):
                reflected_waves = reflected_wave(x_next, seg1, seg2)
                if reflected_waves:
                    out_segments.extend(reflected_waves)
            elif isinstance(seg2, Wave) and isinstance(seg1, WallSegment):
                reflected_waves = reflected_wave(x_next, seg2, seg1)
                if reflected_waves:
                    out_segments.extend(reflected_waves)

            elif isinstance(seg1, Wave) and isinstance(seg2, Wave):
                refracted_waves = riemann_problem(x_next, seg1, seg2, wave_res)
                if refracted_waves:
                    out_segments.extend(refracted_waves)

            elif isinstance(seg1, WallSegment) and isinstance(seg2, WallSegment):
                raise RuntimeError("Self-intersecting geometry not permitted. Please revise your input geometry.")
            
    event_segs = [s for e in events for s in (e if isinstance(e, tuple) else [])]
    for seg in segments_x_i.segments:
        
        if seg not in event_segs:
            if not isinstance(seg, WallSegment):
                out_segments.append(seg)
        else:
            if isinstance(seg, Wave) and seg.x_end == np.inf:
                domain = propagate_termination(domain, seg, x_next)

    
    active_inflections = [e for e in events if isinstance(e, Inflection)]
    wall_segments = geom.get_walls(x_next, active_inflections=active_inflections)
    out_segments.extend(wall_segments)
    segments_x_next = SegmentList(x_next, out_segments)
    # plot_segments(segments_x_next)
    slice_x_next = build_regions(domain, out_segments, x_next, geom, tol=1e-12) 
    domain.add_slice(slice_x_next)
    return domain



def plot_segments(domain: Domain, geom: Geometry, fig):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax, cax = fig.axes

    xlim = (geom.x0, geom.x_end)
    ylim = (0, 0.065)

    # main loop: plot segments
    for sl in domain.slices:
        seglist = sl.as_segmentlist()

        # vertical line at slice x
        ax.axvline(sl.x, color="gray", linestyle="--", linewidth=0.5)

        for seg in seglist.segments:
            x0, y0 = seg.x_start, seg.y_start
            x1, y1 = seg.x_end, seg.y_end

            if not np.isfinite(x1) or not np.isfinite(y1):
                dx, dy = np.cos(seg.sigma), np.sin(seg.sigma)
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

            # plot segment
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=1)

            # plot endpoints
            ax.plot(x0, y0, "ko", markersize=2)
            ax.plot(x1, y1, "ko", markersize=2)
    return fig




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
    return domain