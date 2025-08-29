from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.segment import SegmentList, Segment, WallSegment, Wave, Slipstream, Farfield
from moc_inlet.utils.geometry import Geometry, Inflections
import numpy as np
from itertools import groupby

class Domain:
    def __init__(self):
        self.slices = []
    def add_slice(self, new_slice: "SolutionSlice"):
        self.slices.append(new_slice)
    def get_slice(self, x):
        for sl in self.slices:
            if np.isclose(sl.x, x, atol=1e-12):
                return sl
        return None


class Region:
    def __init__(self, lower_seg, upper_seg, flowstate: FlowState):
        self.lower = lower_seg
        self.upper = upper_seg
        self.flowstate = flowstate  

class SolutionSlice:
    def __init__(self, x, regions):
        if isinstance(regions, Region):
            regions = [regions]
        self.x = x
        self.regions = regions

    def __iter__(self):
        """Allow iteration directly over regions."""
        return iter(self.regions)

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        return self.regions[idx]

    def as_segmentlist(self) -> SegmentList:
        segs = []
        for region in self.regions:
            segs.append(region.lower)
            segs.append(region.upper)
        unique_segs = list({id(s): s for s in segs}.values())
        return SegmentList(self.x, unique_segs)
    
    def get_flowstate(self, x_next: float, y_next: float):
        """
        Return the FlowState at a given x_next and y_next within this slice.
        """
        for region in self.regions:
            y_lower = y_at(region.lower, x_next)
            y_upper = y_at(region.upper, x_next)

            if y_lower > y_upper:
                y_lower, y_upper = y_upper, y_lower

            if y_lower <= y_next <= y_upper:
                return region.flowstate
        return None




def build_regions(seglist: SegmentList, x_i: float, geom: Geometry) -> SolutionSlice:
    segs = list(seglist)
    seg_positions = [(seg, float(y_at(seg, x_i))) for seg in segs]

    tol = 1e-12  # tolerance for clustering same y
    # sort primarily by y (rounded), secondarily by slope (angle)
    seg_positions.sort(key=lambda t: (round(t[1], 12), np.tan(t[0].sigma)))

    # group by y within tolerance
    grouped = []
    for _, group in groupby(seg_positions, key=lambda t: round(t[1], 12)):
        g = list(group)
        # sort group by slope explicitly (so fans split properly)
        g.sort(key=lambda t: np.tan(t[0].sigma))
        grouped.extend(g)

    regions = []
    for (seg_low, y_low), (seg_high, y_high) in zip(grouped[:-1], grouped[1:]):
        fs = None
        above_low, below_low = states_above_below(seg_low)
        above_high, below_high = states_above_below(seg_high)

        if isinstance(seg_low, Farfield) and seg_low is geom.lower_bbox:
            fs = above_high
        elif isinstance(seg_high, Farfield) and seg_high is geom.upper_bbox:
            fs = below_low
        else:
            # regular resolution
            if above_low is not None:
                fs = above_low
            if below_high is not None:
                fs = below_high

            if fs is None:
                if isinstance(seg_low, WallSegment):
                    if seg_low.normal[1] < 0 and above_high is not None:
                        fs = above_high
                    elif seg_low.normal[1] > 0 and below_high is not None:
                        fs = below_high

                if isinstance(seg_high, WallSegment):
                    if seg_high.normal[1] > 0 and below_low is not None:
                        fs = below_low
                    elif seg_high.normal[1] < 0 and above_low is not None:
                        fs = above_low

        regions.append(Region(seg_low, seg_high, fs))

    return SolutionSlice(x_i, regions)


def states_above_below(segment: Segment):
    if isinstance(segment, (Wave, Slipstream)):
        orientation = segment.sigma - segment.pre_state.theta
        if orientation > 0:  
            return segment.pre_state, segment.post_state
        else:
            return segment.post_state, segment.pre_state
    elif isinstance(segment, (WallSegment, Farfield)):
        return None, None
    
def y_at(segment, x_i):
    return segment.y_start + np.tan(segment.sigma) * (x_i - segment.x_start)


def get_inflow(x_next: float, inflection, solution_slice: SolutionSlice):
    """
    Given an inflection point and a SolutionSlice at the same x,
    return the assigned FlowState at that point.
    
    Parameters:
    - x_next: float, the x-coordinate where inflow is being evaluated
    - inflection: object with attributes x_start, y_start, sigma, wall_id
    - solution_slice: SolutionSlice at x_start
    
    Returns:
    - FlowState assigned to the region containing the inflection
    """
    y_inflect = inflection.y

    for region in solution_slice:
        y_lower = y_at(region.lower, x_next)
        y_upper = y_at(region.upper, x_next)

        # ensure y_lower < y_upper
        if y_lower > y_upper:
            y_lower, y_upper = y_upper, y_lower
        if y_lower <= y_inflect <= y_upper:
            return region.flowstate
    return None