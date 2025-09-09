from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.segment import SegmentList, Segment, WallSegment, Wave, Slipstream, Farfield
from moc_inlet.utils.geometry import Geometry, Inflection
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
    
    def get_flowstate(self, x_next: float, y_next: float, eps: float = 1e-7):
        """
        Return the FlowState at a given x_next and y_next within this slice.
        Skips regions that are wall–wall bounded or have no flowstate.
        If nothing is found, retries slightly upstream in x.
        Raises ValueError if still no valid region is found.
        """
        def _search(xq, yq):
            for region in self.regions:
                # Skip wall–wall regions
                if isinstance(region.lower, WallSegment) and isinstance(region.upper, WallSegment):
                    continue
                if region.flowstate is None:
                    continue

                y_lower = region.lower.y_at(xq)
                y_upper = region.upper.y_at(xq)

                if y_lower > y_upper:
                    y_lower, y_upper = y_upper, y_lower

                if y_lower <= yq <= y_upper:
                    return region.flowstate
            return None

        fs = _search(x_next, y_next)
        if fs is not None:
            return fs

        fs = _search(x_next - eps, y_next)
        if fs is not None:
            return fs

        raise ValueError(
            f"No valid flowstate found at (x={x_next}, y={y_next}) in slice at x={self.x}"
        )



def build_regions(domain: Domain, seglist: SegmentList, x_new: float, geom: Geometry, tol=1e-12) -> SolutionSlice:
    """
    Build regions at slice x_new, using prior slice in domain to propagate FlowStates.
    Prevent impossible wall/wall regions from different bodies.
    """
    segs = list(seglist)
    seg_positions = [(seg, y_at(seg, x_new)) for seg in segs]

    # Sort primarily by y, then by slope
    seg_positions.sort(key=lambda t: (round(t[1], 12), np.tan(t[0].sigma)))
    grouped = seg_positions

    regions = []
    prev_slice = domain.slices[-1] if domain.slices else None

    # Helper to determine body_id of a wall segment
    def get_body_id(seg):
        if isinstance(seg, WallSegment):
            for body in [geom.body1, geom.body2]:
                if seg in body.wall1.segments or seg in body.wall2.segments:
                    return body.body_id
        return None

    for (seg_low, y_low), (seg_high, y_high) in zip(grouped[:-1], grouped[1:]):
        # Skip impossible wall/wall regions
        if isinstance(seg_low, WallSegment) and isinstance(seg_high, WallSegment):
            body_low = get_body_id(seg_low)
            body_high = get_body_id(seg_high)
            if body_low != body_high:
                continue


        fs = None
        if prev_slice is not None:
            for old_reg in prev_slice:
                if old_reg.lower is seg_low and old_reg.upper is seg_high:
                    fs = old_reg.flowstate
                    break

        if fs is None:
            above_low, below_low = states_above_below(seg_low)
            above_high, below_high = states_above_below(seg_high)

            if isinstance(seg_low, Farfield) and seg_low is geom.lower_bbox:
                fs = seg_low.state
            elif isinstance(seg_high, Farfield) and seg_high is geom.upper_bbox:
                fs = seg_high.state
            else:
                if above_low is not None:
                    fs = above_low
                if below_high is not None:
                    fs = below_high

                if fs is None:
                    if isinstance(seg_low, (WallSegment, Slipstream)):
                        if seg_low.normal[1] < 0 and above_high is not None:
                            fs = above_high
                        elif seg_low.normal[1] > 0 and below_high is not None:
                            fs = below_high
                    if isinstance(seg_high, (WallSegment, Slipstream)):
                        if seg_high.normal[1] > 0 and below_low is not None:
                            fs = below_low
                        elif seg_high.normal[1] < 0 and above_low is not None:
                            fs = above_low

        regions.append(Region(seg_low, seg_high, fs))

    return SolutionSlice(x_new, regions)



def states_above_below(segment: Segment):
    if isinstance(segment, (Wave)):
        orientation = segment.sigma - segment.pre_state.theta
        if orientation > 0:  
            return segment.pre_state, segment.post_state
        else:
            return segment.post_state, segment.pre_state
    elif isinstance(segment, (WallSegment, Farfield, Slipstream)):
        return None, None
    
def y_at(segment, x_i):
    return segment.y_start + np.tan(segment.sigma) * (x_i - segment.x_start)