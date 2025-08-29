import numpy as np

from moc_inlet.utils.flowstate import FlowState
class Segment:
    """
    Base class: line segment or ray, with angle sigma.
    Finite [x_start, x_end] for walls, initially infinite for waves/sliplines.
    """
    def __init__(self, x_start, y_start, sigma, x_end=np.inf, y_end=np.inf):
        self.x_start = x_start
        self.y_start = y_start
        self.sigma = sigma
        self.x_end = x_end
        self.y_end = y_end

    def y_at(self, x: float) -> float:
        return self.y_start + np.tan(self.sigma) * (x - self.x_start)

    def x_at(self, y: float) -> float:
        return self.x_start + (y - self.y_start) / np.tan(self.sigma)

    def set_end(self, x_end, y_end):
        self.x_end = x_end
        self.y_end = y_end

    def span_x(self):
        return min(self.x_start, self.x_end), max(self.x_start, self.x_end)


class WallSegment(Segment):
    def __init__(self, x0, y0, x1, y1, sigma, normal, wall_id):
        super().__init__(x0, y0, sigma, x_end=x1, y_end=y1)
        self.normal = normal
        self.wall_id = wall_id

class Wave(Segment):
    """
    A shock or expansion wave.
    Inherits geometry, and adds flow-state info.
    """
    def __init__(self, x_start, y_start, sigma, pre_state, post_state, wave_type=None):
        super().__init__(x_start, y_start, sigma)
        self.pre_state = pre_state
        self.post_state = post_state
        self.wave_type = wave_type

class Slipstream(Segment):
    def __init__(self, pre_state, post_state, sigma, x_start, y_start):
        super().__init__(x_start, y_start, sigma)
        self.pre_state = pre_state
        self.post_state = post_state

class Farfield(Segment):
    def __init__(self, x0, y0, x1, y1, sigma, normal):
        super().__init__(x0, y0, sigma, x_end=x1, y_end=y1)
        self.normal = normal

        

        
class SegmentList:
    """
    Stores all segments found at a given x
    """
    def __init__(self, x: float, segments = None):
        self.segments = []
        self.x = x
        if segments is not None:
            self.segments = segments

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]
    
    def add_segment(self, seg: Segment):
        self.segments.append(seg)

    def get_crossings(self, x, tol=1e-12):
        y_coords = []
        sigma_values = []
        segments_out = []

        for seg in self.segments:
            xmin, xmax = seg.span_x()
            if not np.isfinite(xmax):
                xmax = x + 1e6  # extend for infinite segments
            if xmin - tol <= x <= xmax + tol:
                try:
                    y_val = seg.y_at(x)
                except:
                    continue
                y_coords.append(float(y_val))  # ensure scalar
                sigma_values.append(seg.sigma)
                segments_out.append(seg)

        if sigma_values:
            # lexsort: primary key sigma, secondary key y
            indices = np.lexsort((y_coords, sigma_values))
            y_coords = np.array(y_coords)[indices]
            sigma_values = np.array(sigma_values)[indices]
            segments_out = np.array(segments_out, dtype=object)[indices]

        return Crossings(y_coords, sigma_values, segments_out)



class Crossings:
    """Container for all segment intersections (vectorized, sliceable)."""
    def __init__(self, y, sigma, segments):
        self.y = np.asarray(y)
        self.sigma = np.asarray(sigma)
        self.segments = np.asarray(segments, dtype=object)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return Crossings(
            self.y[idx],
            self.sigma[idx],
            self.segments[idx]
        )