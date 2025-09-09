import numpy as np
from moc_inlet.utils.segment import WallSegment, Farfield
from moc_inlet.utils.compute_normals import compute_normals
import matplotlib.pyplot as plt

def plot_normals(points, normals, scale=0.05):
    pts = np.asarray(points)
    mids = 0.5 * (pts[:-1] + pts[1:])  # segment midpoints
    plt.plot(pts[:,0], pts[:,1], 'k-', lw=1)  # polygon / polyline
    plt.quiver(
        mids[:,0], mids[:,1], 
        normals[:,0], normals[:,1], 
        angles='xy', scale_units='xy', scale=1/scale, color='r'
    )
    plt.axis('equal')
    plt.show()

class Inflection:
    def __init__(self, x_s: float, segment: 'WallSegment'):
        self.x_s = x_s
        self.segment = segment


class Inflections:
    def __init__(self, inflections=None):
        self.inflections = inflections or []
    def __len__(self):
        return len(self.inflections)
    def __getitem__(self, idx):
        return self.inflections[idx]
    def append(self, inflection: Inflection):
        self.inflections.append(inflection)
    def extend(self, inflections):
        self.inflections.extend(inflections)
    @property
    def xs(self):
        return np.array([inf.x_s for inf in self.inflections])
    @property
    def segments(self):
        return [inf.segment for inf in self.inflections]

class Wall:
    def __init__(self, segments, inflections, wall_id):
        self.wall_id = wall_id
        self.segments = segments
        self.inflections = inflections

class Body:
    def __init__(self, body_points, body_id):
        self.body_id = body_id
        self.wall1, self.wall2, self.xy_min, self.xy_max = self.generate_walls(body_points)
        self.points = body_points
    def generate_walls(self, body_points):
        body_normals, body_points = compute_normals(body_points)
        # plot_normals(body_points,body_normals)
        wall1_points, wall2_points = self._truncate_at_turnback(body_points)
        n1 = len(wall1_points) - 1
        n2 = len(wall2_points) - 1
        wall1_normals = body_normals[:n1]
        wall2_normals = body_normals[n1:n1+n2]

        seg1 = self.points_to_segments(wall1_points, wall1_normals, "w1")
        seg2 = self.points_to_segments(wall2_points, wall2_normals, "w2")

        inf1 = self.get_inflections(seg1)
        inf2 = self.get_inflections(seg2)

        wall1 = Wall(seg1, inf1, "w1")
        wall2 = Wall(seg2, inf2, "w2")

        x_min = float(min(body_points[:,0]))
        x_max = float(max(body_points[:,0]))

        y_min = float(min(body_points[:,1]))
        y_max = float(max(body_points[:,1]))

        return wall1, wall2, (x_min, y_min), (x_max, y_max)

    def points_to_segments(self, pts, normals, wall_id):
        if pts.shape[0] < 2:
            return []
        if normals.shape[0] != pts.shape[0] - 1:
            raise ValueError(
                f"normals must have length N-1 for N points; "
                f"got {normals.shape[0]} normals and {pts.shape[0]} points"
            )
        segments = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]; x1, y1 = pts[i + 1]
            dx = x1 - x0; dy = y1 - y0

            if (dx < 0) or (np.isclose(dx, 0.0) and (y1 < y0)):
                xs, ys = x1, y1
                xe, ye = x0, y0
                dx = -dx
                dy = -dy
            else:
                xs, ys = x0, y0
                xe, ye = x1, y1
            if np.isclose(dx, 0.0):
                sigma = np.pi / 2 if dy >= 0 else -np.pi / 2
            else:
                sigma = np.arctan2(dy, dx)

            seg = WallSegment(xs, ys, xe, ye, sigma, normals[i], wall_id, self.body_id)
            segments.append(seg)
        segments.sort(key=lambda s: (s.x_start, s.y_start))
        return segments
    
    def get_body_downstream(self, x_cur: float):
        segs = []
        for wall in (self.wall1, self.wall2):
            for seg in wall.segments:
                if seg.x_end >= x_cur:
                    segs.append(seg)
        return segs

    def get_inflections(self, segments):
        inflections = Inflections()
        inflections.append(Inflection(segments[0].x_start, segments[0]))


        for i in range(1, len(segments)):
            prev_sigma = segments[i-1].sigma
            curr_seg = segments[i]
            if not np.isclose(curr_seg.sigma, prev_sigma):
                inflections.append(Inflection(curr_seg.x_start, curr_seg))
        return inflections

    def _truncate_at_turnback(self, points):
        dx = np.diff(points[:, 0])
        turn_idx = np.where(dx <= 0)[0]
        if len(turn_idx) > 0:
            turnback_idx = turn_idx[0] + 1
            wall1 = points[:turnback_idx]  
            wall2 = points[turnback_idx:]  
            return wall1, wall2
        return points, np.empty((0, 2))  # no turnback, return original wall and empty wall
    

    
class Geometry:
    def __init__(self, body1, body2):
        min_x_idx1 = np.argmin(body1[:, 0])
        min_x_idx2 = np.argmin(body2[:, 0])
        min_x1, min_y1 = body1[min_x_idx1]
        min_x2, min_y2 = body2[min_x_idx2]

        if min_x1 < min_x2:
            origin_x, origin_y = min_x1, min_y1
        else:
            origin_x, origin_y = min_x2, min_y2

        body1 -= [origin_x, origin_y]
        body2 -= [origin_x, origin_y]

        self.body1 = Body(body1, "b1")
        self.body2 = Body(body2, "b2")
        self.lower_bbox, self.upper_bbox = self.build_bounding_box()
        self.x0 = self.upper_bbox.x_start
        self.x_end = self.upper_bbox.x_end
        self.inflections = self._combine_inflections()

    def _combine_inflections(self):
        all_inflections = (
            self.body1.wall1.inflections.inflections +
            self.body1.wall2.inflections.inflections +
            self.body2.wall1.inflections.inflections +
            self.body2.wall2.inflections.inflections
        )
        return Inflections(all_inflections)
    
    def build_bounding_box(self):
        x1_min, y1_min = self.body1.xy_min
        x1_max, y1_max = self.body1.xy_max
        x2_min, y2_min = self.body2.xy_min
        x2_max, y2_max = self.body2.xy_max

        x_min = min(x1_min, x2_min); x_max = max(x1_max, x2_max)
        y_min = min(y1_min, y2_min); y_max = max(y1_max, y2_max)

        dx = np.abs(x_max - x_min) / (10 * x_max if x_max != 0 else 1.0)
        dy = np.abs(y_max - y_min) / (10 * y_max if y_max != 0 else 1.0)

        x0 = x_min - dx
        x_end = x_max + dx
        y0 = y_min - dy
        y1 = y_max + dy

        bot = Farfield(x0, y0, x_end, y0, 0, np.array([0, 1]))
        top = Farfield(x0, y1, x_end, y1, 0, np.array([0, -1]))
        return bot, top

    def get_downstream_geo(self, x_cur: float):
        segs1 = self.body1.get_body_downstream(x_cur)
        segs2 = self.body2.get_body_downstream(x_cur)
        all_segs = segs1 + segs2
        return all_segs

    def get_next_inflection(self, x_cur: float, tol=1e-12):
        xs_all = self.inflections.xs
        ds_mask = xs_all > (x_cur + tol)

        if not np.any(ds_mask):
            return None, []  # no inflection downstream

        x_next = float(xs_all[ds_mask].min())
        idxs = np.where(np.isclose(xs_all, x_next, atol=tol))[0]
        infl_objects = [self.inflections[i] for i in idxs]  # these are actual Inflection objects
        return x_next, infl_objects
        

    def get_walls(self, x_cur, active_inflections=None):
        """
        Return wall segments relevant at x_cur. Only includes:
            - the segment spanning x_cur
            - segments starting at active inflections
        """
        active_inflections = active_inflections or []
        segments = []

        for body in [self.body1, self.body2]:
            for wall in [body.wall1, body.wall2]:
                # default segment spanning x_cur
                seg = next((s for s in wall.segments if s.x_start <= x_cur <= s.x_end), None)

                inf = next((inf for inf in active_inflections
                            if inf.segment.wall_id == wall.wall_id
                            and inf.segment.body_id == body.body_id
                            and np.isclose(inf.x_s, x_cur)), None)
                if inf is not None:
                    seg = inf.segment
                if seg is not None:
                    segments.append(seg)

        return segments if segments else None