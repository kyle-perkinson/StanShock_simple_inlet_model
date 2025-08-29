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

# def compute_normals(points):
#     """
#     Compute outward normals for a closed polygon defined by `points`.
#     Returns one unit normal per *segment* (len(points)-1).
#     """
#     pts = np.asarray(points, dtype=float)
#     if not np.allclose(pts[0], pts[-1]):
#         pts = np.vstack([pts, pts[0]])  # close loop

#     tangents = np.diff(pts, axis=0)
#     left = np.column_stack((-tangents[:, 1], tangents[:, 0]))
#     right = -left

#     # Signed area: >0 => CCW, <0 => CW
#     area = 0.5 * np.sum(pts[:-1, 0]*pts[1:, 1] - pts[1:, 0]*pts[:-1, 1])
#     normals = right if area > 0 else left

    


#     normals /= np.linalg.norm(normals, axis=1)[:, None]
#     plot_normals(pts, normals)
#     return normals  # length = len(pts)-1




class Wall:
    def __init__(self, points, wall_id, normals=None):
        self.points = np.asarray(points, dtype=np.float64)
        self.wall_id = wall_id
        self.normals = normals
        self.points = self.points[np.argsort(self.points[:, 0])]
        self._segments = self._create_segments()

    @property
    def x(self): return self.points[:, 0]
    @property
    def y(self): return self.points[:, 1]

    def __len__(self): return len(self.points)
    def __getitem__(self, idx): return self.points[idx]
    def __iter__(self): return iter(self.points)
    
    def y_at(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return None
        return np.interp(x, self.x, self.y)

    def sigma_at(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return None
        idx = np.searchsorted(self.x, x) - 1
        if idx < 0:
            idx = 0
        if idx >= len(self.x) - 1:
            idx = len(self.x) - 2
        dy = self.y[idx + 1] - self.y[idx]
        dx = self.x[idx + 1] - self.x[idx]
        sigma = np.arctan2(dy, dx) if dx != 0 else (np.pi/2 if dy >= 0 else -np.pi/2)
        return sigma
    
    def _create_segments(self):
        segments = []
        for i in range(len(self.points) - 1):
            x_start, y_start = self.points[i]
            x_end, y_end = self.points[i + 1]
            sigma = np.arctan2(y_end - y_start, x_end - x_start)
            normal = self.normals[i]
            segments.append(WallSegment(x_start, y_start, x_end, y_end, sigma, normal, self.wall_id))
        return segments

class Inflections:
    """Container for all wall inflections (vectorized, sliceable)."""
    def __init__(self, x, y, sigma, wall_ids, body_ids, normals):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.sigma = np.asarray(sigma)
        self.wall = np.asarray(wall_ids, dtype=object)
        self.body = np.asarray(body_ids, dtype=object)
        self.normals = np.asarray(normals)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        sigma = self.sigma[idx]
        wall = np.array(self.wall, dtype=object)[idx]
        body = np.array(self.body, dtype=object)[idx]
        normal = self.normals[idx]
        if np.isscalar(x):
            return Inflections([x], [y], [sigma], [wall], [body], [normal])
        else:
            return Inflections(x, y, sigma, wall.tolist(), body.tolist(), normal.tolist())

class Body:
    def __init__(self, wall1_points, wall2_points, body_id):
        self.body_id = body_id

        # Build closed polygon (wall1 forward, wall2 reversed)
        body_poly = np.vstack([wall1_points, wall2_points])
        if not np.allclose(body_poly[0], body_poly[-1]):
            body_poly = np.vstack([body_poly, body_poly[0]])

        # Compute outward normals for the polygon
        body_normals = compute_normals(body_poly)

        # Slice normals back to wall segments
        n1 = len(wall1_points) - 1
        n2 = len(wall2_points) - 1
        wall1_normals = body_normals[:n1]
        wall2_normals = body_normals[n1:n1+n2]

        self.wall1 = Wall(wall1_points, "wall1", wall1_normals)
        self.wall2 = Wall(wall2_points, "wall2", wall2_normals)
        self.inflections = self._find_inflections()

    def _find_inflections(self):
        xs, ys, slopes, walls, bodies, normals = [], [], [], [], [], []
        for wall in [self.wall1, self.wall2]:
            dx = np.diff(wall.x); dy = np.diff(wall.y)
            slopes_local = dy / dx
            change_idx = np.where(np.diff(slopes_local) != 0)[0]
            infl_idx = np.unique(np.concatenate(([0], change_idx + 1)))
            for idx in infl_idx:
                if idx < len(wall) - 1:
                    slope = (wall.y[idx+1] - wall.y[idx]) / (wall.x[idx+1] - wall.x[idx])
                else:
                    slope = slopes_local[-1] if len(slopes_local) > 0 else 0
                xs.append(wall.x[idx]); ys.append(wall.y[idx])
                slopes.append(slope)
                walls.append(wall.wall_id); bodies.append(self.body_id)
                # pick nearest segment normal
                normals.append(wall.normals[min(idx, len(wall.normals)-1)])
        return Inflections(xs, ys, np.arctan(slopes), walls, bodies, normals)
    

class Geometry:
    def __init__(self, body1, body2):
        body1 = self._ensure_closed_polygon(body1)
        body2 = self._ensure_closed_polygon(body2)  

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

        wall1a, wall2a = self._truncate_at_turnback(body1)
        wall1b, wall2b = self._truncate_at_turnback(body2)

        self.body1 = Body(wall1a, wall2a, "body1")
        self.body2 = Body(wall1b, wall2b, "body2")
        self.lower_bbox, self.upper_bbox = self.build_bounding_box()
        self.x0 = self.upper_bbox.x_start
        self.x_end = self.upper_bbox.x_end
        self.inflections = self._combine_inflections()

    def _ensure_closed_polygon(self, body):
        body = np.asarray(body, float)
        if not np.allclose(body[0], body[-1]):
            return np.vstack([body, body[0]])
        return body

    def _truncate_at_turnback(self, wall):
        dx = np.diff(wall[:, 0])
        turn_idx = np.where(dx <= 0)[0]
        if len(turn_idx) > 0:
            turnback_idx = turn_idx[0] + 1
            wall1 = wall[:turnback_idx]  
            wall2 = wall[turnback_idx:]  
            return wall1, wall2
        return wall, np.empty((0, 2))  # no turnback, return original wall and empty wall

    def _combine_inflections(self):
        i1, i2 = self.body1.inflections, self.body2.inflections
        return Inflections(
            np.concatenate([i1.x, i2.x]),
            np.concatenate([i1.y, i2.y]),
            np.concatenate([i1.sigma, i2.sigma]),
            np.concatenate([i1.wall, i2.wall]),
            np.concatenate([i1.body, i2.body]),
            np.concatenate([i1.normals, i2.normals]),
        )
    

    def build_bounding_box(self):
        x = np.concatenate([self.body1.wall1.x, self.body1.wall2.x, self.body2.wall1.x, self.body2.wall2.x])
        y = np.concatenate([self.body1.wall1.y, self.body1.wall2.y, self.body2.wall1.y, self.body2.wall2.y])
        dx = np.abs(max(x) - min(x)) / (10 * max(x))
        dy = np.abs(max(y) - min(y)) / (10 * max(y))

        x0 = min(x) - dx
        x_end = max(x) + dx
        y0 = min(y) - dy
        y1 = max(y) + dy
        bot = Farfield(x0, y0, x_end, y0, 0, np.array([0, 1]) )
        top = Farfield(x0, y1, x_end, y1, 0, np.array([0,  -1]) )
        return bot, top


    def get_walls(self, x_cur, active_inflections=None): 
        active_inflections = active_inflections or []
        infl_walls = {inf.wall[0]: inf for inf in active_inflections}

        def get_wall_segments(body, x_cur, infl_walls):
            segs = []
            for wall in [body.wall1, body.wall2]:
                for seg in wall._create_segments():  # precomputed segments
                    if seg.x_start <= x_cur <= seg.x_end:
                        # If this wall has an active inflection, override sigma
                        if wall.wall_id in infl_walls:
                            seg = WallSegment(seg.x_start, seg.y_start,
                                            seg.x_end, seg.y_end,
                                            infl_walls[wall.wall_id].sigma[0],
                                            seg.normal, wall.wall_id)
                        segs.append(seg)
                        break  # only one active segment per wall at x_cur
            return segs

        segments = []
        segments.extend(get_wall_segments(self.body1, x_cur, infl_walls))
        segments.extend(get_wall_segments(self.body2, x_cur, infl_walls))
        return segments if segments else None