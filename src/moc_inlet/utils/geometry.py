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

class Wall:
    def __init__(self, points, wall_id):
        self.points = np.asarray(points, dtype=np.float64)
        self.wall_id = wall_id
        self.segments = []

    @property
    def x(self): return self.points[:, 0]
    @property
    def y(self): return self.points[:, 1]

    def __len__(self): return len(self.points)
    def __getitem__(self, idx): return self.points[idx]
    def __iter__(self): return iter(self.points)
    
    def assign_segments(self, segments):
        self.segments.extend(segments)

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
    def __init__(self, body_points, body_id):
        self.body_id = body_id
        body_normals, body_points = compute_normals(body_points)
        wall1_points, wall2_points = self._truncate_at_turnback(body_points)
        # plt.figure()
        # plt.plot(wall1_points[:,0], wall1_points[:,1],'k')
        # plt.plot(wall2_points[:,0], wall2_points[:,1],'r')
        # plt.show()
        n1 = len(wall1_points) - 1
        n2 = len(wall2_points) - 1
        wall1_normals = body_normals[:n1]
        wall2_normals = body_normals[n1:n1+n2]

        self.wall1 = Wall(wall1_points, "wall1")
        self.wall2 = Wall(wall2_points, "wall2")
        segs1 = self.generate_wall_segments(self.wall1, wall1_normals)
        segs2 = self.generate_wall_segments(self.wall2, wall2_normals)
        self.wall1.assign_segments(segs1)
        self.wall2.assign_segments(segs2)

        inf1 = self.get_inflections(self.wall1)
        inf2 = self.get_inflections(self.wall2)

        self.inflections = self._combine_inflections(inf1, inf2)

    def generate_wall_segments(self, wall: Wall, normals):
        """
        Generate WallSegment objects for a given wall.
        Guarantees: x_start <= x_end for every segment.
        sigma is computed for the directed segment from x_start -> x_end.
        normals must be length = len(wall.points) - 1 and are kept as given.
        """
        pts = np.asarray(wall.points, dtype=float)
        normals = np.asarray(normals, dtype=float)

        if pts.shape[0] < 2:
            return []

        if normals.shape[0] != pts.shape[0] - 1:
            raise ValueError(
                f"normals must have length N-1 for N points; "
                f"got {normals.shape[0]} normals and {pts.shape[0]} points"
            )

        segments = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            dx = x1 - x0
            dy = y1 - y0

            # Ensure x_start <= x_end (if equal, order by y)
            if (dx < 0) or (np.isclose(dx, 0.0) and (y1 < y0)):
                xs, ys = x1, y1
                xe, ye = x0, y0
                dx = -dx
                dy = -dy
            else:
                xs, ys = x0, y0
                xe, ye = x1, y1

            # Compute sigma for oriented (xs,ys)->(xe,ye)
            if np.isclose(dx, 0.0):
                sigma = np.pi / 2 if dy >= 0 else -np.pi / 2
            else:
                sigma = np.arctan2(dy, dx)

            seg = WallSegment(xs, ys, xe, ye, sigma, normals[i], wall.wall_id)
            segments.append(seg)

        # Sort by x_start, y_start for deterministic ordering
        segments.sort(key=lambda s: (s.x_start, s.y_start))
        return segments
    

    def get_inflections(self, wall: Wall):
        xs, ys, sigmas, walls, bodies, normals = [], [], [], [], [], []
        segments = wall.segments
        first_seg = segments[0]
        xs.append(first_seg.x_start)
        ys.append(first_seg.y_start)
        sigmas.append(first_seg.sigma)
        walls.append(wall.wall_id)
        bodies.append(self.body_id)
        normals.append(first_seg.normal)

        for i in range(1, len(segments)):
            prev_sigma = segments[i-1].sigma
            curr_seg = segments[i]
            if not np.isclose(curr_seg.sigma, prev_sigma):
                xs.append(curr_seg.x_start)
                ys.append(curr_seg.y_start)
                sigmas.append(curr_seg.sigma)
                walls.append(wall.wall_id)
                bodies.append(self.body_id)
                normals.append(curr_seg.normal)

        return Inflections(xs, ys, sigmas, walls, bodies, normals)


    def _truncate_at_turnback(self, points):
        dx = np.diff(points[:, 0])
        turn_idx = np.where(dx <= 0)[0]
        if len(turn_idx) > 0:
            turnback_idx = turn_idx[0] + 1
            wall1 = points[:turnback_idx]  
            wall2 = points[turnback_idx:]  
            return wall1, wall2
        return points, np.empty((0, 2))  # no turnback, return original wall and empty wall
    
    def _combine_inflections(self, inf1, inf2):
        return Inflections(
            np.concatenate([inf1.x, inf2.x]),
            np.concatenate([inf1.y, inf2.y]),
            np.concatenate([inf1.sigma, inf2.sigma]),
            np.concatenate([inf1.wall, inf2.wall]),
            np.concatenate([inf1.body, inf2.body]),
            np.concatenate([inf1.normals, inf2.normals])
        )
    
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

        self.body1 = Body(body1, "body1")
        self.body2 = Body(body2, "body2")
        self.lower_bbox, self.upper_bbox = self.build_bounding_box()
        self.x0 = self.upper_bbox.x_start
        self.x_end = self.upper_bbox.x_end
        self.inflections = self._combine_inflections()



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


    # def get_walls(self, x_cur, active_inflections=None): 
    #     active_inflections = active_inflections or []
    #     infl_walls = {inf.wall[0]: inf for inf in active_inflections}

    #     def get_wall_segments(body, x_cur, infl_walls):
    #         segs = []
    #         for wall in [body.wall1, body.wall2]:
    #             for seg in wall._create_segments():  # precomputed segments
    #                 if seg.x_start <= x_cur <= seg.x_end:
    #                     # If this wall has an active inflection, override sigma
    #                     if wall.wall_id in infl_walls:
    #                         seg = WallSegment(seg.x_start, seg.y_start,
    #                                         seg.x_end, seg.y_end,
    #                                         infl_walls[wall.wall_id].sigma[0],
    #                                         seg.normal, wall.wall_id)
    #                     segs.append(seg)
    #                     break  # only one active segment per wall at x_cur
    #         return segs

    #     segments = []
    #     segments.extend(get_wall_segments(self.body1, x_cur, infl_walls))
    #     segments.extend(get_wall_segments(self.body2, x_cur, infl_walls))
    #     return segments if segments else None

    def get_walls(self, x_cur, active_inflections=None):
        active_inflections = active_inflections or []

        segments = []

        for body in [self.body1, self.body2]:
            for wall in [body.wall1, body.wall2]:
                # Find default segment spanning x_cur
                seg = next((s for s in wall.segments if s.x_start <= x_cur <= s.x_end), None)

                # Check for active inflection for this wall + body at x_cur
                inf = next(
                    (inf for inf in active_inflections
                    if inf.wall[0] == wall.wall_id and inf.body[0] == body.body_id
                        and np.isclose(inf.x[0], x_cur)),
                    None
                )

                if inf is not None:
                    # Pick the segment that begins at the inflection point
                    seg_inf = next(
                        (s for s in wall.segments 
                        if np.isclose(s.x_start, inf.x[0]) and np.isclose(s.y_start, inf.y[0])),
                        seg  # fallback to default if not found
                    )
                    seg = seg_inf

                if seg is not None:
                    segments.append(seg)

        return segments if segments else None