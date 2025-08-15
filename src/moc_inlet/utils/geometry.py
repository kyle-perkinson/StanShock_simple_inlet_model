import numpy as np
from dataclasses import dataclass

class Wall:
    def __init__(self, points, wall_id):
        self.points = np.asarray(points, dtype=np.float64)
        self.wall_id = wall_id

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def __iter__(self):
        return iter(self.points)

@dataclass
class Inflection:
    x: float
    y: float
    slope: float
    wall: Wall  # Direct reference to wall object

class Geometry:
    def __init__(self, body1, body2):
        body1 = self._ensure_closed_polygon(body1)
        body2 = self._ensure_closed_polygon(body2)  

        # Shift origin to min-x of either body
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

        # Assign upper/lower based on mean y
        if np.mean(body1[:, 1]) > np.mean(body2[:, 1]):
            upper = self._truncate_at_turnback(body1)
            lower = self._truncate_at_turnback(body2)
        else:
            upper = self._truncate_at_turnback(body2)
            lower = self._truncate_at_turnback(body1)

        self.upper_wall = Wall(upper, "upper")
        self.lower_wall = Wall(lower, "lower")

        # Store inflections
        self.inflections = self._find_inflections()

    @property
    def x(self):
        """All x points from both walls combined."""
        return np.concatenate([self.upper_wall.x, self.lower_wall.x])

    @property
    def y(self):
        """All y points from both walls combined."""
        return np.concatenate([self.upper_wall.y, self.lower_wall.y])

    def _ensure_closed_polygon(self, body):
        body = np.asarray(body, dtype=np.float64)
        if np.allclose(body[0], body[-1]):
            return body
        if len(body) > 2 and np.allclose(body[1], body[-1]):
            body = np.vstack([body[0:1], body[2:], body[1:2]])
            return body
        return np.vstack([body, body[0:1]])

    def _truncate_at_turnback(self, wall):
        dx = np.diff(wall[:, 0])
        turn_idx = np.where(dx <= 0)[0]
        if len(turn_idx) > 0:
            wall = wall[:turn_idx[0]+1]  # keep points up to first turnback
        return wall

    def _find_inflections(self):
        inflections = []
        for wall in [self.upper_wall, self.lower_wall]:
            dx = np.diff(wall.x)
            dy = np.diff(wall.y)
            slopes = dy / dx
            change_idx = np.where(np.diff(slopes) != 0)[0]
            # Include the very first point as an inflection
            inflection_indices = np.unique(np.concatenate(([0], change_idx + 1)))
            for idx in inflection_indices:
                if idx < len(wall) - 1:
                    slope = (wall.y[idx+1] - wall.y[idx]) / (wall.x[idx+1] - wall.x[idx])
                else:
                    slope = slopes[-1] if len(slopes) > 0 else 0
                inflections.append(
                    Inflection(
                        x=wall.x[idx],
                        y=wall.y[idx],
                        slope=slope,
                        wall=wall
                    )
                )
        return inflections
    
class Region:
    def __init__(self, y_lower, y_upper, flow_state):
        self.y_lower = y_lower
        self.y_upper = y_upper
        self.flow_state = flow_state

class DomainSlice:
    def __init__(self, x):
        self.x = x
        self.regions = []  # list of Region objects

    def add_region(self, y_lower, y_upper, flow_state):
        self.regions.append(Region(y_lower, y_upper, flow_state))