import numpy as np
import matplotlib.pyplot as plt

def compute_normals(points):
    pts = np.asarray(points, dtype=float).copy()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be shape (N,2)")
    diff = np.diff(pts, axis=0)
    keep = np.any(np.abs(diff) > 1e-12, axis=1)
    pts = np.vstack([pts[0], pts[1:][keep]])

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    tangents = np.diff(pts, axis=0)
    lengths = np.linalg.norm(tangents, axis = 1)
    tangents = tangents[lengths > 1e-12]

    left = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    right = -left

    area = 0.5 * np.sum(pts[:-1, 0]*pts[1:, 1] - pts[1:, 0]*pts[:-1, 1])
    normals = right if area > 0 else left
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals, pts
