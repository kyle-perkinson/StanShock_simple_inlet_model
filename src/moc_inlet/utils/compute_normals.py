import numpy as np
import matplotlib.pyplot as plt

def compute_normals(points, closed=True):
    pts = np.asarray(points, dtype=float).copy()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be shape (N,2)")
    if closed:
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

    tangents = np.diff(pts, axis=0)
    lengths = np.linalg.norm(tangents, axis=1)
    if np.any(lengths == 0):
        keep = lengths > 0
        tangents = tangents[keep]

    left = np.column_stack((-tangents[:, 1], tangents[:, 0]))   # rotate +90 (left)
    right = -left                                               # rotate -90 (right)

    if closed:
        area = 0.5 * np.sum(pts[:-1, 0]*pts[1:, 1] - pts[1:, 0]*pts[:-1, 1])
        normals = right if area > 0 else left
    else:
        centroid = np.mean(pts, axis=0)
        midpoints = (pts[:-1] + pts[1:]) / 2.0
        vec = midpoints - centroid
        dot_right = np.einsum('ij,ij->i', vec, right)
        choose_right = dot_right > 0
        normals = np.empty_like(right)
        normals[choose_right] = right[choose_right]
        normals[~choose_right] = left[~choose_right]

    norms = np.linalg.norm(normals, axis=1)
    small = norms < 1e-12
    if np.any(small):
        normals[~small] = normals[~small] / norms[~small][:, None]
        normals[small] = np.array([0.0, 0.0])
    else:
        normals /= norms[:, None]
    # plot_normals(pts, normals)
    return normals


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