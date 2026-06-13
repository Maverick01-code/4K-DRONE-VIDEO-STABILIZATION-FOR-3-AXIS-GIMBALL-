from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


def decompose_homography(H: np.ndarray) -> np.ndarray:
    """
    Extract [tx, ty, rotation_rad, log_scale] from a 3x3 homography.
    Uses the upper-left 2x2 sub-matrix for rotation+scale.
    """
    tx = H[0, 2]
    ty = H[1, 2]
    # compose_homography builds H[0,0]=s·cosθ, H[1,0]=s·sinθ, so recover the
    # angle from the FIRST COLUMN (H[1,0], H[0,0]).  Using H[0,1] here would
    # flip the sign of θ and make rotation corrections amplify shake.
    a = H[0, 0]
    c = H[1, 0]
    angle = np.arctan2(c, a)
    scale = np.sqrt(a ** 2 + c ** 2)
    log_scale = np.log(max(scale, 1e-6))
    return np.array([tx, ty, angle, log_scale], dtype=np.float64)


def compose_homography(
    params: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Build a 3x3 homography from [tx, ty, angle, log_scale].
    Rotation is applied around the frame centre.
    """
    tx, ty, angle, log_scale = params
    scale = np.exp(log_scale)
    cx, cy = frame_w / 2.0, frame_h / 2.0

    cos_a = scale * np.cos(angle)
    sin_a = scale * np.sin(angle)

    H = np.array([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + tx],
        [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx + ty],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)
    return H


def draw_trajectory(
    raw: np.ndarray,
    smoothed_correction: np.ndarray,
    col: int = 0,
    title: str = "Trajectory",
) -> np.ndarray:
    """Render a simple line plot comparing raw vs smoothed trajectory column."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        cumraw = np.cumsum(raw[:, col])
        cumsmooth = cumraw + smoothed_correction[:, col]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(cumraw, label="raw", alpha=0.6)
        ax.plot(cumsmooth, label="smoothed")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except ImportError:
        # matplotlib not available – return blank image
        blank = np.zeros((100, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "matplotlib not installed", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        return blank


def draw_keypoints(
    frame: np.ndarray,
    pts0: Optional[np.ndarray],
    pts1: Optional[np.ndarray],
) -> np.ndarray:
    vis = frame.copy()
    if pts0 is None:
        return vis
    for (x0, y0), (x1, y1) in zip(pts0.astype(int), pts1.astype(int)):
        cv2.circle(vis, (x0, y0), 3, (0, 255, 0), -1)
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), (0, 100, 255), 1, tipLength=0.3)
    return vis
