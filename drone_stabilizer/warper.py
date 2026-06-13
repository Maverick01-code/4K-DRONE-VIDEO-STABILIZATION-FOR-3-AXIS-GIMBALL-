from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from .config import Config
from .utils import compose_homography


_BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT,
    "reflect101": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "wrap": cv2.BORDER_WRAP,
}


def warp_frame(
    frame: np.ndarray,
    correction: np.ndarray,
    cfg: Config,
    prev_correction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply stabilisation correction to a single frame.

    When cfg.use_mesh_warp is True (default) the frame is warped via a
    16×9 mesh with optional per-scanline rolling-shutter compensation.
    Otherwise falls back to a single cv2.warpPerspective call.

    Parameters
    ----------
    correction      : 4-DOF delta [tx, ty, angle, log_scale] for this frame.
    prev_correction : 4-DOF delta for the previous frame (used for RS only).
    """
    h, w = frame.shape[:2]
    H_curr = compose_homography(correction, w, h)
    border = _BORDER_MODES.get(cfg.border_mode, cv2.BORDER_REFLECT)

    if not cfg.use_mesh_warp:
        return cv2.warpPerspective(frame, H_curr, (w, h), borderMode=border)

    H_prev = (
        compose_homography(prev_correction, w, h)
        if prev_correction is not None
        else H_curr
    )

    rs_factor = cfg.rolling_shutter_factor  # 0 = no RS, 1 = full RS blend
    map_x, map_y = _build_mesh_remap(H_curr, H_prev, w, h, cfg.mesh_cols, cfg.mesh_rows, rs_factor)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=border)


def _build_mesh_remap(
    H_curr: np.ndarray,
    H_prev: np.ndarray,
    W: int,
    H: int,
    mesh_cols: int,
    mesh_rows: int,
    rs_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dense (H×W) inverse remap maps from a coarse mesh.

    The mesh has (mesh_rows+1)×(mesh_cols+1) vertices evenly spaced over the
    output frame.  Each vertex position in the SOURCE frame is computed by
    applying the inverse of the per-row correction homography (rolling-shutter
    adjusted if rs_factor > 0) to the output vertex position.

    Within each cell the source coordinates are bilinearly interpolated so we
    avoid per-pixel matrix math while still getting smooth deformation.
    """
    # --- mesh vertex positions in OUTPUT space ---
    xs = np.linspace(0, W - 1, mesh_cols + 1)   # (mesh_cols+1,)
    ys = np.linspace(0, H - 1, mesh_rows + 1)   # (mesh_rows+1,)
    grid_x, grid_y = np.meshgrid(xs, ys)         # (mesh_rows+1, mesh_cols+1)

    # --- per-vertex inverse correction (output → source) ---
    # Rolling-shutter model: the sensor reads top-to-bottom over the frame
    # interval, so each scanline is captured at a slightly different time and
    # therefore a slightly different camera pose.  The *centre* row is the
    # reference and gets the real stabilising transform H_curr; rows above and
    # below get a symmetric skew proportional to how much the correction
    # changed since the previous frame (the motion that occurred during
    # readout).  Referencing the centre row is essential — otherwise the whole
    # frame is offset toward H_prev and the per-row shear injects wobble.
    delta = H_curr - H_prev          # inter-frame change in correction
    src_x = np.zeros_like(grid_x)
    src_y = np.zeros_like(grid_y)

    for vi in range(mesh_rows + 1):
        row_frac = vi / mesh_rows                 # 0 at top, 1 at bottom
        if rs_factor > 0:
            s = (row_frac - 0.5) * rs_factor      # 0 at centre, +/- at edges
            H_row = H_curr + s * delta
        else:
            H_row = H_curr
        H_inv = np.linalg.inv(H_row)

        vx = grid_x[vi]          # (mesh_cols+1,) output x coords
        vy = grid_y[vi]          # (mesh_cols+1,) output y coords
        ones = np.ones(mesh_cols + 1)
        pts = np.stack([vx, vy, ones])    # (3, mesh_cols+1)
        mapped = H_inv @ pts              # (3, mesh_cols+1)
        mapped /= mapped[2:3, :]
        src_x[vi] = mapped[0]
        src_y[vi] = mapped[1]

    # --- bilinear interpolation to build dense remap ---
    # For each output pixel find its mesh cell and fractional position
    cell_w = (W - 1) / mesh_cols
    cell_h = (H - 1) / mesh_rows

    # pixel row/col indices
    px_cols = np.arange(W, dtype=np.float32)       # (W,)
    px_rows = np.arange(H, dtype=np.float32)       # (H,)

    ci_f = px_cols / cell_w                        # fractional col in mesh
    ri_f = px_rows / cell_h                        # fractional row in mesh

    ci = np.clip(ci_f.astype(np.int32), 0, mesh_cols - 1)   # (W,) cell col index
    ri = np.clip(ri_f.astype(np.int32), 0, mesh_rows - 1)   # (H,) cell row index

    fc = (ci_f - ci).astype(np.float32)            # (W,) frac within cell [0,1]
    fr = (ri_f - ri).astype(np.float32)            # (H,) frac within cell [0,1]

    # Expand to (H, W) grids
    ci2d = ci[np.newaxis, :]    # (1, W)
    ri2d = ri[:, np.newaxis]    # (H, 1)
    fc2d = fc[np.newaxis, :]    # (1, W)
    fr2d = fr[:, np.newaxis]    # (H, 1)

    # Four corner vertices of each cell (for x and y separately)
    v00x = src_x[ri2d,     ci2d]       # top-left
    v10x = src_x[ri2d,     ci2d + 1]   # top-right
    v01x = src_x[ri2d + 1, ci2d]       # bottom-left
    v11x = src_x[ri2d + 1, ci2d + 1]   # bottom-right

    v00y = src_y[ri2d,     ci2d]
    v10y = src_y[ri2d,     ci2d + 1]
    v01y = src_y[ri2d + 1, ci2d]
    v11y = src_y[ri2d + 1, ci2d + 1]

    # Bilinear blend
    map_x = (
        (1 - fr2d) * ((1 - fc2d) * v00x + fc2d * v10x) +
        fr2d       * ((1 - fc2d) * v01x + fc2d * v11x)
    ).astype(np.float32)

    map_y = (
        (1 - fr2d) * ((1 - fc2d) * v00y + fc2d * v10y) +
        fr2d       * ((1 - fc2d) * v01y + fc2d * v11y)
    ).astype(np.float32)

    return map_x, map_y
