from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .config import Config
from .utils import compose_homography


def compute_warp_crop_region(
    corrections: np.ndarray,
    frame_w: int,
    frame_h: int,
    cfg: Config,
) -> Tuple[int, int, int, int]:
    """
    Analytically compute the crop rectangle by finding the intersection of the
    valid (non-black) output regions across all frames.

    For each frame the correction homography H_i maps source pixels to output
    space.  The valid output region for frame i is the quadrilateral whose
    vertices are the four input corners projected through H_i.  The crop is
    the inner bounding-box of all these quadrilaterals.

    Returns (x, y, w, h) with w and h rounded down to even numbers.
    """
    W, H = frame_w, frame_h
    if len(corrections) == 0:
        # No motion estimated -> nothing to crop, keep the full (even) frame.
        return 0, 0, W - W % 2, H - H % 2

    # corner order: top-left, top-right, bottom-right, bottom-left
    corners_src = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])

    # Start with the whole frame as the safe region
    x1, y1 = 0.0, 0.0
    x2, y2 = float(W), float(H)

    for corr in corrections:
        Hi = compose_homography(corr, W, H)
        warped = cv2.perspectiveTransform(
            corners_src.reshape(1, -1, 2), Hi
        ).reshape(-1, 2)
        tl, tr, br, bl = warped

        # Inner rectangle guaranteed inside the warped quad: take the inner
        # edge on each side (max of the two left corners, min of the two right
        # corners, etc.).  This avoids leaving black triangles when a frame is
        # rotated, unlike a plain bounding box of the four corners.
        left = max(tl[0], bl[0])
        right = min(tr[0], br[0])
        top = max(tl[1], tr[1])
        bottom = min(bl[1], br[1])

        x1 = max(x1, left)
        y1 = max(y1, top)
        x2 = min(x2, right)
        y2 = min(y2, bottom)

    # Degenerate result (no frames, or motion so large the safe region
    # collapsed) — fall back to the full frame rather than a 2px sliver.
    if x2 <= x1 or y2 <= y1:
        return 0, 0, W - W % 2, H - H % 2

    # Add safety margin on top of the computed border
    margin_x = (x2 - x1) * cfg.crop_margin_pct
    margin_y = (y2 - y1) * cfg.crop_margin_pct
    x1 += margin_x
    y1 += margin_y
    x2 -= margin_x
    y2 -= margin_y

    cx = max(0, int(np.ceil(x1)))
    cy = max(0, int(np.ceil(y1)))
    cw = min(W - cx, int(np.floor(x2)) - cx)
    ch = min(H - cy, int(np.floor(y2)) - cy)

    # Ensure positive and even dimensions (required by most video codecs)
    cw = max(2, cw - cw % 2)
    ch = max(2, ch - ch % 2)

    return cx, cy, cw, ch


def auto_crop_region(
    warped_frames: List[np.ndarray],
    cfg: Config,
) -> Tuple[int, int, int, int]:
    """
    Pixel-scan fallback: sample warped frames to find the largest rectangle
    free of black borders.  Used when correction matrices are unavailable.
    """
    h, w = warped_frames[0].shape[:2]
    ever_black = np.zeros((h, w), dtype=np.uint8)

    step = max(1, len(warped_frames) // 60)
    for frame in warped_frames[::step]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        ever_black = cv2.bitwise_or(ever_black, (gray == 0).astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ever_black = cv2.erode(ever_black, kernel)

    valid = cv2.bitwise_not(ever_black)
    x, y, rw, rh = cv2.boundingRect(valid)

    margin_x = int(rw * cfg.crop_margin_pct)
    margin_y = int(rh * cfg.crop_margin_pct)
    x = min(x + margin_x, w - 1)
    y = min(y + margin_y, h - 1)
    rw = max(2, rw - 2 * margin_x)
    rh = max(2, rh - 2 * margin_y)

    return x, y, rw, rh


def apply_crop(frame: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = region
    return frame[y:y + h, x:x + w]


def inpaint_borders(frame: np.ndarray, radius: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mask = (gray == 0).astype(np.uint8) * 255
    if not mask.any():
        return frame
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)


def finalize_frame(
    frame: np.ndarray,
    region: Optional[Tuple[int, int, int, int]],
    cfg: Config,
) -> np.ndarray:
    if cfg.inpaint_borders:
        frame = inpaint_borders(frame, cfg.inpaint_radius)
    if cfg.auto_crop and region is not None:
        frame = apply_crop(frame, region)
    return frame
