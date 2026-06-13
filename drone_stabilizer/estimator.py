from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Set, Tuple

from .config import Config
from .detector import AKAZEDetector, FarnebackDetector
from .utils import decompose_homography
from .video_io import VideoReader


def _identity_transform() -> np.ndarray:
    return np.zeros(4, dtype=np.float64)  # [tx, ty, angle, log_scale]


def estimate_trajectory(
    reader: VideoReader, cfg: Config
) -> Tuple[np.ndarray, Set[int]]:
    """
    Pass 1: estimate per-frame incremental transforms.

    Returns
    -------
    trajectory : ndarray, shape (N-1, 4)
        Per-frame [tx, ty, angle, log_scale] transforms.
    low_texture_frames : set[int]
        Frame indices where AKAZE failed and Farneback was used.
    """
    akaze_det = AKAZEDetector(cfg) if cfg.use_akaze else None
    farne_det = FarnebackDetector(cfg)

    reader.seek(0)

    trajectory: List[np.ndarray] = []
    low_texture_frames: Set[int] = set()

    # Stream frame pairs, holding only the previous frame in memory.
    # (Loading the whole 4K video into RAM would need tens of GB.)
    prev = reader.read()
    if prev is None:
        return np.zeros((0, 4), dtype=np.float64), low_texture_frames

    i = 0
    while True:
        curr = reader.read()
        if curr is None:
            break

        pts0, pts1, used_farneback = _get_points(
            prev, curr, akaze_det, farne_det, cfg
        )

        if used_farneback:
            low_texture_frames.add(i + 1)

        transform = _estimate_pair(pts0, pts1, cfg)
        trajectory.append(transform)

        if cfg.debug and i % 50 == 0:
            src = "farneback" if used_farneback else "akaze"
            print(
                f"  [estimator] frame {i} ({src})  "
                f"tx={transform[0]:.1f} ty={transform[1]:.1f}"
            )

        prev = curr
        i += 1

    if not trajectory:
        return np.zeros((0, 4), dtype=np.float64), low_texture_frames

    return np.array(trajectory, dtype=np.float64), low_texture_frames


def _get_points(
    prev: np.ndarray,
    curr: np.ndarray,
    akaze_det: Optional[AKAZEDetector],
    farne_det: FarnebackDetector,
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    Try AKAZE first; fall back to Farneback for low-texture scenes.
    Returns (pts0, pts1, used_farneback).
    """
    if akaze_det is not None:
        pts0, pts1 = akaze_det.detect(prev, curr)
        if pts0 is not None and len(pts0) >= cfg.low_texture_kp_threshold:
            return pts0, pts1, False
        # Too few keypoints — low-texture scene
        pts0, pts1 = farne_det.detect(prev, curr)
        return pts0, pts1, True

    pts0, pts1 = farne_det.detect(prev, curr)
    return pts0, pts1, False


def _estimate_pair(
    pts0: Optional[np.ndarray],
    pts1: Optional[np.ndarray],
    cfg: Config,
) -> np.ndarray:
    if pts0 is None or len(pts0) < cfg.min_match_count:
        return _identity_transform()

    H, mask = cv2.findHomography(
        pts0, pts1,
        cv2.RANSAC,
        cfg.ransac_reproj_threshold,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )

    if H is None:
        return _identity_transform()

    return decompose_homography(H)
