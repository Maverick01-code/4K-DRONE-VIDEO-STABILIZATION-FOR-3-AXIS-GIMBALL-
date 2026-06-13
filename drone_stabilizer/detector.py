from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from .config import Config


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


class AKAZEDetector:
    def __init__(self, cfg: Config) -> None:
        self._akaze = cv2.AKAZE_create(threshold=cfg.akaze_threshold)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._min_matches = cfg.min_match_count

    def detect(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        g0, g1 = _to_gray(prev), _to_gray(curr)
        kp0, des0 = self._akaze.detectAndCompute(g0, None)
        kp1, des1 = self._akaze.detectAndCompute(g1, None)

        if des0 is None or des1 is None or len(kp0) < self._min_matches:
            return None, None

        matches = self._matcher.knnMatch(des0, des1, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < self._min_matches:
            return None, None

        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
        return pts0, pts1


class FarnebackDetector:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def detect(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        g0, g1 = _to_gray(prev), _to_gray(curr)
        c = self._cfg
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1,
            None,
            c.farneback_pyr_scale,
            c.farneback_levels,
            c.farneback_winsize,
            c.farneback_iterations,
            c.farneback_poly_n,
            c.farneback_poly_sigma,
            0,
        )
        h, w = g0.shape
        step = 16
        ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
        pts0 = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
        dx = flow[ys, xs, 0].ravel()
        dy = flow[ys, xs, 1].ravel()
        pts1 = pts0 + np.column_stack([dx, dy])
        return pts0, pts1


def build_detector(cfg: Config) -> AKAZEDetector | FarnebackDetector:
    if cfg.use_akaze:
        return AKAZEDetector(cfg)
    return FarnebackDetector(cfg)
