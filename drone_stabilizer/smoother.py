from __future__ import annotations

import numpy as np
from .config import Config


def smooth_trajectory(trajectory: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Kalman RTS smoother over the cumulative trajectory.

    Returns per-frame correction deltas C[i] (shape N-1 x 4) where
    C[i] = smoothed_cumulative[i] - raw_cumulative[i].

    Intentional pans are preserved: when the raw velocity exceeds
    cfg.velocity_pan_threshold the smoothed value is blended back toward
    the raw value, avoiding over-smoothing of deliberate camera moves.
    """
    n, d = trajectory.shape  # (N-1, 4)
    if n == 0:
        return np.zeros((0, d), dtype=np.float64)

    cumulative = np.cumsum(trajectory, axis=0)

    smoothed = np.zeros_like(cumulative)
    for col in range(d):
        smoothed[:, col] = _kalman_smooth_1d(cumulative[:, col], cfg)

    # Pan preservation: when the camera is genuinely panning we let the
    # smoothed path follow the raw path so the correction doesn't grow large
    # (which would push content out of frame / force a huge crop).
    #
    # The threshold MUST be applied to the *smoothed* velocity, not the raw
    # velocity: high-frequency jitter also has large instantaneous raw
    # velocity, so thresholding the raw signal would mistake jitter for a pan
    # and blend the jitter straight back in.  The smoothed velocity reflects
    # only sustained low-frequency motion (the actual pan).
    if n >= 2:
        pan_vel = np.abs(np.gradient(smoothed, axis=0))        # (N-1, 4)
    else:
        pan_vel = np.zeros_like(smoothed)
    thr = cfg.velocity_pan_threshold
    blend = np.clip((pan_vel - thr) / max(thr, 1e-6), 0.0, 1.0)
    smoothed = (1.0 - blend) * smoothed + blend * cumulative

    corrections = smoothed - cumulative
    return corrections


def _kalman_smooth_1d(signal: np.ndarray, cfg: Config) -> np.ndarray:
    n = len(signal)
    q = cfg.process_noise_q
    r = cfg.measurement_noise_r

    # --- forward pass ---
    x = np.zeros(n)
    p = np.zeros(n)
    x[0] = signal[0]
    p[0] = 1.0

    for i in range(1, n):
        x_pred = x[i - 1]
        p_pred = p[i - 1] + q
        k = p_pred / (p_pred + r)
        x[i] = x_pred + k * (signal[i] - x_pred)
        p[i] = (1 - k) * p_pred

    # --- backward RTS smoother pass ---
    xs = x.copy()
    ps = p.copy()

    for i in range(n - 2, -1, -1):
        p_pred = p[i] + q
        gain = p[i] / p_pred
        xs[i] = xs[i] + gain * (xs[i + 1] - xs[i])
        ps[i] = ps[i] + gain ** 2 * (ps[i + 1] - p_pred)

    return xs
