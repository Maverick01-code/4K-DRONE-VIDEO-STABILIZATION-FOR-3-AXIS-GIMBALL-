"""
End-to-end + unit tests for the drone stabilizer.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_pipeline.py   (standalone, no pytest needed)
"""
from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_stabilizer.config import Config
from drone_stabilizer.cropper import compute_warp_crop_region
from drone_stabilizer.estimator import estimate_trajectory
from drone_stabilizer.main import run
from drone_stabilizer.smoother import smooth_trajectory
from drone_stabilizer.utils import compose_homography, decompose_homography
from drone_stabilizer.video_io import VideoReader


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def make_textured_frame(w: int, h: int, seed: int = 0) -> np.ndarray:
    """A high-texture base image so AKAZE finds plenty of keypoints."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    # add some structured shapes for stable corners
    for _ in range(40):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = rng.integers(0, w), rng.integers(0, h)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(base, (x1, y1), (x2, y2), color, 2)
        cv2.circle(base, (x1, y1), rng.integers(5, 30), color, -1)
    return base


def make_shaky_video(path: str, n_frames: int = 40, w: int = 320, h: int = 240,
                     fps: float = 30.0, shake_px: float = 8.0) -> np.ndarray:
    """
    Write a synthetic video: a static textured scene viewed through a camera
    that jitters with high-frequency sinusoidal + random shake.

    Returns the ground-truth per-frame (dx, dy) shake used.
    """
    scene = make_textured_frame(w * 2, h * 2, seed=42)  # larger canvas to pan over
    rng = np.random.default_rng(7)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    assert writer.isOpened(), "could not open synthetic VideoWriter"

    shakes = []
    cx0, cy0 = w // 2, h // 2
    for i in range(n_frames):
        # high-frequency jitter (the thing we want to remove) + tiny slow drift
        dx = shake_px * np.sin(i * 1.7) + rng.normal(0, 1.5)
        dy = shake_px * np.cos(i * 2.1) + rng.normal(0, 1.5)
        shakes.append((dx, dy))
        x = int(cx0 + dx)
        y = int(cy0 + dy)
        crop = scene[y:y + h, x:x + w]
        writer.write(crop)
    writer.release()
    return np.array(shakes)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def measure_shakiness(path: str) -> float:
    """
    Estimate residual camera shake as the std-dev of frame-to-frame
    translation magnitude, measured with phase correlation.
    """
    mags = []
    with VideoReader(path) as r:
        prev = r.read()
        prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
        while True:
            f = r.read()
            if f is None:
                break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # crop to common center region (output may be smaller after crop)
            (dx, dy), _ = cv2.phaseCorrelate(prev_g, g)
            mags.append(np.hypot(dx, dy))
            prev_g = g
    return float(np.std(mags))


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #

def test_homography_roundtrip():
    """decompose(compose(p)) ≈ p for a known transform."""
    params = np.array([12.0, -7.0, 0.05, 0.02])
    H = compose_homography(params, 320, 240)
    back = decompose_homography(H)
    # tx/ty are recovered exactly only when rotation about origin; here rotation
    # is about centre so translation differs — check angle & scale precisely.
    assert abs(back[2] - params[2]) < 1e-6, "angle mismatch"
    assert abs(back[3] - params[3]) < 1e-6, "log_scale mismatch"
    print("  [ok] homography decompose/compose angle+scale roundtrip")


def test_smoother_reduces_variance():
    """Smoothed cumulative path must have lower variance than the raw path."""
    rng = np.random.default_rng(0)
    n = 100
    # slow drift + high-freq noise
    drift = np.cumsum(rng.normal(0, 0.1, n))
    noise = rng.normal(0, 5.0, n)
    raw = drift + noise
    traj = np.zeros((n, 4))
    traj[:, 0] = np.diff(raw, prepend=raw[0])  # tx increments

    cfg = Config()
    corr = smooth_trajectory(traj, cfg)
    cumulative = np.cumsum(traj, axis=0)
    smoothed = cumulative + corr

    raw_jerk = np.std(np.diff(cumulative[:, 0]))
    smooth_jerk = np.std(np.diff(smoothed[:, 0]))
    assert smooth_jerk < raw_jerk, f"smoother did not reduce jerk ({smooth_jerk} !< {raw_jerk})"
    print(f"  [ok] smoother reduced jerk std {raw_jerk:.2f} -> {smooth_jerk:.2f}")


def test_crop_region_within_bounds():
    cfg = Config()
    corr = np.zeros((10, 4))
    corr[:, 0] = np.linspace(-15, 15, 10)  # tx wobble
    corr[:, 1] = np.linspace(10, -10, 10)
    x, y, w, h = compute_warp_crop_region(corr, 320, 240, cfg)
    assert 0 <= x and 0 <= y
    assert x + w <= 320 and y + h <= 240
    assert w % 2 == 0 and h % 2 == 0
    assert w > 0 and h > 0
    print(f"  [ok] crop region ({x},{y},{w},{h}) within bounds & even")


def test_empty_trajectory_no_crash():
    cfg = Config()
    empty = np.zeros((0, 4))
    corr = smooth_trajectory(empty, cfg)
    assert corr.shape == (0, 4)
    region = compute_warp_crop_region(empty, 320, 240, cfg)
    assert region == (0, 0, 320, 240)
    print("  [ok] empty trajectory handled without crash")


# --------------------------------------------------------------------------- #
# End-to-end test
# --------------------------------------------------------------------------- #

def test_end_to_end_reduces_shake():
    tmp = tempfile.mkdtemp(prefix="stab_test_")
    in_path = os.path.join(tmp, "shaky.mp4")
    out_path = os.path.join(tmp, "stable.mp4")

    make_shaky_video(in_path, n_frames=40, shake_px=8.0)
    assert os.path.exists(in_path)

    cfg = Config(
        use_akaze=True,
        skip_vibration_filter=True,   # 30fps Nyquist is below 100Hz anyway
        use_mesh_warp=True,
        auto_crop=True,
        process_noise_q=1e-4,
        measurement_noise_r=1.0,      # smooth aggressively for the test
        velocity_pan_threshold=1e9,   # disable pan preservation for the test
    )
    run(cfg, in_path, out_path)

    assert os.path.exists(out_path), "output video was not created"

    before = measure_shakiness(in_path)
    after = measure_shakiness(out_path)
    print(f"  shakiness: before={before:.3f}  after={after:.3f}")
    assert after < before, f"stabilization did not reduce shake ({after} !< {before})"
    print("  [ok] end-to-end stabilization reduced measured shake")


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [
        test_homography_roundtrip,
        test_smoother_reduces_variance,
        test_crop_region_within_bounds,
        test_empty_trajectory_no_crash,
        test_end_to_end_reduces_shake,
    ]
    failures = 0
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"  [FAIL] {e}")
        except Exception as e:
            failures += 1
            print(f"  [ERROR] {type(e).__name__}: {e}")
    print(f"\n{'='*50}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
