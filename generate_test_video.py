"""
Generate a realistic-looking shaky aerial "drone" test video.

Renders a procedural top-down cityscape (road grid, buildings, parks, cars,
lane markings) onto a large canvas, then flies a virtual camera over it with
drone-style motion:

  * slow ambient drift
  * an intentional rightward PAN in the middle (to test pan preservation)
  * high-frequency propeller-style jitter (translation)
  * small rotational wobble
  * random gusts

Output is texture-rich so AKAZE finds plenty of keypoints.

Usage:
    python generate_test_video.py                       # defaults
    python generate_test_video.py out.mp4 --seconds 8 --w 1280 --h 720
"""
from __future__ import annotations

import argparse
import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Procedural cityscape
# --------------------------------------------------------------------------- #

def build_city(cw: int, ch: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((ch, cw, 3), (60, 75, 60), np.uint8)  # ground / vegetation base

    # speckled ground texture (helps optical flow on "empty" areas)
    noise = rng.integers(-12, 12, size=(ch, cw, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    block = 220          # city block size in px
    road_w = 46          # road width

    # --- city blocks (buildings / rooftops) ---
    for by in range(0, ch, block):
        for bx in range(0, cw, block):
            x0 = bx + road_w // 2
            y0 = by + road_w // 2
            x1 = bx + block - road_w // 2
            y1 = by + block - road_w // 2
            if x1 <= x0 or y1 <= y0:
                continue

            # subdivide the block into a few buildings / lots
            n = rng.integers(1, 4)
            xs = np.sort(rng.integers(x0, x1, size=n - 1)) if n > 1 else np.array([], int)
            edges = [x0, *xs.tolist(), x1]
            for i in range(len(edges) - 1):
                rx0, rx1 = edges[i] + 3, edges[i + 1] - 3
                if rx1 <= rx0:
                    continue
                kind = rng.random()
                if kind < 0.18:
                    # park / greenery
                    color = (int(rng.integers(40, 70)), int(rng.integers(110, 160)),
                             int(rng.integers(40, 70)))
                    cv2.rectangle(img, (rx0, y0), (rx1, y1), color, -1)
                    for _ in range(rng.integers(3, 9)):
                        tx = rng.integers(rx0, rx1)
                        ty = rng.integers(y0, y1)
                        cv2.circle(img, (int(tx), int(ty)),
                                   int(rng.integers(5, 14)),
                                   (30, int(rng.integers(80, 130)), 30), -1)
                else:
                    # rooftop
                    base = int(rng.integers(90, 200))
                    color = (base + int(rng.integers(-20, 20)),
                             base + int(rng.integers(-20, 20)),
                             base + int(rng.integers(-20, 20)))
                    color = tuple(int(np.clip(c, 0, 255)) for c in color)
                    cv2.rectangle(img, (rx0, y0), (rx1, y1), color, -1)
                    # rooftop detail: vents, edges, AC units
                    cv2.rectangle(img, (rx0, y0), (rx1, y1),
                                  tuple(int(c * 0.6) for c in color), 2)
                    for _ in range(rng.integers(2, 6)):
                        ux = rng.integers(rx0, max(rx0 + 1, rx1 - 10))
                        uy = rng.integers(y0, max(y0 + 1, y1 - 10))
                        cv2.rectangle(img, (int(ux), int(uy)),
                                      (int(ux + rng.integers(6, 18)),
                                       int(uy + rng.integers(6, 18))),
                                      tuple(int(np.clip(c + rng.integers(-40, 40), 0, 255))
                                            for c in color), -1)

    # --- roads (drawn over the grid gaps) + lane markings + cars ---
    for y in range(0, ch, block):
        cv2.rectangle(img, (0, y - road_w // 2), (cw, y + road_w // 2), (45, 45, 48), -1)
    for x in range(0, cw, block):
        cv2.rectangle(img, (x - road_w // 2, 0), (x + road_w // 2, ch), (45, 45, 48), -1)

    # dashed lane markings
    for y in range(0, ch, block):
        for x in range(0, cw, 40):
            cv2.line(img, (x, y), (x + 22, y), (210, 210, 180), 2)
    for x in range(0, cw, block):
        for y in range(0, ch, 40):
            cv2.line(img, (x, y), (x, y + 22), (210, 210, 180), 2)

    # cars (small bright rectangles on the roads)
    car_colors = [(0, 0, 200), (200, 0, 0), (220, 220, 220), (30, 30, 30),
                  (0, 160, 220), (200, 200, 0)]
    for _ in range(int(cw * ch / 9000)):
        if rng.random() < 0.5:           # on horizontal road
            y = int(rng.integers(0, ch // block + 1) * block + rng.integers(-12, 12))
            x = int(rng.integers(0, cw))
            cv2.rectangle(img, (x, y - 6), (x + 16, y + 6),
                          car_colors[rng.integers(0, len(car_colors))], -1)
        else:                             # on vertical road
            x = int(rng.integers(0, cw // block + 1) * block + rng.integers(-12, 12))
            y = int(rng.integers(0, ch))
            cv2.rectangle(img, (x - 6, y), (x + 6, y + 16),
                          car_colors[rng.integers(0, len(car_colors))], -1)

    return img


# --------------------------------------------------------------------------- #
# Camera motion
# --------------------------------------------------------------------------- #

def camera_path(n: int, rng: np.random.Generator):
    """Yield (cx, cy, angle_deg) per frame: drift + pan + jitter + wobble."""
    t = np.arange(n)

    # slow ambient drift (smooth, low-freq)
    drift_x = 40 * np.sin(t * 0.015) + 18 * np.sin(t * 0.041 + 1.0)
    drift_y = 30 * np.cos(t * 0.012) + 14 * np.sin(t * 0.033 + 0.5)

    # intentional pan to the right, starting at the midpoint
    pan = np.zeros(n)
    mid = n // 2
    pan[mid:] = np.cumsum(np.full(n - mid, 3.2))   # steady rightward velocity

    # high-frequency propeller-style jitter
    jit_x = (5.0 * np.sin(t * 1.9) + 3.0 * np.sin(t * 3.3 + 0.7)
             + rng.normal(0, 1.6, n))
    jit_y = (5.0 * np.cos(t * 2.2) + 3.0 * np.sin(t * 3.9 + 1.1)
             + rng.normal(0, 1.6, n))

    # occasional gusts
    gust = np.zeros(n)
    for _ in range(max(1, n // 50)):
        g = rng.integers(0, n)
        gust[g:g + 6] += rng.normal(0, 10)

    cx = drift_x + pan + jit_x + gust
    cy = drift_y + jit_y

    # rotational wobble (degrees): slow sway + jitter
    angle = 1.4 * np.sin(t * 0.05) + 0.8 * np.sin(t * 2.7) + rng.normal(0, 0.3, n)

    return cx, cy, angle


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

def render(out_path: str, w: int, h: int, fps: float, seconds: float, seed: int):
    n = int(round(fps * seconds))

    # canvas large enough that the camera never leaves it, with margin for shake
    margin = 260
    cw = w + margin * 2 + 700   # extra width for the pan
    ch = h + margin * 2
    print(f"Building {cw}x{ch} cityscape ...")
    city = build_city(cw, ch, seed)

    rng = np.random.default_rng(seed + 1)
    cx, cy, angle = camera_path(n, rng)

    # center the nominal camera on the canvas
    base_x = (cw - w) / 2 - 300     # start left so the pan moves into new area
    base_y = (ch - h) / 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {out_path}")

    print(f"Rendering {n} frames @ {fps} fps ({seconds}s) -> {out_path}")
    for i in range(n):
        # Top-left of the crop in canvas coords for this frame
        tx = base_x + cx[i]
        ty = base_y + cy[i]

        # Affine that maps canvas -> frame: rotate about the crop centre then
        # translate so (tx,ty) lands at the frame origin.
        center = (tx + w / 2, ty + h / 2)
        M = cv2.getRotationMatrix2D(center, angle[i], 1.0)
        # shift so the rotated crop centre maps to the frame centre
        M[0, 2] += w / 2 - center[0]
        M[1, 2] += h / 2 - center[1]

        frame = cv2.warpAffine(city, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
        if i % 30 == 0:
            print(f"  frame {i}/{n}")

    writer.release()
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="Generate a shaky aerial test video")
    p.add_argument("output", nargs="?", default="shaky_drone.mp4")
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    render(args.output, args.w, args.h, args.fps, args.seconds, args.seed)


if __name__ == "__main__":
    main()
