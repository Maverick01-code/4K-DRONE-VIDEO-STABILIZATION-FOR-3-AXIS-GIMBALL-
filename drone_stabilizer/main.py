"""
drone_stabilizer – CLI entry point.

Usage:
  python -m drone_stabilizer.main input.mp4 output.mp4 [options]
  python stabilizer.py input.mp4 output.mp4 [options]

Run with --help for all options.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .cropper import compute_warp_crop_region, auto_crop_region, finalize_frame
from .estimator import estimate_trajectory
from .smoother import smooth_trajectory
from .utils import draw_trajectory
from .vibration import remove_motor_vibration
from .video_io import VideoReader, VideoWriter
from .warper import warp_frame


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4K drone video stabilizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input video path")
    p.add_argument("output", help="Output video path")

    # detector
    p.add_argument("--no-akaze", dest="use_akaze", action="store_false",
                   help="Force Farneback optical-flow (skips AKAZE entirely)")
    p.add_argument("--low-texture-threshold", type=int, default=20,
                   help="Min AKAZE keypoints before falling back to Farneback")

    # vibration filter
    p.add_argument("--motor-freq", type=float, default=100.0,
                   help="Fundamental motor vibration frequency in Hz")
    p.add_argument("--no-vibration-filter", action="store_true",
                   help="Skip FFT vibration notch filter")

    # smoother
    p.add_argument("--pan-threshold", type=float, default=3.0,
                   help="Velocity threshold (px/frame) above which pans are preserved")

    # warper
    p.add_argument("--rs-factor", type=float, default=0.0,
                   help="Rolling-shutter correction strength [0-1]")
    p.add_argument("--no-mesh", dest="use_mesh_warp", action="store_false",
                   help="Use single warpPerspective instead of 16x9 mesh warp")
    p.add_argument("--mesh-cols", type=int, default=16)
    p.add_argument("--mesh-rows", type=int, default=9)

    # crop
    p.add_argument("--no-crop", dest="auto_crop", action="store_false",
                   help="Disable auto-crop (keep full warped frame)")
    p.add_argument("--inpaint", dest="inpaint_borders", action="store_true",
                   help="Inpaint black borders instead of cropping")

    # output
    p.add_argument("--ffmpeg", dest="use_ffmpeg", action="store_true",
                   help="Encode output via FFmpeg (H.264, requires ffmpeg on PATH)")
    p.add_argument("--crf", type=int, default=18,
                   help="FFmpeg CRF quality (lower = better, used with --ffmpeg)")
    p.add_argument("--codec", default="mp4v", help="OpenCV FourCC codec (no-ffmpeg mode)")

    # debug
    p.add_argument("--debug", action="store_true", help="Verbose progress output")
    p.add_argument("--preview", action="store_true",
                   help="Show live preview window (requires display)")
    p.add_argument("--save-plot", action="store_true",
                   help="Save trajectory comparison plot as trajectory.png")

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        use_akaze=args.use_akaze,
        low_texture_kp_threshold=args.low_texture_threshold,
        skip_vibration_filter=args.no_vibration_filter,
        motor_freq_hz=args.motor_freq,
        rolling_shutter_factor=args.rs_factor,
        use_mesh_warp=args.use_mesh_warp,
        mesh_cols=args.mesh_cols,
        mesh_rows=args.mesh_rows,
        velocity_pan_threshold=args.pan_threshold,
        auto_crop=args.auto_crop,
        inpaint_borders=args.inpaint_borders,
        use_ffmpeg=args.use_ffmpeg,
        ffmpeg_crf=args.crf,
        output_codec=args.codec,
        debug=args.debug,
        preview=args.preview,
        save_trajectory_plot=args.save_plot,
    )


def run(cfg: Config, input_path: str, output_path: str) -> None:
    # ------------------------------------------------------------------ Pass 1
    print(f"[1/5] Pass 1 - trajectory estimation: {input_path}")
    with VideoReader(input_path) as reader:
        fps = reader.fps
        W, H = reader.size
        frame_count = reader.frame_count
        print(f"      {W}x{H}  {fps:.2f} fps  {frame_count} frames")
        trajectory, low_texture_frames = estimate_trajectory(reader, cfg)

    if low_texture_frames:
        pct = 100 * len(low_texture_frames) / max(len(trajectory), 1)
        print(f"      Farneback fallback used on {len(low_texture_frames)} frames ({pct:.1f}%)")

    # --------------------------------------------------------- Vibration filter
    if cfg.skip_vibration_filter:
        print("[2/5] FFT notch filter - skipped (--no-vibration-filter)")
    else:
        print(f"[2/5] FFT notch filter @ {cfg.motor_freq_hz} Hz "
              f"(+/-{cfg.notch_bandwidth_hz/2:.1f} Hz x {cfg.notch_harmonics} harmonics)")
        trajectory = remove_motor_vibration(trajectory, fps, cfg)

    # ------------------------------------------------------------------ Pass 2
    print("[3/5] Pass 2 - Kalman RTS smoothing + correction matrices")
    corrections = smooth_trajectory(trajectory, cfg)  # (N-1, 4)

    # Frame-align corrections: trajectory[i] is the motion BETWEEN frame i and
    # i+1, so its correction belongs to frame i+1.  Frame 0 is the reference
    # and gets a zero correction.  Result length == frame count.
    corr_aligned = np.vstack([np.zeros((1, 4), dtype=np.float64), corrections])

    if cfg.save_trajectory_plot:
        plot = draw_trajectory(trajectory, corrections, col=0, title="tx trajectory")
        cv2.imwrite("trajectory.png", plot)
        print("      saved trajectory.png")

    # ------------------------------------------------------ Global crop region
    crop_region = None
    out_size = (W, H)
    if cfg.auto_crop:
        print("[4a]  Computing global crop from warp extent ...")
        crop_region = compute_warp_crop_region(corr_aligned, W, H, cfg)
        cx, cy, cw, ch = crop_region
        border_pct = ((cx / W + cy / H) / 2) * 100
        print(f"      crop: ({cx},{cy}) -> {cw}x{ch}  (~{border_pct:.1f}% border)")
        out_size = (cw, ch)

    # ------------------------------------------------------ Warp + write frames
    print(f"[4/5] Warping + encoding -> {output_path}  ({out_size[0]}x{out_size[1]} @ {fps:.2f} fps)")

    if cfg.use_ffmpeg:
        _render_ffmpeg(input_path, output_path, corr_aligned, crop_region,
                       cfg, fps, out_size)
    else:
        _render_opencv(input_path, output_path, corr_aligned, crop_region,
                       cfg, fps, out_size)

    print("Done.")


# ------------------------------------------------------------------ renderers

def _render_opencv(
    input_path: str,
    output_path: str,
    corrections: np.ndarray,
    crop_region,
    cfg: Config,
    fps: float,
    out_size: tuple,
) -> None:
    with VideoReader(input_path) as reader, \
         VideoWriter(output_path, fps, out_size, cfg.output_codec) as writer:
        prev_corr = None
        for i, frame in enumerate(reader.frames()):
            corr = corrections[i] if i < len(corrections) else np.zeros(4)
            warped = warp_frame(frame, corr, cfg, prev_correction=prev_corr)
            final = finalize_frame(warped, crop_region, cfg)
            writer.write(final)
            prev_corr = corr

            if cfg.preview:
                preview = cv2.resize(final, (final.shape[1] // 2, final.shape[0] // 2))
                cv2.imshow("stabilized", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if cfg.debug and i % 100 == 0:
                print(f"      rendered {i}/{len(corrections)}")

    if cfg.preview:
        cv2.destroyAllWindows()


def _render_ffmpeg(
    input_path: str,
    output_path: str,
    corrections: np.ndarray,
    crop_region,
    cfg: Config,
    fps: float,
    out_size: tuple,
) -> None:
    cw, ch = out_size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{cw}x{ch}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_path,        # source for audio track
        "-map", "0:v",
        "-map", "1:a?",          # copy audio if present
        "-c:v", "libx264",
        "-preset", cfg.ffmpeg_preset,
        "-crf", str(cfg.ffmpeg_crf),
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    with VideoReader(input_path) as reader:
        prev_corr = None
        for i, frame in enumerate(reader.frames()):
            corr = corrections[i] if i < len(corrections) else np.zeros(4)
            warped = warp_frame(frame, corr, cfg, prev_correction=prev_corr)
            final = finalize_frame(warped, crop_region, cfg)
            proc.stdin.write(final.tobytes())
            prev_corr = corr
            if cfg.debug and i % 100 == 0:
                print(f"      rendered {i}/{len(corrections)}")

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret}")


# ------------------------------------------------------------------ CLI entry

def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(cfg, args.input, args.output)


if __name__ == "__main__":
    main()
