from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # --- detector ---
    use_akaze: bool = True
    akaze_threshold: float = 0.001
    min_match_count: int = 20
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2

    # --- estimator ---
    ransac_reproj_threshold: float = 3.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.995

    # --- smoother (Kalman) ---
    process_noise_q: float = 1e-4
    measurement_noise_r: float = 1e-1
    # number of DOF smoothed: tx, ty, rotation, scale
    state_dim: int = 4

    # --- vibration filter ---
    skip_vibration_filter: bool = False # bypass FFT notch entirely
    motor_freq_hz: float = 100.0        # fundamental motor vibration frequency
    notch_bandwidth_hz: float = 5.0     # width of each notch
    notch_harmonics: int = 3            # how many harmonics to kill

    # --- warper ---
    rolling_shutter_factor: float = 0.0  # 0 = off, 1 = full RS correction
    border_mode: str = "reflect"         # cv2 border mode name

    # --- cropper ---
    auto_crop: bool = True
    inpaint_borders: bool = False
    inpaint_radius: int = 3
    crop_margin_pct: float = 0.05       # extra safety margin after auto-crop

    # --- video I/O ---
    output_fps: float = 0.0             # 0 = preserve source fps
    output_codec: str = "mp4v"
    output_crf: int = 18                # quality hint (used when codec supports it)

    # --- mesh warp ---
    use_mesh_warp: bool = True           # 16×9 mesh-based warping (flowchart spec)
    mesh_cols: int = 16
    mesh_rows: int = 9

    # --- low-texture / Farneback fallback ---
    low_texture_kp_threshold: int = 20   # auto-fall back to Farneback below this

    # --- smoother: pan preservation ---
    velocity_pan_threshold: float = 3.0  # px/frame — preserve pans above this speed

    # --- FFmpeg output ---
    use_ffmpeg: bool = False             # pipe raw frames to ffmpeg for H.264 output
    ffmpeg_preset: str = "slow"
    ffmpeg_crf: int = 18

    # --- debug ---
    debug: bool = False
    preview: bool = False
    save_trajectory_plot: bool = False
