# 4K Drone Video Stabilization for 3-Axis Gimbal

A robust and efficient Python-based video stabilization pipeline designed specifically for 4K drone footage captured with 3-axis gimbals. It employs advanced motion estimation (AKAZE keypoints with Farneback optical flow fallback), high-frequency motor vibration filtering via FFT, mesh-based warping, and intelligent dynamic panning preservation.

## Features

- **Hybrid Motion Estimation**: Uses AKAZE feature tracking for sharp scenes and falls back to Farneback optical flow for low-texture areas.
- **Motor Vibration Filtering**: Employs an FFT-based notch filter to remove high-frequency micro-jitters introduced by drone motors (e.g., 100 Hz).
- **Dynamic Panning Preservation**: Intelligently preserves deliberate camera movements and panning by adaptive velocity thresholding.
- **Mesh-based Warping**: Corrects complex non-linear distortions across the frame, not just simple global shifts.
- **Rolling Shutter Correction**: Built-in support to mitigate rolling shutter artifacts common in CMOS sensors.
- **Auto-Cropping & Inpainting**: Options to either automatically crop the black borders resulting from stabilization or inpaint them.
- **High-Quality Output**: Native integration with FFmpeg for optimal H.264 encoding with configurable CRF settings.

## Installation

Ensure you have Python 3.9+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/Maverick01-code/4K-DRONE-VIDEO-STABILIZATION-FOR-3-AXIS-GIMBALL-.git
   cd 4K-DRONE-VIDEO-STABILIZATION-FOR-3-AXIS-GIMBALL-
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *(Note: For the `--ffmpeg` encoding option to work, you must have FFmpeg installed and accessible in your system's PATH).*

## Usage

The main entry point for the stabilizer is `stabilizer.py`.

### Basic Usage

```bash
python stabilizer.py input_video.mp4 output_video.mp4
```

### Advanced Usage with FFmpeg Encoding

```bash
python stabilizer.py input_video.mp4 output_video.mp4 --ffmpeg --crf 18
```

### Available Options

```text
positional arguments:
  input                 Input video path
  output                Output video path

optional arguments:
  -h, --help            show this help message and exit
  --no-akaze            Force Farneback optical-flow (skips AKAZE entirely)
  --low-texture-threshold MIN
                        Min AKAZE keypoints before falling back to Farneback (default: 20)
  --motor-freq MOTOR_FREQ
                        Fundamental motor vibration frequency in Hz (default: 100.0)
  --no-vibration-filter
                        Skip FFT vibration notch filter (default: False)
  --pan-threshold PAN_THRESHOLD
                        Velocity threshold (px/frame) above which pans are preserved (default: 3.0)
  --rs-factor RS_FACTOR
                        Rolling-shutter correction strength [0-1] (default: 0.0)
  --no-mesh             Use single warpPerspective instead of 16x9 mesh warp
  --mesh-cols MESH_COLS
  --mesh-rows MESH_ROWS
  --no-crop             Disable auto-crop (keep full warped frame)
  --inpaint             Inpaint black borders instead of cropping (default: False)
  --ffmpeg              Encode output via FFmpeg (H.264, requires ffmpeg on PATH)
  --crf CRF             FFmpeg CRF quality (lower = better) (default: 18)
  --codec CODEC         OpenCV FourCC codec (no-ffmpeg mode) (default: mp4v)
  --debug               Verbose progress output (default: False)
  --preview             Show live preview window (requires display) (default: False)
  --save-plot           Save trajectory comparison plot as trajectory.png (default: False)
```

## How It Works

1. **Detection & Estimation**: The script calculates the frame-to-frame transformation using AKAZE and Farneback optical flow.
2. **Smoothing & Filtering**: The trajectory is smoothed using a sliding window Gaussian filter, and the high-frequency motor vibrations are removed using an FFT notch filter.
3. **Warping**: The current frame is mapped to the smoothed trajectory, correcting for rolling shutter and applying a grid mesh warp to correct complex distortions.
4. **Cropping/Inpainting**: The resulting black edges are removed dynamically by cropping or filled by OpenCV's inpainting algorithm.
5. **Encoding**: Finally, the stabilized frame is encoded back to an MP4 container.