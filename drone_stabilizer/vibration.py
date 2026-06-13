from __future__ import annotations

import numpy as np
from .config import Config


def remove_motor_vibration(trajectory: np.ndarray, fps: float, cfg: Config) -> np.ndarray:
    """
    FFT-based notch filter suppressing the motor vibration frequency band
    and its harmonics.  Operates in the frequency domain on each DOF column.

    For typical 4K drone footage at 30 fps the Nyquist is 15 Hz, so the
    default 80-120 Hz notch (above Nyquist) has no effect — pass a high-fps
    source or adjust motor_freq_hz accordingly.  For standard footage the
    filter gracefully no-ops for out-of-band frequencies.
    """
    result = trajectory.copy()
    N = len(result)
    nyq = fps / 2.0

    # Build frequency axis for rfft (only positive freqs)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    half_bw = cfg.notch_bandwidth_hz / 2.0

    for col in range(result.shape[1]):
        spectrum = np.fft.rfft(result[:, col])

        for h in range(1, cfg.notch_harmonics + 1):
            f_center = cfg.motor_freq_hz * h
            f_low = f_center - half_bw
            f_high = f_center + half_bw
            if f_low >= nyq:
                break  # harmonic above Nyquist — nothing to suppress
            f_high = min(f_high, nyq)
            notch_mask = (freqs >= f_low) & (freqs <= f_high)
            spectrum[notch_mask] = 0.0

        result[:, col] = np.fft.irfft(spectrum, n=N)

    return result
