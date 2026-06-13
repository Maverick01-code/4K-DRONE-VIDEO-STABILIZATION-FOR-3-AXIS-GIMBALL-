import sys
sys.path.insert(0, "tests")
from test_pipeline import measure_shakiness
from drone_stabilizer.config import Config
from drone_stabilizer.main import run
import io
from contextlib import redirect_stdout

IN = "shaky_drone.mp4"
b = measure_shakiness(IN)
print(f"before = {b:.2f}\n")

combos = [
    dict(measurement_noise_r=0.1, velocity_pan_threshold=3.0),
    dict(measurement_noise_r=0.1, velocity_pan_threshold=1e9),
    dict(measurement_noise_r=1.0, velocity_pan_threshold=3.0),
    dict(measurement_noise_r=1.0, velocity_pan_threshold=1e9),
    dict(measurement_noise_r=5.0, velocity_pan_threshold=3.0),
    dict(measurement_noise_r=5.0, velocity_pan_threshold=1e9),
]

for c in combos:
    cfg = Config(use_akaze=True, skip_vibration_filter=True, **c)
    with redirect_stdout(io.StringIO()):
        run(cfg, IN, "_t.mp4")
    a = measure_shakiness("_t.mp4")
    print(f"r={c['measurement_noise_r']:<4} pan_thr={c['velocity_pan_threshold']:<7} "
          f"after={a:6.2f}  reduction={100*(1-a/b):+.0f}%")
