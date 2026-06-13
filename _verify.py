import sys, io
sys.path.insert(0, "tests")
from contextlib import redirect_stdout
from test_pipeline import measure_shakiness as m
from drone_stabilizer.config import Config
from drone_stabilizer.main import run

IN = "shaky_drone.mp4"
b = m(IN)
print(f"before          = {b:.2f}")

for label, kw in [
    ("default (rs=0)", dict()),
    ("rs-factor 0.4 ", dict(rolling_shutter_factor=0.4)),
    ("rs-factor 1.0 ", dict(rolling_shutter_factor=1.0)),
]:
    cfg = Config(use_akaze=True, skip_vibration_filter=True, **kw)
    with redirect_stdout(io.StringIO()):
        run(cfg, IN, "_v.mp4")
    a = m("_v.mp4")
    print(f"{label}  = {a:6.2f}  ({100*(1-a/b):+.0f}%)")
