"""Step 2 of a v1.2 head run solved offline with the camera angles Step 1.5 now gives in the range mode.

2026-10-16: Step 1.5 estimates the optical-axis rotation in the design-range mode again, and Step 2 holds
the camera at those angles. This re-solves Step 2 of a saved run (scratch/step2_rerun_20261008_1750.py) twice:
as run, and with the head offsets and camera angles of scratch/head_camera_roll_check.py's "design range"
solve. With --design-x the design camera x of the range is changed too (run head_camera_roll_check.py with
the same --design-x first). Offline; outputs in archive/reports/step2_rerun_<date_time>/.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\step2_with_step15_roll.py <run folder> <timestamp>
         "<roll-check run name>" [--design-x 0.045]
"""
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
design_x = float(sys.argv[sys.argv.index("--design-x") + 1]) if "--design-x" in sys.argv else None
run_name = sys.argv[3]
sys.argv = sys.argv[:3]
m = runpy.run_path(str(ROOT / "scratch" / "step2_rerun_20261008_1750.py"), run_name="rerun")
suffix = "" if design_x is None else f"_x{design_x * 1000:.1f}"
roll = json.loads((ROOT / f"archive/reports/head_camera_roll_check_20261016{suffix}/summary.json").read_text(encoding="utf-8"))
step15 = roll[run_name]["design range"]
m["row"]("as_run", m["solve"]("as_run"))
m["HEAD_STEP15"].update({k: float(v) for k, v in step15["head_offsets_deg"].items()})
m["CAM_STEP15"][:] = step15["mount_to_cam"]
if design_x is not None:
    m["DESIGN"][0] = design_x
m["row"]("step15_roll" + suffix, m["solve"]("step15_roll" + suffix))
