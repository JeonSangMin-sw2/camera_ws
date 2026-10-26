"""Does a marker bracket roll other than the design 90 deg fit a saved v1.2 Step 2 run better?

v1.2 holds the bracket roll at its design angle (setting marker.lock_bracket_roll). 2026-10-19: after
applying, the right hand at the check pose looked rolled about the robot x axis (z off 2-3 mm) and the right
J0 off, an error the run's Step 2 data reject when it is put on the joints (scratch/check_pose_error_cost.py).
This re-solves the run's Step 2 (scratch/step2_rerun.py) with one bracket's roll moved by a
few tenths of a degree and prints the arm offsets and the noise-weighted residual. Offline, nothing applied.
usage: .venv\Scripts\python.exe -X utf8 scratch\bracket_roll_scan.py <run folder> <timestamp> [right|left]
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
side = sys.argv[3] if len(sys.argv) > 3 else "right"
sys.argv = sys.argv[:3]
m = runpy.run_path(str(ROOT / "scratch" / "step2_rerun.py"), run_name="rerun")
bracket = m["SAVED"]["ee_to_marker_used"][side]
design_roll = bracket[3]
for delta in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
    bracket[3] = design_roll + delta
    m["row"](f"{side} roll{delta:+.1f}", m["solve"](f"{side}_roll{delta:+.1f}"))
bracket[3] = design_roll
