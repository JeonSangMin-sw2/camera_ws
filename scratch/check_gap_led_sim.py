"""Simulator check (127.0.0.1 only) of the wizard's held gather / spread and the calibration LED calls.

Check pose with zero offsets -> gather, cancelled after 3 s (a released button) -> gather again to the end ->
spread back; prints the ee gap (link_torso_5 y) and timing after each, then the three LED states.
usage: python scratch/check_gap_led_sim.py <scratch dir>
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.calibration import CalibrationCore  # noqa: E402
from core.robot.motion import compute_fk  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
zero = OUT / "zero_offsets.json"
zero.write_text(json.dumps({"joint_offset_deg": [0.0] * 14, "head_joint_offset_deg": [0.0, 0.0]}), encoding="utf-8")

logs = []
core = CalibrationCore()
core.on_event = lambda kind, value: logs.append(str(value)) if kind == "log" else None
core.connect_robot("127.0.0.1:50051", "m", include_head=True, simulated=True)
core.bind_robot(core.robot, "1.2")
dyn = core.robot.get_dynamics()


def gap_mm():
    q = np.array(core.robot.get_state().position)
    return round((compute_fk(core.robot, dyn, q, "ee_left")[1][1, 3] - compute_fk(core.robot, dyn, q, "ee_right")[1][1, 3]) * 1000, 1)


def run(title, cancel_after=None, **options):
    timer = threading.Timer(cancel_after, core.cancel) if cancel_after else None
    start = time.time()
    if timer:
        timer.start()
    result = core.run("home", **options)
    if timer:
        timer.cancel()
    print(f"{title:<28} status={result.status:<10} {time.time() - start:5.1f} s  gap={gap_mm()} mm  {result.error[:60]}", flush=True)


run("check pose", task_type="move_check", json_path=str(zero), label="Check", arm="both", include_head=True)
run("gather, let go after 3 s", cancel_after=3.0, task_type="check_gap", json_path=str(zero), label="Gather", direction="gather")
run("gather again (carry on)", task_type="check_gap", json_path=str(zero), label="Gather", direction="gather")
run("gather once more", task_type="check_gap", json_path=str(zero), label="Gather", direction="gather")
run("spread back", task_type="check_gap", json_path=str(zero), label="Spread", direction="spread")
for state in ("off", "alert", "release"):
    logs.clear()
    core._set_led(state)
    print(f"LED {state:<8}", logs or "accepted", flush=True)
core.close()
