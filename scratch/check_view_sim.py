"""Simulator check (127.0.0.1, model m v1.2) of the post-Step 2 check-pose view (sequences/verify.py).

The simulated camera follows simulation.yaml's truth. Bracket errors are zeroed in a copied config so the model's
brackets match it. Case A hands the verification the true offsets (gap ~ 0); case B makes the left J0 0.7 deg too
large, which tilts the check pose by about 4 mm (2026-11-24 field case). The simulated sensor does not model the
field of view, so this checks the motion and the numbers, not whether a real camera sees the markers there.
usage: python scratch/check_view_sim.py <scratch dir>
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])
if (OUT / "config").exists():
    shutil.rmtree(OUT / "config")
shutil.copytree(ROOT / "config", OUT / "config")
(OUT / "config" / "home_reset_baseline.json").unlink(missing_ok=True)
truth_path = OUT / "config" / "simulation.yaml"
truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
for side in ("right", "left"):
    truth["offsets"][side]["bracket_pos"] = [0.0, 0.0, 0.0]
    truth["offsets"][side]["bracket_rpy"] = [0.0, 0.0, 0.0]
truth_path.write_text(yaml.safe_dump(truth, sort_keys=False), encoding="utf-8")

from core import storage  # noqa: E402
storage.StoragePaths.root = OUT
for key, value in storage.CONFIG_PATHS.items():
    storage.CONFIG_PATHS[key] = str(OUT / Path(value).relative_to(ROOT))

from threading import Event  # noqa: E402
from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.sequences import verify  # noqa: E402
from core.calibration.sequences.result import SequenceContext  # noqa: E402
from core.calibration.sequences.run_warnings import stage_warnings  # noqa: E402

logs = []
core = CalibrationCore()
core.on_event = lambda kind, value: logs.append(str(value)) if kind == "log" else None
core.connect_robot("127.0.0.1:50051", "m", include_head=True, simulated=True)
core.bind_robot(core.robot, "1.2")
core.connect_camera(sim=True, robot=core.robot, robot_version="1.2")
core.include_head_motion = True
for side in ("right", "left"):
    core.marker_calibrator.camera_config[f"Tf_to_marker_{side}"] = list(truth["brackets"]["1.2"][side])


def true_offsets(side):
    values = truth["offsets"][side]
    return [values[f"joint{i}" if i != 5 else "joint5_v12"] for i in range(7)]


def case(title, left_j0_error=0.0):
    left = true_offsets("left")
    left[0] += left_j0_error
    optimization = {"right_arm_joint_offset_deg": true_offsets("right"), "left_arm_joint_offset_deg": left,
                    "head_joint_offset_deg": [truth["offsets"]["head"]["pan"], truth["offsets"]["head"]["tilt"]],
                    "mount_to_cam_new": list(truth["mount_to_cam"])}
    ctx = SequenceContext("step2", Event(), None)
    logs.clear()
    result = verify.run_post_step2_verification(core, ctx, optimization=optimization)
    view = result["check_pose_view"]
    print(f"--- {title}", flush=True)
    print("\n".join(line for line in logs if "check pose" in line.lower() or "check-pose" in line.lower()
                    or "flanges, as measured" in line or "model vs camera" in line), flush=True)
    print("run-line warnings:", stage_warnings("step2", {"optimization": optimization}, ctx.result.partial), flush=True)
    return view


case("A: true offsets")
case("B: left J0 +0.7 deg too large", left_j0_error=0.7)
case("A again: true offsets")
core.close()
