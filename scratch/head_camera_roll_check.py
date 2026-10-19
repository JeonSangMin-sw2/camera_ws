"""Is the camera of the 2026-10-16 v1.2 robot turned about its optical axis, and were the earlier robots?

Step 2 with the camera angles free (scratch/step2_rerun_20261008_1750.py, old_mode) put the 19:11 camera at
-0.87 deg about the optical axis with half the position residual, and moved both J1 the same way. This solves
Step 1.5 (head tilt/pan sweeps, independent of the arms) of each v1.2 head run on the new bracket again with the
production solver, twice: the camera position held to its design range (the optical-axis rotation then stays at
the design value, as run) and free (rotation estimated, the mode before 2026-10-08). Design camera:
camera_info.yaml D405 now (z 63 mm). Offline, nothing is applied or written to config.
Outputs: archive/reports/head_camera_roll_check_20261016/.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\head_camera_roll_check.py [--design-x 0.045]
  --design-x: another design camera x (m) to try, same -3/+1 mm range; outputs get an _x<mm> suffix.
  --run "<name>": only that run (merged into the folder's summary.json).
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core.calibration  # noqa: E402,F401

head_module = sys.modules["core.calibration.HeadCameraCalibrator"]
DESIGN_X = float(sys.argv[sys.argv.index("--design-x") + 1]) if "--design-x" in sys.argv else 0.0433
OUT = ROOT / "archive" / "reports" / ("head_camera_roll_check_20261016" +
                                      ("" if DESIGN_X == 0.0433 else f"_x{DESIGN_X * 1000:.1f}"))
OUT.mkdir(parents=True, exist_ok=True)
URDF = ROOT / "archive" / "reports" / "step2_rerun_20261008" / "model_m_v1.2.urdf"
DESIGN = [DESIGN_X, 0.009, 0.063, -90.0, 0.0, -90.0]
RANGE_M = np.array([(-0.003, 0.001), (0.0, 0.0), (-0.0005, 0.0005)], dtype=float)
RUNS = {
    "10-08 09:51 robot3": ROOT / "archive/runs/20261008_robot3_head_newbracket",
    "10-08 17:50 robot": ROOT / "archive/runs/20261008_1750_newrobot_head/result_txt",
    "10-16 19:11 robot": ROOT / "archive/runs/20261016_1911_v12_robot/result_txt",
    "10-16 20:39 robot": ROOT / "archive/runs/20261016_2039_v12_robot_optical_roll/result_txt",
    "10-16 21:25 robot": ROOT / "archive/runs/20261016_2125_v12_robot_applied/result_txt",
}
if "--run" in sys.argv:
    RUNS = {name: RUNS[name] for name in [sys.argv[sys.argv.index("--run") + 1]]}
LINE = re.compile(r"(Tilt|Pan)=\s*([-+\d.]+).*?t=([\d.]+) side=(\w+) -> Marker Cam Pos: "
                  r"\[\s*([-+\d.]+),\s*([-+\d.]+),\s*([-+\d.]+)\]")


class OfflineRobot:
    def __init__(self, urdf):
        self._model = rby.Model_M()
        count = len(self._model.robot_joint_names)
        self._dynamics = getattr(rby.dynamics, f"Robot_{count}")(rby.dynamics.load_robot_from_urdf_data(urdf, "base"))
        self._state = SimpleNamespace(position=np.zeros(count))

    def model(self):
        return self._model

    def get_dynamics(self):
        return self._dynamics

    def get_state(self):
        return self._state


def load_phases(path):
    phases = {name: {"joint": joint, "name": name, "missing": 0, "samples": []} for name, joint in (("Tilt", 1), ("Pan", 0))}
    for line in open(path, encoding="utf-8"):
        match = LINE.search(line)
        if match:
            name, deg, t, side, x, y, z = match.groups()
            phases[name]["samples"].append({"t": float(t), "side": side, "deg": float(deg),
                                            "p": np.array([float(x), float(y), float(z)]) / 1000.0})
    for phase in phases.values():
        order = np.argsort([s["t"] for s in phase["samples"]])
        phase["enc_t"] = np.array([phase["samples"][i]["t"] for i in order])
        phase["enc_deg"] = np.array([phase["samples"][i]["deg"] for i in order])
    return [phases["Tilt"], phases["Pan"]]


robot = OfflineRobot(URDF.read_text(encoding="utf-8"))
r_nom = Rotation.from_euler("ZYX", [DESIGN[5], DESIGN[4], DESIGN[3]], degrees=True).as_matrix()
summary = {}
for name, folder in RUNS.items():
    saved = json.loads((folder / "head_camera_result_latest.json").read_text(encoding="utf-8"))
    priors = {side: (np.array(saved["marker_positions_t5_mm"][side]) - np.array(saved["marker_prior_shift_mm"][side])) / 1000.0
              for side in saved["markers_used"]}
    phases = load_phases(folder / "head_camera_sweep_debug.txt")
    summary[name] = {}
    for mode, position_range in (("design range", RANGE_M), ("free", None)):
        calibrator = head_module.HeadCameraCalibrator(None, robot)
        calibrator.robot_version = "1.2"
        with contextlib.redirect_stdout(io.StringIO()):
            res = calibrator._solve_head_camera(phases, list(DESIGN), r_nom, marker_priors=priors,
                                                position_range_m=position_range)
        cam = np.array(res["calibrated_mount_to_cam"], dtype=float)
        rot = Rotation.from_matrix(r_nom.T @ Rotation.from_euler("ZYX", cam[[5, 4, 3]], degrees=True).as_matrix())
        rotvec_deg = np.degrees(rot.as_rotvec())      # in the design camera frame: z is the optical axis
        quality = res["quality"]
        summary[name][mode] = {"mount_to_cam": cam.tolist(), "camera_rotation_deg_xyz_in_camera_frame": rotvec_deg.tolist(),
                               "head_offsets_deg": res["head_offsets_deg"], "rmse_3d_marker_mm": quality["rmse_3d_marker_mm"]}
        print(f"{name:20s} {mode:12s} | camera {np.round((cam[:3] - DESIGN[:3]) * 1000, 2)} mm from design | "
              f"rotation about camera x {rotvec_deg[0]:+.3f} y {rotvec_deg[1]:+.3f} optical z {rotvec_deg[2]:+.3f} deg | "
              f"head pan {res['head_offsets_deg']['pan']:+.3f} tilt {res['head_offsets_deg']['tilt']:+.3f} | "
              f"RMSE {quality['rmse_3d_marker_mm']:.3f} mm")
previous = json.loads((OUT / "summary.json").read_text(encoding="utf-8")) if (OUT / "summary.json").exists() else {}
(OUT / "summary.json").write_text(json.dumps({**previous, **summary}, indent=2), encoding="utf-8")
print(f"outputs: {OUT}")
