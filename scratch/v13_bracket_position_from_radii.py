"""Can the v1.3 marker bracket position be solved from the three bracket-sweep radii, as on v1.2?

v1.3's J4/J5/J6 axes meet at the wrist centre, so the marker origin p (in the ee frame) sits at
distance r_k = |(p - w) x a_k| from each swept axis, where w is the wrist centre and a_k the axis
direction in the ee frame (URDF, at the marker ready pose). Each bracket sweep's encoder-angle circle
fit (fit_circle_3d_and_6dof_misalignment, the radius v1.2 uses) gives r_k; p is solved by least
squares from the nominal bracket. Compared with the simulator truth (nominal + bracket_pos).

Reads sweep_points_<arm>_marker_axis_<4|5|6>.txt (last block) from a result_txt folder. Offline.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_bracket_position_from_radii.py <result_txt folder>
"""
import sys
from pathlib import Path

import numpy as np
import rby1_sdk as rby
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

FOLDER = Path(sys.argv[1])
URDF = ROOT / "archive" / "reports" / "v13_sweep_clearance" / "model_m_v1.3.urdf"
truth = yaml.safe_load((ROOT / "config" / "simulation.yaml").read_text(encoding="utf-8"))
ready = yaml.safe_load((ROOT / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))["v1.3"]["marker"]

model = rby.Model_M()
names = list(model.robot_joint_names)
dyn = getattr(rby.dynamics, f"Robot_{len(names)}")(rby.dynamics.load_robot_from_urdf_data(URDF.read_text(encoding="utf-8"), "base"))


def wrist_geometry(side):
    links = ["base", f"link_{side}_arm_4", f"link_{side}_arm_5", f"ee_{side}"]
    state = dyn.make_state(links, names)
    q = np.zeros(len(names))
    q[list(getattr(model, f"{side}_arm_idx"))] = np.radians(ready[f"{side}_arm"])
    state.set_q(q)
    dyn.compute_forward_kinematics(state)
    T4, T5, Tee = (dyn.compute_transformation(state, 0, i) for i in (1, 2, 3))
    to_ee = np.linalg.inv(Tee)
    w = (to_ee @ T5[:, 3])[:3]
    axes = {4: to_ee[:3, :3] @ T4[:3, 2], 5: to_ee[:3, :3] @ T4[:3, 1], 6: to_ee[:3, :3] @ T5[:3, 0]}
    return w, axes


def last_block(path):
    angles, poses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "NEW ITERATION" in line:
            angles, poses = [], []
        elif line and not line.startswith("#"):
            v = [float(x) for x in line.split(",") if x.strip()]
            if len(v) >= 26:
                angles.append(v[0])
                poses.append(np.array(v[10:26]).reshape(4, 4))
    return angles, poses


for side in ("right", "left"):
    w, axes = wrist_geometry(side)
    radii = {}
    for k in (4, 5, 6):
        angles, poses = last_block(FOLDER / f"sweep_points_{side}_marker_axis_{k}.txt")
        radii[k] = float(BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, angles, robust=True)["radius"]) / 1000.0
    nominal = np.array(truth["brackets"]["1.3"][side][:3], dtype=float)
    actual = nominal + np.array(truth["offsets"][side]["bracket_pos"], dtype=float)

    def model_radii(p):
        return np.array([np.linalg.norm(np.cross(p - w, axes[k])) for k in (4, 5, 6)])

    fit = least_squares(lambda p: model_radii(p) - np.array([radii[k] for k in (4, 5, 6)]), nominal)
    d = nominal - w
    print(f"{side}: wrist centre in ee frame {np.round(w * 1000, 1)} mm; marker - wrist (nominal) {np.round(d * 1000, 1)} mm")
    print(f"   axes in ee frame: J4 {np.round(axes[4], 3)}  J5 {np.round(axes[5], 3)}  J6 {np.round(axes[6], 3)}")
    print(f"   radii measured (encoder fit) {np.round([radii[k] * 1000 for k in (4, 5, 6)], 2)} mm | nominal model "
          f"{np.round(model_radii(nominal) * 1000, 2)} | true-bracket model {np.round(model_radii(actual) * 1000, 2)}")
    print(f"   solved bracket position {np.round(fit.x * 1000, 2)} mm vs truth {np.round(actual * 1000, 2)} "
          f"(error {np.round((fit.x - actual) * 1000, 2)} mm), nominal {np.round(nominal * 1000, 2)}")
    J = np.array([[np.dot(np.cross(np.cross(nominal - w, axes[k]), axes[k]) * -1, e) / max(np.linalg.norm(np.cross(nominal - w, axes[k])), 1e-9)
                   for e in np.eye(3)] for k in (4, 5, 6)])
    print(f"   sensitivity d(radius)/d(position) singular values {np.round(np.linalg.svd(J, compute_uv=False), 3)}")
