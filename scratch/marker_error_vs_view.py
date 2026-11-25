"""Does the marker measurement error grow with where the marker sits in the image, how obliquely it is seen, or how
far it is? Per sample of a real Step 2 run (and its saved check-pose views): measured minus model marker pose in the
camera frame, with the run's own offsets, camera and brackets, against image position, viewing angle and range.
FK from the running simulator of the same model (127.0.0.1, model m, v1.2); nothing is written.
usage: python scratch/marker_error_vs_view.py <result json> <dataset npz> <intrinsics yaml>
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rby1_sdk as rby  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
data = np.load(sys.argv[2], allow_pickle=True)
K = np.asarray(yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))["camera_matrix"], dtype=float)
view = result.get("check_pose_view") or {}
brackets = {s: BaseCalibrator.make_transform(result["setting_snapshot"]["marker"][f"Tf_to_marker_{s}"])
            for s in ("right", "left")}
T_mount_cam = BaseCalibrator.make_transform(result["mount_to_cam_new"])
offsets = np.radians(np.r_[result["right_arm_joint_offset_deg"], result["left_arm_joint_offset_deg"]])
head_off = np.radians(result["head_joint_offset_deg"])

robot = rby.create_robot("127.0.0.1:50051", "m")
assert robot.connect(), "simulator not reachable"
dyn, model = robot.get_dynamics(), robot.model()
names = list(model.robot_joint_names)
arm_idx = list(model.right_arm_idx) + list(model.left_arm_idx)
head_idx = [names.index("head_0"), names.index("head_1")]

samples = [("step2", q, h, m) for q, h, m in zip(data["q_arm"], data["q_head"], data["marker"])]
for v in [view] + list(view.get("other_views") or []) + [view.get("facing_view") or {}]:
    if v.get("seen"):
        head = v.get("head_deg") or v["view"]["head_deg"]
        tag = "facing" if "wrist_deg" in v else "check"
        samples.append((f"{tag} head {head[1]:.0f}", v["q_arm"], v["q_head"],
                        [np.asarray(v["markers"]["right"]), np.asarray(v["markers"]["left"])]))

rows = []
for tag, q_arm, q_head, markers in samples:
    q = np.zeros(model.robot_dof)
    q[arm_idx] = np.asarray(q_arm) + offsets
    q[head_idx] = np.asarray(q_head) + head_off
    T_cam = BaseCalibrator.compute_fk(robot, dyn, q, "link_head_2", "link_torso_5") @ T_mount_cam
    for i, side in enumerate(("right", "left")):
        T_model = np.linalg.inv(T_cam) @ BaseCalibrator.compute_fk(robot, dyn, q, f"ee_{side}", "link_torso_5") @ brackets[side]
        T_meas = np.asarray(markers[i])
        p = T_meas[:3, 3]
        u, v_px = (K @ (p / p[2]))[:2]
        oblique = np.degrees(np.arccos(np.clip(-T_meas[:3, 2] @ (-p / np.linalg.norm(p)), -1, 1)))
        rows.append(dict(tag=tag, side=side, u=u, v=v_px, oblique=oblique, range_mm=np.linalg.norm(p) * 1000,
                         dpos=(p - T_model[:3, 3]) * 1000,
                         drot=R.from_matrix(T_meas[:3, :3] @ T_model[:3, :3].T).as_rotvec(degrees=True)))
robot.disconnect()

step2 = [r for r in rows if r["tag"] == "step2"]
for side in ("right", "left"):
    s = [r for r in step2 if r["side"] == side]
    print(f"{side}: Step 2 samples {len(s)}  u {min(r['u'] for r in s):.0f}..{max(r['u'] for r in s):.0f} px, "
          f"v {min(r['v'] for r in s):.0f}..{max(r['v'] for r in s):.0f} px, oblique {min(r['oblique'] for r in s):.0f}.."
          f"{max(r['oblique'] for r in s):.0f} deg, range {min(r['range_mm'] for r in s):.0f}..{max(r['range_mm'] for r in s):.0f} mm")
    pos = np.array([np.linalg.norm(r["dpos"]) for r in s])
    rot = np.array([np.linalg.norm(r["drot"]) for r in s])
    print(f"   residual |pos| median {np.median(pos):.2f} max {pos.max():.2f} mm, |rot| median {np.median(rot):.2f} max {rot.max():.2f} deg")
    for key in ("u", "v", "oblique", "range_mm"):
        x = np.array([r[key] for r in s])
        cp, cr = np.corrcoef(x, pos)[0, 1], np.corrcoef(x, rot)[0, 1]
        # slope of |error| per 10 units of the variable, to extrapolate to the check pose
        sp, sr = np.polyfit(x, pos, 1)[0] * 10, np.polyfit(x, rot, 1)[0] * 10
        print(f"   vs {key:8s}: corr pos {cp:+.2f} rot {cr:+.2f} | slope per 10: pos {sp:+.3f} mm, rot {sr:+.3f} deg")
print("check-pose views (same model, same offsets):")
for r in rows:
    if r["tag"] != "step2":
        print(f"   {r['tag']:14s} {r['side']:5s} u {r['u']:.0f} v {r['v']:.0f} px, oblique {r['oblique']:.0f} deg, "
              f"range {r['range_mm']:.0f} mm | |pos| {np.linalg.norm(r['dpos']):.2f} mm, |rot| {np.linalg.norm(r['drot']):.2f} deg")
