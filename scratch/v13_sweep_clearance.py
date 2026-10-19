"""Clearance of the v1.3 wrist-stage sweeps with grippers off and marker brackets on (offline).

perform_move_to_ready_pose first sends the other arm to zero (hanging down), so a wrist stage runs
with the sweeping arm at the v1.3 wrist-stage ready pose, the other arm at zero and the torso at zero.
The URDF's link_*_arm_5 collision capsule (r 75 mm, 250 mm) is the gripper envelope, removed for
calibration, so this builds its own approximate shapes as sampled points with a radius:
  URDF capsules of arm links 2-4 and the torso, a 40 mm sphere at the wrist (link_*_arm_5 origin),
  a 35 mm sphere at the flange (ee_*), a 10 mm bar from the flange to the marker and the 80 mm
  marker plate (v1.3 bracket, setting.yaml Tf_to_marker_*_v13).
For each sweep angle it reports the smallest gap between the sweeping arm's wrist/flange/bracket/
plate and its own upper arm (links 2-3), the other arm, or the torso. Approximate shapes: a gap of a few mm is not a clearance.
The URDF is read once (read-only) from the v1.3 simulator, 127.0.0.1:50051, and kept in OUT.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_sweep_clearance.py
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import rby1_sdk as rby
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.calibration.calibration_optimizer import make_transform  # noqa: E402

OUT = ROOT / "archive" / "reports" / "v13_sweep_clearance"
OUT.mkdir(parents=True, exist_ok=True)
URDF = OUT / "model_m_v1.3.urdf"
# (mode, joint, half arc before 2026-10-14, half arc now), CalibratorBase.JOINT_CONFIGS.
SWEEPS = [("wrist_roll_v13", 6, 20.0, 20.0), ("wrist_roll_v13", 5, 15.0, 15.0),
          ("wrist_pitch_v13", 6, 15.0, 22.5), ("wrist_pitch_v13", 4, 15.0, 30.0)]
CHECK_DEG = np.arange(-40.0, 40.01, 2.5)
WRIST_R, FLANGE_R, BAR_R, PLATE_HALF = 0.040, 0.035, 0.010, 0.040

if not URDF.exists():
    client = rby.create_robot("127.0.0.1:50051", "m")
    deadline = time.time() + 30.0
    while not client.connect():
        if time.time() > deadline:
            raise SystemExit("simulator not reachable (needed once, read-only, for the URDF)")
        time.sleep(1.0)
    URDF.write_text(client.get_robot_model(), encoding="utf-8")
    client.disconnect()
urdf = URDF.read_text(encoding="utf-8")
model = rby.Model_M()
names = list(model.robot_joint_names)
dyn = getattr(rby.dynamics, f"Robot_{len(names)}")(rby.dynamics.load_robot_from_urdf_data(urdf, "base"))

capsules = {}
for link, body in re.findall(r'<link name="([^"]+)">(.*?)</link>', urdf, re.S):
    m = re.search(r'<collision>\s*<origin xyz="([^"]+)" rpy="([^"]+)"\s*/>\s*<geometry>\s*<capsule radius="([^"]+)" length="([^"]+)"', body)
    if m:
        capsules[link] = (np.array(m.group(1).split(), float), np.array(m.group(2).split(), float),
                          float(m.group(3)), float(m.group(4)))
STATIC_LINKS = ["link_torso_0", "link_torso_1", "link_torso_2", "link_torso_4", "link_torso_5"]
links = ["base"] + sorted(set(STATIC_LINKS + [f"link_{s}_arm_{i}" for s in ("right", "left") for i in range(6)]
                              + ["ee_right", "ee_left"]))
state = dyn.make_state(links, names)
setting = yaml.safe_load((ROOT / "config" / "setting.yaml").read_text(encoding="utf-8"))
markers = setting.get("marker", setting.get("camera", {}))
ready = yaml.safe_load((ROOT / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))["v1.3"]
# --ready-j5 DEG: J5 of the wrist_pitch ready pose for this check only.
if "--ready-j5" in sys.argv:
    for arm in ("right_arm", "left_arm"):
        ready["joint"]["wrist_pitch"][arm][5] = float(sys.argv[sys.argv.index("--ready-j5") + 1])
# --roll-arcs A B: wrist_roll_v13 J6 / J5 half arcs for this check only.
if "--roll-arcs" in sys.argv:
    a = sys.argv.index("--roll-arcs")
    SWEEPS[:2] = [("wrist_roll_v13", 6, 20.0, float(sys.argv[a + 1])), ("wrist_roll_v13", 5, 15.0, float(sys.argv[a + 2]))]
# --arcs A B: wrist_pitch_v13 J6 / J4 half arcs for this check only.
if "--arcs" in sys.argv:
    a = sys.argv.index("--arcs")
    SWEEPS[2:] = [("wrist_pitch_v13", 6, 15.0, float(sys.argv[a + 1])), ("wrist_pitch_v13", 4, 15.0, float(sys.argv[a + 2]))]


def pose(link):
    return dyn.compute_transformation(state, 0, links.index(link))


def capsule_points(link):
    xyz, rpy, radius, length = capsules[link]
    local = make_transform(list(xyz) + list(np.degrees(rpy)))
    world = pose(link) @ local
    ts = np.linspace(-length / 2.0, length / 2.0, 7)
    return [(world[:3, 3] + t * world[:3, 2], radius) for t in ts]


def hand_points(side):
    """Wrist sphere, flange sphere, bracket bar and marker plate of one arm (point, radius)."""
    ee = pose(f"ee_{side}")
    marker = ee @ make_transform(markers[f"Tf_to_marker_{side}_v13"])
    points = [(pose(f"link_{side}_arm_5")[:3, 3], WRIST_R), (ee[:3, 3], FLANGE_R)]
    points += [(ee[:3, 3] + t * (marker[:3, 3] - ee[:3, 3]), BAR_R) for t in np.linspace(0.0, 1.0, 6)]
    grid = np.linspace(-PLATE_HALF, PLATE_HALF, 5)
    points += [((marker @ np.array([x, y, 0.0, 1.0]))[:3], 0.002) for x in grid for y in grid]
    return points


# --centres: distances between the sampled centres only (no guessed radii), for comparing arcs.
CENTRES = "--centres" in sys.argv


def gap(a, b):
    return min(np.linalg.norm(p - q) - (0.0 if CENTRES else r + s) for p, r in a for q, s in b)


def clearance(q, side):
    state.set_q(q)
    dyn.compute_forward_kinematics(state)
    other = "left" if side == "right" else "right"
    # The wrist sphere always sits next to its own forearm (link_3) whatever the sweep, so the
    # gaps below are for the flange, bracket and plate only.
    moving = hand_points(side)[1:]
    own_arm = sum((capsule_points(f"link_{side}_arm_{i}") for i in (2, 3)), [])
    other_arm = hand_points(other) + sum((capsule_points(f"link_{other}_arm_{i}") for i in (2, 3, 4)), [])
    torso = sum((capsule_points(link) for link in STATIC_LINKS if link in capsules), [])
    return min(gap(moving, own_arm), gap(moving, other_arm), gap(moving, torso)) * 1000.0, \
        gap(moving, own_arm) * 1000.0


def ready_q(mode, side):
    q = np.zeros(len(names))
    key = "wrist_pitch" if mode == "wrist_pitch_v13" else "wrist_roll"
    q[list(getattr(model, f"{side}_arm_idx"))] = np.radians(ready["joint"][key][f"{side}_arm"])
    return q


print("Sweeping arm at the v1.3 wrist-stage ready pose, other arm and torso at zero. Gaps in mm"
      + (" between sampled centres (no radii)." if CENTRES else " (approximate shapes)."))
print("arm   mode             joint | before: arc / min gap to anything, own upper arm | now: arc / same | "
      "min gap at -40 ... +40 deg (10 deg steps)")
for side in ("right", "left"):
    idx = list(getattr(model, f"{side}_arm_idx"))
    for mode, joint, arc_before, arc_now in SWEEPS:
        q0 = ready_q(mode, side)
        rows = {}
        for d in CHECK_DEG:
            q = q0.copy()
            q[idx[joint]] += np.radians(d)
            rows[round(float(d), 1)] = clearance(q, side)

        def worst(arc):
            values = [v for d, v in rows.items() if abs(d) <= arc + 1e-9]
            return min(v[0] for v in values), min(v[1] for v in values)

        b, n = worst(arc_before), worst(arc_now)
        profile = " ".join(f"{rows[round(float(d), 1)][0]:5.0f}" for d in np.arange(-40.0, 40.01, 10.0))
        print(f"{side:5s} {mode:16s} J{joint}   | +-{arc_before:4.1f}: {b[0]:6.1f} {b[1]:6.1f} | "
              f"+-{arc_now:4.1f}: {n[0]:6.1f} {n[1]:6.1f} | {profile}")
