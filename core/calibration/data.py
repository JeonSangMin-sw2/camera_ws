from core.storage import ConfigStorage
from core.storage import DatasetStorage
"""
Calibration Core Dataset & Configuration Helper Module

Provides dataset validation, loading/saving (.npz), marker capture utilities,
and kinematic/nominal configuration lookups for the calibration suite.
"""

import os
from pathlib import Path
import numpy as np

from core.storage import CONFIG_PATHS


np.set_printoptions(suppress=True, precision=6)

SETTING_PATH = Path(CONFIG_PATHS["setting_yaml"])
DEFAULT_LAMBDA_CAM_POS = 1.0
DEFAULT_LAMBDA_CAM_ROT = 1.0
# Used only when setting.yaml predates the step2.min_marker_x_gap_m key (an installed copy
# is never overwritten by a newer bundle); the caller logs that the default was used.
DEFAULT_MIN_MARKER_X_GAP_M = 0.11
# Same fallback rule for exposure_check.max_marker_jitter_rms_mm (wizard exposure slide).
DEFAULT_MAX_MARKER_JITTER_MM = 0.2
# Same fallback rule for step2.min_rot_noise_deg: the floor on Step 2's estimated marker-orientation
# noise (2026-09-29). Without it the orientation weight grew as J4 absorbed an orientation bias.
DEFAULT_STEP2_MIN_ROT_NOISE_DEG = 0.3

ARM_SIDES = ("right", "left")
D2R = np.pi / 180.0


# ============================================================
# Dataset I/O & Validation Helpers
# ============================================================





def validate_dataset(q_arm, q_head, T_meas, optimize_head, active_arms):
    """Validate shapes and dimensions of a calibration dataset."""
    if not active_arms or len(set(active_arms)) != len(active_arms) or any(side not in ARM_SIDES for side in active_arms):
        raise RuntimeError("active_arms must contain distinct right/left arm names")
    if len(q_arm) == 0:
        raise RuntimeError("Calibration dataset is empty")
    for name, values in (("q_arm", q_arm), ("q_head", q_head), ("marker", T_meas)):
        if values is not None:
            try:
                finite = np.all(np.isfinite(values))
            except TypeError:
                finite = False
            if not finite:
                raise RuntimeError(f"Dataset {name} must contain finite numeric values")
    if q_head is not None and (q_head.ndim != 2 or q_head.shape[1] != 2):
        raise RuntimeError(f"Expected q_head shape (N, 2), got {q_head.shape}")
    if len(q_arm) != len(T_meas):
        raise RuntimeError(
            f"Dataset size mismatch: q_arm={len(q_arm)}, marker={len(T_meas)}"
        )

    if q_head is not None and len(q_head) != len(q_arm):
        raise RuntimeError(
            f"Dataset size mismatch: q_head={len(q_head)}, q_arm={len(q_arm)}"
        )

    if q_head is None and optimize_head:
        raise RuntimeError(
            "Head-mounted camera calibration requires `q_head`, but the loaded npz does not contain it."
        )

    if q_arm.ndim != 2:
        raise RuntimeError(f"Expected q_arm to be a 2D array, got shape {q_arm.shape}")

    expected_q_arm_len = 7 * len(active_arms)
    if q_arm.shape[1] != expected_q_arm_len:
        raise RuntimeError(
            f"Unsupported q_arm width {q_arm.shape[1]}. Expected {expected_q_arm_len} for arms: {active_arms}."
        )

    if len(active_arms) > 1:
        if T_meas.ndim != 4 or T_meas.shape[1:] != (len(active_arms), 4, 4):
            raise RuntimeError(
                f"Expected marker measurements with shape (N, {len(active_arms)}, 4, 4), got {T_meas.shape}"
            )
    else:
        if T_meas.ndim != 3 or T_meas.shape[1:] != (4, 4):
            raise RuntimeError(
                f"Expected marker measurements with shape (N, 4, 4), got {T_meas.shape}"
            )


def split_arm_offsets(q_offset):
    """Split concatenated 14-element arm offsets into (right, left)."""
    q_offset = np.asarray(q_offset, dtype=np.float64).reshape(-1)
    if len(q_offset) == 14:
        return q_offset[:7], q_offset[7:]
    return q_offset, None


# ============================================================
# Nominal & Model Configuration Helpers
# ============================================================

def load_camera_nominals(version="1.2"):
    """Load nominal camera and marker transforms from setting.yaml."""
    if not os.path.exists(SETTING_PATH):
        raise FileNotFoundError(f"[CRITICAL ERROR] setting.yaml not found at {SETTING_PATH}!")
    try:
        config = ConfigStorage.load(SETTING_PATH)
    except Exception as e:
        raise RuntimeError(f"[CRITICAL ERROR] Failed to parse {SETTING_PATH}: {e}")

    camera_cfg = config.get("camera", {})
    marker_cfg = config.get("marker", {})
    mount_to_cam_nom = camera_cfg.get("mount_to_cam")
    head_base_to_cam_nom = camera_cfg.get("head_base_to_cam")

    if mount_to_cam_nom is None:
        raise KeyError(f"[CRITICAL ERROR] Missing 'mount_to_cam' under 'camera' in setting.yaml!")

    if str(version).replace("v", "").strip() == "1.3":
        if "Tf_to_marker_left_v13" not in marker_cfg or "Tf_to_marker_right_v13" not in marker_cfg:
            raise KeyError("[CRITICAL ERROR] Missing required 'Tf_to_marker_left_v13' or 'Tf_to_marker_right_v13' in setting.yaml!")
        ee_to_marker_left = marker_cfg["Tf_to_marker_left_v13"]
        ee_to_marker_right = marker_cfg["Tf_to_marker_right_v13"]
    else:
        if "Tf_to_marker_left_v12" not in marker_cfg or "Tf_to_marker_right_v12" not in marker_cfg:
            raise KeyError("[CRITICAL ERROR] Missing required 'Tf_to_marker_left_v12' or 'Tf_to_marker_right_v12' in setting.yaml!")
        ee_to_marker_left = marker_cfg["Tf_to_marker_left_v12"]
        ee_to_marker_right = marker_cfg["Tf_to_marker_right_v12"]

    return {
        "mount_to_cam_nom": mount_to_cam_nom,
        "head_base_to_cam_nom": head_base_to_cam_nom,
        "camera_mount_link": camera_cfg.get("camera_mount_link", "link_head_2"),
        "ee_to_marker_left": ee_to_marker_left,
        "ee_to_marker_right": ee_to_marker_right,
    }


class MarkersTooCloseError(RuntimeError):
    """The two arm markers start too close together along the camera x axis for Step 2 motion.

    `partial` is merged into the sequence result so the UI can show the measured gap."""
    def __init__(self, gap_m, min_gap_m):
        super().__init__(f"Markers too close: camera x gap {gap_m * 100:.1f} cm is below "
                         f"{min_gap_m * 100:.1f} cm. Widen the arms before Step 2 motion.")
        self.gap_m, self.min_gap_m = float(gap_m), float(min_gap_m)
        self.partial = {"marker_x_gap": {"gap_m": self.gap_m, "min_gap_m": self.min_gap_m,
                                         "blocked": True}}


def load_min_marker_x_gap_m(path=SETTING_PATH):
    """Minimum camera-x gap between the markers before Step 2 motion, and whether the value
    came from setting.yaml (False = DEFAULT_MIN_MARKER_X_GAP_M, the key is missing)."""
    config = ConfigStorage.load(path) if os.path.exists(path) else {}
    value = (config.get("step2") or {}).get("min_marker_x_gap_m")
    if value is None:
        return DEFAULT_MIN_MARKER_X_GAP_M, False
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"step2.min_marker_x_gap_m in {path} must be a positive distance in meters, got {value}")
    return value, True


READY_POSES_PATH = Path(CONFIG_PATHS["ready_poses_yaml"])


def load_step2_wrist_diversity(robot_version, setting_path=SETTING_PATH, ready_poses_path=READY_POSES_PATH):
    """Extra Step 2 poses that turn the wrists (J4/J5/J6) well beyond the base plan, for head robots.

    2026-09-30: the base plan moves each joint only a few degrees, so some J2/J4/J6 offset
    combinations are nearly invisible to Step 2 yet move the hands 1-2 mm per degree at the check
    pose. The fixed list in ready_poses.yaml (<version>.step2_wrist_diversity) was chosen offline:
    J0-J3 stay at the Step 2 baseline, both markers stay in view on three recorded baselines.
    Returns (steps, enabled): steps is a list of {"right": [dJ4, dJ5, dJ6], "left": [...],
    "head": [dpan, dtilt]} in degrees, empty when switched off (setting.yaml
    step2.wrist_diversity_poses) or when the version has no list.
    """
    setting = ConfigStorage.load(setting_path) if os.path.exists(setting_path) else {}
    enabled = bool((setting.get("step2") or {}).get("wrist_diversity_poses", False))
    if not enabled:
        return [], False
    poses = ConfigStorage.load(ready_poses_path) if os.path.exists(ready_poses_path) else {}
    ver = str(robot_version).lstrip("v")
    entries = (poses.get(f"v{ver}") or poses.get(ver) or {}).get("step2_wrist_diversity") or []
    steps = []
    for i, entry in enumerate(entries):
        try:
            step = {key: [float(v) for v in entry[key]] for key in ("right", "left", "head")}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{ready_poses_path} v{ver}.step2_wrist_diversity[{i}] must have right/left "
                             f"[dJ4, dJ5, dJ6] and head [dpan, dtilt] in degrees: {error}") from error
        if len(step["right"]) != 3 or len(step["left"]) != 3 or len(step["head"]) != 2 \
                or not np.all(np.isfinite(step["right"] + step["left"] + step["head"])):
            raise ValueError(f"{ready_poses_path} v{ver}.step2_wrist_diversity[{i}] must have right/left "
                             f"[dJ4, dJ5, dJ6] and head [dpan, dtilt] in degrees, got {entry}")
        steps.append(step)
    return steps, True


def marker_x_gap_m(T_right, T_left):
    """Distance between the two marker origins along the camera x axis (image horizontal).

    Both transforms are camera -> marker, as the detector and the simulator return them."""
    p_right = np.asarray(T_right, dtype=np.float64).reshape(4, 4)[:3, 3]
    p_left = np.asarray(T_left, dtype=np.float64).reshape(4, 4)[:3, 3]
    if not (np.all(np.isfinite(p_right)) and np.all(np.isfinite(p_left))):
        raise ValueError("Marker positions must be finite to measure their gap")
    return float(abs(p_right[0] - p_left[0]))


def marker_position_jitter_mm(points):
    """Spread of repeated marker positions of a marker that is not moving, in millimetres.

    points: (N, 3) camera-frame positions in metres. Returns {"rms_mm", "max_mm"}: the RMS and
    the largest distance from the mean position, or None with fewer than two points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return None
    dist = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    return {"rms_mm": float(np.sqrt(np.mean(dist ** 2)) * 1000.0), "max_mm": float(dist.max() * 1000.0)}


def load_max_marker_jitter_mm(path=SETTING_PATH):
    """Largest RMS marker jitter (mm) the exposure check accepts, and whether the value came
    from setting.yaml (False = DEFAULT_MAX_MARKER_JITTER_MM, the key is missing)."""
    config = ConfigStorage.load(path) if os.path.exists(path) else {}
    value = (config.get("exposure_check") or {}).get("max_marker_jitter_rms_mm")
    if value is None:
        return DEFAULT_MAX_MARKER_JITTER_MM, False
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"exposure_check.max_marker_jitter_rms_mm in {path} must be a positive distance in mm, got {value}")
    return value, True


def load_step2_min_rot_noise_deg(path=SETTING_PATH):
    """Floor (deg) on the marker-orientation noise Step 2 estimates from its residuals, and whether
    it came from setting.yaml (False = DEFAULT_STEP2_MIN_ROT_NOISE_DEG, the key is missing).

    Step 2 weights each residual by 1/sigma. On 2026-09-29 (D405) the orientation residual fell to
    0.114 deg because J4 bent to absorb a marker-orientation bias, which raised the orientation
    weight further and pulled J4 by up to 1.6 deg and both J0 by 1.5-1.8 deg. The floor keeps the
    orientation weight at the level the camera's orientation measurement actually supports
    (0.2-0.36 deg drift with image position, design 5.2)."""
    config = ConfigStorage.load(path) if os.path.exists(path) else {}
    value = (config.get("step2") or {}).get("min_rot_noise_deg")
    if value is None:
        return DEFAULT_STEP2_MIN_ROT_NOISE_DEG, False
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"step2.min_rot_noise_deg in {path} must be a positive angle in degrees, got {value}")
    return value, True


def marker_monitor_summary(history, max_jitter_mm=None):
    """Recognition and jitter per marker from recent samples.

    history: {side: [position (3,) in metres, or None when not detected, ...]} oldest first.
    Returns {side: {"visible", "rate", "samples", "jitter", "stable"}}: visible = detected in
    the latest sample, rate = fraction of samples detected, jitter = marker_position_jitter_mm
    of the detected positions (None with fewer than two), stable = visible with an RMS jitter
    at or below max_jitter_mm (None when no limit is given)."""
    summary = {}
    for side, samples in history.items():
        samples = list(samples)
        detected = [p for p in samples if p is not None]
        visible = bool(samples) and samples[-1] is not None
        jitter = marker_position_jitter_mm(detected) if len(detected) >= 2 else None
        stable = None
        if max_jitter_mm is not None:
            stable = bool(visible and jitter is not None and jitter["rms_mm"] <= max_jitter_mm)
        summary[side] = {
            "visible": visible,
            "rate": len(detected) / len(samples) if samples else 0.0,
            "samples": len(samples),
            "jitter": jitter,
            "stable": stable,
        }
    return summary


def get_arm_config(model, arm, version="1.2"):
    """Get single-arm configuration dictionary."""
    camera_nominals = load_camera_nominals(version=version)
    base_config = {
        "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
        "head_base_to_cam_nom": camera_nominals["head_base_to_cam_nom"],
        "camera_mount_link": camera_nominals["camera_mount_link"],
    }

    if arm == "right":
        base_config.update({
            "arm_idx": model.right_arm_idx[:7],
            "ee_link": "ee_right",
            "ee_to_marker_nom": camera_nominals["ee_to_marker_right"],
        })
    else:
        base_config.update({
            "arm_idx": model.left_arm_idx[:7],
            "ee_link": "ee_left",
            "ee_to_marker_nom": camera_nominals["ee_to_marker_left"],
        })
    return base_config


def get_both_arm_config(model, version="1.2"):
    """Get dual-arm configuration dictionary."""
    camera_nominals = load_camera_nominals(version=version)
    return {
        "arm_idx": np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]]),
        "ee_links": {
            "right": "ee_right",
            "left": "ee_left",
        },
        "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
        "head_base_to_cam_nom": camera_nominals["head_base_to_cam_nom"],
        "camera_mount_link": camera_nominals["camera_mount_link"],
        "ee_to_marker_nom": {
            "right": camera_nominals["ee_to_marker_right"],
            "left": camera_nominals["ee_to_marker_left"],
        },
    }


def get_head_config(model):
    """Get head kinematics configuration dictionary."""
    camera_nominals = load_camera_nominals()
    head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
    return {
        "head_idx": head_idx,
        "camera_link": camera_nominals["camera_mount_link"],
    }


# ============================================================
# Live Sample Capture Helper
# ============================================================

def capture_one_sample(robot, arm_idx, marker_transform, sampling_time=1, side="all", head_idx=None):
    """
    Capture a single data point containing robot joint angles and marker transform.
    """
    state = robot.get_state()
    if state is None or getattr(state, 'position', None) is None:
        return None, None, None
    q_full = np.array(state.position)
    q_arm = q_full[arm_idx].copy()
    q_head = np.array([float(q_full[i]) for i in list(head_idx)], dtype=np.float64) if head_idx is not None else None

    result = marker_transform.get_marker_transform(sampling_time=sampling_time, side=side)
    if result is None:
        return None, None, None

    if side == "all":
        if len(result) < 2 or result[0] is None or result[1] is None:
            return None, None, None

        def _to_tf(flat_tf):
            arr = np.asarray(flat_tf, dtype=np.float64).reshape(-1)
            if arr.size != 16:
                raise RuntimeError(
                    f"Expected one marker transform to contain 16 values, got shape {np.asarray(flat_tf).shape}"
                )
            return arr.reshape(4, 4)

        T_right = _to_tf(result[0])
        T_left = _to_tf(result[1])
        return q_arm, q_head, np.stack([T_right, T_left], axis=0)

    T_meas = np.asarray(result, dtype=np.float64).reshape(-1)
    if T_meas.size != 16:
        raise RuntimeError(
            f"Expected marker transform with 16 values for side='{side}', got shape {np.asarray(result).shape}"
        )
    return q_arm, q_head, T_meas.reshape(4, 4)

load_npz_dataset = DatasetStorage.load_calibration
save_npz_dataset = DatasetStorage.save_calibration
