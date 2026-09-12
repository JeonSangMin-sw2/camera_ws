"""
Calibration Core Dataset & Configuration Helper Module

Provides dataset validation, loading/saving (.npz), marker capture utilities,
and kinematic/nominal configuration lookups for the calibration suite.
"""

import os
from pathlib import Path
import numpy as np
import yaml

from core.paths import CONFIG_PATHS

try:
    from core.marker_detection import Marker_Transform
except ImportError:
    from marker_detection import Marker_Transform

try:
    from .homeoffset_core import reset_home_offsets
except ImportError:
    from core.calibration.homeoffset_core import reset_home_offsets

try:
    from core.robot_motion import check_calibration_state
except ImportError:
    from robot_motion import check_calibration_state

np.set_printoptions(suppress=True, precision=6)

SETTING_PATH = Path(CONFIG_PATHS.get("setting", Path(__file__).resolve().parent.parent.parent / "config" / "setting.yaml"))
DEFAULT_LAMBDA_CAM_POS = 1.0
DEFAULT_LAMBDA_CAM_ROT = 1.0

ARM_SIDES = ("right", "left")
D2R = np.pi / 180.0


# ============================================================
# Dataset I/O & Validation Helpers
# ============================================================

def load_npz_dataset(path):
    """Load calibration dataset from an .npz file."""
    data = np.load(path)
    q_arm = data["q_arm"] if "q_arm" in data else data["q"]
    q_head = data["q_head"] if "q_head" in data else None
    return q_arm, q_head, data["marker"]


def save_npz_dataset(path, q_arm, T_meas, q_head=None):
    """Save calibration dataset to a compressed .npz file."""
    save_kwargs = {
        "q": q_arm,
        "q_arm": q_arm,
        "marker": T_meas,
    }
    if q_head is not None:
        save_kwargs["q_head"] = q_head
    np.savez_compressed(path, **save_kwargs)


def validate_dataset(q_arm, q_head, T_meas, optimize_head, active_arms):
    """Validate shapes and dimensions of a calibration dataset."""
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
        with open(SETTING_PATH, "r") as f:
            config = yaml.safe_load(f) or {}
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
