import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import rby1_sdk as rby
import yaml

from core.marker_detection import Marker_Transform
from core.homeoffset_core import reset_home_offsets
from core.robot_motion import check_calibration_state


np.set_printoptions(suppress=True, precision=6)

import sys
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
from core.config_store import CONFIG_PATHS, RobotConfig
DEFAULT_LAMBDA_CAM_POS = 1.0
DEFAULT_LAMBDA_CAM_ROT = 1.0

ARM_SIDES = ("right", "left")
D2R = np.pi / 180.0


# ============================================================
# Auto data-collection motion
# ============================================================

# (Auto motion logic has been moved to core.robot_motion)


from core.calibration_optimizer import (
    adjoint, make_transform, so3_exp, se3_exp, so3_log, se3_log, rot_to_euler_zyx,
    CalibrationOptimizer, QPCalibrationOptimizer, compute_fk, prepare_q_full,
    DEFAULT_LAMBDA_CAM_POS, DEFAULT_LAMBDA_CAM_ROT
)
# ============================================================
# Config / helpers
# ============================================================
def create_robot(ip, model_name="a", power_regex="48v", servo_regex=None, *,
                 include_head=False, unlimited_mode_enabled=False):
    from core.robot_motion import initialize_robot_connection
    return initialize_robot_connection(ip, model_name, power=power_regex, servo=servo_regex,
        include_head=include_head, unlimited_mode_enabled=unlimited_mode_enabled)

def load_npz_dataset(path, return_metadata=False):
    import json
    with np.load(path, allow_pickle=False) as data:
        q_arm = data["q_arm"] if "q_arm" in data else data["q"]
        q_head = data["q_head"] if "q_head" in data else None
        result = (q_arm, q_head, data["marker"])
        metadata = json.loads(str(data['metadata_json'])) if 'metadata_json' in data else {
            'schema_version': 0, 'source': 'legacy_unknown', 'truth': None}
    return (*result, metadata) if return_metadata else result

def save_npz_dataset(path, q_arm, T_meas, q_head=None, metadata=None):
    import json
    from datetime import datetime, timezone
    save_kwargs = {
        "q": q_arm,
        "q_arm": q_arm,
        "marker": T_meas,
        "metadata_json": json.dumps({**(metadata or {'source': 'unspecified'}),
                                    'saved_at_utc': datetime.now(timezone.utc).isoformat()}, allow_nan=False),
    }
    if q_head is not None:
        save_kwargs["q_head"] = q_head
    np.savez_compressed(path, **save_kwargs)


def validate_dataset(q_arm, q_head, T_meas, optimize_head, active_arms):
    if len(q_arm) == 0 or not np.all(np.isfinite(q_arm)) or not np.all(np.isfinite(T_meas)):
        raise ValueError('Dataset must contain finite samples')
    if q_head is not None and (np.asarray(q_head).shape != (len(q_arm), 2) or not np.all(np.isfinite(q_head))):
        raise ValueError('Head encoders must be a finite (N, 2) array')
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
    q_offset = np.asarray(q_offset, dtype=np.float64).reshape(-1)
    if len(q_offset) == 14:
        return q_offset[:7], q_offset[7:]
    return q_offset, None

def load_camera_nominals(version="1.2", *, camera_config=None):
    """Camera initialization plus canonical design, never mixed with bracket estimates.

    Replays pass their recorded/explicit camera context. Only interactive callers
    without a context read the current settings through the centralized path.
    """
    if camera_config is None:
        setting_path = CONFIG_PATHS['setting_yaml']
        with open(setting_path, "r", encoding="utf-8") as stream:
            camera_cfg = (yaml.safe_load(stream) or {})['camera']
    else:
        camera_cfg = camera_config
    mount_to_cam_nom = camera_cfg.get("mount_to_cam")
    head_base_to_cam_nom = camera_cfg.get("head_base_to_cam")

    if mount_to_cam_nom is None:
        raise KeyError(f"[CRITICAL ERROR] Missing 'mount_to_cam' under 'camera' in setting.yaml!")

    design = RobotConfig.load().nominal_brackets[str(version).removeprefix('v')]

    return {
        "mount_to_cam_nom": mount_to_cam_nom,
        "head_base_to_cam_nom": head_base_to_cam_nom,
        "camera_mount_link": camera_cfg.get("camera_mount_link", "link_head_2"),
        "ee_to_marker_left": design['left'],
        "ee_to_marker_right": design['right'],
    }

def get_arm_config(model, arm, version="1.2", *, camera_config=None):
    camera_nominals = load_camera_nominals(version=version, camera_config=camera_config)
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

def get_both_arm_config(model, version="1.2", *, camera_config=None):
    camera_nominals = load_camera_nominals(version=version, camera_config=camera_config)
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

def get_head_config(model, *, camera_config=None):
    head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
    return {
        "head_idx": head_idx,
        "camera_link": (camera_config or {}).get('camera_mount_link', 'link_head_2'),
    }


# ============================================================
# Capture dataset
# ============================================================

def create_live_marker_transform():
    marker_transform = Marker_Transform(
        serial_number=None
    )
    marker_transform.marker_detection.set_marker_type("plate")
    return marker_transform


def capture_one_sample(robot, arm_idx, marker_transform, sampling_time=1, side="all", head_idx=None):
    state = robot.get_state()
    if state is None or getattr(state, 'position', None) is None:
        return None, None, None
    q_full = np.array(state.position)
    q_arm = q_full[arm_idx].copy()
    q_head = np.array([float(q_full[i]) for i in list(head_idx)], dtype=np.float64) if head_idx is not None else None

    result = marker_transform.get_marker_transform(sampling_time=sampling_time, side=side, q_encoder=q_full)
    if result is None:
        return None, None, None

    # side="all" returns [right, left] where each entry is a flattened 4x4.
    # If either side is missing, skip this sample gracefully.
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


def _joint_result_accepted(result):
    return bool(result and result.get('measurement_accepted', True) and result.get('converged', False))


def _joint_store_key(mode):
    if mode in ('wrist_yaw2', 'wrist_roll_v13'):
        return 'joint6'
    if mode in ('wrist_pitch', 'wrist_pitch_v13'):
        return 'joint5'
    return 'joint3'


class FullAutoCalibrationService:
    """Measured Step 1 sequence; receives domain calibrators and notification callbacks."""
    def __init__(self, joint_calibrator, marker_calibrator, stop_event=None, joint_offsets_store=None, save_debug=False,
                 log_callback=None, status_callback=None, bracket_callback=None, joint_callback=None,
                 reset_initial_state=False, head_camera_calibrator=None):
        self.log_callback = log_callback or (lambda message: None)
        self.status_callback = status_callback or (lambda detected: None)
        self.bracket_callback = bracket_callback or (lambda result: None)
        self.joint_callback = joint_callback or (lambda result: None)
        self.joint_calibrator = joint_calibrator
        self.marker_calibrator = marker_calibrator
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.joint_offsets_store = joint_offsets_store if joint_offsets_store is not None else {}
        self.save_debug = save_debug
        self.error_msg = None
        self.arm_convergence = {}
        self.stage_results = {}
        self.reset_initial_state = reset_initial_state
        self.head_camera_calibrator = head_camera_calibrator

    def get_robot_version(self):
        return self.marker_calibrator.get_robot_version()

    def _run_joint_calibration(self, arm_side, mode, **kwargs):
        key = {'wrist_pitch_v13': 'wrist_pitch', 'wrist_roll_v13': 'wrist_roll'}.get(mode, mode)
        try:
            return self.joint_calibrator.perform_joint_calibration(arm_side, mode, **kwargs)
        finally:
            approved = self.joint_offsets_store[arm_side][_joint_store_key(mode)]
            for calibrator in (self.joint_calibrator, self.marker_calibrator):
                calibrator.joint_offsets.setdefault(arm_side, {})[key] = approved

    def run(self):
        from copy import deepcopy
        snapshots = [(calibrator, deepcopy(calibrator.joint_offsets), deepcopy(calibrator.camera_config))
                     for calibrator in (self.joint_calibrator, self.marker_calibrator)]
        head_result = deepcopy(getattr(self.head_camera_calibrator, 'calibrated_results', None))
        if self.reset_initial_state:
            for side in ('right', 'left'):
                for key in self.joint_offsets_store.get(side, {}):
                    self.joint_offsets_store[side][key] = 0.
                for calibrator, _, _ in snapshots:
                    for key in calibrator.joint_offsets.get(side, {}):
                        calibrator.joint_offsets[side][key] = 0.
            self.joint_offsets_store['head'] = dict(pan=0., tilt=0.)
            if self.head_camera_calibrator is not None:
                self.head_camera_calibrator.calibrated_results = None
        try:
            from scipy.spatial.transform import Rotation as R_scipy
            self.log_callback("Starting FULL AUTO sequential calibration...")
            version_num = self.get_robot_version()
            is_v13 = (version_num == "1.3")
            mode5 = 'wrist_pitch_v13' if is_v13 else 'wrist_pitch'
            mode6 = 'wrist_roll_v13' if is_v13 else 'wrist_yaw2'
            key6 = 'wrist_roll' if is_v13 else 'wrist_yaw2'
            self.arm_convergence = {}
            max_passes = 2

            for arm_side in ["right", "left"]:
                self.arm_convergence[arm_side] = False
                pass1_joint_results = {"wrist_pitch": None, "elbow": None}
                self.stage_results[arm_side] = pass1_joint_results
                # Backup of parameters before Pass 1 for early exit / change checking
                prev_j6 = self.joint_offsets_store[arm_side]["joint6"]
                prev_j5 = self.joint_offsets_store[arm_side]["joint5"]
                prev_j3 = self.joint_offsets_store[arm_side]["joint3"]

                # We need nominal bracket baseline to calculate bracket parameter changes
                ver_key = "1.3" if is_v13 else "1.2"
                nominal_vec = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
                tf_vec_init = self.marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
                if tf_vec_init is not None and len(tf_vec_init) == 6:
                    prev_bracket_pos = np.array(tf_vec_init[:3]) * 1000.0
                    prev_bracket_rot = np.array(tf_vec_init[3:])
                else:
                    prev_bracket_pos = np.array(nominal_vec[:3]) * 1000.0
                    prev_bracket_rot = np.array(nominal_vec[3:])

                res_4 = None
                res_5 = None
                res_6 = None
                bracket_completed = False

                for pass_idx in range(1, max_passes + 1):
                    self.log_callback("\n" + "="*50)
                    self.log_callback(f"   STARTING PASS {pass_idx}/{max_passes} FOR {arm_side.upper()} ARM")
                    self.log_callback("="*50 + "\n")
                    self.log_callback(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")

                    for calibrator in [self.joint_calibrator, self.marker_calibrator]:
                        if arm_side not in calibrator.joint_offsets:
                            calibrator.joint_offsets[arm_side] = {}
                        calibrator.joint_offsets[arm_side]["wrist_pitch"] = self.joint_offsets_store[arm_side]["joint5"]
                        if is_v13:
                            calibrator.joint_offsets[arm_side]["wrist_roll"] = self.joint_offsets_store[arm_side]["joint6"]
                            calibrator.joint_offsets[arm_side]["wrist_yaw2"] = 0.0
                        else:
                            calibrator.joint_offsets[arm_side]["wrist_roll"] = 0.0
                            calibrator.joint_offsets[arm_side]["wrist_yaw2"] = self.joint_offsets_store[arm_side]["joint6"]
                        calibrator.joint_offsets[arm_side]["elbow"] = self.joint_offsets_store[arm_side]["joint3"]

                    # --- Step 1: Sequential Calibration Execution ---
                    # Both versions use measured J5 -> J6 -> bracket -> J3 acceptance.
                    # 1. Calibrate J5 (Wrist Pitch) FIRST
                    pass1_res_pitch = pass1_joint_results.get("wrist_pitch")
                    if pass_idx >= 2 and _joint_result_accepted(pass1_res_pitch):
                        self.log_callback(f"[FULL AUTO 1/3] J5 (Wrist Pitch) previously converged ({pass1_res_pitch['recommended_joint_offset']:.4f}°). Skipping Pass {pass_idx} sweep.")
                        opt_pitch = pass1_res_pitch["recommended_joint_offset"]
                        self.joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                        self.marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                        self.joint_offsets_store[arm_side]["joint5"] = opt_pitch
                        self.joint_callback(pass1_res_pitch)
                    else:
                        self.log_callback(f"[FULL AUTO 1/3] Calibrating J5 (Wrist Pitch) first on v{version_num} {arm_side} arm...")
                        for calibrator in [self.joint_calibrator, self.marker_calibrator]:
                            calibrator.joint_offsets[arm_side]["wrist_pitch"] = self.joint_offsets_store[arm_side]["joint5"]
                            calibrator.joint_offsets[arm_side]["wrist_yaw2" if is_v13 else "wrist_roll"] = 0.0
                            calibrator.joint_offsets[arm_side][key6] = self.joint_offsets_store[arm_side]["joint6"]
                            calibrator.joint_offsets[arm_side]["elbow"] = self.joint_offsets_store[arm_side]["joint3"]

                        if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, mode5, log_callback=self.log_callback):
                            raise RuntimeError(f"Failed to move to ready pose for wrist_pitch on {arm_side} arm")
                        if self.stop_event.is_set(): return

                        joint_res_pitch = self._run_joint_calibration(
                            arm_side, mode5,
                            log_callback=self.log_callback,
                            status_callback=self.status_callback,
                            current_offset_deg=self.joint_offsets_store[arm_side]["joint5"],
                            save_debug=self.save_debug,
                            pass_idx=pass_idx,
                            pass1_res=pass1_res_pitch
                        )
                        if not joint_res_pitch:
                            raise RuntimeError(f"Wrist pitch joint calibration failed on {arm_side} arm")
                        pass1_joint_results["wrist_pitch"] = joint_res_pitch
                        joint_res_pitch['arm_side'] = arm_side
                        joint_res_pitch['mode'] = mode5
                        joint_res_pitch['pass_idx'] = pass_idx

                        if _joint_result_accepted(joint_res_pitch):
                            opt_pitch = joint_res_pitch["recommended_joint_offset"]
                            self.joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                            self.marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                            self.joint_offsets_store[arm_side]["joint5"] = opt_pitch

                        plot_path = self.joint_calibrator.save_calibration_comparison_plot(
                            arm_side, mode5, pass1_res_pitch if pass1_res_pitch else joint_res_pitch, joint_res_pitch,
                            log_callback=self.log_callback, force_overwrite=True
                        )
                        if plot_path:
                            joint_res_pitch['plot_path_combined'] = plot_path

                        self.joint_callback(joint_res_pitch)
                        time.sleep(0.5)
                    if self.stop_event.is_set(): return

                    # 2. Calibrate v1.2 J6 using the September 2 saved-reference method before bracket fitting
                    if not _joint_result_accepted(pass1_joint_results.get("wrist_pitch")):
                        self.log_callback("[FULL AUTO] J5 remains unconverged; J6, bracket and elbow deferred.")
                        if pass_idx == max_passes:
                            raise RuntimeError(f"{arm_side}: J5 prerequisite did not converge after {max_passes} passes")
                        continue
                    pass1_res_yaw2 = pass1_joint_results.get("wrist_yaw2")
                    if pass_idx >= 2 and _joint_result_accepted(pass1_res_yaw2):
                        self.log_callback(f"[FULL AUTO 2/3] J6 (Wrist Yaw 2) previously converged ({pass1_res_yaw2['recommended_joint_offset']:.4f}°). Skipping Pass {pass_idx} sweep.")
                        opt_roll = pass1_res_yaw2["recommended_joint_offset"]
                        self.joint_offsets_store[arm_side]["joint6"] = opt_roll
                        self.joint_calibrator.joint_offsets[arm_side][key6] = opt_roll
                        self.marker_calibrator.joint_offsets[arm_side][key6] = opt_roll
                        self.joint_callback(pass1_res_yaw2)
                    else:
                        self.log_callback(f"\n[FULL AUTO] Calibrating J6 (Wrist Yaw 2) using September 2 encoder/circle method...")
                        if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, mode6, log_callback=self.log_callback):
                            raise RuntimeError(f"Failed to move to wrist_yaw2 ready pose on {arm_side} arm")
                        joint_res_roll = self._run_joint_calibration(
                            arm_side, mode6,
                            log_callback=self.log_callback,
                            status_callback=self.status_callback,
                            current_offset_deg=self.joint_offsets_store[arm_side]["joint6"],
                            save_debug=self.save_debug,
                            pass_idx=pass_idx,
                            pass1_res=pass1_res_yaw2
                        )
                        if not joint_res_roll:
                            raise RuntimeError(f"J6 (Wrist Yaw 2) calibration failed on {arm_side} arm")
                        pass1_joint_results["wrist_yaw2"] = joint_res_roll

                        if _joint_result_accepted(joint_res_roll):
                            opt_roll = joint_res_roll["recommended_joint_offset"]
                            self.log_callback(f"[FULL AUTO] Staging J6 offset: {opt_roll:.4f}°")
                            self.joint_offsets_store[arm_side]["joint6"] = opt_roll
                            self.joint_calibrator.joint_offsets[arm_side][key6] = opt_roll
                            self.marker_calibrator.joint_offsets[arm_side][key6] = opt_roll

                        plot_path = self.joint_calibrator.save_calibration_comparison_plot(
                            arm_side, mode6, pass1_res_yaw2 if pass1_res_yaw2 else joint_res_roll, joint_res_roll,
                            log_callback=self.log_callback, force_overwrite=True
                        )
                        if plot_path:
                            joint_res_roll['plot_path_combined'] = plot_path

                        joint_res_roll['arm_side'] = arm_side
                        joint_res_roll['mode'] = mode6
                        joint_res_roll['pass_idx'] = pass_idx
                        self.joint_callback(joint_res_roll)
                        time.sleep(0.5)
                        if self.stop_event.is_set(): return

                    pending = [mode for mode in ("wrist_pitch", "wrist_yaw2")
                               if not _joint_result_accepted(pass1_joint_results.get(mode))]
                    if pending:
                        self.log_callback(f"[FULL AUTO] Pending joints: {', '.join(pending)}. Bracket and elbow deferred.")
                        if pass_idx == max_passes:
                            reason = (pass1_joint_results.get('wrist_yaw2') or {}).get('failure_reason')
                            raise RuntimeError(f"{arm_side}: J5/J6 prerequisites did not converge after {max_passes} passes; bracket was not fitted. {reason or ''}")
                        continue

                    if not bracket_completed:
                        # 3. Marker sweeps with J5/J6 already calibrated and fixed
                        self.log_callback(f"[FULL AUTO 2/3] Performing Marker Bracket Sweeps for both wrist versions {arm_side} arm (Pass {pass_idx}/{max_passes})...")
                        self.log_callback(f"[FULL AUTO] Moving {arm_side} arm to ready pose...")
                        if not self.marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=self.log_callback):
                            raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                        if self.stop_event.is_set(): return

                        state = self.joint_calibrator.robot.get_state()
                        model = self.joint_calibrator.robot.model()
                        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                        first_starting_pose = list(state.position[arm_idx])
                        self.log_callback(f"[FULL AUTO] Sweeping Axis 4...")
                        res_4 = self.marker_calibrator.perform_calibration_sweep(
                            arm_side, 4, log_callback=self.log_callback, status_callback=self.status_callback,
                            save_debug=self.save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                        )
                        if not res_4: raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
                        res_4['axis_mode'] = 4
                        res_4['axis'] = res_4['axis_opt']
                        if self.stop_event.is_set(): return

                        self.log_callback(f"[FULL AUTO] Sweeping Axis 6...")
                        res_6 = self.marker_calibrator.perform_calibration_sweep(
                            arm_side, 6, log_callback=self.log_callback, status_callback=self.status_callback,
                            save_debug=self.save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                        )
                        if not res_6: raise RuntimeError(f"Axis 6 marker sweep failed on {arm_side} arm")
                        res_6['axis_mode'] = 6
                        res_6['axis'] = res_6['axis_opt']
                        if self.stop_event.is_set(): return

                        self.log_callback(f"[FULL AUTO] Sweeping Axis 5...")
                        res_5 = self.marker_calibrator.perform_calibration_sweep(
                            arm_side, 5, log_callback=self.log_callback, status_callback=self.status_callback,
                            save_debug=self.save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                        )
                        if not res_5: raise RuntimeError(f"Axis 5 marker sweep failed on {arm_side} arm")
                        res_5['axis_mode'] = 5
                        res_5['axis'] = res_5['axis_opt']
                        if self.stop_event.is_set(): return

                        # 3. Compute Marker Bracket (1-time lock)
                        self.log_callback("\n[FULL AUTO] Computing unified marker bracket calibration for both wrist versions...")
                        unified_res = self.marker_calibrator.fit_observed_bracket(
                            res_4, res_5, res_6, arm_side
                        )
                        if not unified_res.get('measurement_accepted', False):
                            raise RuntimeError(unified_res.get('failure_reason', 'Bracket measurement rejected'))

                        unified_res['res_5'] = res_5
                        unified_res['res_6'] = res_6
                        if res_4 is not None:
                            unified_res['res_4'] = res_4
                        unified_res['arm_side'] = arm_side
                        unified_res['pass_idx'] = pass_idx

                        plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                        plot_saved = self.marker_calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, arm_side, is_v13, plot_path)
                        if plot_saved:
                            unified_res['plot_path_combined'] = plot_path

                        x_m, y_m, z_m = unified_res['x_e']/1000.0, unified_res['y_e']/1000.0, unified_res['z_e']/1000.0
                        new_vals = [x_m, y_m, z_m, unified_res['roll_e'], unified_res['pitch_e'], unified_res['yaw_e']]
                        key = f"Tf_to_marker_{arm_side}"
                        self.marker_calibrator.camera_config[key] = new_vals
                        self.joint_calibrator.camera_config[key] = new_vals

                        self.bracket_callback(unified_res)
                        pass1_joint_results['bracket'] = unified_res
                        time.sleep(0.5)
                        if self.stop_event.is_set(): return


                        bracket_completed = True

                    # 5. Calibrate J3 Elbow
                    pass1_res_elbow = pass1_joint_results.get("elbow")
                    if pass_idx >= 2 and _joint_result_accepted(pass1_res_elbow):
                        self.log_callback(f"[FULL AUTO 3/3] J3 (Elbow) previously converged ({pass1_res_elbow['recommended_joint_offset']:.4f}°). Skipping Pass {pass_idx} sweep.")
                        opt_elbow = pass1_res_elbow["recommended_joint_offset"]
                        self.joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                        self.marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                        self.joint_offsets_store[arm_side]["joint3"] = opt_elbow
                        self.joint_callback(pass1_res_elbow)
                    else:
                        self.log_callback("[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...")
                        if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=self.log_callback):
                            raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                        if self.stop_event.is_set(): return

                        joint_res_elbow = self._run_joint_calibration(
                            arm_side, "elbow",
                            log_callback=self.log_callback,
                            status_callback=self.status_callback,
                            current_offset_deg=self.joint_offsets_store[arm_side]["joint3"],
                            save_debug=self.save_debug,
                            pass_idx=pass_idx,
                            pass1_res=pass1_res_elbow
                        )
                        if not joint_res_elbow:
                            raise RuntimeError(f"Elbow joint calibration failed on {arm_side} arm")
                        pass1_joint_results["elbow"] = joint_res_elbow
                        joint_res_elbow['arm_side'] = arm_side
                        joint_res_elbow['mode'] = "elbow"
                        joint_res_elbow['pass_idx'] = pass_idx

                        if _joint_result_accepted(joint_res_elbow):
                            opt_elbow = joint_res_elbow["recommended_joint_offset"]
                            self.joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                            self.marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                            self.joint_offsets_store[arm_side]["joint3"] = opt_elbow

                        plot_path = self.joint_calibrator.save_calibration_comparison_plot(
                            arm_side, "elbow", pass1_res_elbow if pass1_res_elbow else joint_res_elbow, joint_res_elbow,
                            log_callback=self.log_callback, force_overwrite=True
                        )
                        if plot_path:
                            joint_res_elbow['plot_path_combined'] = plot_path

                        self.joint_callback(joint_res_elbow)
                        time.sleep(0.5)

                    if not _joint_result_accepted(pass1_joint_results.get("elbow")):
                        self.log_callback(f"[FULL AUTO] {arm_side} elbow remains unconverged; only failed joints will be retried.")
                        if pass_idx == max_passes:
                            raise RuntimeError(f"{arm_side}: elbow did not converge after {max_passes} passes")
                        continue

                    # Pass Evaluation & Convergence Check
                    j6_change = abs(self.joint_offsets_store[arm_side]["joint6"] - prev_j6)
                    j5_change = abs(self.joint_offsets_store[arm_side]["joint5"] - prev_j5)
                    j3_change = abs(self.joint_offsets_store[arm_side]["joint3"] - prev_j3)

                    tf_vec_now = self.marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
                    if tf_vec_now is not None and len(tf_vec_now) == 6:
                        now_bracket_pos = np.array(tf_vec_now[:3]) * 1000.0
                        now_bracket_rot = np.array(tf_vec_now[3:])
                    else:
                        now_bracket_pos = prev_bracket_pos
                        now_bracket_rot = prev_bracket_rot

                    pos_change = np.linalg.norm(now_bracket_pos - prev_bracket_pos)

                    # Compute rotation change in degrees
                    R_prev = R_scipy.from_euler('ZYX', [prev_bracket_rot[2], prev_bracket_rot[1], prev_bracket_rot[0]], degrees=True)
                    R_now = R_scipy.from_euler('ZYX', [now_bracket_rot[2], now_bracket_rot[1], now_bracket_rot[0]], degrees=True)
                    rot_change = np.rad2deg(np.linalg.norm((R_now * R_prev.inv()).as_rotvec()))

                    self.log_callback(f"\n[PASS {pass_idx} EVALUATION] Staged parameter changes for {arm_side.upper()} Arm:")
                    self.log_callback(f"  * Joint 6 Change      : {j6_change:.4f}°")
                    self.log_callback(f"  * Joint 5 Change      : {j5_change:.4f}°")
                    self.log_callback(f"  * Joint 3 Change      : {j3_change:.4f}°")
                    self.log_callback(f"  * Bracket Pos Change  : {pos_change:.4f} mm")
                    self.log_callback(f"  * Bracket Rot Change  : {rot_change:.4f}°")

                    self.arm_convergence[arm_side] = True
                    self.log_callback(f"[PASS {pass_idx} EVALUATION] All measured joint checks and bracket fit accepted.")
                    break

                if self.arm_convergence[arm_side]:
                    self.log_callback(f"[INFO] {arm_side.upper()} arm sequential calibration converged.")
                else:
                    raise RuntimeError(f"{arm_side}: calibration failed after {max_passes} passes; results must not be applied")
                if self.stop_event.is_set(): return
                time.sleep(1.0)

            self.log_callback("\n" + "="*50)
            self.log_callback("   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!")
            self.log_callback("="*50 + "\n")

            # Print Final Calibrated Results Report in the same style as simulated ground truth
            self.log_callback("[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):")
            for arm in ["right", "left"]:
                j_store = self.joint_offsets_store.get(arm, {})
                j6_cal = j_store.get("joint6", 0.0)
                j5_cal = j_store.get("joint5", 0.0)
                j3_cal = j_store.get("joint3", 0.0)

                tf_vec = self.marker_calibrator.camera_config.get(f"Tf_to_marker_{arm}")
                if tf_vec is not None and len(tf_vec) == 6:
                    x_cal = tf_vec[0] * 1000.0
                    y_cal = tf_vec[1] * 1000.0
                    z_cal = tf_vec[2] * 1000.0
                    r_cal = tf_vec[3]
                    p_cal = tf_vec[4]
                    y_cal_deg = tf_vec[5]
                else:
                    ver_key = "1.3" if is_v13 else "1.2"
                    nominal_vec = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
                    x_cal = nominal_vec[0] * 1000.0
                    y_cal = nominal_vec[1] * 1000.0
                    z_cal = nominal_vec[2] * 1000.0
                    r_cal = nominal_vec[3]
                    p_cal = nominal_vec[4]
                    y_cal_deg = nominal_vec[5]

                ver_key = "1.3" if is_v13 else "1.2"
                nominal_vec = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
                x_nom = nominal_vec[0] * 1000.0
                y_nom = nominal_vec[1] * 1000.0
                z_nom = nominal_vec[2] * 1000.0
                r_nom = nominal_vec[3]
                p_nom = nominal_vec[4]
                y_nom_deg = nominal_vec[5]

                dx = x_cal - x_nom
                dy = y_cal - y_nom
                dz = z_cal - z_nom

                from scipy.spatial.transform import Rotation as R_scipy
                R_ideal = R_scipy.from_euler('ZYX', [y_nom_deg, p_nom, r_nom], degrees=True)
                R_actual = R_scipy.from_euler('ZYX', [y_cal_deg, p_cal, r_cal], degrees=True)
                R_offset = R_actual * R_ideal.inv()
                yaw_off, pitch_off, roll_off = R_offset.as_euler('ZYX', degrees=True)

                roll_off = (roll_off + 180) % 360 - 180
                pitch_off = (pitch_off + 180) % 360 - 180
                yaw_off = (yaw_off + 180) % 360 - 180

                self.log_callback(f"  --- {arm.upper()} ARM ---")
                self.log_callback(f"  * Bracket Pos: X: {dx:+.1f}, Y: {dy:+.1f}, Z: {dz:+.1f} mm")
                self.log_callback(f"  * Bracket Rot: R: {roll_off:+.2f}, P: {pitch_off:+.2f}, Y: {yaw_off:+.2f} deg")
                self.log_callback(f"  * Joint Offsets: Joint 6: {j6_cal:+.2f}°, Joint 5: {j5_cal:+.2f}°, Joint 3: {j3_cal:+.2f}°")
            self.log_callback("==================================================\n")
        except Exception as e:
            self.error_msg = str(e)
            self.log_callback(f"[ERROR] Full Auto sequential calibration failed: {e}")
            import traceback
            self.log_callback(traceback.format_exc())
        finally:
            if self.error_msg or self.stop_event.is_set() or not all(self.arm_convergence.get(side, False) for side in ('right', 'left')):
                for calibrator, offsets, camera in snapshots:
                    # Retain normalized zero keys for previously absent offsets,
                    # while restoring all pre-run physical compensation values.
                    for side in ('right', 'left'):
                        current = calibrator.joint_offsets.setdefault(side, {})
                        for key in current:
                            current[key] = offsets.get(side, {}).get(key, 0.)
                        current.update(offsets.get(side, {}))
                    calibrator.camera_config.clear()
                    calibrator.camera_config.update(camera)
                if self.head_camera_calibrator is not None:
                    self.head_camera_calibrator.calibrated_results = head_result
            if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
                self.joint_calibrator.clear_user_taught_ready_poses()


@dataclass
class OptimizerContext:
    """Immutable-by-convention input snapshot, independent of any UI object."""
    robot: object
    model: object
    camera_config: dict
    robot_version: str = '1.2'
    include_head_motion: bool = True
    joint_offsets_store: dict = field(default_factory=dict)
    apply_joint_offset_limits: bool = False
    head_camera_result: dict | None = None
    capture_metadata: dict | None = None
    home_reset_baseline_path: str | None = None
    comparison_baseline_path: str | None = None
    simulation_offsets: dict | None = None
    nominal_brackets: dict = field(default_factory=dict)


def _transform_pose_degrees(pose):
    from scipy.spatial.transform import Rotation
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_euler('ZYX', list(pose[3:])[::-1], degrees=True).as_matrix()
    return result


def run_calibration_optimizer(
    context,
    active_arms,
    optimize_head,
    optimize_camera,
    q_arm_list,
    q_head_list,
    T_meas_list,
    result_path,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    solver_type="QP Solver",
    use_sag=False,
    log_callback=None,
):
    log = log_callback or (lambda message: None)
    if context.model is None:
        raise RuntimeError("Robot is not connected.")

    if len(q_arm_list.shape) == 2 and q_arm_list.shape[1] == 7 and len(active_arms) == 2:
        log("[WARN] q_arm_list has 7 joints but active_arms has 2 arms. Falling back active_arms to single arm ['right'].")
        active_arms = ["right"]

    if len(active_arms) == 1:
        cfg = get_arm_config(context.model, active_arms[0], version=context.robot_version, camera_config=context.camera_config)
        ee_links = {active_arms[0]: cfg["ee_link"]}
        ee_to_marker_nom = {active_arms[0]: cfg["ee_to_marker_nom"]}
    else:
        cfg = get_both_arm_config(context.model, version=context.robot_version, camera_config=context.camera_config)
        ee_links = cfg["ee_links"]
        ee_to_marker_nom = cfg["ee_to_marker_nom"]

    # Override ee_to_marker_nom with actual calibrated values from memory
    for side in active_arms:
        key = f"Tf_to_marker_{side}"
        if key in context.camera_config:
            ee_to_marker_nom[side] = context.camera_config[key]
            log(f"[INFO] Using calibrated marker bracket values for {side}: {ee_to_marker_nom[side]}")

    head_cfg = get_head_config(context.model, camera_config=context.camera_config)
    from core.marker_detection import uses_head_camera
    use_head_kinematics = (
        uses_head_camera(context.camera_config, context.model) and
        (q_head_list is not None) and
        (head_cfg.get("head_idx") is not None) and
        len(head_cfg.get("head_idx", [])) >= 2
    )
    head_idx = head_cfg["head_idx"] if use_head_kinematics else None
    optimize_head = optimize_head and use_head_kinematics and context.include_head_motion
    if uses_head_camera(context.camera_config, context.model) and q_head_list is None:
        raise ValueError('Head-mounted camera dataset needs recorded head encoders, even when head motion is disabled.')

    # Determine initial head offsets if previously calibrated
    q_head_offset_init = None
    head_stored = context.joint_offsets_store.get("head", {})
    if head_stored and head_idx is not None and len(head_idx) >= 2:
        q_head_offset_init = np.radians([head_stored.get("pan", 0.0), head_stored.get("tilt", 0.0)])
        log(f"[INFO] Using stored head offsets as Step 2 initialization: {head_stored}")

    apply_limits = context.apply_joint_offset_limits
    joint_offsets = None
    if apply_limits:
        joint_offsets = {}
        for side in active_arms:
            side_dict = context.joint_offsets_store.get(side, {})
            if not all(k in side_dict for k in ('joint3', 'joint5', 'joint6')):
                raise ValueError(f'Missing Step 1 measurements for {side}; zero bounds will not be invented.')
            joint_offsets[side] = {
                "joint3": side_dict.get("joint3", 0.0),
                "joint5": side_dict.get("joint5", 0.0),
                "joint6": side_dict.get("joint6", 0.0),
            }
        log(f"[INFO] Applying joint offset bounds: {joint_offsets}")

    # Check if Step 1.5 camera results are present
    mount_cam_from_step1_5 = None
    if context.head_camera_result is not None:
        res15 = context.head_camera_result
        if res15 and res15.get('success') and not res15.get("skipped", False) and "calibrated_mount_to_cam" in res15 and context.camera_config.get('extrinsic_source') != 'independent_measurement':
            mount_cam_from_step1_5 = res15["calibrated_mount_to_cam"]

    if len(active_arms) == 2 and q_arm_list.shape[1] >= 14:
        log("\n[INFO] === UNIFIED DUAL-ARM JOINT-CAMERA CALIBRATION WORKFLOW ===")
        cfg_both = get_both_arm_config(context.model, version=context.robot_version, camera_config=context.camera_config)
        mount_cam_init = mount_cam_from_step1_5 or context.camera_config.get("mount_to_cam", cfg_both["mount_to_cam_nom"])
        if mount_cam_from_step1_5:
            log(f"[INFO] Using Step 1.5 effective camera initialization (not independent truth): {mount_cam_init}")
        optimizer = QPCalibrationOptimizer(
            robot=context.robot,
            arm_idx=cfg_both["arm_idx"],
            ee_links=cfg_both["ee_links"],
            mount_to_cam_nom=mount_cam_init,
            head_base_to_cam_nom=cfg_both.get("head_base_to_cam_nom"),
            ee_to_marker_nom=ee_to_marker_nom,
            active_arms=["right", "left"],
            optimize_arm=True,
            optimize_head=optimize_head,
            optimize_camera=optimize_camera,
            head_idx=head_idx,
            use_head_kinematics=use_head_kinematics,
            lambda_cam_pos=lambda_cam_pos,
            lambda_cam_rot=lambda_cam_rot,
            head_tilt_reference_rad=(np.deg2rad(context.camera_config['head_tilt_reference_deg'])
                if context.camera_config.get('head_tilt_reference_deg') is not None else None),
            head_zero_convention=context.camera_config.get('head_zero_convention', 'camera_forward'),
            use_sag=use_sag,
            estimate_measurement_noise=True,
            apply_joint_offset_limits=apply_limits,
            joint_offsets_to_apply=joint_offsets,
            camera_pos_bound_m=0.010,
            camera_rot_bound_rad=3.0 * D2R,
            eps=1e-7,
            max_iter=50,
        )
        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
            q_arm_list, q_head_list, T_meas_list, q_head_offset_init=q_head_offset_init
        )
    else:
        log("\n[INFO] === SINGLE-ARM JOINT-CAMERA CALIBRATION WORKFLOW ===")
        mount_cam_init = mount_cam_from_step1_5 or context.camera_config.get("mount_to_cam", cfg["mount_to_cam_nom"])
        if mount_cam_from_step1_5:
            log(f"[INFO] Using Step 1.5 effective camera initialization (not independent truth): {mount_cam_init}")
        opt_single = QPCalibrationOptimizer(
            robot=context.robot,
            arm_idx=cfg["arm_idx"],
            ee_links=ee_links,
            mount_to_cam_nom=mount_cam_init,
            head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
            ee_to_marker_nom=ee_to_marker_nom,
            active_arms=active_arms,
            optimize_arm=True,
            optimize_head=optimize_head,
            optimize_camera=optimize_camera,
            head_idx=head_idx,
            use_head_kinematics=use_head_kinematics,
            lambda_cam_pos=lambda_cam_pos,
            lambda_cam_rot=lambda_cam_rot,
            use_sag=use_sag,
            head_tilt_reference_rad=(np.deg2rad(context.camera_config['head_tilt_reference_deg'])
                if context.camera_config.get('head_tilt_reference_deg') is not None else None),
            head_zero_convention=context.camera_config.get('head_zero_convention', 'camera_forward'),
            estimate_measurement_noise=True,
            apply_joint_offset_limits=apply_limits,
            joint_offsets_to_apply=joint_offsets,
            camera_pos_bound_m=0.005,
            camera_rot_bound_rad=2.0 * D2R,
            eps=1e-7,
            max_iter=50,
        )
        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = opt_single.optimize(
            q_arm_list, q_head_list, T_meas_list, q_head_offset_init=q_head_offset_init
        )
        optimizer = opt_single

    if len(active_arms) == 1:
        if active_arms[0] == "right":
            right_arm_offset = q_arm_offset
            left_arm_offset = None
        else:
            right_arm_offset = None
            left_arm_offset = q_arm_offset
    else:
        right_arm_offset = q_arm_offset[:7]
        left_arm_offset = q_arm_offset[7:]

    head_base_to_cam_new = [float(x) for x in head_base_to_cam_new] if head_base_to_cam_new else None
    mount_to_cam_new = [float(x) for x in mount_to_cam_new] if mount_to_cam_new else None

    log("\n===== RESULT =====")
    log(f"Calibration diagnostics: {optimizer.last_diagnostics}")
    if not optimizer.last_diagnostics.get('converged') or not optimizer.last_diagnostics.get('observable'):
        raise RuntimeError('Calibration did not converge to an observable solution; no applicable result was saved.')
    log(f"lambda_cam_pos = {lambda_cam_pos}")
    log(f"lambda_cam_rot = {lambda_cam_rot}")
    log(f"measurement_noise = {optimizer.noise_estimator.format()}")

    if right_arm_offset is not None:
        log(f"Right arm joint offset (deg): {np.rad2deg(right_arm_offset)}")

    if left_arm_offset is not None:
        log(f"Left arm joint offset (deg): {np.rad2deg(left_arm_offset)}")
    if q_head_offset is not None:
        h = np.rad2deg(q_head_offset)
        log(f"Head model offset (deg): Pan={h[0]:+.4f}, Tilt={h[1]:+.4f} ({optimizer.last_diagnostics['head_tilt_mode']})")
    forward_zero = optimizer.camera_forward_zero
    if forward_zero is not None:
        pan, tilt = forward_zero['encoder_zero_deg']
        log(f"[CAMERA ZERO] Camera command (Pan=0, Tilt=0) -> encoder Pan={pan:+.4f}°, Tilt={tilt:+.4f}°.")
        log("[CAMERA ZERO] Optical +Z faces torso (link_torso_5) +X in the fitted model; image roll is unchanged.")
        log("[CAMERA ZERO] Separate software command reference, NOT mechanical home offsets. No motion or home write performed.")

    if use_head_kinematics:
        log(f"mount_to_cam xi: {xi_cam}")
        log(f"mount_to_cam_new: {mount_to_cam_new}")
    else:
        log(f"head_base-to-camera xi: {xi_cam}")
        log(f"head_base_to_cam_new: {head_base_to_cam_new}")

    result_dict = {
        "joint_offset_deg": np.rad2deg(q_arm_offset).tolist(),
        "right_arm_joint_offset_deg": np.rad2deg(right_arm_offset).tolist() if right_arm_offset is not None else None,
        "left_arm_joint_offset_deg": np.rad2deg(left_arm_offset).tolist() if left_arm_offset is not None else None,
        "head_joint_offset_deg": np.rad2deg(q_head_offset).tolist() if optimize_head and q_head_offset is not None else None,
        "xi_cam": np.array(xi_cam).tolist(),
        "measurement_noise": optimizer.noise_estimator.as_dict(),
        "diagnostics": optimizer.last_diagnostics,
        "offset_convention": optimizer.last_diagnostics['offset_convention'],
        "capture_metadata": context.capture_metadata,
        "head_tilt_independent": optimizer.last_diagnostics['head_tilt_independent'],
        "camera_forward_zero": forward_zero,
        "head_offset_convention": "model_q_plus_delta; camera_command_zero_is_separate",
    }
    if mount_to_cam_new is not None:
        result_dict["mount_to_cam_new"] = mount_to_cam_new
    if head_base_to_cam_new is not None:
        result_dict["head_base_to_cam_new"] = head_base_to_cam_new

    if context.home_reset_baseline_path is not None and Path(context.home_reset_baseline_path).exists():
        result_dict["home_reset_baseline_path"] = str(context.home_reset_baseline_path)

    if use_head_kinematics:
        result_dict["xi_mount_cam"] = result_dict["xi_cam"]
    else:
        result_dict["xi_head_base_cam"] = result_dict["xi_cam"]

    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    history_path = os.path.join(os.path.dirname(result_path), "calibration_history.txt")
    try:
        with open(history_path, "a") as f:
            import datetime
            f.write(f"\n--- Calibration Iteration: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Result Path: {result_path}\n")
            f.write(f"Right Arm Joint Offset (deg): {result_dict.get('right_arm_joint_offset_deg')}\n")
            f.write(f"Left Arm Joint Offset (deg): {result_dict.get('left_arm_joint_offset_deg')}\n")
            f.write(f"Head Joint Offset (deg): {result_dict.get('head_joint_offset_deg')}\n")
            f.write(f"Head Offset Convention: {result_dict.get('head_offset_convention')}\n")
            f.write(f"Camera Forward Zero: {json.dumps(result_dict.get('camera_forward_zero'))}\n")
            f.write(f"Camera xi: {result_dict.get('xi_cam')}\n")
            f.write(f"Measurement Noise: {json.dumps(result_dict.get('measurement_noise'))}\n")
    except Exception as e:
        log(f"[ERROR] Failed to append to history: {e}")

    # The UI updates its path only after this function returns successfully.
    log(f"Result saved to {result_path}")
    log(f"History appended to {history_path}")

    # Baseline Comparison Output
    baseline_file = context.comparison_baseline_path
    if baseline_file is not None and os.path.exists(baseline_file):
        try:
            with open(baseline_file, "r") as bf:
                b_data = json.load(bf)
            log("\n=========================================================")
            log("  BASE LINE COMPARISON (config/home_reset_baseline.json)")
            log("=========================================================")
            if right_arm_offset is not None and "right_arm_joint_offset_deg" in b_data:
                calc_r = np.rad2deg(right_arm_offset)
                base_r = np.array(b_data["right_arm_joint_offset_deg"])
                diff_r = np.abs(calc_r - base_r)
                log(" [RIGHT ARM]")
                for i in range(len(calc_r)):
                    log(f"   J{i}: Calc = {calc_r[i]:+8.4f}° | Baseline = {base_r[i]:+8.4f}° | Diff = {diff_r[i]:6.4f}°")
            if left_arm_offset is not None and "left_arm_joint_offset_deg" in b_data:
                calc_l = np.rad2deg(left_arm_offset)
                base_l = np.array(b_data["left_arm_joint_offset_deg"])
                diff_l = np.abs(calc_l - base_l)
                log(" [LEFT ARM]")
                for i in range(len(calc_l)):
                    log(f"   J{i}: Calc = {calc_l[i]:+8.4f}° | Baseline = {base_l[i]:+8.4f}° | Diff = {diff_l[i]:6.4f}°")
            log("=========================================================\n")
        except Exception as e:
            log(f"[WARN] Failed to compare with baseline: {e}")

    # Simulation Ground-Truth Comparison Output (Only runs in simulation mode)
    is_sim = context.simulation_offsets is not None

    if is_sim:
        try:
            # Nominal designs are explicit reporting inputs.
            mock_gt = context.simulation_offsets
            is_v13 = (context.robot_version == "1.3")
            ver_key = "1.3" if is_v13 else "1.2"

            log("\n=========================================================")
            log("  SIMULATION GROUND-TRUTH COMPARISON REPORT")
            log("=========================================================")

            # 1. Joint Offsets Comparison
            if right_arm_offset is not None and "right" in mock_gt:
                r_calc = np.rad2deg(right_arm_offset)
                r_gt = [
                    mock_gt["right"].get("joint0", 0.0),
                    mock_gt["right"].get("joint1", 0.0),
                    mock_gt["right"].get("joint2", 0.0),
                    mock_gt["right"].get("joint3", 0.0),
                    mock_gt["right"].get("joint4", 0.0),
                    mock_gt["right"].get("joint5_v13" if is_v13 else "joint5_v12", 0.0),
                    mock_gt["right"].get("joint6", 0.0),
                ]
                log(" [RIGHT ARM JOINTS]")
                for i in range(7):
                    diff = abs(r_calc[i] - r_gt[i])
                    log(f"   J{i}: Calc = {r_calc[i]:+8.4f}° | GT = {r_gt[i]:+8.4f}° | Error = {diff:6.4f}°")

            if left_arm_offset is not None and "left" in mock_gt:
                l_calc = np.rad2deg(left_arm_offset)
                l_gt = [
                    mock_gt["left"].get("joint0", 0.0),
                    mock_gt["left"].get("joint1", 0.0),
                    mock_gt["left"].get("joint2", 0.0),
                    mock_gt["left"].get("joint3", 0.0),
                    mock_gt["left"].get("joint4", 0.0),
                    mock_gt["left"].get("joint5_v13" if is_v13 else "joint5_v12", 0.0),
                    mock_gt["left"].get("joint6", 0.0),
                ]
                log(" [LEFT ARM JOINTS]")
                for i in range(7):
                    diff = abs(l_calc[i] - l_gt[i])
                    log(f"   J{i}: Calc = {l_calc[i]:+8.4f}° | GT = {l_gt[i]:+8.4f}° | Error = {diff:6.4f}°")

            # 2. Head Joint Offsets Comparison
            if optimize_head and q_head_offset is not None and "head" in mock_gt:
                h_calc = np.rad2deg(q_head_offset)
                h_gt = [
                    mock_gt["head"].get("pan", 0.0),
                    mock_gt["head"].get("tilt", 0.0),
                ]
                log(" [HEAD JOINTS]")
                log(f"   Pan:  Calc = {h_calc[0]:+8.4f}° | GT = {h_gt[0]:+8.4f}° | Error = {abs(h_calc[0] - h_gt[0]):6.4f}°")
                if optimizer.last_diagnostics.get('head_tilt_mode') == 'camera_forward_gauge':
                    log(f"   Tilt: camera-forward effective offset = {h_calc[1]:+.4f}° (includes coaxial camera mounting tilt; not a physical GT offset)")
                elif optimizer.last_diagnostics.get('head_tilt_mode') == 'effective_zero_gauge':
                    log(f"   Tilt: effective gauge = {h_calc[1]:+.4f}° (not an independently estimated physical offset; excluded from GT accuracy)")
                else:
                    log(f"   Tilt: Calc = {h_calc[1]:+8.4f}° | GT = {h_gt[1]:+8.4f}° | Error = {abs(h_calc[1] - h_gt[1]):6.4f}°")

            # 3. Marker Bracket Offsets Comparison (relative to Nominal)
            for side in ["right", "left"]:
                key = f"Tf_to_marker_{side}"
                if key in context.camera_config and side in mock_gt:
                    calc_val = context.camera_config[key]
                    nom_val = context.nominal_brackets[ver_key][side]
                    T_nom = _transform_pose_degrees(nom_val)
                    T_cal = _transform_pose_degrees(calc_val)

                    # T_bracket_calc represents the actual translation/rotation of the bracket relative to flange
                    T_bracket_calc = T_cal @ np.linalg.inv(T_nom)
                    # Assembly translation is additive in flange axes.
                    calc_pos_offset = T_cal[:3, 3] - T_nom[:3, 3]

                    from scipy.spatial.transform import Rotation as R_scipy
                    calc_rot_offset = R_scipy.from_matrix(T_bracket_calc[:3, :3]).as_euler('ZYX', degrees=True)[::-1]

                    # Normalize rotation differences to [-180, 180]
                    calc_rot_offset = (calc_rot_offset + 180) % 360 - 180

                    gt_pos_offset = np.array(mock_gt[side]["bracket_pos"])
                    gt_rot_offset = np.array(mock_gt[side]["bracket_rpy"])

                    log(f" [{side.upper()} ARM BRACKET OFFSETS]")
                    pos_norm_mm = np.linalg.norm(calc_pos_offset) * 1000.0
                    if pos_norm_mm > 40.0:
                        log(f"   [ERROR] Bracket position offset exceeded safety threshold: {pos_norm_mm:.1f}mm > 40.0mm!")
                    # Position in mm
                    log(f"   Pos X (mm): Calc = {calc_pos_offset[0]*1000.0:+7.2f} | GT = {gt_pos_offset[0]*1000.0:+7.2f} | Error = {abs(calc_pos_offset[0] - gt_pos_offset[0])*1000.0:5.2f}")
                    log(f"   Pos Y (mm): Calc = {calc_pos_offset[1]*1000.0:+7.2f} | GT = {gt_pos_offset[1]*1000.0:+7.2f} | Error = {abs(calc_pos_offset[1] - gt_pos_offset[1])*1000.0:5.2f}")
                    log(f"   Pos Z (mm): Calc = {calc_pos_offset[2]*1000.0:+7.2f} | GT = {gt_pos_offset[2]*1000.0:+7.2f} | Error = {abs(calc_pos_offset[2] - gt_pos_offset[2])*1000.0:5.2f}")
                    # Rotation in deg
                    log(f"   Rot R (deg): Calc = {calc_rot_offset[0]:+7.2f}° | GT = {gt_rot_offset[0]:+7.2f}° | Error = {abs(calc_rot_offset[0] - gt_rot_offset[0]):5.2f}°")
                    log(f"   Rot P (deg): Calc = {calc_rot_offset[1]:+7.2f}° | GT = {gt_rot_offset[1]:+7.2f}° | Error = {abs(calc_rot_offset[1] - gt_rot_offset[1]):5.2f}°")
                    log(f"   Rot Y (deg): Calc = {calc_rot_offset[2]:+7.2f}° | GT = {gt_rot_offset[2]:+7.2f}° | Error = {abs(calc_rot_offset[2] - gt_rot_offset[2]):5.2f}°")

            log("=========================================================\n")
        except Exception as e:
            log(f"[WARN] Failed to print simulation GT comparison: {e}")

    return result_dict


def capture_calibration_sample(robot, model, marker_transform, *, robot_version='1.2', head_idx=None):
    """Capture dual-arm observations using actual encoder feedback only."""
    if robot is None:
        raise RuntimeError("Robot is not connected. Camera/Sim marker capture requires a connected robot.")
    config = get_both_arm_config(model, version=robot_version)
    return capture_one_sample(robot, config['arm_idx'], marker_transform,
                              head_idx=head_idx, side='all')


def build_capture_metadata(camera_config, robot_version, include_head_motion, *,
                           source_metadata=None, intrinsics=None):
    from copy import deepcopy
    metadata = deepcopy(source_metadata) if source_metadata is not None else {
        'schema_version': 1, 'source': 'live_camera', 'robot_version': robot_version}
    metadata['estimation_camera_snapshot'] = deepcopy(camera_config)
    metadata['intrinsics'] = deepcopy(intrinsics)
    metadata['head_motion_enabled'] = include_head_motion
    return metadata


def prepare_sample_dataset(arm_samples, head_samples, marker_samples, *, active_arms,
                           optimize_head):
    q_arm = np.asarray(arm_samples)
    q_head = np.asarray(head_samples) if len(head_samples) else None
    markers = np.asarray(marker_samples)
    selected = ['right', 'left'] if q_arm.ndim == 2 and q_arm.shape[1] == 14 else active_arms
    validate_dataset(q_arm, q_head, markers, optimize_head, selected)
    return q_arm, q_head, markers, selected


def select_calibration_dataset(arm_samples, head_samples, marker_samples, active_arms):
    q_arm = np.asarray(arm_samples)
    q_head = np.asarray(head_samples) if head_samples is not None and len(head_samples) else None
    markers = np.asarray(marker_samples)
    selected = list(active_arms)
    if q_arm.ndim == 2 and q_arm.shape[1] == 7 and len(selected) == 2:
        selected = ['right']
    if len(selected) == 1:
        side_index = 0 if selected[0] == 'right' else 1
        if q_arm.shape[1] == 14:
            q_arm = q_arm[:, side_index * 7:(side_index + 1) * 7]
        if markers.ndim == 4 and markers.shape[1] == 2:
            markers = markers[:, side_index]
    return q_arm, q_head, markers, selected


@dataclass
class CollectionState:
    motion_plan: list | None = None
    pose_index: int = 0
    ready: bool = False
    base_head_q: object = None
    arm_samples: list = field(default_factory=list)
    head_samples: list = field(default_factory=list)
    marker_samples: list = field(default_factory=list)


@dataclass(frozen=True)
class CapturedSample:
    """Captured observation with its ordinal fixed before queued notification."""
    ordinal: int
    q_arm: np.ndarray
    q_head: np.ndarray | None
    marker: np.ndarray


class AutoCollectionService:
    """Motion and measured-data sequencing with explicit state and notifications."""
    def __init__(self, robot, model, dyn_model, marker_transform, config, state, *,
                 robot_version='1.2', include_head_motion=True, capture_head_idx=None,
                 stop_callback=None, log_callback=None, progress_callback=None, sample_callback=None):
        self.robot = robot
        self.model = model
        self.dyn_model = dyn_model
        self.marker_transform = marker_transform
        self.config = config
        self.state = state
        self.robot_version = robot_version
        self.include_head_motion = include_head_motion
        self.capture_head_idx = capture_head_idx
        self.stopped = stop_callback or (lambda: False)
        self.log = log_callback or (lambda message: None)
        self.progress = progress_callback or (lambda state: None)
        self.sample_callback = sample_callback or (lambda sample: None)

    def build_plan(self):
        from core.robot_motion import build_incremental_motion_plan
        self.state.motion_plan = build_incremental_motion_plan(
            self.robot, self.dyn_model, self.config, ['right', 'left'],
            include_head_motion=self.include_head_motion)
        return self.state.motion_plan

    def step(self):
        from core.robot_motion import execute_auto_motion_step
        state = self.state
        if self.robot is None or self.model is None:
            raise RuntimeError('Robot is not connected.')
        if not state.motion_plan:
            raise RuntimeError('Auto motion plan is empty; build a valid plan before collecting samples')
        if state.pose_index >= len(state.motion_plan):
            self.log('Auto motions have already been executed.')
            return True
        if not state.ready:
            raise RuntimeError('Please move to Init Pose first.')
        if state.pose_index == 0:
            self.log(f'Building motion plan based on current pose... (Angle={self.config.angle_step_deg}deg, Pos={self.config.position_step_m}m, StepX={self.config.step_x_m}m, MaxX={self.config.max_x}m)')
            self.build_plan()
            if not state.motion_plan:
                raise RuntimeError('Auto motion plan is empty; build a valid plan before collecting samples')
        if self.include_head_motion and state.base_head_q is None:
            head_idx = get_head_config(self.model)['head_idx']
            feedback = self.robot.get_state() if head_idx is not None else None
            if feedback is not None and getattr(feedback, 'position', None) is not None:
                state.base_head_q = np.asarray(feedback.position)[list(head_idx)].copy()
                self.log(f'Auto base head pose (deg): {np.round(np.rad2deg(state.base_head_q), 3)}')
            else:
                self.include_head_motion = False
        if self.stopped():
            return False
        step = state.motion_plan[state.pose_index]
        execute_auto_motion_step(robot=self.robot, config=self.config, motion_plan_step=step,
                                 active_arms=['right', 'left'], include_head_motion=self.include_head_motion)
        self.log(f"Auto motion done: {step['desc']}")
        if self.stopped():
            return False
        sample = capture_calibration_sample(self.robot, self.model, self.marker_transform,
            robot_version=self.robot_version, head_idx=self.capture_head_idx)
        q_arm, q_head, marker = sample
        state.pose_index += 1
        if q_arm is None:
            self.progress(state)
            self.log('Capture failed after motion. This pose is skipped.')
            return False
        state.arm_samples.append(q_arm)
        if q_head is not None:
            state.head_samples.append(q_head)
        state.marker_samples.append(marker)
        self.sample_callback(CapturedSample(len(state.arm_samples), q_arm, q_head, marker))
        self.progress(state)
        return True

    def run(self):
        if not self.state.motion_plan:
            raise RuntimeError('Auto motion plan is empty; build a valid plan before collecting samples')
        failures = 0
        while self.state.pose_index < len(self.state.motion_plan):
            if self.stopped():
                self.log('Auto Motion stopped by user.')
                return False
            ok = self.step()
            if self.stopped():
                self.log('Auto Motion stopped by user.')
                return False
            failures = 0 if ok else failures + 1
            if failures:
                self.log(f'[WARNING] Step capture failed ({failures}/3). Skipping this pose...')
                if failures >= 3:
                    raise RuntimeError('Marker not detected 3 consecutive times. Calibration aborted.')
            time.sleep(.2)
        self.log('Auto motions completed.')
        return True


def calibrate_marker_bracket(calibrator, arm_side, use_head_tracking=True, tolerance=.5,
                             save_debug=False, log_callback=None, status_callback=None):
    log = log_callback or (lambda message: None)
    status = status_callback or (lambda detected: None)
    try:
        version_num = calibrator.get_robot_version()
        is_v13 = calibrator.is_v13()

        # Automatically move to ready pose first to guarantee calibration starting pose consistency
        log("[INFO] Automatically moving active arm to marker ready pose...")
        success = calibrator.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=log)
        if not success:
            log("[ERROR] Failed to move to marker ready pose at startup. Aborting.")
            return None
        state = calibrator.robot.get_state()
        model = calibrator.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        first_starting_pose = list(state.position[arm_idx])
        if True: # Always sweep J4 for both v1.2 and v1.3 to get full 3D calibration
            # Stage 1 Axis 4 sweep starts immediately from the initial/current pose
            if getattr(calibrator, 'stop_requested', False):
                return None

            log("\n" + "="*50)
            log("   [Stage 1/3] Sweeping Axis 4 (Wrist Yaw)...")
            log("="*50 + "\n")
            res_4 = calibrator.perform_calibration_sweep(
                arm_side, 4,
                log_callback=log,
                status_callback=status,
                use_head_tracking=use_head_tracking,
                save_debug=save_debug,
                initial_joint_pos=first_starting_pose
            )
            if not res_4:
                log("[ERROR] Stage 1 (Axis 4) sweep failed. Aborting.")
                return None
            res_4['axis_mode'] = 4
            res_4['axis'] = res_4['axis_opt']

            if getattr(calibrator, 'stop_requested', False):
                return None

            time.sleep(1.0)

        # Stage 2/3 Axis 6 Sweep
        log("\n" + "="*50)
        log("   [Stage 2/3] Sweeping Axis 6 (Roll)...")
        log("="*50 + "\n")

        res_6 = calibrator.perform_calibration_sweep(
            arm_side, 6,
            log_callback=log,
            status_callback=status,
            use_head_tracking=use_head_tracking,
            save_debug=save_debug,
            initial_joint_pos=first_starting_pose
        )
        if not res_6:
            log("[ERROR] Stage 6 sweep failed. Aborting.")
            return None

        res_6['axis_mode'] = 6
        res_6['axis'] = res_6['axis_opt']

        if getattr(calibrator, 'stop_requested', False):
            return None

        time.sleep(1.0)

        # Stage 3/3 Axis 5 Sweep
        log("\n" + "="*50)
        log("   [Stage 3/3] Sweeping Axis 5 (Pitch)...")
        log("="*50 + "\n")

        res_5 = calibrator.perform_calibration_sweep(
            arm_side, 5,
            log_callback=log,
            status_callback=status,
            use_head_tracking=use_head_tracking,
            save_debug=save_debug,
            initial_joint_pos=first_starting_pose
        )
        if not res_5:
            log("[ERROR] Stage 5 sweep failed. Aborting.")
            return None

        res_5['axis_mode'] = 5
        res_5['axis'] = res_5['axis_opt']

        # Compute unified bracket calibration
        log("\n[PROCESSING] Computing unified bracket calibration parameters...")
        unified_res = calibrator.fit_observed_bracket(
            res_4, res_5, res_6, arm_side
        )
        if not unified_res.get('measurement_accepted', False):
            raise RuntimeError(unified_res.get('failure_reason', 'Bracket measurement rejected'))

        unified_res['res_5'] = res_5
        unified_res['res_6'] = res_6
        if res_4 is not None:
            unified_res['res_4'] = res_4

        # Save plot using the calibrator method
        plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
        plot_saved = calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, arm_side, is_v13, plot_path)

        if plot_saved:
            unified_res['plot_path_combined'] = plot_path
        return unified_res
    except Exception as e:
        log(f"[ERROR] Worker exception: {e}")
        log(traceback.format_exc())
        return None


def prepare_full_auto_calibration(joint_calibrator, marker_calibrator, log_callback=None):
    log = log_callback or (lambda message: None)
    try:
        error = None
        log("Moving robot arms to Full Auto initial ready poses...")
        version_num = marker_calibrator.get_robot_version()
        is_v13 = marker_calibrator.is_v13()
        log(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")


        for arm_side in ["right", "left"]:
            log(f"Preparing {arm_side.upper()} arm...")
            if not is_v13:
                log(f"Moving {arm_side} arm to wrist pitch ready pose...")
                if not joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch", log_callback=log):
                    raise RuntimeError(f"Failed to move {arm_side} arm to wrist pitch ready pose.")
            else:
                log(f"Moving {arm_side} arm to marker ready pose...")
                if not marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=log):
                    raise RuntimeError(f"Failed to move {arm_side} arm to marker ready pose.")
        log("All arms moved to initial ready poses successfully.")
    except Exception as e:
        error = str(e)
        log(f"[ERROR] Ready pose movement failed: {e}")
    return error


def prepare_taught_ready_pose(robot, marker_calibrator, arm_side, active_mode, log_callback=None):
    """Preserve calibration-axis reference angles and validate taught encoder limits."""
    log = log_callback or (lambda message: None)
    state = robot.get_state()
    model = robot.model()
    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
    taught_pose = list(state.position[arm_idx])
    norm_mode = "wrist_pitch" if active_mode == "wrist_pitch_v13" else ("wrist_roll" if active_mode == "wrist_roll_v13" else active_mode)

    version_key = "v1.3" if (marker_calibrator is not None and marker_calibrator.is_v13()) else "v1.2"
    if norm_mode == "marker":
        type_key = "marker"
        ready_mode = None
    else:
        type_key = "joint"
        if norm_mode == "wrist_pitch":
            ready_mode = "wrist_pitch"
        elif norm_mode in ("wrist_roll", "wrist_yaw2"):
            ready_mode = "wrist_roll_v13" if version_key == "v1.3" else "wrist_yaw2"
        elif norm_mode == "elbow":
            ready_mode = "elbow"
        else:
            ready_mode = "wrist_pitch"

    if marker_calibrator is not None:
        try:
            nom_pose = marker_calibrator.get_ready_pose(version_key, type_key, ready_mode, arm_side)
            if norm_mode == "elbow":
                taught_pose[3] = nom_pose[3]
            elif norm_mode == "wrist_pitch":
                taught_pose[5] = nom_pose[5]
            elif norm_mode in ("marker", "wrist_roll", "wrist_yaw2"):
                taught_pose[5] = nom_pose[5]
        except Exception as e:
            log(f"[WARN] Could not enforce nominal target angle on taught pose: {e}")

    # Check if the joint values are valid within the robot's operating range
    dyn_model = robot.get_dynamics()
    state_lim = dyn_model.make_state([f"ee_{arm_side}"], model.robot_joint_names)
    q_lower_all = np.array(dyn_model.get_limit_q_lower(state_lim))
    q_upper_all = np.array(dyn_model.get_limit_q_upper(state_lim))

    invalid_joints = []
    for i in range(7):
        j_val = taught_pose[i]
        g_idx = arm_idx[i]
        low_lim = q_lower_all[g_idx]
        upp_lim = q_upper_all[g_idx]
        if j_val < low_lim or j_val > upp_lim:
            invalid_joints.append((i, np.degrees(j_val), np.degrees(low_lim), np.degrees(upp_lim)))

    return taught_pose, norm_mode, invalid_joints
