from core.storage import ConfigStorage
from core.storage import FileStorage
import os
import time
import threading
import yaml
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares
from .CalibratorBase import BaseCalibrator

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

class HeadCameraCalibrator(BaseCalibrator):
    """
    Step 1.5 Calibrator:
    Decouples Head Pan/Tilt joints and Camera Extrinsics (mount_to_cam) from the arm/shoulder joints.
    Uses stationary arm markers and sweeps only the head pan and tilt joints.
    """
    def __init__(self, marker_st=None, robot=None):
        super().__init__(marker_st, robot)
        self.calibrated_results = None

    @staticmethod
    def fit_plane_normal_svd(points):
        """
        Fits plane normal using SVD on 3D points.
        Returns unit normal vector and centroid.
        """
        pts = np.array(points)
        if len(pts) < 3:
            raise ValueError(f"Need at least 3 points for plane fit, got {len(pts)}")
        centroid = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[2, :]
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-9:
            raise ValueError("Degenerate plane fit in SVD")
        return normal / norm_val, centroid

    def perform_move_to_ready_pose(self, arm_side="both", log_callback=None, stop_event=None):
        """
        Moves the robot to the dual-arm Cartesian Init Pose (same as Step 2)
        so that both arm markers are positioned in front of the camera.
        """
        if log_callback:
            log_callback("[INFO] Moving robot to Head-Camera Calibration Ready Pose (Dual-arm Init Pose)...")

        if self.robot is None:
            if log_callback: log_callback("[ERROR] Robot is not connected.")
            return False

        has_head = getattr(self, "include_head_motion", True)
        if hasattr(self, "robot") and hasattr(self.robot, "model"):
            try:
                m = self.robot.model()
                has_head = has_head and (hasattr(m, 'head_idx') and m.head_idx is not None and len(m.head_idx) >= 2)
            except Exception:
                pass

        if not has_head:
            if log_callback:
                log_callback("[INFO] Headless mode detected (no head joints). Head & Camera calibration is skipped.")
            return True

        if stop_event and stop_event.is_set():
            return False

        try:
            from core.robot.motion import move_to_auto_ready_pose
            robot_ver = getattr(self, "robot_version", "1.2")
            move_to_auto_ready_pose(
                robot=self.robot,
                active_arms=["right", "left"],
                minimum_time=5.0,
                priority=10,
                include_head_motion=True,
                robot_version=robot_ver
            )
            time.sleep(1.0)
            # Verify marker visibility (check right or left)
            if self.marker_st is not None:
                m_pose_r = self.marker_st.get_marker_transform(sampling_time=1.0, side="right")
                m_pose_l = self.marker_st.get_marker_transform(sampling_time=1.0, side="left")
                if m_pose_r is not None or m_pose_l is not None:
                    found_sides = []
                    if m_pose_r is not None: found_sides.append("right")
                    if m_pose_l is not None: found_sides.append("left")
                    if log_callback: log_callback(f"[SUCCESS] Ready Pose reached. Visible markers: {', '.join(found_sides)}.")
                    return True
                else:
                    if log_callback: log_callback("[WARN] Ready Pose reached, but neither marker was immediately detected. Check lighting.")
                    return True
            if log_callback: log_callback("[SUCCESS] Ready Pose reached.")
            return True
        except Exception as e:
            if log_callback: log_callback(f"[ERROR] Failed to move to Ready Pose: {e}")
            return False

    def _detect_marker_point(self, arm_side="auto", sampling_time=0.5):
        """Helper to sample 3D marker position from camera stream."""
        if self.marker_st is None:
            return None, None
        m_res = None
        detected_side = arm_side
        if arm_side in ["right", "left"]:
            m_res = self.marker_st.get_marker_transform(sampling_time=sampling_time, side=arm_side)
        else:
            # Check right marker first, fallback to left marker
            m_res = self.marker_st.get_marker_transform(sampling_time=sampling_time, side="right")
            if m_res:
                detected_side = "right"
            else:
                m_res = self.marker_st.get_marker_transform(sampling_time=sampling_time, side="left")
                if m_res:
                    detected_side = "left"
        if not m_res:
            return None, None
        if isinstance(m_res, list):
            T_cam_to_marker = np.array(m_res[0]).reshape(4, 4)
        elif isinstance(m_res, dict):
            T_cam_to_marker = np.array(list(m_res.values())[0]).reshape(4, 4)
        else:
            T_cam_to_marker = np.array(m_res).reshape(4, 4)
        return T_cam_to_marker[:3, 3], detected_side

    # Head joints are swept continuously within these ranges (marker must stay in view; on the real
    # v1.2 ready pose the marker left the image from about +12 deg tilt).
    MAX_HEAD_SWEEP_RANGE_DEG = 15.0
    MAX_HEAD_TILT_RANGE_DEG = 10.0
    MIN_HEAD_SWEEP_POINTS = 20
    MAX_HEAD_SWEEP_SOLVER_POINTS = 150

    def perform_head_sweep(
        self,
        arm_side="auto",
        pan_range_deg=15.0,
        tilt_range_deg=10.0,
        num_steps=11,
        step_delay=0.8,
        log_callback=None,
        status_callback=None,
        stop_event=None,
        save_debug=True,
        allow_readjust=True,
        sweep_duration_s=15.0
    ):
        """
        Executes continuous Tilt and Pan sweeps with stationary arm markers:
        1. Sweep Tilt -range -> +range -> -range while Pan=0, sampling the marker continuously.
        2. Sweep Pan  -range -> +range -> -range while Tilt=0, sampling the marker continuously.
        3. Solve camera mount rotation (y/z) and Head Pan & Tilt joint zero offsets.
        num_steps only sets the coverage bins used to detect where the marker was out of view.
        """
        for label, value, limit in (("Pan", pan_range_deg, self.MAX_HEAD_SWEEP_RANGE_DEG), ("Tilt", tilt_range_deg, self.MAX_HEAD_TILT_RANGE_DEG)):
            if value > limit and log_callback:
                log_callback(f"[WARN] {label} sweep range ±{value:.1f}° limited to ±{limit:.1f}°.")
        pan_range_deg = min(float(pan_range_deg), self.MAX_HEAD_SWEEP_RANGE_DEG)
        tilt_range_deg = min(float(tilt_range_deg), self.MAX_HEAD_TILT_RANGE_DEG)
        num_steps = max(int(num_steps), 2)

        if log_callback:
            log_callback("=" * 60)
            log_callback(f"[Head-Camera Calib] Commencing Head-Camera Extrinsic Calibration")
            log_callback(f"  Target Marker Arm : {arm_side}")
            log_callback(f"  Pan Sweep Range   : ±{pan_range_deg:.1f}° (continuous, {sweep_duration_s:.0f}s per direction)")
            log_callback(f"  Tilt Sweep Range  : ±{tilt_range_deg:.1f}° (continuous, {sweep_duration_s:.0f}s per direction)")
            log_callback("=" * 60)

        # Per-step "Marker Cam Pos" readings are noisy at GUI scale (up to
        # num_steps=11 lines per sweep phase) and only useful for offline
        # debugging -- route them to a debug file instead of log_callback;
        # the GUI still sees phase headers, [WARN] (marker lost), and the
        # final [Step 1.5 Calibration Results] block below.
        try:
            from core.storage import CONFIG_PATHS
            debug_txt_dir = CONFIG_PATHS["txt_dir"]
            FileStorage.ensure_dir(debug_txt_dir, exist_ok=True)
            sweep_debug_path = os.path.join(debug_txt_dir, "head_camera_sweep_debug.txt")
            FileStorage.open(sweep_debug_path, "w", encoding="utf-8").close()
        except Exception:
            sweep_debug_path = None

        def _sweep_debug_write(line):
            if not sweep_debug_path:
                return
            try:
                with FileStorage.open(sweep_debug_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        has_head = getattr(self, "include_head_motion", True)
        head_idx = [0, 1]
        if hasattr(self, "robot") and hasattr(self.robot, "model"):
            try:
                m = self.robot.model()
                if hasattr(m, 'head_idx') and m.head_idx is not None and len(m.head_idx) >= 2:
                    head_idx = list(m.head_idx)
                else:
                    has_head = False
            except Exception:
                pass

        nom_mount = list(self.camera_config.get("mount_to_cam_nominal", self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])))

        if not has_head:
            if log_callback:
                log_callback("[INFO] Headless mode: Robot has no head joints. Skipping Head-Camera extrinsic calibration.")
            res = {
                "success": True,
                "skipped": True,
                "head_offsets_deg": {"pan": 0.0, "tilt": 0.0},
                "camera_delta_rpy_deg": [0.0, 0.0, 0.0],
                "calibrated_mount_to_cam": nom_mount,
                "nominal_mount_to_cam": nom_mount,
            }
            self.calibrated_results = res
            return res

        if self.robot is None or not hasattr(self.robot, "get_state"):
            if log_callback: log_callback("[ERROR] Head sweep requires connected robot.")
            return None

        # Nominal CAD parameters (Loaded directly from setting.yaml camera_config)
        nominal_mount_to_cam = nom_mount
        R_nom = R_scipy.from_euler('ZYX', [nominal_mount_to_cam[5], nominal_mount_to_cam[4], nominal_mount_to_cam[3]], degrees=True).as_matrix()

        # Check marker visibility at Ready Pose (head = [0, 0])
        obs_r_0, _ = self._detect_marker_point(arm_side="right", sampling_time=0.5)
        obs_l_0, _ = self._detect_marker_point(arm_side="left", sampling_time=0.5)

        found_sides = []
        if obs_r_0 is not None: found_sides.append("right")
        if obs_l_0 is not None: found_sides.append("left")
        if log_callback:
            log_callback(f"[INFO] Stationary markers detected at center: {', '.join(found_sides) if found_sides else 'None'}")

        if arm_side in ["right", "left"]:
            active_side = arm_side
        else:
            active_side = "right" if obs_r_0 is not None else ("left" if obs_l_0 is not None else "right")

        if log_callback:
            log_callback(f"[INFO] Active marker for Head sweep: {active_side}")

        # Every marker visible at the head centre is tracked: two stationary markers break the
        # head-pan <-> marker-position degeneracy of a single marker and decouple tilt from the
        # camera position.
        tracked_sides = [s for s, obs in (("right", obs_r_0), ("left", obs_l_0)) if obs is not None] or [active_side]
        if log_callback:
            log_callback(f"[INFO] Markers used for Head sweep: {', '.join(tracked_sides)}")

        marker_priors = {}
        for side in tracked_sides:
            try:
                dyn_model = self.robot.get_dynamics()
                q_current = np.array(self.robot.get_state().position)
                T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_current, f"ee_{side}", "link_torso_5")
                suffix = "_v13" if self.is_v13() else "_v12"
                tf_vec = self.camera_config.get(f"Tf_to_marker_{side}{suffix}")
                if tf_vec is None:
                    tf_vec = self.camera_config.get(f"Tf_to_marker_{side}")
                if tf_vec is None:
                    tf_vec = self.NOMINAL_BRACKET_TEMPLATES["1.3" if self.is_v13() else "1.2"][side]
                T_ee_to_marker = BaseCalibrator.make_transform(tf_vec)
                marker_priors[side] = (T_t5_to_ee @ T_ee_to_marker)[:3, 3]
            except Exception:
                pass

        def head_cmd(joint_pos, deg):
            return [0.0, deg * D2R] if joint_pos == 1 else [deg * D2R, 0.0]

        def sweep_head_joint(name, joint_pos, range_deg):
            """
            Continuously sweeps one head joint (joint_pos: 0 = pan, 1 = tilt) -range -> +range -> -range.
            Records the head encoder history (time, angle) and, per frame, every tracked marker's
            camera-frame position with its capture time, so the solver can estimate the
            camera-vs-encoder latency instead of relying on the out-and-back cancellation alone.
            Returns a phase dict or None on stop.
            """
            if not self.movej(self.robot, head=head_cmd(joint_pos, -range_deg), minimum_time=2.0, apply_offsets=False):
                if log_callback: log_callback(f"  [WARN] Head move to {name} start ({-range_deg:.1f}°) failed.")
            time.sleep(step_delay)

            enc_t, enc_deg, samples = [], [], []
            last_pose = {side: None for side in tracked_sides}
            for target_deg in (range_deg, -range_deg):
                motion = {"ok": False}

                def run_motion(target=target_deg):
                    motion["ok"] = self.movej(self.robot, head=head_cmd(joint_pos, target),
                                              minimum_time=sweep_duration_s, apply_offsets=False)

                thread = threading.Thread(target=run_motion, name=f"head-{name.lower()}-sweep", daemon=True)
                thread.start()
                try:
                    while thread.is_alive():
                        if stop_event and stop_event.is_set():
                            try:
                                self.robot.cancel_control()
                            except Exception:
                                pass
                            thread.join()
                            if log_callback: log_callback("[INFO] Calibration aborted by user.")
                            return None
                        try:
                            t_q = time.monotonic()
                            q_cur = np.array(self.robot.get_state().position)
                        except Exception:
                            time.sleep(0.01)
                            continue
                        enc_now = float(np.degrees(q_cur[head_idx[joint_pos]]))
                        enc_t.append(t_q)
                        enc_deg.append(enc_now)
                        for side in tracked_sides:
                            t_call = time.monotonic()
                            res = self.marker_st.get_marker_transform(sampling_time=0, side=side, use_filter=False, q_encoder=q_cur) if self.marker_st is not None else None
                            t_img = 0.5 * (t_call + time.monotonic())
                            if not res:
                                continue
                            flat = res[0] if isinstance(res, list) else (list(res.values())[0] if isinstance(res, dict) else res)
                            pose = np.array(flat, dtype=float).reshape(4, 4)
                            if np.linalg.norm(pose[:3, 3]) <= 0.01 or (last_pose[side] is not None and np.allclose(last_pose[side], pose, atol=1e-7)):
                                continue
                            last_pose[side] = pose
                            samples.append({"t": t_img, "side": side, "deg": enc_now, "p": pose[:3, 3].copy()})
                            _sweep_debug_write(f"  {name}={enc_now:+7.3f}° t={t_img:.4f} side={side} -> Marker Cam Pos: [{pose[0, 3]*1000:+7.2f}, {pose[1, 3]*1000:+7.2f}, {pose[2, 3]*1000:+7.2f}] mm")
                        time.sleep(0.01)
                finally:
                    if thread.is_alive():
                        thread.join()
                if not motion["ok"] and log_callback:
                    log_callback(f"  [WARN] Head {name} sweep motion toward {target_deg:+.1f}° did not finish cleanly.")

            # Coverage: split the range into num_steps bins; a bin with no detection = marker out of view there.
            edges = np.linspace(-range_deg, range_deg, num_steps + 1)
            angles = np.asarray([s["deg"] for s in samples], dtype=float)
            counts, _ = np.histogram(angles, bins=edges) if len(angles) else (np.zeros(num_steps), edges)
            missing = int(np.sum(counts == 0))
            if missing and log_callback:
                empty = [f"{edges[i]:+.1f}~{edges[i+1]:+.1f}°" for i in range(num_steps) if counts[i] == 0]
                log_callback(f"  [WARN] {name} sweep: marker not detected in {missing}/{num_steps} range bins ({', '.join(empty)}).")
            if log_callback:
                per_side = ", ".join(f"{side} {sum(1 for s in samples if s['side'] == side)}" for side in tracked_sides)
                log_callback(f"  [INFO] {name} sweep collected {len(samples)} marker samples ({per_side}).")
            return {"joint": joint_pos, "name": name, "enc_t": np.asarray(enc_t), "enc_deg": np.asarray(enc_deg),
                    "samples": samples, "missing": missing}

        def downsample(phase):
            kept = []
            for side in tracked_sides:
                side_samples = [s for s in phase["samples"] if s["side"] == side]
                if len(side_samples) > self.MAX_HEAD_SWEEP_SOLVER_POINTS:
                    idx = np.round(np.linspace(0, len(side_samples) - 1, self.MAX_HEAD_SWEEP_SOLVER_POINTS)).astype(int)
                    side_samples = [side_samples[i] for i in idx]
                kept.extend(side_samples)
            return dict(phase, samples=kept)

        def request_readjustment(name, missing):
            """2+ missed steps: back to the ready pose and ask the user to readjust once. True = re-run sweeps."""
            if not allow_readjust:
                if log_callback: log_callback(f"[WARN] {name} sweep still missed {missing} range bin(s) after readjustment; continuing with the collected points.")
                return False
            callback = getattr(self, "marker_problem_callback", None)
            if log_callback:
                log_callback(f"[WARN] {name} sweep missed {missing}/{num_steps} range bins (marker out of view). Returning to ready pose for readjustment...")
            self.movej(self.robot, head=[0.0, 0.0], minimum_time=2.0, apply_offsets=False)
            if not self.perform_move_to_ready_pose(arm_side="both", log_callback=log_callback, stop_event=stop_event):
                if log_callback: log_callback("[WARN] Ready pose move failed; continuing with the collected points.")
                return False
            if callback is None:
                if log_callback: log_callback("[WARN] No readjustment prompt available; continuing with the collected points.")
                return False
            if not callback(active_side, mode="head_camera"):
                if log_callback: log_callback("[WARN] Readjustment cancelled; continuing with the collected points.")
                return False
            if log_callback: log_callback("[INFO] Posture readjusted. Restarting head Tilt and Pan sweeps...")
            return True

        # ----------------------------------------------------
        # Phase A: Head Tilt Sweep (Pan = 0, Tilt: -range to +range)
        # ----------------------------------------------------
        if log_callback: log_callback("\n[Phase A] Sweeping Head Tilt Joint (-tilt to +tilt and back, continuous)...")
        tilt_phase = sweep_head_joint("Tilt", 1, tilt_range_deg)
        if tilt_phase is None:
            return None
        self.partial_data = {"tilt_phase": tilt_phase}
        if tilt_phase["missing"] >= 2 and request_readjustment("Tilt", tilt_phase["missing"]):
            return self.perform_head_sweep(arm_side, pan_range_deg, tilt_range_deg, num_steps, step_delay,
                                           log_callback, status_callback, stop_event, save_debug, allow_readjust=False,
                                           sweep_duration_s=sweep_duration_s)

        if len(tilt_phase["samples"]) < self.MIN_HEAD_SWEEP_POINTS:
            raise RuntimeError(f"Insufficient marker points collected during Tilt sweep ({len(tilt_phase['samples'])} points). Calibration cannot proceed.")

        # Return head to zero center before Pan sweep
        if log_callback: log_callback("\n[INFO] Returning head to center before Pan sweep...")
        self.movej(self.robot, head=[0.0, 0.0], minimum_time=1.5, apply_offsets=False)
        time.sleep(1.0)

        # ----------------------------------------------------
        # Phase B: Head Pan Sweep (Tilt = 0, Pan: -range to +range)
        # ----------------------------------------------------
        if log_callback: log_callback("\n[Phase B] Sweeping Head Pan Joint (-pan to +pan and back, continuous)...")
        pan_phase = sweep_head_joint("Pan", 0, pan_range_deg)
        if pan_phase is None:
            return None
        self.partial_data["pan_phase"] = pan_phase
        if pan_phase["missing"] >= 2 and request_readjustment("Pan", pan_phase["missing"]):
            # The arms moved: tilt data from before the readjustment is no longer consistent.
            return self.perform_head_sweep(arm_side, pan_range_deg, tilt_range_deg, num_steps, step_delay,
                                           log_callback, status_callback, stop_event, save_debug, allow_readjust=False,
                                           sweep_duration_s=sweep_duration_s)

        if len(pan_phase["samples"]) < self.MIN_HEAD_SWEEP_POINTS:
            raise RuntimeError(f"Insufficient marker points collected during Pan sweep ({len(pan_phase['samples'])} points). Calibration cannot proceed.")

        # Return head to zero center
        self.movej(self.robot, head=[0.0, 0.0], minimum_time=2.0, apply_offsets=False)

        # ----------------------------------------------------
        # Phase C: Mathematical Solution for Decoupled Head-Camera Calib
        # ----------------------------------------------------
        return self._solve_head_camera(
            [downsample(tilt_phase), downsample(pan_phase)], nominal_mount_to_cam, R_nom,
            marker_priors=marker_priors, log_callback=log_callback
        )

    # Step 1.5 estimation bounds
    CAMERA_POS_BOUND_M = 0.025
    MARKER_POS_BOUND_M = 0.02
    LATENCY_BOUNDS_S = (-0.05, 0.25)
    DIRECTION_TILT_WARN_DEG = 0.1

    def _compute_head_camera_solution(
        self,
        pts_tilt_cam,
        pts_pan_cam,
        captured_tilt_angles,
        captured_pan_angles,
        nominal_mount_to_cam,
        R_nom,
        obs_r_0=None,
        obs_l_0=None,
        active_side="right",
        P_marker_t5_nom=None,
        log_callback=None
    ):
        """Single-marker, untimed sweep points (legacy input form) -> _solve_head_camera."""
        def phase(joint_pos, name, pts, angles):
            return {"joint": joint_pos, "name": name, "enc_t": None, "enc_deg": None, "missing": 0,
                    "samples": [{"t": None, "side": active_side, "deg": float(a), "p": np.asarray(p, dtype=float)}
                                for p, a in zip(pts, angles)]}
        priors = {active_side: P_marker_t5_nom} if P_marker_t5_nom is not None else {}
        return self._solve_head_camera(
            [phase(1, "Tilt", pts_tilt_cam, captured_tilt_angles), phase(0, "Pan", pts_pan_cam, captured_pan_angles)],
            nominal_mount_to_cam, R_nom, marker_priors=priors, log_callback=log_callback)

    def _solve_head_camera(self, phases, nominal_mount_to_cam, R_nom, marker_priors=None, log_callback=None):
        """
        Joint least-squares over continuous head Tilt/Pan sweep samples.

        Unknowns:
          - camera-mount rotation about the optical (camera z) axis only. Rotation about the camera
            x (head tilt) and y (head pan) axes points the camera exactly like the head joints, so by
            convention (notion References 3.2, "camera forward") the head offsets absorb it.
          - head tilt / pan zero offsets
          - camera-mount position (±CAMERA_POS_BOUND_M around CAD). Real 2026-09-15 data: with position
            held at CAD the tilt stayed near 0.2 deg although the head visibly needed ~1.2 deg; freeing it
            gave tilt 1.17 deg, camera x -2.1 mm and 3x lower residual.
          - camera-vs-encoder latency (seconds), when capture times are available: the measured frame
            is matched to the head angle at (t - latency) interpolated from the encoder history.
            Out-only and back-only solves then agree (1.160 / 1.169 deg on the same data).
          - stationary position (link_torso_5) of every tracked marker (nuisance, ±MARKER_POS_BOUND_M).
        """
        marker_priors = dict(marker_priors or {})
        nominal_mount_to_cam = list(nominal_mount_to_cam)
        nom_t = np.array(nominal_mount_to_cam[:3], dtype=np.float64)
        nom_roll, nom_pitch, nom_yaw = (float(v) for v in nominal_mount_to_cam[3:6])

        head_idx = [0, 1]
        try:
            m = self.robot.model()
            if hasattr(m, 'head_idx') and m.head_idx is not None and len(m.head_idx) >= 2:
                head_idx = list(m.head_idx)
        except Exception:
            pass
        dyn_model = self.robot.get_dynamics()
        q_base = np.array(self.robot.get_state().position, dtype=np.float64)

        def fk_t5_to_head2(pan_val, tilt_val):
            q_full = q_base.copy()
            q_full[head_idx[0]] = pan_val
            q_full[head_idx[1]] = tilt_val
            return BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")

        # Flatten samples: (phase index, joint_pos, time, encoder deg, side, measured point, out/back flag)
        flat = []
        for ph_i, ph in enumerate(phases):
            enc_t, enc_deg = ph.get("enc_t"), ph.get("enc_deg")
            t_turn = None
            if enc_t is not None and len(enc_t) > 1:
                t_turn = float(enc_t[int(np.argmax(enc_deg))])
            for s in ph["samples"]:
                outbound = True if (t_turn is None or s["t"] is None) else (s["t"] <= t_turn)
                flat.append((ph_i, ph["joint"], s["t"], float(s["deg"]), s["side"], np.asarray(s["p"], dtype=float), outbound))
        if not flat:
            raise RuntimeError("Head sweep produced no marker samples.")
        sides = sorted({f[4] for f in flat})
        timed = all(f[2] is not None for f in flat) and all(ph.get("enc_t") is not None and len(ph["enc_t"]) > 1 for ph in phases)

        # Stationary marker priors: FK/bracket estimate if available, else the nominal-model position
        # of the sample closest to the head centre.
        for side in sides:
            if marker_priors.get(side) is None:
                closest = min((f for f in flat if f[4] == side), key=lambda f: abs(f[3]))
                pan0, tilt0 = (0.0, closest[3] * D2R) if closest[1] == 1 else (closest[3] * D2R, 0.0)
                H = fk_t5_to_head2(pan0, tilt0)
                marker_priors[side] = H[:3, :3] @ (R_nom @ closest[5] + nom_t) + H[:3, 3]
            marker_priors[side] = np.asarray(marker_priors[side], dtype=np.float64)

        n_core = 7 if timed else 6   # ez, dtilt, dpan, cam dx, dy, dz, [latency]

        # Vectorised head kinematics: T_t5_head2(p, t) = G1(0) Rot(a1, p) G12(0) Rot(a2, t), taken from
        # the robot's own FK and verified against it (exact on the real v1.2 model). Falls back to
        # per-sample FK calls if the factorisation does not reproduce FK.
        def rot_stack(axis, angles):
            axis = axis / np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
            return np.eye(3)[None] + s * K[None] + (1.0 - c) * (K @ K)[None]

        head_chain = None
        try:
            def fk_link(link, base, pan_val, tilt_val):
                q_full = q_base.copy()
                q_full[head_idx[0]] = pan_val
                q_full[head_idx[1]] = tilt_val
                return BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, link, base)
            step = 1e-3
            G1_0 = fk_link("link_head_1", "link_torso_5", 0.0, 0.0)
            G12_0 = fk_link("link_head_2", "link_head_1", 0.0, 0.0)
            a1 = R_scipy.from_matrix(G1_0[:3, :3].T @ fk_link("link_head_1", "link_torso_5", step, 0.0)[:3, :3]).as_rotvec() / step
            a2 = R_scipy.from_matrix(G12_0[:3, :3].T @ fk_link("link_head_2", "link_head_1", 0.0, step)[:3, :3]).as_rotvec() / step
            chain = (G1_0, a1, G12_0, a2)
            test = np.array([[0.13, -0.08], [-0.21, 0.17], [0.05, 0.05]])
            Rp, Rt = rot_stack(a1, test[:, 0]), rot_stack(a2, test[:, 1])
            rot = G1_0[:3, :3][None] @ Rp @ G12_0[:3, :3][None] @ Rt
            trans = G1_0[:3, 3][None] + (G1_0[:3, :3][None] @ Rp @ G12_0[:3, 3][None, :, None])[..., 0]
            err = max(max(np.abs(rot[i] - fk_t5_to_head2(*test[i])[:3, :3]).max(),
                          np.abs(trans[i] - fk_t5_to_head2(*test[i])[:3, 3]).max()) for i in range(len(test)))
            if np.isfinite(np.linalg.norm(a1)) and np.linalg.norm(a1) > 0.5 and np.linalg.norm(a2) > 0.5 and err < 1e-6:
                head_chain = chain
        except Exception:
            head_chain = None

        joint_arr = np.array([f[1] for f in flat])
        phase_arr = np.array([f[0] for f in flat])
        t_arr = np.array([f[2] if f[2] is not None else np.nan for f in flat], dtype=float)
        deg_arr = np.array([f[3] for f in flat], dtype=float)
        side_arr = np.array([sides.index(f[4]) for f in flat])

        def unpack(x, fixed_latency=None):
            ez, dtilt, dpan = x[0], x[1], x[2]
            dpos = x[3:6]
            latency = (x[6] if timed else 0.0) if fixed_latency is None else fixed_latency
            pw = {side: x[n_core + 3 * k: n_core + 3 * k + 3] for k, side in enumerate(sides)}
            return ez, dtilt, dpan, dpos, latency, pw

        def predict(x, subset, fixed_latency=None):
            """subset: integer index array into flat."""
            ez, dtilt, dpan, dpos, latency, pw = unpack(x, fixed_latency)
            R_cam = R_nom @ R_scipy.from_rotvec([0.0, 0.0, ez]).as_matrix()
            t_cam = nom_t + dpos
            deg = deg_arr[subset].copy()
            if timed:
                for ph_i, ph in enumerate(phases):
                    sel = phase_arr[subset] == ph_i
                    if np.any(sel):
                        deg[sel] = np.interp(t_arr[subset][sel] - latency, ph["enc_t"], ph["enc_deg"])
            a = deg * D2R
            is_tilt = joint_arr[subset] == 1
            pan_vals = np.where(is_tilt, 0.0, a) + dpan
            tilt_vals = np.where(is_tilt, a, 0.0) + dtilt
            pw_rows = np.stack([pw[sides[k]] for k in side_arr[subset]])
            if head_chain is not None:
                G1_0, a1, G12_0, a2 = head_chain
                Rp, Rt = rot_stack(a1, pan_vals), rot_stack(a2, tilt_vals)
                A = G1_0[:3, :3][None] @ Rp
                T_rot = A @ G12_0[:3, :3][None] @ Rt
                T_trans = G1_0[:3, 3][None] + (A @ G12_0[:3, 3][None, :, None])[..., 0]
            else:
                Ts = [fk_t5_to_head2(p, t) for p, t in zip(pan_vals, tilt_vals)]
                T_rot = np.array([T[:3, :3] for T in Ts])
                T_trans = np.array([T[:3, 3] for T in Ts])
            cam_rot = T_rot @ R_cam[None]
            cam_origin = (T_rot @ t_cam[None, :, None])[..., 0] + T_trans
            return (np.transpose(cam_rot, (0, 2, 1)) @ (pw_rows - cam_origin)[..., None])[..., 0]

        meas_arr = np.array([f[5] for f in flat])
        outbound_arr = np.array([f[6] for f in flat], dtype=bool)

        def run_solve(subset, fixed_latency=None):
            meas = meas_arr[subset]
            x0 = np.zeros(n_core + 3 * len(sides))
            lo = [-np.deg2rad(5.0), -np.deg2rad(20.0), -np.deg2rad(20.0)] + [-self.CAMERA_POS_BOUND_M] * 3
            hi = [np.deg2rad(5.0), np.deg2rad(20.0), np.deg2rad(20.0)] + [self.CAMERA_POS_BOUND_M] * 3
            if timed:
                lo.append(self.LATENCY_BOUNDS_S[0]); hi.append(self.LATENCY_BOUNDS_S[1])
                x0[6] = 0.0 if fixed_latency is None else fixed_latency
            for k, side in enumerate(sides):
                x0[n_core + 3 * k: n_core + 3 * k + 3] = marker_priors[side]
                lo.extend(marker_priors[side] - self.MARKER_POS_BOUND_M)
                hi.extend(marker_priors[side] + self.MARKER_POS_BOUND_M)
            lo, hi = np.array(lo), np.array(hi)
            if timed and fixed_latency is not None:
                lo[6] = fixed_latency - 1e-9; hi[6] = fixed_latency + 1e-9
            sol = least_squares(lambda x: (predict(x, subset, fixed_latency) - meas).ravel(), x0,
                                bounds=(lo, hi), method='trf', x_scale='jac', xtol=1e-10, ftol=1e-10)
            try:
                res = (predict(sol.x, subset, fixed_latency) - meas).reshape(-1, 3)
                rmse = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))) * 1000.0)
                side_rmse = {side: float(np.sqrt(np.mean(np.sum(res[side_arr[subset] == k] ** 2, axis=1))) * 1000.0)
                             for k, side in enumerate(sides) if np.any(side_arr[subset] == k)}
            except Exception:
                rmse, side_rmse = float("inf"), {}
            return sol, rmse, side_rmse

        sol, rmse_3d_marker_mm, rmse_by_side = run_solve(np.arange(len(flat)))
        try:
            ez_sol, dtilt_sol, dpan_sol, dpos_sol, latency_sol, pw_sol = unpack(sol.x)
        except Exception:
            ez_sol, dtilt_sol, dpan_sol, dpos_sol, latency_sol, pw_sol = 0.0, 0.0, 0.0, np.zeros(3), 0.0, {}
        decoupled_success = bool(sol.success) and np.isfinite(rmse_3d_marker_mm) and rmse_3d_marker_mm < 5.0

        # Out-and-back consistency with the latency held at the joint estimate.
        direction_check = None
        if decoupled_success and timed:
            out_subset = np.where(outbound_arr)[0]
            back_subset = np.where(~outbound_arr)[0]
            if len(out_subset) >= self.MIN_HEAD_SWEEP_POINTS and len(back_subset) >= self.MIN_HEAD_SWEEP_POINTS:
                s_out, r_out, _ = run_solve(out_subset, fixed_latency=latency_sol)
                s_back, r_back, _ = run_solve(back_subset, fixed_latency=latency_sol)
                direction_check = {
                    "tilt_out_deg": round(float(np.degrees(s_out.x[1])), 4),
                    "tilt_back_deg": round(float(np.degrees(s_back.x[1])), 4),
                    "pan_out_deg": round(float(np.degrees(s_out.x[2])), 4),
                    "pan_back_deg": round(float(np.degrees(s_back.x[2])), 4),
                    "cam_pos_out_mm": [round(float(v) * 1000.0, 3) for v in s_out.x[3:6]],
                    "cam_pos_back_mm": [round(float(v) * 1000.0, 3) for v in s_back.x[3:6]],
                    "rmse_out_mm": round(r_out, 3),
                    "rmse_back_mm": round(r_back, 3),
                }

        R_cam_est = R_nom @ R_scipy.from_rotvec([0.0, 0.0, ez_sol]).as_matrix()
        try:
            from .calibration_optimizer import rot_to_euler_zyx
        except ImportError:
            from core.calibration.calibration_optimizer import rot_to_euler_zyx
        est_roll_deg, est_pitch_deg, est_yaw_deg = (float(v) for v in rot_to_euler_zyx(R_cam_est) * R2D)
        diff_roll, diff_pitch, diff_yaw = est_roll_deg - nom_roll, est_pitch_deg - nom_pitch, est_yaw_deg - nom_yaw
        calibrated_t = nom_t + dpos_sol
        head_pan_offset_deg = float(np.degrees(dpan_sol))
        head_tilt_offset_deg = float(np.degrees(dtilt_sol))

        # Plane-fit diagnostics per sweep (first tracked marker)
        def plane_rmse(joint_pos):
            pts = np.array([f[5] for f in flat if f[1] == joint_pos and f[4] == sides[0]])
            if len(pts) < 3:
                return float("nan"), None
            n, c = self.fit_plane_normal_svd(pts)
            return float(np.sqrt(np.mean(np.dot(pts - c, n) ** 2)) * 1000.0), n
        rmse_tilt_plane, n_tilt = plane_rmse(1)
        rmse_pan_plane, n_pan = plane_rmse(0)
        ortho_err_deg = float(abs(np.arcsin(np.clip(abs(np.dot(n_tilt, n_pan)), 0.0, 1.0))) * R2D) if n_tilt is not None and n_pan is not None else float("nan")

        calibrated_mount_to_cam = [round(float(v), 6) for v in calibrated_t] + [round(est_roll_deg, 4), round(est_pitch_deg, 4), round(est_yaw_deg, 4)]
        tilt_angles = [f[3] for f in flat if f[1] == 1]
        pan_angles = [f[3] for f in flat if f[1] == 0]

        results = {
            "success": decoupled_success,
            "nominal_mount_to_cam": nominal_mount_to_cam,
            "calibrated_mount_to_cam": calibrated_mount_to_cam,
            "cam_rot_diff_deg": {"roll": round(diff_roll, 4), "pitch": round(diff_pitch, 4), "yaw": round(diff_yaw, 4)},
            "camera_position_delta_mm": [round(float(v) * 1000.0, 3) for v in dpos_sol],
            "head_offsets_deg": {"pan": round(head_pan_offset_deg, 4), "tilt": round(head_tilt_offset_deg, 4)},
            "latency_ms": round(float(latency_sol) * 1000.0, 2) if timed else None,
            "markers_used": sides,
            "marker_positions_t5_mm": {side: [round(float(v) * 1000.0, 2) for v in pw_sol.get(side, [])] for side in sides},
            "marker_prior_shift_mm": {side: [round(float(v) * 1000.0, 2) for v in (pw_sol[side] - marker_priors[side])]
                                      for side in sides if side in pw_sol and len(pw_sol[side]) == 3},
            "direction_consistency": direction_check,
            "quality": {
                "rmse_tilt_plane_mm": round(rmse_tilt_plane, 3),
                "rmse_pan_plane_mm": round(rmse_pan_plane, 3),
                "rmse_3d_marker_mm": round(rmse_3d_marker_mm, 3) if np.isfinite(rmse_3d_marker_mm) else None,
                "rmse_by_marker_mm": {k: round(v, 3) for k, v in rmse_by_side.items()},
                "ortho_error_deg": round(ortho_err_deg, 4),
                "decoupled": decoupled_success,
            },
            "pts_tilt_count": len(tilt_angles),
            "pts_pan_count": len(pan_angles),
        }

        self.calibrated_results = results if decoupled_success else None
        self._record_step1_5_result(results, tilt_angles, pan_angles, log_callback)

        if log_callback:
            log_callback("\n" + "=" * 60)
            log_callback(" [Step 1.5 Calibration Results]")
            log_callback("=" * 60)
            log_callback(f" Markers used: {', '.join(sides)} | samples: tilt {len(tilt_angles)}, pan {len(pan_angles)}")
            log_callback(f" Camera Mount Extrinsics (Euler ZYX; only the optical-axis rotation is estimated):")
            log_callback(f"   Roll  : Nom {nom_roll:+7.2f}° -> Calib {est_roll_deg:+7.2f}° (Δ {diff_roll:+6.3f}°)")
            log_callback(f"   Pitch : Nom {nom_pitch:+7.2f}° -> Calib {est_pitch_deg:+7.2f}° (Δ {diff_pitch:+6.3f}°)")
            log_callback(f"   Yaw   : Nom {nom_yaw:+7.2f}° -> Calib {est_yaw_deg:+7.2f}° (Δ {diff_yaw:+6.3f}°)")
            log_callback(f"   Position Δ: [{dpos_sol[0]*1000:+.2f}, {dpos_sol[1]*1000:+.2f}, {dpos_sol[2]*1000:+.2f}] mm")
            log_callback(f" Head Joint Offsets:")
            log_callback(f"   Head Pan  : {head_pan_offset_deg:+6.3f}°")
            log_callback(f"   Head Tilt : {head_tilt_offset_deg:+6.3f}°")
            if timed:
                log_callback(f" Camera latency: {latency_sol*1000:+.1f} ms")
            log_callback(f" Fit Quality Diagnostics:")
            log_callback(f"   3D Reprojection RMSE: {rmse_3d_marker_mm:.3f} mm (Decoupled: {decoupled_success}) {results['quality']['rmse_by_marker_mm']}")
            log_callback(f"   Tilt Plane Fit RMSE : {rmse_tilt_plane:.3f} mm")
            log_callback(f"   Pan  Plane Fit RMSE : {rmse_pan_plane:.3f} mm")
            log_callback(f"   Axis Ortho Error    : {ortho_err_deg:.3f}°")
            if direction_check:
                d_tilt = direction_check["tilt_out_deg"] - direction_check["tilt_back_deg"]
                log_callback(f"   Out/Back tilt: {direction_check['tilt_out_deg']:+.3f}° / {direction_check['tilt_back_deg']:+.3f}°, "
                             f"pan: {direction_check['pan_out_deg']:+.3f}° / {direction_check['pan_back_deg']:+.3f}°")
                if abs(d_tilt) > self.DIRECTION_TILT_WARN_DEG:
                    log_callback(f"   [WARN] Out and back sweeps disagree on head tilt by {d_tilt:+.3f}° (> {self.DIRECTION_TILT_WARN_DEG}°).")
            log_callback("=" * 60)

        return results

    @staticmethod
    def _record_step1_5_result(results, captured_tilt_angles, captured_pan_angles, log_callback=None):
        """Persist every Step 1.5 solve: latest result as JSON + one appended history entry."""
        try:
            import json
            import datetime
            from core.storage import CONFIG_PATHS
            txt_dir = CONFIG_PATHS["txt_dir"]
            FileStorage.ensure_dir(txt_dir, exist_ok=True)
            record = dict(results)
            record["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
            record["captured_tilt_angles_deg"] = [round(float(a), 3) for a in captured_tilt_angles]
            record["captured_pan_angles_deg"] = [round(float(a), 3) for a in captured_pan_angles]
            FileStorage.write_text(
                os.path.join(txt_dir, "head_camera_result_latest.json"),
                json.dumps(record, indent=2, ensure_ascii=False)
            )
            with FileStorage.open(os.path.join(txt_dir, "head_camera_result_history.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to record Step 1.5 result file: {e}")

    def apply_calibration_results(self, results=None, log_callback=None):
        """
        Saves calibrated camera extrinsics into setting.yaml and updates head joint offsets.
        """
        if results is None:
            results = self.calibrated_results

        if not results or not results.get("success", False):
            if log_callback: log_callback("[ERROR] No calibration results to apply!")
            return False

        if results.get("skipped"):
            if log_callback:
                log_callback("[INFO] Headless mode: No head parameters to apply.")
            return True

        calib_mount_to_cam = results.get("calibrated_mount_to_cam")
        head_offsets = results.get("head_offsets_deg", {})

        if not calib_mount_to_cam:
            if log_callback: log_callback("[ERROR] Missing calibrated_mount_to_cam in results.")
            return False

        try:
            # 1. Update in-memory camera_config
            self.camera_config["mount_to_cam"] = calib_mount_to_cam

            # 2. Update setting.yaml file
            from core.storage import CONFIG_PATHS
            setting_path = CONFIG_PATHS.get("setting_yaml") or CONFIG_PATHS.get("setting")
            if setting_path and os.path.exists(setting_path):
                cfg = ConfigStorage.load(setting_path)

                if "camera" not in cfg:
                    cfg["camera"] = {}
                changes = {("camera", "mount_to_cam"): list(calib_mount_to_cam)}
                if "mount_to_cam_nominal" not in cfg.get("camera", {}):
                    changes[("camera", "mount_to_cam_nominal")] = list(self.camera_config.get(
                        "mount_to_cam_nominal", self.camera_config.get(
                            "mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])))
                if "head_base_to_cam_nominal" not in cfg.get("camera", {}):
                    changes[("camera", "head_base_to_cam_nominal")] = list(self.camera_config.get(
                        "head_base_to_cam_nominal", self.camera_config.get(
                            "head_base_to_cam", [0.098, 0.009, 0.012, -90.0, 0.0, -90.0])))
                changes[("joint_offset", "head", "pan")] = head_offsets.get("pan", 0.0)
                changes[("joint_offset", "head", "tilt")] = head_offsets.get("tilt", 0.0)
                # Was a whole-file yaml.dump with sort_keys defaulting to True, which reordered
                # every key in setting.yaml on each Step 1.5 apply.
                ConfigStorage.update_values(setting_path, changes)

                if log_callback:
                    log_callback(f"[SUCCESS] Updated setting.yaml with calibrated mount_to_cam: {calib_mount_to_cam}")
                    log_callback(f"[SUCCESS] Updated setting.yaml with head offsets: {head_offsets}")

            # 3. Update in-memory joint offsets store and camera configs if app reference exists

            return True
        except Exception as e:
            if log_callback: log_callback(f"[ERROR] Failed to save calibration results: {e}")
            return False
