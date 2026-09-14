from core.storage import ConfigStorage
from core.storage import FileStorage
import os
import time
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

    def perform_head_sweep(
        self,
        arm_side="auto",
        pan_range_deg=15.0,
        tilt_range_deg=20.0,
        num_steps=11,
        step_delay=0.8,
        log_callback=None,
        status_callback=None,
        stop_event=None,
        save_debug=True
    ):
        """
        Executes Tilt and Pan sweeps with stationary arm markers:
        1. Sweep Tilt from -tilt_range to +tilt_range while Pan=0.
        2. Sweep Pan from -pan_range to +pan_range while Tilt=0.
        3. Fit circle/plane normals in camera coordinates via SVD.
        4. Solve camera mount extrinsic rotation R_mount_to_cam.
        5. Solve Head Pan & Tilt joint zero offsets.
        """
        if log_callback:
            log_callback("=" * 60)
            log_callback(f"[Head-Camera Calib] Commencing Head-Camera Extrinsic Calibration")
            log_callback(f"  Target Marker Arm : {arm_side}")
            log_callback(f"  Pan Sweep Range   : ±{pan_range_deg:.1f}° ({num_steps} steps)")
            log_callback(f"  Tilt Sweep Range  : ±{tilt_range_deg:.1f}° ({num_steps} steps)")
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

        P_marker_t5_nom = None
        try:
            dyn_model = self.robot.get_dynamics()
            q_current = np.array(self.robot.get_state().position)
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_current, f"ee_{active_side}", "link_torso_5")
            suffix = "_v13" if self.is_v13() else "_v12"
            tf_vec = self.camera_config.get(f"Tf_to_marker_{active_side}{suffix}")
            if tf_vec is None:
                tf_vec = self.camera_config.get(f"Tf_to_marker_{active_side}")
            if tf_vec is None:
                tf_vec = self.NOMINAL_BRACKET_TEMPLATES["1.3" if self.is_v13() else "1.2"][active_side]
            T_ee_to_marker = BaseCalibrator.make_transform(tf_vec)
            P_marker_t5_nom = (T_t5_to_ee @ T_ee_to_marker)[:3, 3]
        except Exception:
            pass

        # ----------------------------------------------------
        # Phase A: Head Tilt Sweep (Pan = 0, Tilt: -range to +range)
        # ----------------------------------------------------
        if log_callback: log_callback("\n[Phase A] Sweeping Head Tilt Joint (-tilt to +tilt)...")
        tilt_angles_deg = np.linspace(-tilt_range_deg, tilt_range_deg, num_steps)
        pts_tilt_cam = []
        captured_tilt_angles = []
        self.partial_data = {"tilt_points": pts_tilt_cam, "tilt_angles": captured_tilt_angles}

        for idx, t_deg in enumerate(tilt_angles_deg):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Calibration aborted by user.")
                return None

            t_rad = t_deg * D2R
            ok = self.movej(self.robot, head=[0.0, t_rad], minimum_time=1.5, apply_offsets=False)
            if not ok:
                if log_callback: log_callback(f"  [WARN] Head move to Tilt={t_deg:.1f}° failed. Retrying...")
                time.sleep(0.5)

            time.sleep(step_delay)

            # Detect marker
            p_marker, det_side = self._detect_marker_point(arm_side=active_side, sampling_time=0.5)
            if p_marker is None:
                if log_callback: log_callback(f"  [WARN] Step {idx+1}/{num_steps} (Tilt={t_deg:.1f}°): Marker not visible, skipping.")
                continue

            actual_t_deg = t_deg
            if hasattr(self.robot, "get_state"):
                try:
                    q_cur = self.robot.get_state().position
                    actual_t_deg = float(np.degrees(q_cur[head_idx[1]]))
                except Exception:
                    pass

            pts_tilt_cam.append(p_marker)
            captured_tilt_angles.append(actual_t_deg)
            _sweep_debug_write(f"  [{idx+1}/{num_steps}] Tilt={actual_t_deg:+5.1f}° -> Marker Cam Pos: [{p_marker[0]*1000:+6.1f}, {p_marker[1]*1000:+6.1f}, {p_marker[2]*1000:+6.1f}] mm")

        if len(pts_tilt_cam) < 5:
            raise RuntimeError(f"Insufficient marker points collected during Tilt sweep ({len(pts_tilt_cam)} points). Calibration cannot proceed.")

        # Return head to zero center before Pan sweep
        if log_callback: log_callback("\n[INFO] Returning head to center before Pan sweep...")
        self.movej(self.robot, head=[0.0, 0.0], minimum_time=1.5, apply_offsets=False)
        time.sleep(1.0)

        # ----------------------------------------------------
        # Phase B: Head Pan Sweep (Tilt = 0, Pan: -range to +range)
        # ----------------------------------------------------
        if log_callback: log_callback("\n[Phase B] Sweeping Head Pan Joint (-pan to +pan)...")
        pan_angles_deg = np.linspace(-pan_range_deg, pan_range_deg, num_steps)
        pts_pan_cam = []
        captured_pan_angles = []
        self.partial_data.update(pan_points=pts_pan_cam, pan_angles=captured_pan_angles)

        for idx, p_deg in enumerate(pan_angles_deg):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Calibration aborted by user.")
                return None

            p_rad = p_deg * D2R
            ok = self.movej(self.robot, head=[p_rad, 0.0], minimum_time=1.5, apply_offsets=False)
            if not ok:
                if log_callback: log_callback(f"  [WARN] Head move to Pan={p_deg:.1f}° failed. Retrying...")
                time.sleep(0.5)

            time.sleep(step_delay)

            p_marker, det_side = self._detect_marker_point(arm_side=active_side, sampling_time=0.5)
            if p_marker is None:
                if log_callback: log_callback(f"  [WARN] Step {idx+1}/{num_steps} (Pan={p_deg:.1f}°): Marker not visible, skipping.")
                continue

            actual_p_deg = p_deg
            if hasattr(self.robot, "get_state"):
                try:
                    q_cur = self.robot.get_state().position
                    actual_p_deg = float(np.degrees(q_cur[head_idx[0]]))
                except Exception:
                    pass

            pts_pan_cam.append(p_marker)
            captured_pan_angles.append(actual_p_deg)
            _sweep_debug_write(f"  [{idx+1}/{num_steps}] Pan={actual_p_deg:+5.1f}°  -> Marker Cam Pos: [{p_marker[0]*1000:+6.1f}, {p_marker[1]*1000:+6.1f}, {p_marker[2]*1000:+6.1f}] mm")

        if len(pts_pan_cam) < 5:
            raise RuntimeError(f"Insufficient marker points collected during Pan sweep ({len(pts_pan_cam)} points). Calibration cannot proceed.")

        # Return head to zero center
        self.movej(self.robot, head=[0.0, 0.0], minimum_time=2.0, apply_offsets=False)

        # ----------------------------------------------------
        # Phase C: Mathematical Solution for Decoupled Head-Camera Calib
        # ----------------------------------------------------
        return self._compute_head_camera_solution(
            pts_tilt_cam, pts_pan_cam, captured_tilt_angles, captured_pan_angles,
            nominal_mount_to_cam, R_nom, obs_r_0=obs_r_0, obs_l_0=obs_l_0,
            active_side=active_side, P_marker_t5_nom=P_marker_t5_nom, log_callback=log_callback
        )

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
        pts_tilt_cam = np.array(pts_tilt_cam)
        pts_pan_cam = np.array(pts_pan_cam)

        n_tilt_cam, c_tilt = self.fit_plane_normal_svd(pts_tilt_cam)
        n_pan_cam, c_pan = self.fit_plane_normal_svd(pts_pan_cam)

        # Calculate plane fit RMSE (diagnostics only, not used in the solve below)
        d_tilt = np.abs(np.dot(pts_tilt_cam - c_tilt, n_tilt_cam))
        rmse_tilt_plane = np.sqrt(np.mean(d_tilt**2)) * 1000.0 # mm

        d_pan = np.abs(np.dot(pts_pan_cam - c_pan, n_pan_cam))
        rmse_pan_plane = np.sqrt(np.mean(d_pan**2)) * 1000.0 # mm

        # Sign consistency: Nominal Tilt axis in camera coords is [-1, 0, 0], Pan is [0, -1, 0]
        if n_tilt_cam[0] > 0: n_tilt_cam_signed = -n_tilt_cam
        else: n_tilt_cam_signed = n_tilt_cam
        if n_pan_cam[1] > 0: n_pan_cam_signed = -n_pan_cam
        else: n_pan_cam_signed = n_pan_cam

        # Orthogonality diagnostic
        dot_ortho = np.dot(n_tilt_cam_signed, n_pan_cam_signed)
        ortho_err_deg = abs(np.arcsin(np.clip(dot_ortho, -1.0, 1.0))) * R2D

        nom_roll = float(nominal_mount_to_cam[3])
        nom_pitch = float(nominal_mount_to_cam[4])
        nom_yaw = float(nominal_mount_to_cam[5])

        # ----------------------------------------------------
        # Phase C: Joint least-squares solution for camera rotation (3 DOF) + Head
        # Pan/Tilt zero offsets, using the raw 3D sweep points (not just plane
        # normals). Camera mount TRANSLATION is held fixed at the CAD nominal and
        # is deliberately NOT re-derived from the estimated rotation: analysis
        # showed that tying position to rotation (t_cam = R_rel @ nom_t) makes the
        # position term cancel out of the observation model, which reopens the
        # exact rank-1 degeneracy between camera-mount pitch (eps_y) and the head
        # tilt joint offset (dtilt). Holding position independently FIXED is what
        # keeps that pair identifiable from data instead of by convention.
        # Unknowns: eps (camera rotation error, axis-angle, in nominal mount
        # frame), dtilt, dpan (head joint zero offsets), Pw (stationary marker
        # position in link_torso_5 frame, nuisance parameter).
        # ----------------------------------------------------
        nom_t = np.array(nominal_mount_to_cam[:3], dtype=np.float64)

        head_idx = [0, 1]
        try:
            m = self.robot.model()
            if hasattr(m, 'head_idx') and m.head_idx is not None and len(m.head_idx) >= 2:
                head_idx = list(m.head_idx)
        except Exception:
            pass

        dyn_model = self.robot.get_dynamics()
        q_base = np.array(self.robot.get_state().position, dtype=np.float64)
        tilt_rad = np.deg2rad(np.asarray(captured_tilt_angles, dtype=np.float64))
        pan_rad = np.deg2rad(np.asarray(captured_pan_angles, dtype=np.float64))
        Pw0 = np.array(P_marker_t5_nom, dtype=np.float64) if P_marker_t5_nom is not None else np.mean(
            np.vstack([pts_tilt_cam, pts_pan_cam]), axis=0
        )

        def fk_t5_to_head2(pan_val, tilt_val):
            q_full = q_base.copy()
            q_full[head_idx[0]] = pan_val
            q_full[head_idx[1]] = tilt_val
            return BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")

        def predict_all(params):
            eps = params[0:3]
            dtilt = params[3]
            dpan = params[4]
            Pw = params[5:8]
            R_cam = R_nom @ R_scipy.from_rotvec(eps).as_matrix()
            preds = []
            for t in tilt_rad:
                T = fk_t5_to_head2(0.0 + dpan, t + dtilt)
                T_rot, T_trans = T[:3, :3], T[:3, 3]
                cam_rot_in_t5 = T_rot @ R_cam
                cam_origin_in_t5 = T_rot @ nom_t + T_trans
                preds.append(cam_rot_in_t5.T @ (Pw - cam_origin_in_t5))
            for p in pan_rad:
                T = fk_t5_to_head2(p + dpan, 0.0 + dtilt)
                T_rot, T_trans = T[:3, :3], T[:3, 3]
                cam_rot_in_t5 = T_rot @ R_cam
                cam_origin_in_t5 = T_rot @ nom_t + T_trans
                preds.append(cam_rot_in_t5.T @ (Pw - cam_origin_in_t5))
            return np.array(preds)

        meas = np.vstack([pts_tilt_cam, pts_pan_cam])

        def residual(params):
            return (predict_all(params) - meas).ravel()

        # Marker position bound: this direction is EXACTLY degenerate with head Pan
        # offset (dpan) using pan-sweep data alone -- the achievable dpan accuracy
        # is set almost entirely by this bound's width (empirically ~1 deg of dpan
        # error per ~35mm of position slack), not by data/noise. 20mm is chosen to
        # roughly match the single-joint FK lever-arm error expected from
        # not-yet-calibrated shoulder joints (J0/J1/J2/J4) at Step 1.5 time -- tighter
        # would risk biasing against real uncalibrated-shoulder error; looser
        # reopens the degeneracy (see project memory on this identifiability limit).
        marker_pos_bound_m = 0.02
        x0 = np.concatenate([np.zeros(5), Pw0])
        bounds_lo = np.concatenate([np.full(3, -np.deg2rad(5.0)), [-np.deg2rad(20.0), -np.deg2rad(20.0)], Pw0 - marker_pos_bound_m])
        bounds_hi = np.concatenate([np.full(3, np.deg2rad(5.0)), [np.deg2rad(20.0), np.deg2rad(20.0)], Pw0 + marker_pos_bound_m])

        sol = least_squares(residual, x0, bounds=(bounds_lo, bounds_hi), method='trf', xtol=1e-13, ftol=1e-13)
        eps_sol = sol.x[0:3]
        dtilt_sol = float(sol.x[3])
        dpan_sol = float(sol.x[4])

        R_cam_est = R_nom @ R_scipy.from_rotvec(eps_sol).as_matrix()

        try:
            from .calibration_optimizer import rot_to_euler_zyx
        except ImportError:
            from core.calibration.calibration_optimizer import rot_to_euler_zyx
        rpy_est_deg = rot_to_euler_zyx(R_cam_est) * R2D
        est_roll_deg = float(rpy_est_deg[0])
        est_pitch_deg = float(rpy_est_deg[1])
        est_yaw_deg = float(rpy_est_deg[2])

        diff_roll = est_roll_deg - nom_roll
        diff_pitch = est_pitch_deg - nom_pitch
        diff_yaw = est_yaw_deg - nom_yaw

        # Camera mount translation is reported at CAD nominal (not re-derived from
        # the solved rotation) -- see note above on why tying it to rotation
        # reopens the eps_y/dtilt degeneracy. Fine position correction (a few mm)
        # is left to Step 2's bounded camera-position optimization.
        calibrated_t = nom_t

        head_pan_offset_deg = float(np.degrees(dpan_sol))
        head_tilt_offset_deg = float(np.degrees(dtilt_sol))

        final_residuals = residual(sol.x).reshape(-1, 3)
        rmse_3d_marker_mm = float(np.sqrt(np.mean(np.sum(final_residuals ** 2, axis=1))) * 1000.0)

        # Genuine joint identification (not a gauge-fixing convention): success
        # tracks solver convergence rather than a hardcoded value.
        decoupled_success = bool(sol.success) and rmse_3d_marker_mm < 5.0

        calibrated_mount_to_cam = [
            round(float(calibrated_t[0]), 6),
            round(float(calibrated_t[1]), 6),
            round(float(calibrated_t[2]), 6),
            round(est_roll_deg, 4),
            round(est_pitch_deg, 4),
            round(est_yaw_deg, 4)
        ]

        results = {
            "success": decoupled_success,
            "nominal_mount_to_cam": nominal_mount_to_cam,
            "calibrated_mount_to_cam": calibrated_mount_to_cam,
            "cam_rot_diff_deg": {
                "roll": round(diff_roll, 4),
                "pitch": round(diff_pitch, 4),
                "yaw": round(diff_yaw, 4),
            },
            "head_offsets_deg": {
                "pan": round(head_pan_offset_deg, 4),
                "tilt": round(head_tilt_offset_deg, 4),
            },
            "quality": {
                "rmse_tilt_plane_mm": round(rmse_tilt_plane, 3),
                "rmse_pan_plane_mm": round(rmse_pan_plane, 3),
                "rmse_3d_marker_mm": round(rmse_3d_marker_mm, 3) if rmse_3d_marker_mm is not None else None,
                "ortho_error_deg": round(ortho_err_deg, 4),
                "decoupled": decoupled_success,
            },
            "pts_tilt_count": len(pts_tilt_cam),
            "pts_pan_count": len(pts_pan_cam)
        }

        self.calibrated_results = results if decoupled_success else None

        if log_callback:
            log_callback("\n" + "=" * 60)
            log_callback(" [Step 1.5 Calibration Results]")
            log_callback("=" * 60)
            log_callback(f" Camera Mount Extrinsics (Euler ZYX):")
            log_callback(f"   Roll  : Nom {nom_roll:+7.2f}° -> Calib {est_roll_deg:+7.2f}° (Δ {diff_roll:+6.3f}°)")
            log_callback(f"   Pitch : Nom {nom_pitch:+7.2f}° -> Calib {est_pitch_deg:+7.2f}° (Δ {diff_pitch:+6.3f}°)")
            log_callback(f"   Yaw   : Nom {nom_yaw:+7.2f}° -> Calib {est_yaw_deg:+7.2f}° (Δ {diff_yaw:+6.3f}°)")
            log_callback(f" Head Joint Offsets:")
            log_callback(f"   Head Pan  : {head_pan_offset_deg:+6.3f}°")
            log_callback(f"   Head Tilt : {head_tilt_offset_deg:+6.3f}°")
            log_callback(f" Fit Quality Diagnostics:")
            if rmse_3d_marker_mm is not None:
                log_callback(f"   3D Reprojection RMSE: {rmse_3d_marker_mm:.3f} mm (Decoupled: {decoupled_success})")
            log_callback(f"   Tilt Plane Fit RMSE : {rmse_tilt_plane:.3f} mm")
            log_callback(f"   Pan  Plane Fit RMSE : {rmse_pan_plane:.3f} mm")
            log_callback(f"   Axis Ortho Error    : {ortho_err_deg:.3f}°")
            log_callback("=" * 60)

        return results

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
                cfg["camera"]["mount_to_cam"] = calib_mount_to_cam
                if "mount_to_cam_nominal" not in cfg["camera"]:
                    cfg["camera"]["mount_to_cam_nominal"] = list(self.camera_config.get("mount_to_cam_nominal", self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])))
                if "head_base_to_cam_nominal" not in cfg["camera"]:
                    cfg["camera"]["head_base_to_cam_nominal"] = list(self.camera_config.get("head_base_to_cam_nominal", self.camera_config.get("head_base_to_cam", [0.098, 0.009, 0.012, -90.0, 0.0, -90.0])))

                # Store head joint offsets in joint_offset section
                if "joint_offset" not in cfg:
                    cfg["joint_offset"] = {}
                if "head" not in cfg["joint_offset"]:
                    cfg["joint_offset"]["head"] = {}
                cfg["joint_offset"]["head"]["pan"] = head_offsets.get("pan", 0.0)
                cfg["joint_offset"]["head"]["tilt"] = head_offsets.get("tilt", 0.0)

                FileStorage.write_text(setting_path, yaml.dump(cfg, default_flow_style=None))

                if log_callback:
                    log_callback(f"[SUCCESS] Updated setting.yaml with calibrated mount_to_cam: {calib_mount_to_cam}")
                    log_callback(f"[SUCCESS] Updated setting.yaml with head offsets: {head_offsets}")

            # 3. Update in-memory joint offsets store and camera configs if app reference exists

            return True
        except Exception as e:
            if log_callback: log_callback(f"[ERROR] Failed to save calibration results: {e}")
            return False
