import os
import time
import yaml
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator
from scipy.optimize import least_squares

def camera_command_to_encoder(command_rad, reference):
    """Map zero-relative head JOINT commands, not absolute optical Euler angles.

    The caller must validate the robot/session, head enable state and limits
    before motion. This helper never commands a robot or writes home offsets.
    """
    command = np.asarray(command_rad, dtype=float)
    zero = np.deg2rad(np.asarray(reference['encoder_zero_deg'], dtype=float))
    if command.shape != (2,) or zero.shape != (2,) or not np.all(np.isfinite([command, zero])):
        raise ValueError('Camera head command and encoder zero must be finite Pan/Tilt pairs')
    if not reference.get('accepted') or reference.get('reference_frame') != 'link_torso_5':
        raise ValueError('An accepted torso-frame camera zero is required')
    return command + zero


def camera_forward_zero(head_fk, mount_to_cam, head_offset, lower, upper,
                        redistribute_tilt=True):
    """Return an equivalent head/camera pair and the encoder pose facing torso +X.

    head_fk takes model Pan/Tilt radians and returns torso-to-head-mount SE(3).
    Bounds are encoder limits. The search is limited to +/-15 degrees around
    encoder zero, rejecting reversed/unreachable camera installations.
    """
    camera = np.asarray(mount_to_cam, dtype=float)
    offset = np.asarray(head_offset, dtype=float)
    lower = np.maximum(np.asarray(lower, dtype=float), np.deg2rad([-15., -15.]))
    upper = np.minimum(np.asarray(upper, dtype=float), np.deg2rad([15., 15.]))
    if (camera.shape != (4, 4) or offset.shape != (2,) or lower.shape != (2,)
            or upper.shape != (2,) or not np.all(np.isfinite(camera))
            or not np.all(np.isfinite([offset, lower, upper])) or np.any(lower >= upper)):
        raise ValueError('Invalid camera transform, head offsets or camera-zero search limits')
    if (not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-8)
            or np.linalg.det(camera[:3, :3]) < 0):
        raise ValueError('Camera transform must be a proper SE(3) transform')

    target = np.array([1., 0., 0.])

    def direction(encoder):
        return (head_fk(encoder + offset) @ camera)[:3, 2]

    fit = least_squares(lambda q: direction(q) - target,
                        np.clip(-offset, lower + 1e-10, upper - 1e-10),
                        bounds=(lower, upper), ftol=1e-13, xtol=1e-13, gtol=1e-13)
    error_deg = float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(direction(fit.x), target)),
                                           np.dot(direction(fit.x), target))))
    if not fit.success or error_deg > 1e-5 or np.linalg.matrix_rank(fit.jac, tol=1e-7) != 2:
        raise ValueError(f'Camera forward zero is unreachable or degenerate (error={error_deg:.4f} deg)')

    new_offset = offset.copy()
    new_camera = camera.copy()
    if redistribute_tilt:
        new_offset[1] = -fit.x[1]
        # Exact terminal-joint gauge: H(q+d_old) C_old = H(q+d_new) C_new.
        # Use FK, including the joint pivot translation; never copy an RPY scalar.
        delta = np.array([0., offset[1] - new_offset[1]])
        new_camera = np.linalg.inv(head_fk(np.zeros(2))) @ head_fk(delta) @ camera
        for q in (np.zeros(2), np.array([.12, -.09]), np.array([-.08, .13])):
            if not np.allclose(head_fk(q + offset) @ camera,
                               head_fk(q + new_offset) @ new_camera, atol=1e-9, rtol=0):
                raise ValueError('Selected camera link does not have a terminal Tilt gauge')

    reference = {
        'accepted': True,
        'reference_frame': 'link_torso_5',
        'forward_axis': [1., 0., 0.],
        'camera_optical_axis': [0., 0., 1.],
        'encoder_zero_deg': np.rad2deg(fit.x).tolist(),
        'command_offset_deg': np.rad2deg(-fit.x).tolist(),
        'command_convention': 'q_encoder = q_camera_command + encoder_zero',
        'independent_physical_offset': False,
        'image_roll_corrected': False,
        'predicted_alignment_error_deg': error_deg,
        'encoder_search_lower_deg': np.rad2deg(lower).tolist(),
        'encoder_search_upper_deg': np.rad2deg(upper).tolist(),
    }
    return new_offset, new_camera, reference


D2R = np.pi / 180.0
R2D = 180.0 / np.pi

class HeadCameraCalibrator(BaseCalibrator):
    """
    Step 1.5 Calibrator:
    Fits effective camera extrinsics from stationary markers and head encoders.
    Pan zero is unobservable here; terminal tilt/camera needs a gauge reference.
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

        if self.robot is None or not hasattr(self.robot, "get_state"):
            if log_callback: log_callback("[ERROR] Robot is not connected")
            return False

        has_head = self.is_head_active() and self.uses_head_camera()
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
            from core.robot_motion import move_to_auto_ready_pose
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
        tilt_range_deg=10.0,
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

        if self.robot is None or not hasattr(self.robot, "get_state"):
            raise RuntimeError("Connect a robot before collecting head observations")
        has_head = self.is_head_active() and self.uses_head_camera()
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
                log_callback("[INFO] Head sweep disabled or head-mounted camera unavailable. Skipping Head-Camera extrinsic calibration.")
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

        # ----------------------------------------------------
        # Phase A: Head Tilt Sweep (Pan = 0, Tilt: -range to +range)
        # ----------------------------------------------------
        if log_callback: log_callback("\n[Phase A] Sweeping Head Tilt Joint (-tilt to +tilt)...")
        tilt_angles_deg = np.linspace(-tilt_range_deg, tilt_range_deg, num_steps)
        pts_tilt_cam = []
        captured_tilt_angles = []
        captured_tilt_head_deg = []

        def read_head_angles():
            state = self.robot.get_state()
            angles = np.rad2deg(np.asarray(state.position)[head_idx])
            if angles.shape != (2,) or not np.all(np.isfinite(angles)):
                raise RuntimeError('Invalid head encoder feedback; commanded angles are not measurements')
            return angles

        for idx, t_deg in enumerate(tilt_angles_deg):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Calibration aborted by user.")
                return None

            t_rad = t_deg * D2R
            ok = self.movej(self.robot, head=[0.0, t_rad], minimum_time=1.5, apply_offsets=False)
            if not ok:
                raise RuntimeError(f'Head tilt motion failed at {t_deg:.1f} degrees')

            time.sleep(step_delay)

            # Detect marker
            p_marker, det_side = self._detect_marker_point(arm_side=active_side, sampling_time=0.5)
            if p_marker is None:
                if log_callback: log_callback(f"  [WARN] Step {idx+1}/{num_steps} (Tilt={t_deg:.1f}°): Marker not visible, skipping.")
                continue

            head_deg = read_head_angles()
            actual_t_deg = float(head_deg[1])

            pts_tilt_cam.append(p_marker)
            captured_tilt_angles.append(actual_t_deg)
            captured_tilt_head_deg.append(head_deg)
            if log_callback:
                log_callback(f"  [{idx+1}/{num_steps}] Tilt={actual_t_deg:+5.1f}° -> Marker Cam Pos: [{p_marker[0]*1000:+6.1f}, {p_marker[1]*1000:+6.1f}, {p_marker[2]*1000:+6.1f}] mm")

        if len(pts_tilt_cam) < 5:
            raise RuntimeError(f"Insufficient marker points collected during Tilt sweep ({len(pts_tilt_cam)} points). Calibration cannot proceed.")

        # Return head to zero center before Pan sweep
        if log_callback: log_callback("\n[INFO] Returning head to center before Pan sweep...")
        if not self.movej(self.robot, head=[0.0, 0.0], minimum_time=1.5, apply_offsets=False):
            raise RuntimeError('Head centering failed before pan sweep')
        time.sleep(1.0)

        # ----------------------------------------------------
        # Phase B: Head Pan Sweep (Tilt = 0, Pan: -range to +range)
        # ----------------------------------------------------
        if log_callback: log_callback("\n[Phase B] Sweeping Head Pan Joint (-pan to +pan)...")
        pan_angles_deg = np.linspace(-pan_range_deg, pan_range_deg, num_steps)
        pts_pan_cam = []
        captured_pan_angles = []
        captured_pan_head_deg = []

        for idx, p_deg in enumerate(pan_angles_deg):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Calibration aborted by user.")
                return None

            p_rad = p_deg * D2R
            ok = self.movej(self.robot, head=[p_rad, 0.0], minimum_time=1.5, apply_offsets=False)
            if not ok:
                raise RuntimeError(f'Head pan motion failed at {p_deg:.1f} degrees')

            time.sleep(step_delay)

            p_marker, det_side = self._detect_marker_point(arm_side=active_side, sampling_time=0.5)
            if p_marker is None:
                if log_callback: log_callback(f"  [WARN] Step {idx+1}/{num_steps} (Pan={p_deg:.1f}°): Marker not visible, skipping.")
                continue

            head_deg = read_head_angles()
            actual_p_deg = float(head_deg[0])

            pts_pan_cam.append(p_marker)
            captured_pan_angles.append(actual_p_deg)
            captured_pan_head_deg.append(head_deg)
            if log_callback:
                log_callback(f"  [{idx+1}/{num_steps}] Pan={actual_p_deg:+5.1f}°  -> Marker Cam Pos: [{p_marker[0]*1000:+6.1f}, {p_marker[1]*1000:+6.1f}, {p_marker[2]*1000:+6.1f}] mm")

        if len(pts_pan_cam) < 5:
            raise RuntimeError(f"Insufficient marker points collected during Pan sweep ({len(pts_pan_cam)} points). Calibration cannot proceed.")

        # Return head to zero center
        if not self.movej(self.robot, head=[0.0, 0.0], minimum_time=2.0, apply_offsets=False):
            raise RuntimeError('Head centering failed after pan sweep')

        # ----------------------------------------------------
        # Phase C: Mathematical Solution for Decoupled Head-Camera Calib
        # ----------------------------------------------------
        return self._compute_head_camera_solution(
            pts_tilt_cam, pts_pan_cam, captured_tilt_angles, captured_pan_angles,
            nominal_mount_to_cam, R_nom, obs_r_0=obs_r_0, obs_l_0=obs_l_0,
            active_side=active_side, log_callback=log_callback,
            tilt_head_deg=captured_tilt_head_deg, pan_head_deg=captured_pan_head_deg
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
        log_callback=None,
        tilt_head_deg=None,
        pan_head_deg=None
    ):
        # Fit encoder-indexed trajectories, not noisy small-arc plane normals.
        # A stationary, unknown marker gives no independent pan-zero reference.
        # Terminal tilt vs camera mounting is an exact gauge as well.
        from scipy.optimize import least_squares
        from core.calibration_optimizer import se3_exp, rot_to_euler_zyx
        if self.robot is None or not hasattr(self.robot, "get_dynamics"):
            raise RuntimeError("Head sweep fitting requires connected robot kinematics; canned mock results are disabled.")
        tilt_pts, pan_pts = np.asarray(pts_tilt_cam), np.asarray(pts_pan_cam)
        ta, pa = np.asarray(captured_tilt_angles), np.asarray(captured_pan_angles)
        if len(tilt_pts) < 5 or len(pan_pts) < 5 or np.ptp(ta) < 10 or np.ptp(pa) < 10:
            raise ValueError("Insufficient encoder sweep coverage (at least 5 points and 10 degrees per axis).")
        points = np.vstack((tilt_pts, pan_pts))
        if points.shape != (len(ta) + len(pa), 3) or not np.all(np.isfinite(points)):
            raise ValueError("Invalid head sweep measurements.")
        commands = np.vstack((np.column_stack((np.zeros_like(ta), ta)),
                              np.column_stack((pa, np.zeros_like(pa)))))
        if tilt_head_deg is not None or pan_head_deg is not None:
            tilt_head, pan_head = np.asarray(tilt_head_deg), np.asarray(pan_head_deg)
            if tilt_head.shape != (len(ta), 2) or pan_head.shape != (len(pa), 2):
                raise ValueError('Both head encoders are required for every head sweep sample')
            commands = np.vstack((tilt_head, pan_head))
            if not np.all(np.isfinite(commands)):
                raise ValueError('Invalid head encoder angles')
        model, dynamics = self.robot.model(), self.robot.get_dynamics()
        q0 = np.asarray(self.robot.get_state().position).copy()
        reference = self.camera_config.get("head_tilt_reference_deg")
        tilt_reference = 0.0 if reference is None else float(reference)
        transforms = []
        for angles in commands:
            q = q0.copy()
            q[model.head_idx] = np.deg2rad(angles + [0.0, tilt_reference])
            transforms.append(BaseCalibrator.compute_fk(self.robot, dynamics, q, "link_head_2", "link_head_0"))
        transforms = np.asarray(transforms)
        nominal = BaseCalibrator.make_transform(nominal_mount_to_cam)
        initial_points = [H[:3, :3] @ (nominal[:3, :3] @ p + nominal[:3, 3]) + H[:3, 3]
                          for H, p in zip(transforms, points)]
        x0 = np.r_[np.zeros(6), np.median(initial_points, axis=0)]
        def residual(x):
            camera = nominal @ se3_exp(x[:6])
            world = np.einsum('nij,nj->ni', transforms[:, :3, :3],
                              points @ camera[:3, :3].T + camera[:3, 3]) + transforms[:, :3, 3]
            return (world - x[6:]).ravel()
        bounds = np.r_[np.full(3, np.deg2rad(3.0)), np.full(3, 0.010), np.full(3, np.inf)]
        fit = least_squares(residual, x0, bounds=(-bounds, bounds), jac="3-point",
                            x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=300)
        errors = residual(fit.x).reshape(-1, 3)
        rmse_mm = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))) * 1000)
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        rank = int(np.sum(singular > singular[0] * 1e-7))
        at_bounds = bool(np.any(np.abs(fit.x[:6]) > 0.995 * bounds[:6]))
        success = bool(fit.success and rank == 9 and rmse_mm <= 1.0 and not at_bounds)
        camera = nominal @ se3_exp(fit.x[:6])
        rpy = np.rad2deg(rot_to_euler_zyx(camera[:3, :3]))
        results = {
            "success": success,
            "nominal_mount_to_cam": list(nominal_mount_to_cam),
            "calibrated_mount_to_cam": [*camera[:3, 3].tolist(), *rpy.tolist()],
            "head_offsets_deg": {"pan": 0.0, "tilt": tilt_reference},
            "head_offset_convention": "physical_error_q_plus_delta",
            "head_tilt_mode": "independent_reference" if reference is not None else "effective_zero_gauge",
            "camera_independently_calibrated": False,
            "cam_rot_diff_deg": dict(zip(("roll", "pitch", "yaw"), (rpy - np.asarray(nominal_mount_to_cam[3:])).tolist())),
            "quality": {"rmse_3d_marker_mm": rmse_mm, "data_rank": rank,
                        "free_parameters": 9, "condition_number": float(singular[0] / max(singular[-1], 1e-16)),
                        "at_bounds": at_bounds, "decoupled": False,
                        "reason": "Stationary-marker fit; pan zero and terminal tilt require independent references."},
            "pts_tilt_count": len(tilt_pts), "pts_pan_count": len(pan_pts)}
        self.calibrated_results = results
        if log_callback:
            log_callback(f"[Step 1.5] Encoder trajectory fit: RMSE={rmse_mm:.4f} mm, rank={rank}/9, accepted={success}")
            log_callback("[GAUGE] Camera estimate is an initialization only; it must not be locked as an independent measurement in Step 2.")
        return results

    def apply_calibration_results(self, results=None, log_callback=None):
        """Commit validated settings before publishing calibration to memory."""
        from core.config_store import update_yaml
        from core.config_store import CONFIG_PATHS
        if results is None:
            results = self.calibrated_results
        if not results or not results.get("success", False):
            if log_callback:
                log_callback("[ERROR] No accepted head calibration results to apply.")
            return False
        if results.get("skipped"):
            return True
        try:
            values = results.get("calibrated_mount_to_cam")
            if values is None or len(values) != 6:
                raise ValueError("calibrated_mount_to_cam must contain six numbers")
            camera_updates = {
                "mount_to_cam": [float(v) for v in values],
                "extrinsic_source": "sweep_effective_initialization",
                "head_tilt_mode": results.get("head_tilt_mode", "effective_zero_gauge")}
            head = {key: float(results.get("head_offsets_deg", {}).get(key, 0.0))
                    for key in ("pan", "tilt")}

            def mutate(cfg):
                camera = cfg.setdefault("camera", {})
                camera.setdefault("mount_to_cam_nominal", list(self.camera_config.get(
                    "mount_to_cam_nominal", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])))
                camera.setdefault("head_base_to_cam_nominal", list(self.camera_config.get(
                    "head_base_to_cam_nominal", [0.098, 0.009, 0.012, -90.0, 0.0, -90.0])))
                camera.update(camera_updates)
                cfg.setdefault("joint_offset", {}).setdefault("head", {}).update(head)

            committed = update_yaml(CONFIG_PATHS["setting_yaml"], mutate)
        except Exception as exc:
            if log_callback:
                log_callback(f"[ERROR] Failed to save calibration results: {exc}")
            return False

        camera_updates.update({key: committed["camera"][key]
                               for key in ("mount_to_cam_nominal", "head_base_to_cam_nominal")})
        self.camera_config.update(camera_updates)
        app = getattr(self, "app", None)
        if app is not None:
            if not hasattr(app, "joint_offsets_store"):
                app.joint_offsets_store = {}
            app.joint_offsets_store.setdefault("head", {}).update(head)
            for name in ("marker_calibrator", "joint_calibrator"):
                calibrator = getattr(app, name, None)
                if calibrator is not None:
                    calibrator.camera_config.update(camera_updates)
        if log_callback:
            log_callback(f"[SUCCESS] Saved head-camera calibration to {CONFIG_PATHS['setting_yaml']}")
        return True
