import time
import logging
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator
# Gross pose-consistency limits, independent of the 0.06 degree joint
# convergence rule. A constant pose bias can pass these checks.
AXIS_CONSISTENCY_DEG = 0.5
MIN_INLIER_FRACTION = 0.8
MIN_FRAMES = 10


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm < 1e-8:
        raise ValueError('Invalid or degenerate sweep axis')
    return vector / norm


def _distance_deg(a, b):
    return np.rad2deg(np.arccos(np.clip(np.asarray(a) @ b, -1., 1.)))


def _marker_axis(poses, camera_axis):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4,4) or len(poses) < MIN_FRAMES:
        raise ValueError(f'At least {MIN_FRAMES} camera poses are required per sweep')
    if not np.all(np.isfinite(poses)):
        raise ValueError('Non-finite camera pose')
    rotations = poses[:, :3, :3]
    if (np.max(np.abs(rotations.transpose(0,2,1) @ rotations - np.eye(3))) > 1e-3
            or np.max(np.abs(np.linalg.det(rotations) - 1.)) > 1e-3):
        raise ValueError('Invalid camera rotation matrix')
    vectors = np.einsum('nji,j->ni', rotations, _unit(camera_axis))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    center = _unit(np.median(vectors, axis=0))
    distances = _distance_deg(vectors, center)
    inliers = distances <= AXIS_CONSISTENCY_DEG
    fraction = float(np.mean(inliers))
    # A smooth drift or two different branches must not be hidden by an
    # average, even if individual frames form locally tight groups.
    width = max(1, len(vectors)//3)
    early = _unit(np.median(vectors[:width], axis=0))
    late = _unit(np.median(vectors[-width:], axis=0))
    early_late = float(_distance_deg(early, late))
    accepted = fraction >= MIN_INLIER_FRACTION and early_late <= AXIS_CONSISTENCY_DEG
    diagnostics = dict(accepted=bool(accepted), frames=len(vectors),
        inlier_fraction=fraction, p90_deviation_deg=float(np.percentile(distances,90)),
        early_late_deg=early_late, consistency_limit_deg=AXIS_CONSISTENCY_DEG)
    if not accepted:
        return None, diagnostics
    axis = _unit(np.mean(vectors[inliers], axis=0))
    return axis, diagnostics


def estimate_j6_reference(poses_a, axis_a, poses_b, axis_b,
                          nominal_rotation):
    """Return the measured DELTA correction at the current staged command.

    Positive axes must come from ordered marker circles. Coaxial bracket
    twist remains inseparable from J6; this is an effective reference only.
    """
    diagnostics = {}
    try:
        nominal = np.asarray(nominal_rotation, dtype=float)
        if (nominal.shape != (3,3) or not np.all(np.isfinite(nominal))
                or not np.allclose(nominal.T @ nominal, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(nominal), 1., atol=1e-6)):
            raise ValueError('Invalid nominal bracket rotation')
        ref_j5 = nominal.T @ [0.,1.,0.]
        n6, diagnostics['j6_sweep'] = _marker_axis(poses_a, axis_a)
        n5, diagnostics['j5_sweep'] = _marker_axis(poses_b, axis_b)
        if n6 is None or n5 is None:
            raise ValueError('Inconsistent marker-frame sweep axes; no unique stable reference selected')
        if abs(float(_distance_deg(n5, n6))-90.) > AXIS_CONSISTENCY_DEG:
            raise ValueError('Observed J5/J6 axes violate the fixed perpendicular wrist geometry')
        actual = _unit(n5 - np.dot(n5,n6)*n6)
        reference = _unit(ref_j5 - np.dot(ref_j5,n6)*n6)
        cross = _unit(np.cross(n6,reference))
        raw = float(np.rad2deg(np.arctan2(np.dot(actual,cross), np.dot(actual,reference))))
        return dict(measurement_accepted=True, raw_diff_deg=raw,
                    optimal_offset=raw,
                    n6_marker_actual=n6, n5_marker_actual=n5,
                    quality_diagnostics=diagnostics, j6_mode='effective_bracket_reference')
    except ValueError as error:
        # Deliberately omit optimal_offset. The caller must not manufacture
        # a zero correction or issue another inner-iteration correction.
        return dict(measurement_accepted=False, failure_reason=str(error),
                    quality_diagnostics=diagnostics, j6_mode='effective_bracket_reference')


class DebugLogger:
    def __init__(self, original_log_callback, file_path):
        self.original_log_callback = original_log_callback
        self.file_path = file_path
        self.buffer = []
        
    def log(self, msg):
        self.buffer.append(msg)
        msg_upper = msg.upper()
        if (
            "[SAFETY WARNING]" in msg_upper or
            "[SUCCESS]" in msg_upper or
            "[ERROR]" in msg_upper or
            "[WARN]" in msg_upper or
            "[INFO]" in msg_upper or
            "[MEASUREMENT REJECTED]" in msg_upper or
            "[SWEEP QUALITY]" in msg_upper or
            "[SWEEP COMMAND]" in msg_upper or
            "[J6 REFERENCE]" in msg_upper or
            "[VALIDATION SWEEP]" in msg_upper or
            "[ITERATION" in msg_upper or
            "RECOMMENDED ABSOLUTE OFFSET" in msg_upper or
            "STEP CORRECTION" in msg_upper or
            "COMMENCING" in msg_upper or
            "SWEPT" in msg_upper or
            "SWEEP COMPLETE" in msg_upper or
            "STARTING" in msg_upper
        ):
            if self.original_log_callback:
                self.original_log_callback(msg)
                
    def save(self):
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n=== NEW ITERATION ===\n")
                f.write("\n".join(self.buffer) + "\n")
        except Exception:
            pass

class JointCalibrator(BaseCalibrator):
    def __init__(self, marker_st=None, robot=None):
        super().__init__(marker_st, robot)


    def perform_joint_calibration(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=None, save_debug=False, pass_idx=1, pass1_res=None):
        if sweep_duration is None:
            sweep_duration = self.JOINT_SWEEP_SECONDS[mode]

        config_dir = os.path.abspath(os.path.dirname(__file__))
        from core.config_store import CONFIG_PATHS
        result_txt_dir = CONFIG_PATHS["txt_dir"]
        os.makedirs(result_txt_dir, exist_ok=True)
        debug_file_path = os.path.join(result_txt_dir, f"joint_calib_debug_{arm_side}_{mode}.txt")
        
        # 처음 시작(pass 1)일 때는 덮어쓰기 위해 기존 파일들 삭제
        if pass_idx == 1:
            if os.path.exists(debug_file_path):
                try: os.remove(debug_file_path)
                except: pass
                
            jcfg = self.JOINT_CONFIGS.get(mode, {})
            for key_type, j_key in [("joint_A", "sweep_joint_A"), ("joint_B", "sweep_joint_B")]:
                axis = jcfg.get(j_key)
                if axis is not None:
                    fname = os.path.join(result_txt_dir, f"sweep_points_{arm_side}_{key_type}_axis_{axis}.txt")
                    if os.path.exists(fname):
                        try: os.remove(fname)
                        except: pass

        logger = DebugLogger(log_callback, debug_file_path)
        original_log = log_callback
        log_callback = logger.log

        try:
            self.last_staged_offset = None
            self.last_diff_angle = None
            if original_log:
                log_callback("\n" + "="*60)
                log_callback("   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE")
                log_callback(f"   Target Arm: {arm_side.upper()} | Joint Target: {mode.upper()}")
                log_callback("="*60 + "\n")
                
            first_starting_pose = None
            if self.robot:
                try:
                    state = self.robot.get_state()
                    model = self.robot.model()
                    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                    first_starting_pose = list(state.position[arm_idx])
                except Exception:
                    pass

            # Preserve every iteration for replay, including the failed one.
            def run_single_sweep(offset):
                return self.perform_calibration_sweep_continuous(
                    arm_side, mode, log_callback=log_callback, status_callback=status_callback,
                    current_offset_deg=offset, sweep_duration=sweep_duration,
                    save_debug=save_debug,
                    first_starting_pose=first_starting_pose
                )
                
            max_iterations = 6
            staged_offset = current_offset_deg
            staged_offsets_history = [staged_offset]
            final_res = None
            first_res = None
            converged = False
            measurement_accepted = True
            
            # Sign-reversal tracking state
            prev_error = None
            prev_step_correction = 0.0
            direction_multiplier = 1.0
            dynamic_damping = 1.0
            prev_step_correction = 0.0
            
            for i in range(1, max_iterations + 1):
                # Update self.joint_offsets with staged_offset for proper FK offset subtraction in this iteration
                jcfg = self.JOINT_CONFIGS.get(mode, {})
                offset_key = jcfg.get("offset_key")
                if offset_key:
                    if arm_side in self.joint_offsets:
                        self.joint_offsets[arm_side][offset_key] = staged_offset
                    else:
                        self.joint_offsets[offset_key] = staged_offset

                if getattr(self, 'stop_requested', False):
                    if log_callback: log_callback("[INFO] Joint calibration aborted due to stop request.")
                    return None

                if log_callback:
                    log_callback(f"\n[ITERATION {i}/{max_iterations}] Sweeping physically with staged offset {staged_offset:.4f}°...")
                
                # Same SDK motion and sensor interface for either marker source.
                res = run_single_sweep(staged_offset)
                if not res:
                    if log_callback: log_callback(f"[ERROR] Iteration {i} sweep failed. Aborting calibration.")
                    return None

                if not res.get('measurement_accepted', True):
                    measurement_accepted = False
                    final_res = res
                    if log_callback:
                        log_callback(f"[MEASUREMENT REJECTED] {res.get('failure_reason', 'Invalid sweep')}. "
                                     f"Keeping staged offset {staged_offset:.4f}°; no correction from this measurement.")
                        log_callback(f"[SWEEP QUALITY] {res.get('quality_diagnostics', {})}")
                    break
                
                if i == 1:
                    first_res = res
                final_res = res
        
                angle_error = res.get('angle_between_normals', 0.0)
                sign = res.get('sign', 1.0)
                
                # wrist_roll_v13 (J6 vs J5) and wrist_pitch_v13 (J6 vs J4) have perpendicular axes (target 90 deg)
                if mode in ("wrist_roll_v13", "wrist_pitch_v13"):
                    angle_dev = abs(angle_error - 90.0)
                    center_dist = res.get('perp_dist_after', 999.0)
                else:
                    angle_dev = angle_error
                    center_dist = res.get('center_dist', 999.0)
                    
                r_A = res.get('r_A', 0.0)
                r_B = res.get('r_B', 0.0)
                size_error = abs(r_A - r_B)
                current_error = max(size_error, center_dist)
                
                # Print iteration summary
                if log_callback:
                    if mode in ("wrist_roll_v13", "wrist_pitch_v13"):
                        log_callback(f"  * Angle Error (Deviation)          : {angle_dev:.4f}°")
                        log_callback(f"  * Perpendicular Distance (After)   : {center_dist:.4f} mm")
                        log_callback(f"  * Perpendicular Distance (Before)  : {res.get('perp_dist_before', 999.0):.4f} mm")
                    else:
                        log_callback(f"  * Angle Error (Deviation)     : {angle_error:.4f}°")
                        if mode == "wrist_pitch_v13":
                            log_callback(f"  * Forearm Length (Center Dist): {center_dist:.4f} mm")
                            log_callback(f"  * Radii Difference (r3 - r5)  : {size_error:.4f} mm")
                        else:
                            log_callback(f"  * Circle Size Error (r_A-r_B) : {size_error:.4f} mm")
                            log_callback(f"  * Center Distance Error       : {center_dist:.4f} mm")
                            log_callback(f"  * Max Fitting Error Metric    : {current_error:.4f} mm")
                
                # Use the pre-calculated damped optimal offset correction to ensure convergence
                raw_optimal_offset = res.get('optimal_offset', 0.0)
                
                # Every solver returns a measured relative correction; staging is command-only.
                raw_delta = raw_optimal_offset
                if not np.isfinite(raw_delta):
                    raise ValueError("Non-finite joint correction; calibration rejected")
                if i > 1 and raw_delta * prev_step_correction < 0:
                    dynamic_damping *= 0.8
                elif i == 1:
                    dynamic_damping = 1.0
                    
                step_correction = direction_multiplier * raw_delta * dynamic_damping
                
                # Calculate relative step delta for convergence check
                step_correction_delta = raw_delta

                # Convergence check:
                # step correction delta < 0.06° to handle bracket RPY noise
                converged_criteria = (abs(step_correction_delta) < 0.06)
                
                if converged_criteria:
                    converged = True
                    if log_callback:
                        log_callback(f"\n[SUCCESS] Calibration CONVERGED successfully:")
                        log_callback(f"  * Step Correction: {step_correction_delta:.4f}° < 0.06° (reached resolution limit)")
                        log_callback(f"  * Recommended Absolute Offset: {staged_offset:.4f}°")
                    break
                
                # Normal update: apply correction
                prev_error = angle_dev
                prev_step_correction = step_correction_delta
                staged_offset += step_correction
                
                # Safety: clamp staged_offset to the joint's configured offset range
                jcfg = self.JOINT_CONFIGS.get(mode, {})
                off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
                if staged_offset < off_min or staged_offset > off_max:
                    if log_callback:
                        log_callback(f"  [SAFETY WARNING] Staged offset {staged_offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                    staged_offset = float(np.clip(staged_offset, off_min, off_max))
                    
                staged_offsets_history.append(staged_offset)
                if log_callback:
                    log_callback(f"  * Updated Absolute Offset     : {staged_offset:.4f}°")
                    
            # Damping fallback for oscillation/noise-floor:
            if measurement_accepted and not converged and len(staged_offsets_history) >= 3:
                avg_offset = float(np.mean(staged_offsets_history[-3:]))
                if log_callback:
                    log_callback(f"\n[INFO] Joint {mode} did not meet 0.06° convergence tolerance; cause is not determined. Fallback remains unconverged.")
                    log_callback(f"       Damping fallback: Averaged last 3 offsets ({', '.join(f'{v:.4f}°' for v in staged_offsets_history[-3:])}) -> {avg_offset:.4f}°")
                staged_offset = avg_offset

            # Final range safety: clamp to configured offset_range
            jcfg = self.JOINT_CONFIGS.get(mode, {})
            off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
            if staged_offset < off_min or staged_offset > off_max:
                if log_callback:
                    log_callback(f"  [SAFETY WARNING] Recommended final offset {staged_offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                staged_offset = float(np.clip(staged_offset, off_min, off_max))
        
            if getattr(self, 'stop_requested', False):
                if log_callback: log_callback("[INFO] Joint calibration aborted before final report.")
                return None
        
            # Build clean final output dict — UI only needs these fields
            final_output = {
                'mode': mode,
                'offset_convention': 'home_correction_minus_physical_error',
                'j6_mode': 'effective_bracket_reference' if mode in ('wrist_roll_v13', 'wrist_yaw2') else None,
                'recommended_joint_offset': staged_offset,
                'optimal_offset': staged_offset,
                'converged': converged,
                'measurement_accepted': measurement_accepted,
                'failure_reason': final_res.get('failure_reason') if final_res else None,
                'quality_diagnostics': final_res.get('quality_diagnostics', {}) if final_res else {},
                'perp_dist_before': final_res.get('perp_dist_before', float('nan')) if final_res else float('nan'),
                'perp_dist_after': final_res.get('perp_dist_after', float('nan')) if final_res else float('nan'),
                'axial_offset_mm': final_res.get('axial_offset_mm', float('nan')) if final_res else float('nan'),
                'lateral_offset_mm': final_res.get('lateral_offset_mm', float('nan')) if final_res else float('nan'),
                'r_A': final_res.get('r_A', float('nan')) if final_res else float('nan'),
                'r_B': final_res.get('r_B', float('nan')) if final_res else float('nan'),
            }
        
            # Save first_res and final_res inside final_output so that the caller (FullAutoWorker) can retrieve them
            final_output['first_res'] = first_res
            final_output['final_res'] = final_res

            # Plot generation logic
            validation_res = final_res
            if measurement_accepted and validation_res and (first_res or pass1_res):
                if pass_idx == 2 and pass1_res is not None:
                    # True cross-pass BEFORE (Pass 1 start) vs AFTER (Pass 2 validation) comparison plot
                    first_res_for_plot = pass1_res.get('first_res', first_res)
                    plot_path = self.save_calibration_comparison_plot(
                        arm_side, mode, first_res_for_plot, validation_res, 
                        log_callback=log_callback, force_overwrite=True
                    )
                else:
                    # In Pass 1 or manual mode, save a comparison plot of the current pass.
                    # In Pass 1, this will be overwritten later when Pass 2 completes.
                    plot_path = self.save_calibration_comparison_plot(
                        arm_side, mode, first_res, validation_res, 
                        log_callback=log_callback, force_overwrite=True
                    )
                final_output['plot_path_combined'] = plot_path
            
            return final_output
        finally:
            logger.save()




    def save_calibration_comparison_plot(self, arm_side, mode, first_res, final_res, log_callback=None, force_overwrite=False):
        if final_res and not final_res.get('measurement_accepted', True):
            return None
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            def extract_plot_dict(res_obj, default_stage="first"):
                if not res_obj or not isinstance(res_obj, dict):
                    return {}
                if '_plot_data' in res_obj:
                    return res_obj
                if default_stage == "first":
                    if 'first_res' in res_obj and isinstance(res_obj['first_res'], dict):
                        return extract_plot_dict(res_obj['first_res'], "first")
                    if 'final_res' in res_obj and isinstance(res_obj['final_res'], dict):
                        return extract_plot_dict(res_obj['final_res'], "final")
                else:
                    if 'final_res' in res_obj and isinstance(res_obj['final_res'], dict):
                        return extract_plot_dict(res_obj['final_res'], "final")
                    if 'first_res' in res_obj and isinstance(res_obj['first_res'], dict):
                        return extract_plot_dict(res_obj['first_res'], "first")
                return res_obj

            first_res_actual = extract_plot_dict(first_res, "first")
            final_res_actual = extract_plot_dict(final_res, "final")

            def plot_column(res, col_idx, stage_name):
                # Read from internal _plot_data bundle
                pd = res.get('_plot_data', res)
                pts_a      = pd.get('pts_a_cam')
                pts_b      = pd.get('pts_b_cam')
                c_A        = pd.get('c_A')
                c_B        = pd.get('c_B')
                n_A        = pd.get('n_A')
                n_B        = pd.get('n_B')
                r_A        = pd.get('r_A', res.get('r_A', 1.0))
                r_B        = pd.get('r_B', res.get('r_B', 1.0))
                angle_error  = pd.get('angle_between_normals', res.get('angle_between_normals', 0.0))
                center_dist  = pd.get('center_dist', res.get('center_dist', 0.0))

                if pts_a is None or c_A is None or n_A is None:
                    # No plot data available for this sweep — leave panel blank
                    for row in range(2):
                        axes[row, col_idx].set_title(f'[{stage_name}] No plot data')
                        axes[row, col_idx].axis('off')
                    return

                # Compute local frames algebraically from normals (Z axes)
                def get_local_vectors(n):
                    n = n / np.linalg.norm(n)
                    if abs(n[0]) < 0.9:
                        u = np.cross(n, [1, 0, 0])
                    else:
                        u = np.cross(n, [0, 1, 0])
                    u = u / np.linalg.norm(u)
                    v = np.cross(n, u)
                    v = v / np.linalg.norm(v)
                    return u, v

                u_A, v_A = get_local_vectors(n_A)
                u_B, v_B = get_local_vectors(n_B)

                theta = np.linspace(0, 2 * np.pi, 200)
                circle_pts_a = c_A + r_A * (np.cos(theta)[:, None] * u_A + np.sin(theta)[:, None] * v_A)
                circle_pts_b = c_B + r_B * (np.cos(theta)[:, None] * u_B + np.sin(theta)[:, None] * v_B)

                # --- 1. TOP VIEW (Row 0, Col col_idx): X-Y Projection ---
                ax_top = axes[0, col_idx]
                ax_top.scatter(pts_a[:, 0], pts_a[:, 1], c='red', s=15, alpha=0.5, label='Sweep A Raw')
                ax_top.scatter(pts_b[:, 0], pts_b[:, 1], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
                ax_top.plot(circle_pts_a[:, 0], circle_pts_a[:, 1], 'r-', linewidth=1.5, label='Sweep A Fit')
                ax_top.plot(circle_pts_b[:, 0], circle_pts_b[:, 1], 'b-', linewidth=1.5, label='Sweep B Fit')
                ax_top.scatter([c_A[0]], [c_A[1]], c='darkred', marker='X', s=100, label='Center A')
                ax_top.scatter([c_B[0]], [c_B[1]], c='darkblue', marker='X', s=100, label='Center B')
                ax_top.plot([c_A[0], c_B[0]], [c_A[1], c_B[1]], color='purple', linestyle=':', linewidth=2, label='Center Shift')

                scale = min(r_A, r_B) * 0.4
                ax_top.arrow(c_A[0], c_A[1], n_A[0]*scale, n_A[1]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
                ax_top.arrow(c_B[0], c_B[1], n_B[0]*scale, n_B[1]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')
                ax_top.set_xlabel('X (mm)')
                ax_top.set_ylabel('Y (mm)')
                ax_top.set_title(f'[{stage_name}] Top View (X-Y Projection)', fontsize=15, fontweight='bold')
                ax_top.set_aspect('equal')
                ax_top.grid(True)
                if col_idx == 0:
                    ax_top.legend(loc='upper right', fontsize=10)

                # --- 2. SIDE VIEW (Row 1, Col col_idx): Y-Z Projection ---
                ax_side = axes[1, col_idx]
                ax_side.scatter(pts_a[:, 1], pts_a[:, 2], c='red', s=15, alpha=0.5, label='Sweep A Raw')
                ax_side.scatter(pts_b[:, 1], pts_b[:, 2], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
                ax_side.plot(circle_pts_a[:, 1], circle_pts_a[:, 2], 'r-', linewidth=1.5, label='Sweep A Fit')
                ax_side.plot(circle_pts_b[:, 1], circle_pts_b[:, 2], 'b-', linewidth=1.5, label='Sweep B Fit')
                ax_side.scatter([c_A[1]], [c_A[2]], c='darkred', marker='X', s=100, label='Center A')
                ax_side.scatter([c_B[1]], [c_B[2]], c='darkblue', marker='X', s=100, label='Center B')
                ax_side.plot([c_A[1], c_B[1]], [c_A[2], c_B[2]], color='purple', linestyle=':', linewidth=2, label='Center Shift')
                ax_side.arrow(c_A[1], c_A[2], n_A[1]*scale, n_A[2]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
                ax_side.arrow(c_B[1], c_B[2], n_B[1]*scale, n_B[2]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')
                ax_side.set_xlabel('Y (mm)')
                ax_side.set_ylabel('Z (mm)')
                ax_side.set_title(f'[{stage_name}] Side View (Y-Z Projection)\nAngle Dev: {angle_error:.3f}° | Center Dist: {center_dist:.2f}mm', fontsize=15, fontweight='bold')
                ax_side.set_aspect('equal')
                ax_side.grid(True)

            def compute_shortest_distance_between_lines(cA, nA, cB, nB):
                nA_norm = nA / np.linalg.norm(nA)
                nB_norm = nB / np.linalg.norm(nB)
                cross = np.cross(nA_norm, nB_norm)
                cross_norm = np.linalg.norm(cross)
                diff = cB - cA
                if cross_norm > 1e-4:
                    return abs(np.dot(diff, cross)) / cross_norm
                else:
                    return np.linalg.norm(diff - np.dot(diff, nA_norm) * nA_norm)

            nominal_dist_35 = None
            if mode == "wrist_pitch_v13" and self.robot:
                try:
                    dyn_model = self.robot.get_dynamics()
                    names = self.robot.model().robot_joint_names
                    state_3_5 = dyn_model.make_state(
                        [f"link_{arm_side}_arm_3", f"link_{arm_side}_arm_5"],
                        names
                    )
                    state_3_5.set_q(np.zeros(len(names)))
                    dyn_model.compute_forward_kinematics(state_3_5)
                    T_3_5 = dyn_model.compute_transformation(state_3_5, 0, 1)
                    nominal_dist_35 = np.linalg.norm(T_3_5[:3, 3]) * 1000.0
                except Exception:
                    pass

            plot_column(first_res_actual, 0, "BEFORE")
            plot_column(final_res_actual, 1, "AFTER")

            before_dist_str = ""
            after_dist_str = ""
            if mode == "wrist_pitch_v13":
                first_pd = first_res_actual.get('_plot_data', first_res_actual)
                final_pd = final_res_actual.get('_plot_data', final_res_actual)
                if all(k in first_pd for k in ('c_A', 'n_A', 'c_B', 'n_B')):
                    dist_before = compute_shortest_distance_between_lines(
                        first_pd['c_A'], first_pd['n_A'], first_pd['c_B'], first_pd['n_B']
                    )
                    before_dist_str = f" | Axis 3-5 Dist = {dist_before:.2f} mm"
                if all(k in final_pd for k in ('c_A', 'n_A', 'c_B', 'n_B')):
                    dist_after = compute_shortest_distance_between_lines(
                        final_pd['c_A'], final_pd['n_A'], final_pd['c_B'], final_pd['n_B']
                    )
                    after_dist_str = f" | Axis 3-5 Dist = {dist_after:.2f} mm"
                    if nominal_dist_35 is not None:
                        after_dist_str += f" (Nom: {nominal_dist_35:.2f} mm)"

            first_pd = first_res_actual.get('_plot_data', first_res_actual)
            final_pd = final_res_actual.get('_plot_data', final_res_actual)
            fig.suptitle(
                f"Joint Calibration: {arm_side.upper()} Arm - {mode.upper()}\n"
                f"Before: Angle Dev = {first_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {first_pd.get('center_dist', 0.0):.2f} mm{before_dist_str}\n"
                f"After : Angle Dev = {final_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {final_pd.get('center_dist', 0.0):.2f} mm{after_dist_str}",
                fontsize=16, fontweight='bold'
            )
            plt.tight_layout()

            from core.config_store import CONFIG_PATHS
            result_dir = CONFIG_PATHS["plot_dir"]
            os.makedirs(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"circle_fit_{arm_side}_{mode}_joint_calib.png"))
            if not force_overwrite and os.path.exists(plot_save_path):
                plt.close()
                if log_callback:
                    log_callback(f"[INFO] Comparison plot already exists at: {plot_save_path}, skipping overwrite.")
            else:
                plt.savefig(plot_save_path, dpi=150)
                plt.close()
                if log_callback:
                    log_callback(f"[SUCCESS] Saved combined calibration comparison plot to: {plot_save_path}")
            return plot_save_path
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to save combined calibration comparison plot: {e}")
            import traceback
            if log_callback:
                log_callback(traceback.format_exc())
            return None

    def perform_calibration_sweep_continuous(self, arm_side, mode, log_callback=None, status_callback=None,
            current_offset_deg=0.0, sweep_duration=None,
            save_debug=False, first_starting_pose=None):
        if self.stop_requested or not self.robot or self.marker_st is None:
            return None
        duration = sweep_duration or self.JOINT_SWEEP_SECONDS[mode]
        cfg = self.JOINT_CONFIGS[mode]
        # Encoder feedback is used to construct a motion command, never passed
        # to the geometric estimator.
        indices = getattr(self.robot.model(), arm_side + '_arm_idx')
        baseline = np.array(first_starting_pose if first_starting_pose is not None
                            else self.robot.get_state().position[indices], copy=True)
        nominal = self.get_ready_pose('v' + self.get_robot_version(), 'joint', mode, arm_side)
        key = cfg['offset_key']
        taught = getattr(self, 'user_taught_ready_poses', {}).get(arm_side, {}).get(key)
        if taught is not None:
            baseline = np.array(taught, copy=True)
        baseline[cfg['cand_joint']] = nominal[cfg['cand_joint']] + np.deg2rad(current_offset_deg)
        if not self.marker_st.get_marker_transform(sampling_time=2., side=arm_side):
            if status_callback: status_callback(False)
            if log_callback: log_callback('[ERROR] Marker is not visible in ready pose')
            return None
        if status_callback: status_callback(True)
        datasets = []
        for label in ('A', 'B'):
            axis, span = cfg['sweep_joint_' + label], cfg['sweep_range_' + label]
            poses = self.perform_single_joint_sweep(arm_side, axis, baseline, -span, span, duration,
                        label='Joint ' + label, log_callback=log_callback, mode=mode)
            if poses is None: return None
            if save_debug:
                self.save_observed_points(arm_side, axis, poses, 'joint_' + label)
            datasets.append(poses)
        candidate = None
        if cfg['cand_joint'] != 6:
            # Measure the sign-reference axis instead of predicting it using
            # encoder/FK. Negative-to-zero also respects the J3 upper limit.
            candidate = self.perform_single_joint_sweep(arm_side, cfg['cand_joint'], baseline,
                        -15., 0., duration, label='Direction reference', log_callback=log_callback, mode=mode)
            if candidate is None: return None
            if save_debug:
                self.save_observed_points(arm_side, cfg['cand_joint'], candidate, 'joint_C')
        return self.compute_calibration_results(arm_side, mode, *datasets,
                        dataset_C=candidate, log_callback=log_callback)

    def compute_calibration_results(self, arm_side, mode, dataset_A, dataset_B,
            *, log_callback=None, dataset_C=None):
        """Observed circles only: no encoder, FK, or staged offset inputs."""
        try:
            a = self.fit_observed_circle(dataset_A)
            b = self.fit_observed_circle(dataset_B)
            na, nb = a['axis'], b['axis']
            angle = float(np.rad2deg(np.arccos(np.clip(na @ nb, -1., 1.))))
            quality = {'circle_A': a['residual_rms_m'], 'circle_B': b['residual_rms_m']}
            if mode in ('wrist_yaw2', 'wrist_roll_v13'):
                parameters = getattr(self, 'robot_parameters', self._default_parameters)
                vector = parameters.nominal_brackets[self.get_robot_version()][arm_side]
                reference = R_scipy.from_euler('xyz', vector[3:6], degrees=True).as_matrix()
                result = estimate_j6_reference(dataset_A, na, dataset_B, nb, reference)
                result['bracket_reference_source'] = 'robot_config.nominal_brackets'
                if not result['measurement_accepted']: return result
            else:
                c = self.fit_observed_circle(dataset_C)
                a, b, c = self.refine_adjacent_circles((dataset_A, dataset_B, dataset_C), (a, b, c))
                na, nb = a['axis'], b['axis']
                angle = float(np.rad2deg(np.arccos(np.clip(na @ nb, -1., 1.))))
                quality = {'circle_A': a['residual_rms_m'], 'circle_B': b['residual_rms_m'],
                           'circle_C': c['residual_rms_m'], 'fit': 'observed_adjacent_axes_free_centers'}
                nc = c['axis']
                # A and B rotate about the same fixed candidate axis. Project
                # minor out-of-plane observation error, not onto a model axis.
                pa, pb = na - (na @ nc)*nc, nb - (nb @ nc)*nc
                if min(np.linalg.norm(pa), np.linalg.norm(pb)) < .5:
                    raise ValueError('Direction-reference circle is not independent')
                signed = float(np.rad2deg(np.arctan2(np.cross(pa, pb) @ nc, pa @ pb)))
                delta = signed + 90. if mode == 'wrist_pitch_v13' else -signed
                if abs(delta) > 30.:
                    raise ValueError('Observed circle relationship is outside calibration bounds')
                result = dict(measurement_accepted=True, optimal_offset=delta, quality_diagnostics=quality)
            delta_center = (b['center_m'] - a['center_m'])*1000.
            distance = float(np.linalg.norm(delta_center - np.dot(delta_center, na)*na))
            center_distance_3d = float(np.linalg.norm(delta_center))
            if mode in ('elbow', 'wrist_pitch'):
                size_error = abs(a['radius'] - b['radius'])
                if max(center_distance_3d, size_error) > 100. or (
                        abs(result['optimal_offset']) < .06 and max(center_distance_3d, size_error) > .5):
                    raise ValueError(f'Parallel-joint circles do not coincide: center={center_distance_3d:.3f} mm, radius difference={size_error:.3f} mm')
            result.update(mode=mode, converged=False, angle_between_normals=angle,
                center_dist=center_distance_3d, perp_dist_after=distance, r_A=a['radius'], r_B=b['radius'],
                _plot_data=dict(pts_a_cam=np.asarray(dataset_A)[:,:3,3]*1000.,
                    pts_b_cam=np.asarray(dataset_B)[:,:3,3]*1000., c_A=a['c_opt'], c_B=b['c_opt'],
                    n_A=na, n_B=nb, r_A=a['radius'], r_B=b['radius'],
                    angle_between_normals=angle, center_dist=distance))
            if log_callback:
                log_callback(f"[SWEEP QUALITY] {mode}: relative correction={result['optimal_offset']:.5f} deg; circles RMS={a['rmse']:.4f}/{b['rmse']:.4f} mm")
            return result
        except ValueError as error:
            return dict(measurement_accepted=False, failure_reason=str(error))
