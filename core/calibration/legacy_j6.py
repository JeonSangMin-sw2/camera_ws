"""September 2 (dfe5f4f) encoder-constrained circle fitting, isolated for v1.2 J6.

Historical numerical routines below retain their formulas (residuals copy parameter
slices to avoid mutating optimizer inputs); the adapter validates inputs
and never converts failed measurements to a zero correction.
"""
import numpy as np
from scipy.optimize import least_squares

# Physical A/B trial switch. False bypasses ONLY the historical five-frame
# moving median used to initialize the circle fit. It does not disable robust
# geometric residual rejection or change sample/encoder pairing. Set True to
# restore the September 2 initializer; no timestamp synchronization is added.
USE_J6_INITIAL_MEDIAN = False


class LegacyCircleFit:

    @staticmethod
    def fit_circle_3d(points, robust=True):
        """
        Fits a 3D circle to points.
        If robust is True, applies robust worst-inlier outlier rejection and moving median filter.
        Otherwise, performs a smooth closed-form algebraic fit for noise-free kinematics.
        Returns (center_3d, R_circle, radius, rmse, pts_2d, uc, vc)
        """
        points = np.array(points)
        
        if not robust:
            centroid = np.mean(points, axis=0)
            pts_centered = points - centroid
            _, _, vh = np.linalg.svd(pts_centered)
            normal = vh[2, :]
            ex = vh[0, :]
            ey = vh[1, :]
            pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)
            A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
            b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            uc, vc = res[0], res[1]
            center_3d = centroid + uc * ex + vc * ey
            R_circle = np.column_stack((ex, ey, normal))
            radius_3d = np.mean(np.linalg.norm(points - center_3d, axis=1))
            return center_3d, R_circle, radius_3d, 0.0, pts_2d, uc, vc
        
        # Apply 3D Moving Median Filter (window size 5) to smooth out camera sensor jitter
        if USE_J6_INITIAL_MEDIAN and len(points) >= 5:
            smoothed = np.copy(points)
            for i in range(2, len(points) - 2):
                smoothed[i] = np.median(points[i - 2 : i + 3], axis=0)
            points = smoothed
            
        inlier_mask = np.ones(len(points), dtype=bool)
        
        for out_iter in range(15):
            pts_in = points[inlier_mask]
            if len(pts_in) < 10:
                break
                
            centroid = np.mean(pts_in, axis=0)
            pts_centered = pts_in - centroid
            
            _, _, vh = np.linalg.svd(pts_centered)
            normal = vh[2, :]
            ex = vh[0, :]
            ey = vh[1, :]
            pts_2d_in = np.dot(pts_centered, np.vstack((ex, ey)).T)

            A = np.c_[2 * pts_2d_in[:, 0], 2 * pts_2d_in[:, 1], np.ones(len(pts_2d_in))]
            b = pts_2d_in[:, 0]**2 + pts_2d_in[:, 1]**2
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            uc, vc = res[0], res[1]
            radius = np.sqrt(max(0.001, res[2] + uc**2 + vc**2))
            
            def residuals(params):
                u, v, R = params
                return np.sqrt((pts_2d_in[:, 0] - u)**2 + (pts_2d_in[:, 1] - v)**2) - R
                
            opt = least_squares(residuals, [uc, vc, radius], loss='huber')
            uc_opt, vc_opt, R_opt = opt.x
            
            # Recompute errors for all original points
            all_errors = []
            for pt in points:
                pt_centered_all = pt - centroid
                u_all = np.dot(pt_centered_all, ex)
                v_all = np.dot(pt_centered_all, ey)
                dist_to_center = np.sqrt((u_all - uc_opt)**2 + (v_all - vc_opt)**2)
                err = abs(dist_to_center - R_opt)
                all_errors.append(err)
            all_errors = np.array(all_errors)
            
            # Find worst point among inliers
            inlier_indices = np.where(inlier_mask)[0]
            inlier_errors = all_errors[inlier_mask]
            worst_inlier_idx_in_inliers = np.argmax(inlier_errors)
            worst_global_idx = inlier_indices[worst_inlier_idx_in_inliers]
            worst_error = inlier_errors[worst_inlier_idx_in_inliers]
            
            if worst_error > 0.1:
                inlier_mask[worst_global_idx] = False
            else:
                break
                
        # Final fit on clean inliers
        pts_in = points[inlier_mask]
        centroid = np.mean(pts_in, axis=0)
        pts_centered = pts_in - centroid
        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[2, :]
        ex = vh[0, :]
        ey = vh[1, :]
        pts_2d_in = np.dot(pts_centered, np.vstack((ex, ey)).T)
        A = np.c_[2 * pts_2d_in[:, 0], 2 * pts_2d_in[:, 1], np.ones(len(pts_2d_in))]
        b = pts_2d_in[:, 0]**2 + pts_2d_in[:, 1]**2
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        uc, vc = res[0], res[1]
        radius = np.sqrt(max(0.001, res[2] + uc**2 + vc**2))
        
        def residuals_final(params):
            u, v, R = params
            return np.sqrt((pts_2d_in[:, 0] - u)**2 + (pts_2d_in[:, 1] - v)**2) - R
            
        opt = least_squares(residuals_final, [uc, vc, radius], loss='huber')
        uc_opt, vc_opt, R_opt = opt.x
        rmse = np.sqrt(np.mean(opt.fun**2))
        center_3d = centroid + uc_opt * ex + vc_opt * ey
        
        # 2D representation for all points
        pts_centered_all = points - centroid
        pts_2d_all = np.dot(pts_centered_all, np.vstack((ex, ey)).T)
        
        R_circle = np.column_stack((ex, ey, normal))
        
        radius_3d = np.mean(np.linalg.norm(pts_in - center_3d, axis=1))
        
        return center_3d, R_circle, radius_3d, rmse, pts_2d_all, uc_opt, vc_opt

    @staticmethod
    def fit_circle_3d_and_6dof_misalignment(relative_poses, captured_angles, axis_prior=None, return_plot_data=False, robust=True):
        points = np.array([T[:3, 3] * 1000.0 for T in relative_poses])
        angles_rad_base = np.radians(captured_angles)
        
        # Robust check for NaN or Inf in inputs
        if len(points) == 0:
            raise ValueError("fit_circle_3d_and_6dof_misalignment: Input points list is empty.")
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Input points contain NaN or Inf values! points={points}")
        if np.any(np.isnan(angles_rad_base)) or np.any(np.isinf(angles_rad_base)):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Input captured_angles contain NaN or Inf values! angles={captured_angles}")
            
        # Initial Center and Normal estimation using unified circle fit
        c_fit, R_fit, radius_fit, rmse_fit, _, _, _ = LegacyCircleFit.fit_circle_3d(points, robust=robust)
        
        # Check if initial circle fit yielded valid numbers
        if np.any(np.isnan(c_fit)) or np.any(np.isinf(c_fit)) or np.isnan(radius_fit):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Initial circle fit (fit_circle_3d) returned NaN or Inf values! c_fit={c_fit}, radius_fit={radius_fit}")
        
        # Nominal Normal
        if axis_prior is not None:
            n_nominal = np.array(axis_prior, dtype=float)
            n_nominal /= np.linalg.norm(n_nominal)
        else:
            n_nominal = np.array([0.0, 0.0, 1.0])
            
        # Initial Normal
        if axis_prior is not None:
            best_normal = n_nominal.copy()
        else:
            best_normal = R_fit[:, 2] # Normal is the Z-axis of R_fit
            
        centroid = np.mean(points, axis=0)
        ex = R_fit[:, 0]
        ey = R_fit[:, 1]

        # Sagitta formula
        C_chord_vec = points[-1] - points[0]
        C_chord = np.linalg.norm(C_chord_vec)
        p_mid_chord = (points[0] + points[-1]) / 2.0
        p_mid_arc = points[len(points) // 2]
        v_sag = p_mid_arc - p_mid_chord
        H_sag = np.linalg.norm(v_sag)
        
        if H_sag > 0.05 and C_chord > 1.0:
            R_geom = (C_chord ** 2) / (8.0 * H_sag) + H_sag / 2.0
        else:
            R_geom = 280.0 if (axis_prior is not None and abs(axis_prior[2]) > 0.8) else 75.0
            
        R_init = np.clip(R_geom, 50.0, 800.0)
        
        if 50.0 <= R_geom <= 800.0 and H_sag > 0.05:
            u_sag = v_sag / H_sag
            c_init = p_mid_arc - R_init * u_sag
        else:
            R_init = np.clip(radius_fit, 50.0, 800.0)
            c_init = c_fit
        
        best_opt = None
        best_rmse = float('inf')
        best_sign = 1
        
        for sign in [1, -1]:
            angles_rad = angles_rad_base * sign
            r_dir_init = points[0] - c_init
            r_dir_init -= np.dot(r_dir_init, best_normal) * best_normal
            if np.linalg.norm(r_dir_init) > 1e-6:
                r_dir_init /= np.linalg.norm(r_dir_init)
            
            init_params = np.hstack([c_init, best_normal, r_dir_init, [R_init]])
            lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
            # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
            init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
            
            def total_residuals(params):
                c = params[0:3]
                axis = params[3:6]
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    
                r_init = params[6:9].copy()
                r_init -= np.dot(r_init, axis) * axis
                r_init_norm = np.linalg.norm(r_init)
                if r_init_norm > 1e-6:
                    r_init = r_init / r_init_norm
                R = params[9]
                
                cos_t = np.cos(angles_rad)[:, None]
                sin_t = np.sin(angles_rad)[:, None]
                cross_term = np.cross(axis, r_init)
                dot_term = np.dot(axis, r_init)
                
                pred_pts = c + R * (r_init[None, :] * cos_t + 
                                   cross_term[None, :] * sin_t + 
                                   (axis * dot_term)[None, :] * (1.0 - cos_t))
                return (points - pred_pts).ravel()
                
            try:
                opt_res = least_squares(total_residuals, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            except ValueError as e:
                raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 1 failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
            rmse = np.sqrt(np.mean(opt_res.fun**2))
            if rmse < best_rmse:
                # [FIX] If axis_prior is given, accept the fitted axis only if it lies in the same half-space as the prior direction.
                # -> Prevents cases where noise overfitting results in a mathematically lower RMSE with the wrong sign (-1).
                if axis_prior is not None:
                    axis_candidate = opt_res.x[3:6]
                    axis_candidate_norm = axis_candidate / (np.linalg.norm(axis_candidate) + 1e-9)
                    if np.dot(axis_candidate_norm, n_nominal) > 0:
                        best_rmse = rmse
                        best_opt = opt_res
                        best_sign = sign
                else:
                    best_rmse = rmse
                    best_opt = opt_res
                    best_sign = sign

        # Fallback: if axis_prior direction check rejected all candidates (edge case),
        # fall back to the lowest-RMSE result to avoid best_opt being None
        if best_opt is None:
            for sign in [1, -1]:
                angles_rad = angles_rad_base * sign
                r_dir_init = points[0] - c_init
                r_dir_init -= np.dot(r_dir_init, best_normal) * best_normal
                if np.linalg.norm(r_dir_init) > 1e-6:
                    r_dir_init /= np.linalg.norm(r_dir_init)
                init_params = np.hstack([c_init, best_normal, r_dir_init, [R_init]])
                lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
                upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
                init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
                try:
                    opt_res = least_squares(total_residuals, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
                    rmse = np.sqrt(np.mean(opt_res.fun**2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_opt = opt_res
                        best_sign = sign
                except Exception:
                    pass

        if best_opt is None:
            raise ValueError('No valid encoder-constrained circle fit')

        # Extract optimal
        c_init = best_opt.x[0:3]
        best_normal = best_opt.x[3:6]
        best_normal /= np.linalg.norm(best_normal)
        r_final_dir = best_opt.x[6:9]
        r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
        if np.linalg.norm(r_final_dir) > 1e-6:
            r_final_dir /= np.linalg.norm(r_final_dir)
        R_init = best_opt.x[9]
        
        # Worst-outlier rejection
        inlier_mask = np.ones(len(points), dtype=bool)
        for out_iter in range(3):
            angles_rad = angles_rad_base * best_sign
            pts_in = points[inlier_mask]
            rad_in = angles_rad[inlier_mask]
            
            if len(pts_in) < 6:
                break
                
            init_params = np.hstack([c_init, best_normal, r_final_dir, [R_init]])
            lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
            # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
            init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
            
            def total_residuals_in(params):
                c = params[0:3]
                axis = params[3:6]
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                r_init = params[6:9].copy()
                r_init -= np.dot(r_init, axis) * axis
                r_init_norm = np.linalg.norm(r_init)
                if r_init_norm > 1e-6:
                    r_init = r_init / r_init_norm
                R = params[9]
                
                cos_t = np.cos(rad_in)[:, None]
                sin_t = np.sin(rad_in)[:, None]
                cross_term = np.cross(axis, r_init)
                dot_term = np.dot(axis, r_init)
                
                pred_pts = c + R * (r_init[None, :] * cos_t + 
                                   cross_term[None, :] * sin_t + 
                                   (axis * dot_term)[None, :] * (1.0 - cos_t))
                return (pts_in - pred_pts).ravel()
                
            try:
                opt_res = least_squares(total_residuals_in, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            except ValueError as e:
                raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 2 (outlier loop) failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
            c_init = opt_res.x[0:3]
            best_normal = opt_res.x[3:6]
            best_normal /= np.linalg.norm(best_normal)
            r_final_dir = opt_res.x[6:9]
            r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
            if np.linalg.norm(r_final_dir) > 1e-6:
                r_final_dir /= np.linalg.norm(r_final_dir)
            R_init = opt_res.x[9]
            
            cos_t = np.cos(angles_rad)[:, None]
            sin_t = np.sin(angles_rad)[:, None]
            cross_term = np.cross(best_normal, r_final_dir)
            dot_term = np.dot(best_normal, r_final_dir)
            
            pred_pts = c_init + R_init * (r_final_dir[None, :] * cos_t + 
                                         cross_term[None, :] * sin_t + 
                                         (best_normal * dot_term)[None, :] * (1.0 - cos_t))
            all_errors = np.linalg.norm(points - pred_pts, axis=1)
            
            inlier_indices = np.where(inlier_mask)[0]
            inlier_errors = all_errors[inlier_mask]
            worst_inlier_idx_in_inliers = np.argmax(inlier_errors)
            worst_global_idx = inlier_indices[worst_inlier_idx_in_inliers]
            worst_error = inlier_errors[worst_inlier_idx_in_inliers]
            
            if worst_error > 0.5:
                inlier_mask[worst_global_idx] = False
            else:
                break
                
        # Final Optimization
        pts_in = points[inlier_mask]
        rad_in = angles_rad_base[inlier_mask] * best_sign
        init_params = np.hstack([c_init, best_normal, r_final_dir, [R_init]])
        
        def total_residuals_final(params):
            c = params[0:3]
            axis = params[3:6]
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
            r_init = params[6:9].copy()
            r_init -= np.dot(r_init, axis) * axis
            r_init_norm = np.linalg.norm(r_init)
            if r_init_norm > 1e-6:
                r_init = r_init / r_init_norm
            R = params[9]
            
            cos_t = np.cos(rad_in)[:, None]
            sin_t = np.sin(rad_in)[:, None]
            cross_term = np.cross(axis, r_init)
            dot_term = np.dot(axis, r_init)
            
            pred_pts = c + R * (r_init[None, :] * cos_t + 
                               cross_term[None, :] * sin_t + 
                               (axis * dot_term)[None, :] * (1.0 - cos_t))
            return (pts_in - pred_pts).ravel()
            
        lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
        upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
        # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
        init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
        
        try:
            opt_res = least_squares(total_residuals_final, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
        except ValueError as e:
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 3 (final) failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
        c_opt = opt_res.x[0:3]
        axis_opt = opt_res.x[3:6]
        axis_opt /= np.linalg.norm(axis_opt)
        
        r_init_opt = opt_res.x[6:9]
        r_init_opt -= np.dot(r_init_opt, axis_opt) * axis_opt
        r_init_opt /= np.linalg.norm(r_init_opt)
        radius_opt = opt_res.x[9]
        
        rmse = np.sqrt(np.mean(opt_res.fun**2))
        
        # Coordinate frames for plotting
        if axis_opt[0] < 0.9:
            ex = np.cross(axis_opt, [1, 0, 0])
        else:
            ex = np.cross(axis_opt, [0, 1, 0])
        ex /= np.linalg.norm(ex)
        ey = np.cross(axis_opt, ex)
        
        pts_centered = points - c_opt
        pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)
        uc_opt = 0.0
        vc_opt = 0.0
        
        # Calculate 6-DOF misalignment
        tilt_angle = np.rad2deg(np.arccos(np.clip(np.dot(axis_opt, n_nominal), -1.0, 1.0)))
        
        # Projection for yaw
        n_proj = axis_opt - np.dot(axis_opt, n_nominal) * n_nominal
        if np.linalg.norm(n_proj) > 1e-6:
            n_proj /= np.linalg.norm(n_proj)
            if n_nominal[2] > 0.8:
                yaw_angle = np.rad2deg(np.arctan2(n_proj[1], n_proj[0]))
            else:
                yaw_angle = 0.0
        else:
            yaw_angle = 0.0
            
        # Calculate individual tilt angles for each pose to compute jitter/stddev
        tilt_list = []
        for T in relative_poses:
            if axis_prior is not None:
                if abs(axis_prior[0]) > 0.8:
                    axis_i = T[:3, 0]
                elif abs(axis_prior[1]) > 0.8:
                    axis_i = T[:3, 1]
                else:
                    axis_i = T[:3, 2]
            else:
                axis_i = T[:3, 2]
            axis_norm = np.linalg.norm(axis_i)
            if axis_norm > 1e-6:
                axis_i /= axis_norm
            tilt_i = np.rad2deg(np.arccos(np.clip(np.dot(axis_i, n_nominal), -1.0, 1.0)))
            tilt_list.append(tilt_i)
            
        res_dict = {
            'c_opt': c_opt,
            'axis_opt': axis_opt,
            'radius': radius_opt,
            'rmse': rmse,
            'tilt': tilt_angle,
            'yaw': yaw_angle,
            'pts_2d': pts_2d,
            'uc_opt': uc_opt,
            'vc_opt': vc_opt,
            'inlier_mask': inlier_mask,
            'tilt_list': tilt_list
        }
        return res_dict


def estimate_legacy_j6(poses_a, encoder_a, poses_b, encoder_b, *, arm_indices,
                      initial_arm, axis_a, axis_b, reference_rotation, staged_offset_deg):
    """Historical midpoint estimator, adapted from absolute to staged-relative output."""
    quality = {'method': 'legacy_v12_20260902'}
    observation_stage = False
    try:
        def unit(v):
            v = np.asarray(v, dtype=float)
            if not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-9:
                raise ValueError('Degenerate J6 reference axis')
            return v / np.linalg.norm(v)
        indices = np.asarray(arm_indices, dtype=int)
        initial = np.asarray(initial_arm, dtype=float)
        reference = np.asarray(reference_rotation, dtype=float)
        if indices.shape != (7,) or initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError('Invalid arm encoder configuration')
        if reference.shape != (3,3) or not np.all(np.isfinite(reference)) or not np.allclose(reference.T @ reference, np.eye(3), atol=1e-5) or np.linalg.det(reference) < 0:
            raise ValueError('Invalid bracket reference rotation')
        if not np.isfinite(staged_offset_deg):
            raise ValueError('Invalid staged offset')
        captures = []
        for poses, encoders in ((poses_a, encoder_a), (poses_b, encoder_b)):
            poses = np.array(poses, dtype=float, copy=True)
            encoders = np.asarray(encoders, dtype=float)
            if poses.ndim != 3 or poses.shape[1:] != (4,4) or len(poses) < 10 or encoders.ndim != 2 or len(encoders) != len(poses):
                raise ValueError('J6 requires matching marker poses and encoder samples')
            if min(indices) < 0 or max(indices) >= encoders.shape[1] or not np.all(np.isfinite(encoders)) or not np.all(np.isfinite(poses)):
                raise ValueError('Invalid J6 encoder or marker samples')
            selection = np.round(np.linspace(0, len(poses)-1, min(200,len(poses)))).astype(int)
            captures.append((poses[selection], encoders[selection]))
        (pa, qa), (pb, qb) = captures
        aa, ab = unit(axis_a), unit(axis_b)
        observation_stage = True
        for (p,q), j in zip(captures, (6,5)):
            if np.degrees(np.ptp(q[:,indices[j]])) < 5.:
                raise ValueError('Poor circle observation: encoder sweep arc below 5 deg')
            if np.linalg.matrix_rank(p[:,:3,3]-p[:,:3,3].mean(axis=0), tol=1e-9) < 2:
                raise ValueError('Degenerate circle observation')
        fits = [LegacyCircleFit.fit_circle_3d_and_6dof_misalignment(
                    p, np.degrees(q[:,indices[j]]-initial[j]), axis_prior=axis, robust=True)
                for (p,q), j, axis in zip(captures, (6,5), (aa,ab))]
        for (p,q), fit in zip(captures, fits):
            normal = unit(fit['axis_opt'])
            points = p[:,:3,3]*1000. - fit['c_opt']
            radial = points - (points @ normal)[:,None]*normal
            x = unit(radial[0])
            y = np.cross(normal,x)
            arc = np.degrees(np.ptp(np.unwrap(np.arctan2(radial @ y, radial @ x))))
            residual = np.hypot(points @ normal, np.linalg.norm(radial, axis=1)-fit['radius'])
            fit['geometric_rms_mm'] = float(np.sqrt(np.mean(residual**2)))
            if arc < 5. or fit['geometric_rms_mm'] > .5:
                raise ValueError(f"Poor circle observation: arc={arc:.2f} deg, RMS={fit['geometric_rms_mm']:.3f} mm")
        observation_stage = False
        a,b = fits
        na,nb = unit(a['axis_opt']),unit(b['axis_opt'])
        na *= 1 if na @ aa >= 0 else -1
        nb *= 1 if nb @ ab >= 0 else -1
        n6 = unit(pa[len(pa)//2,:3,:3].T @ na)
        n5 = unit(pb[len(pb)//2,:3,:3].T @ nb)
        ref_y = reference.T @ np.array([0.,1.,0.])
        ref_z = reference.T @ np.array([0.,0.,1.])
        n6 *= 1 if n6 @ ref_z >= 0 else -1
        n5 *= 1 if n5 @ ref_y >= 0 else -1
        projected = unit(n5 - (n5 @ n6)*n6)
        ref_x = unit(np.cross(n6, ref_y))
        raw = float(np.degrees(np.arctan2(projected @ ref_x, projected @ ref_y)))
        absolute = .8*raw + float(np.degrees(qb[0,indices[6]]))
        delta = absolute - staged_offset_deg
        if not np.all(np.isfinite([raw, absolute, a['rmse'], b['rmse']])):
            raise ValueError('Non-finite J6 fit')
        centers = b['c_opt']-a['c_opt']
        distance = float(np.linalg.norm(centers - (centers @ na)*na))
        angle = float(np.degrees(np.arccos(np.clip(na @ nb,-1,1))))
        quality.update(circle_A=a['geometric_rms_mm']/1000., circle_B=b['geometric_rms_mm']/1000., relative_correction_deg=delta)
        return dict(measurement_accepted=True, converged=False, mode='wrist_yaw2',
            optimal_offset=delta, raw_diff_deg=raw, legacy_absolute_offset_deg=absolute,
            quality_diagnostics=quality, angle_between_normals=angle,
            center_dist=float(np.linalg.norm(centers)), perp_dist_after=distance,
            r_A=a['radius'], r_B=b['radius'],
            _plot_data=dict(pts_a_cam=pa[:,:3,3]*1000., pts_b_cam=pb[:,:3,3]*1000.,
                c_A=a['c_opt'],c_B=b['c_opt'],n_A=na,n_B=nb,r_A=a['radius'],r_B=b['radius'],
                angle_between_normals=angle,center_dist=distance))
    except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as error:
        return dict(measurement_accepted=False, failure_reason=str(error),
                    retryable_observation=observation_stage, quality_diagnostics=quality)
