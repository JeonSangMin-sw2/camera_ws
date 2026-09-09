import time
import logging
import os
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator, SweepObservationError

class MarkerCalibrator(BaseCalibrator):

    @staticmethod
    def rodrigues_rotation(vector, axis, theta_rad):
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return vector * cos_t + np.cross(axis, vector) * sin_t + axis * np.dot(axis, vector) * (1 - cos_t)

    def perform_move_to_center(self, arm_side, log_callback=None, stop_event=None, target_dist=300.0, max_attempts=3):
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system not initialized.")
            return False
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to camera center (target: {target_dist}mm, max_attempts: {max_attempts})...")
        
        # Get rotation only from mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        R_cam_to_rob = R_scipy.from_euler('ZYX', [mount_to_cam[5], mount_to_cam[4], mount_to_cam[3]], degrees=True).as_matrix()
        p_target_cam = np.array([0.0, 0.0, target_dist / 1000.0])

        for attempt in range(max_attempts):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Move canceled by user.")
                self.robot.cancel_control()
                return False
                
            if log_callback: log_callback(f"[Attempt {attempt + 1}/{max_attempts}] Capturing marker pose...")
            time.sleep(1.0)
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not res:
                if log_callback: log_callback("  [ERROR] Marker not visible.")
                return False
            
            if isinstance(res, list):
                T_cam_to_marker = np.array(res[0]).reshape(4, 4)
            else:
                T_cam_to_marker = np.array(list(res.values())[0]).reshape(4, 4)
                
            cam_pos = T_cam_to_marker[:3, 3]
            cam_rot = T_cam_to_marker[:3, :3]
            
            pos_err_mm = np.linalg.norm(cam_pos - p_target_cam) * 1000.0
            rot_err_mat = cam_rot.T
            rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
            err_norm = np.linalg.norm([pos_err_mm, rot_err_deg])
 
            if log_callback:
                log_callback(f"  Current: X={cam_pos[0]*1000:.1f}, Y={cam_pos[1]*1000:.1f}, Z={cam_pos[2]*1000:.1f} mm")
                log_callback(f"  Error Norm: {err_norm:.2f} (Pos:{pos_err_mm:.1f}mm, Ang:{rot_err_deg:.1f}deg)")
 
            if err_norm <= 0.5:
                if log_callback: log_callback(f"  [SUCCESS] Reached center aligned pose! (Norm: {err_norm:.2f})")
                break
 
            if log_callback: log_callback("  Calculating joint command and moving...")
            
            dp_cam = p_target_cam - cam_pos
            dR_cam = cam_rot.T  # relative rotation error to identity
            
            # Rotate errors to robot frame (using only rotation R_cam_to_rob)
            dp_rob = R_cam_to_rob @ dp_cam
            dR_rob = R_cam_to_rob @ dR_cam @ R_cam_to_rob.T
            
            ee_name = f"ee_{arm_side}"
            T_rob_to_ee = self.compute_fk(self.robot, self.robot.get_dynamics(), self.robot.get_state().position, ee_name, "link_torso_5")
            p_ee = T_rob_to_ee[:3, 3]
            R_ee = T_rob_to_ee[:3, :3]
            
            T_rob_to_ee_new = np.eye(4)
            T_rob_to_ee_new[:3, :3] = dR_rob @ R_ee
            T_rob_to_ee_new[:3, 3] = p_ee + dp_rob
            
            cb = rby.CartesianCommandBuilder().set_minimum_time(3.0)
            cb.add_target("link_torso_5", ee_name, T_rob_to_ee_new.astype(np.float32), 0.2, 0.5, 1.0)
            cb.set_stop_orientation_tracking_error(1e-4)
            cb.set_stop_position_tracking_error(1e-3)
            
            body_cmd = rby.BodyComponentBasedCommandBuilder()
            if arm_side == "right":
                body_cmd.set_right_arm_command(cb)
            else:
                body_cmd.set_left_arm_command(cb)
                
            rc = rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
            )
            rv = self.robot.send_command(rc, 10).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                if log_callback: log_callback(f"  [ERROR] Failed to move: {rv.finish_code}")
                return False
            time.sleep(0.5)
        return True

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None,
            status_callback=None, use_head_tracking=True, save_debug=False,
            initial_joint_pos=None, pass_idx=1, sweep_duration=10.0, defer_recovery=False):
        if self.stop_requested or self.robot is None or self.marker_st is None:
            return None
        if not self.marker_st.get_marker_transform(sampling_time=2., side=arm_side):
            if status_callback: status_callback(False)
            if defer_recovery:
                raise SweepObservationError('Marker not visible in bracket ready pose')
            return None
        if status_callback: status_callback(True)
        axis = int(str(axis_mode).split('_')[-1])
        cfg = self.MARKER_CONFIGS[f'axis_{axis}']
        indices = getattr(self.robot.model(), arm_side + '_arm_idx')
        center = np.array(initial_joint_pos if initial_joint_pos is not None
                          else self.robot.get_state().position[indices], copy=True)
        # J6 is held at a DECLARED reference command, not an encoder-derived
        # geometric reference. Its remaining coaxial error is a bracket gauge.
        nominal = self.get_ready_pose('v'+self.get_robot_version(), 'marker', 'marker', arm_side)
        key6 = 'wrist_roll' if self.is_v13() else 'wrist_yaw2'
        center[6] = nominal[6] + np.deg2rad(self.joint_offsets.get(arm_side, {}).get(key6, 0.))
        poses = self.perform_single_joint_sweep(arm_side, axis, center,
                    cfg['start_deg'], cfg['end_deg'], sweep_duration,
                    label=f'Marker Axis {axis}', log_callback=log_callback, mode='marker',
                    defer_recovery=defer_recovery)
        if poses is None: return None
        result = self.fit_observed_circle(poses, int(np.sign(cfg['end_deg']-cfg['start_deg'])))
        result.update(captured_poses=poses, commanded_reference_j6_deg=float(np.rad2deg(nominal[6])),
                      sweep_direction=int(np.sign(cfg['end_deg']-cfg['start_deg'])))
        if save_debug: self.save_observed_points(arm_side, axis, poses, 'marker')
        return result

    @staticmethod
    def observed_axis_line(data):
        """Axis line in marker coordinates; independent of camera and arm pose."""
        poses = np.asarray(data['captured_poses'], dtype=float)
        axis, center = data['axis'], data['center_m']
        rotations = poses[:, :3, :3]
        directions = np.einsum('nji,j->ni', rotations, axis)
        direction = directions.mean(axis=0)
        direction /= np.linalg.norm(direction)
        deviation = np.rad2deg(np.arccos(np.clip(directions @ direction, -1., 1.)))
        if np.percentile(deviation, 90) > .5:
            raise ValueError('Inconsistent marker-frame axis directions')
        origins = np.einsum('nji,nj->ni', rotations, center - poses[:,:3,3])
        origin = np.mean(origins, axis=0)
        # Projection removes the arbitrary position along the rotation line.
        projection = np.eye(3) - np.outer(direction, direction)
        projected_origins = origins @ projection
        line_spread_mm = np.linalg.norm(projected_origins - origin @ projection, axis=1)*1000.
        width = max(1, len(origins)//3)
        drift_mm = np.linalg.norm(np.median(projected_origins[:width], axis=0)
                                  - np.median(projected_origins[-width:], axis=0))*1000.
        if np.percentile(line_spread_mm, 90) > .5 or drift_mm > .5:
            raise ValueError('Inconsistent marker-frame axis origins; marker orientation or position changed')
        return direction, projection @ origin, projection

    @staticmethod
    def fit_common_wrist_pivot(datasets):
        """Use p_camera + R_camera_marker * pivot_marker = pivot_camera.

        Same intersecting-wrist-axis assumption as the circle-line fit, but
        retain every observed rigid transform rather than extrapolating short
        arcs to noisy circle centers. No encoder or timestamp model is used.
        """
        poses = np.concatenate(datasets)
        matrix = np.concatenate((poses[:, :3, :3],
            np.broadcast_to(-np.eye(3), (len(poses), 3, 3))), axis=2).reshape(-1, 6)
        target = -poses[:, :3, 3].reshape(-1)
        initial, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
        if rank != 6 or singular[-1] / np.sqrt(len(poses)) < .01:
            raise ValueError('Common wrist pivot is not observable')
        fit = least_squares(lambda x: matrix @ x - target, initial,
                            # SciPy robust loss scales the returned Jacobian in
                            # place; the observation matrix must stay unchanged.
                            jac=lambda x: matrix.copy(), loss='soft_l1', f_scale=.0001)
        residual = (matrix @ fit.x - target).reshape(-1, 3)
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        if not fit.success or not np.isfinite(rms) or rms > .0005:
            raise ValueError(f'Common wrist pivot changed across sweeps: RMS={rms*1000:.4f} mm')
        return fit.x[:3], rms

    def lock_bracket_j6_twist(self, rot, pivot, side):
        """Choose the CAD J6 gauge, preserving the complete pivot transform.

        Decompose R_measured R_CAD^-1 = twist(J6) * swing. Only the
        two-DOF swing belongs to the bracket; J6 owns the removed twist.
        This is not an Euler-component clamp. Translation must be rotated
        about the wrist pivot too, especially for an eccentric bracket.
        """
        axis = np.array([1., 0., 0.] if self.is_v13() else [0., 0., 1.])
        parameters = getattr(self, 'robot_parameters', self._default_parameters)
        nominal = self.make_transform(parameters.nominal_brackets[self.get_robot_version()][side])[:3, :3]
        quaternion = R_scipy.from_matrix(rot @ nominal.T).as_quat()
        projection = float(quaternion[:3] @ axis)
        if np.hypot(projection, quaternion[3]) < 1e-8:
            raise ValueError('Bracket is 180 degrees from CAD; J6 twist gauge is undefined')
        angle = 2. * np.arctan2(projection, quaternion[3])
        angle = (angle + np.pi) % (2. * np.pi) - np.pi
        locked = R_scipy.from_rotvec(-angle * axis).as_matrix() @ rot
        wrist = np.array([0., 0., parameters.tool_lengths[self.get_robot_version()]])
        return locked, wrist - locked @ pivot, float(np.rad2deg(angle))

    def fit_observed_bracket(self, data4, data5, data6, side):
        """Recover the common wrist pivot and bracket, without encoder or FK.

        Wrist axes J4/5/6 intersect by fixed link design in both robot versions.
        Their observed lines locate that pivot in marker coordinates. J6/J5
        orient the effective flange frame; J6 coaxial twist is not identifiable
        independently of bracket twist.
        """
        try:
            # All three sweeps use the same upstream/camera posture. Refine
            # their measured planes before converting to marker-frame lines;
            # neither coincident centers nor a common pivot is imposed here.
            ordered = (data4, data6, data5)
            directions = tuple(data.get('sweep_direction', 1) for data in ordered)
            datasets = tuple(data['captured_poses'] for data in ordered)
            circles = tuple(dict(data, **self.fit_observed_circle(poses, direction))
                            for data, poses, direction in zip(ordered, datasets, directions))
            data4, data6, data5 = self.refine_adjacent_circles(datasets, circles, directions)
            lines = [self.observed_axis_line(data) for data in (data4, data5, data6)]
            for i in (0, 2):
                angle = np.rad2deg(np.arccos(np.clip(lines[i][0] @ lines[1][0], -1., 1.)))
                if abs(angle-90.) > .5:
                    raise ValueError('Observed wrist axes violate the fixed perpendicular geometry')
            matrix = np.vstack([line[2] for line in lines])
            target = np.concatenate([line[1] for line in lines])
            pivot, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
            residual_mm = float(np.sqrt(np.mean((matrix @ pivot-target)**2))*1000.)
            if rank != 3 or singular[-1] < .1 or residual_mm > .5:
                raise ValueError(f'Wrist-axis intersection is poor: rank={rank}, RMS={residual_mm:.4f} mm')
            pivot, pivot_rms = self.fit_common_wrist_pivot(datasets)
            axis6 = np.array([1.,0.,0.] if self.is_v13() else [0.,0.,1.])
            command6 = float(data5['commanded_reference_j6_deg'])
            axis5 = R_scipy.from_rotvec(-np.deg2rad(command6)*axis6).apply([0.,1.,0.])
            rotation, _ = R_scipy.align_vectors(np.array([axis6, axis5]), np.array([lines[2][0], lines[1][0]]))
            rot = rotation.as_matrix()
            rot, position, removed_twist = self.lock_bracket_j6_twist(rot, pivot, side)
            rpy = R_scipy.from_matrix(rot).as_euler('xyz', degrees=True)
            values = dict(zip(('x_e','y_e','z_e','roll_e','pitch_e','yaw_e'), [*(position*1000.), *rpy]))
            values.update(measurement_accepted=True, success=True, data_rank=int(rank),
                axis_intersection_rms_mm=residual_mm, method='observed_wrist_axis_intersection',
                common_pivot_rms_mm=pivot_rms*1000.,
                j6_mode='cad_locked_bracket_twist', removed_j6_twist_deg=removed_twist,
                wrist_pivot_marker_m=pivot.tolist())
            return values
        except (ValueError, KeyError, TypeError) as error:
            return dict(measurement_accepted=False, success=False, failure_reason=str(error))

    def generate_marker_plot(self, res_5, res_6, res_4, unified_res, arm_side, is_v13, save_path):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        try:
            for ax, data, number in zip(axes.flat, (res_4, res_5, res_6), (4, 5, 6)):
                xy = data['pts_2d']
                ax.scatter(xy[:,0], xy[:,1], s=4)
                ax.add_patch(plt.Circle((data['uc_opt'], data['vc_opt']), data['radius'], fill=False))
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_title(f"J{number}: circle RMS {data['rmse']:.3f} mm")
            axes[1,1].axis('off')
            axes[1,1].text(0., .9, f"Observed wrist-axis intersection\nRMS: {unified_res['axis_intersection_rms_mm']:.4f} mm\nJ6/bracket twist: CAD locked", va='top')
            fig.tight_layout()
            fig.savefig(save_path)
            return True
        finally:
            plt.close(fig)
