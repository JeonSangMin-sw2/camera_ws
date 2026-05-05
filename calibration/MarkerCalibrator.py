import time
import logging
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares

class MarkerCalibrator:
    def __init__(self, marker_st=None, robot=None):
        self.marker_st = marker_st
        self.robot = robot

    @staticmethod
    def initialize_robot(address, model, power=".*", servo=".*"):
        robot = rby.create_robot(address, model)
        if not robot.connect():
            logging.error(f"Failed to connect robot {address}")
            return None
        if not robot.is_power_on(power):
            if not robot.power_on(power):
                logging.error(f"Failed to turn power ({power}) on")
                return None
        if not robot.is_servo_on(servo):
            if not robot.servo_on(servo):
                logging.error(f"Failed to servo ({servo}) on")
                return None
        if robot.get_control_manager_state().state in [
            rby.ControlManagerState.State.MajorFault,
            rby.ControlManagerState.State.MinorFault,
        ]:
            if not robot.reset_fault_control_manager():
                logging.error(f"Failed to reset control manager")
                return None
        if not robot.enable_control_manager():
            logging.error(f"Failed to enable control manager")
            return None
        return robot

    @staticmethod
    def movej(robot, torso=None, right_arm=None, left_arm=None, minimum_time=0):
        if not robot:
            return False
            
        rc = rby.BodyComponentBasedCommandBuilder()
        if torso is not None:
            rc.set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(torso)
            )
        if right_arm is not None:
            rc.set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(right_arm)
            )
        if left_arm is not None:
            rc.set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(left_arm)
            )

        rv = robot.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(rc)
            ),
            1,
        ).get()

        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            logging.error("Failed to conduct movej.")
            return False

        return True

    @staticmethod
    def fit_circle_kinematic(points, angles_deg, return_plot_data=False):
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        pts_centered = points - centroid
        
        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[2, :]
        if normal[2] < 0:
            normal = -normal

        ex = vh[0, :]
        ey = vh[1, :]
        pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)

        # 대수적 방정식으로 초기 중심/방경 추정
        A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
        b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        uc_init, vc_init, offset_init = res[0], res[1], res[2]
        radius_init = np.sqrt(max(0, offset_init + uc_init**2 + vc_init**2))
        
        best_rmse = float('inf')
        best_opt = None

        # 투영 축(ex, ey)의 부호나 모터 회전 방향의 역전을 방지하기 위해 + / - 회전 모두 테스트
        for sign in [1, -1]:
            angles_rad = np.radians(angles_deg) * sign
            
            def residuals(params):
                uc, vc, R, alpha = params
                model_x = uc + R * np.cos(alpha + angles_rad)
                model_y = vc + R * np.sin(alpha + angles_rad)
                return np.sqrt((pts_2d[:, 0] - model_x)**2 + (pts_2d[:, 1] - model_y)**2)
            
            # 시작 각도 위상(alpha) 초기값 추정
            alpha_init = np.arctan2(pts_2d[0, 1] - vc_init, pts_2d[0, 0] - uc_init) - angles_rad[0]
            initial_guess = [uc_init, vc_init, radius_init, alpha_init]
            
            # 이상치에 강건한 Huber 최적화
            opt_result = least_squares(residuals, initial_guess, loss='huber')
            rmse = np.sqrt(np.mean(opt_result.fun**2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_opt = opt_result

        uc_opt, vc_opt, R_opt, _ = best_opt.x
        center_3d = centroid + uc_opt * ex + vc_opt * ey
        
        if return_plot_data:
            return center_3d, normal, R_opt, best_rmse, pts_2d, uc_opt, vc_opt
        return center_3d, normal, R_opt, best_rmse

    def perform_move_to_center(self, arm_side, log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback:
            log_callback("\n" + "="*40)
            log_callback("   STARTING MOVE TO CENTER [0, 0, 180]")
            log_callback("="*40)

        for attempt in range(3):
            if log_callback: log_callback(f"[Attempt {attempt + 1}/3] Capturing marker...")
            time.sleep(1.0)
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not res:
                if log_callback: log_callback("  [ERROR] Marker not visible.")
                return False
            
            if isinstance(res, list):
                pose = np.array(res[0]).reshape(4, 4)
            else:
                pose = np.array(list(res.values())[0]).reshape(4, 4)
                
            cam_x, cam_y, cam_z = pose[:3, 3] * 1000.0
            err_x = 0.0 - cam_x
            err_y = 0.0 - cam_y
            err_z = 180.0 - cam_z
            
            dist = np.sqrt(err_x**2 + err_y**2 + err_z**2)
            if log_callback:
                log_callback(f"  Current: X={cam_x:.1f}, Y={cam_y:.1f}, Z={cam_z:.1f} mm")
                log_callback(f"  Error: dX={err_x:.1f}, dY={err_y:.1f}, dZ={err_z:.1f} (Dist: {dist:.2f} mm)")

            if abs(err_x) <= 0.5 and abs(err_y) <= 0.5 and abs(err_z) <= 0.5:
                if log_callback: log_callback("  [SUCCESS] Reached target center (all axes error <= 0.5mm)!")
                break
                
            if log_callback: log_callback("  Moving robot to correct error...")
            # Camera (X, Y, Z) -> Robot (-Y, -Z, X)
            dx_rob = err_z / 1000.0
            dy_rob = -err_x / 1000.0
            dz_rob = -err_y / 1000.0
            
            model = self.robot.model()
            dyn_robot = self.robot.get_dynamics()
            ee_name = f"ee_{arm_side}"
            dyn_state = dyn_robot.make_state(["base", ee_name], model.robot_joint_names)
            dyn_state.set_q(self.robot.get_state().position)
            dyn_robot.compute_forward_kinematics(dyn_state)
            T_ref = dyn_robot.compute_transformation(dyn_state, 0, 1)
            
            T_target = T_ref.copy()
            T_target[0, 3] += dx_rob
            T_target[1, 3] += dy_rob
            T_target[2, 3] += dz_rob
            
            cb = rby.CartesianCommandBuilder().set_minimum_time(3.0)
            cb.add_target("base", ee_name, T_target, 1.5, np.pi*1.5, 1.0)
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
                if log_callback: log_callback("  [ERROR] Failed to move Cartesian.")
                return False
                
            time.sleep(0.5)

        if log_callback: log_callback("Move to Center finished.\n")
        return True

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None, status_callback=None):
        if log_callback:
            log_callback("\n" + "="*40)
            log_callback(f"   STARTING {axis_mode}-AXIS CALIBRATION SWEEP")
            log_callback("="*40)

        # Pre-check Marker Presence
        if log_callback: log_callback("  - Checking if marker is visible before starting...")
        initial_check = self.marker_st.get_marker_transform(sampling_time=1.0, side=arm_side)
        if not initial_check:
            if log_callback: log_callback("\n[ERROR] 마커가 위치해 있지 않습니다. 시작할 수 없습니다.")
            if status_callback: status_callback(False)
            return None
        
        if status_callback: status_callback(True)
        captured_poses = []
        captured_angles = []

        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected. Cannot perform automated sweep.")
            return None

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])
        
        # Configure Sweep based on Axis Mode
        if axis_mode == 6:
            max_points = 11
            start_deg = -20
            step_deg = 4
            joint_i = 6
        else:
            max_points = 11
            start_deg = -10
            step_deg = 2
            joint_i = 5

        if log_callback: log_callback(f"[INFO] Initial Joint Pose: {np.round(initial_joint_pos, 2)}")
        
        for i in range(max_points):
            if log_callback: log_callback(f"\n[STEP {i + 1}/{max_points}]")
            
            target_offset_deg = start_deg + (i * step_deg)
            target_joint_pos = list(initial_joint_pos)
            target_joint_pos[joint_i] = initial_joint_pos[joint_i] + np.radians(target_offset_deg)
            
            if target_offset_deg == 0:
                if log_callback: log_callback(f"  - Moving to Center Pose (0 deg)...")
            else:
                if log_callback: log_callback(f"  - Moving axis {axis_mode} to {target_offset_deg:.1f} deg offset...")
            
            if arm_side == "left":
                move_status = self.movej(self.robot, left_arm=target_joint_pos, minimum_time=1.5)
            else:
                move_status = self.movej(self.robot, right_arm=target_joint_pos, minimum_time=1.5)

            if move_status:
                time.sleep(1.0) # Settling time
            else:
                if log_callback: log_callback(f"  [ERROR] Arm movement failed.")
                break

            if log_callback: log_callback(f"  - Capturing {arm_side} marker with LPF (2.0s)...")
            lpf_results = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            
            captured_pose = None
            if lpf_results and len(lpf_results) > 0:
                if status_callback: status_callback(True)
                if isinstance(lpf_results, list):
                    captured_pose = np.array(lpf_results[0]).reshape(4, 4)
                elif isinstance(lpf_results, dict):
                    first_key = list(lpf_results.keys())[0]
                    captured_pose = np.array(lpf_results[first_key]).reshape(4, 4)
            else:
                if status_callback: status_callback(False)

            if captured_pose is not None:
                captured_pose[:3, 3] *= 1000.0 # m to mm
                captured_poses.append(captured_pose)
                captured_angles.append(target_offset_deg)
                if log_callback: log_callback(f"  - Pose Saved: Pos={np.round(captured_pose[:3, 3], 2)}")
            else:
                if log_callback: log_callback("  [ERROR] Marker lost during sweep. Aborting.")
                break 
        
        # Return to Initial Pose
        if log_callback: log_callback("\n[INFO] Sweep complete. Returning to initial pose...")
        if arm_side == "left":
            self.movej(self.robot, left_arm=initial_joint_pos, minimum_time=2.0)
        else:
            self.movej(self.robot, right_arm=initial_joint_pos, minimum_time=2.0)

        # Calculation logic
        if len(captured_poses) >= max_points:
            T_cam_ref = captured_poses[0]
            T_ref_cam = np.linalg.inv(T_cam_ref)
            relative_poses = [T_ref_cam @ T for T in captured_poses]
            points = [T[:3, 3] for T in relative_poses]
            
            center, axis, radius, rmse, pts_2d, uc_opt, vc_opt = self.fit_circle_kinematic(points, captured_angles, return_plot_data=True)
            
            result_dict = {
                'axis_mode': axis_mode,
                'radius': radius,
                'rmse': rmse,
                'axis': axis,
                'center': center,
                'pts_2d': pts_2d,
                'uc_opt': uc_opt,
                'vc_opt': vc_opt
            }

            tilt_list = []
            yaw_list = []
            
            for T_i in relative_poses:
                R_i = T_i[:3, :3]
                axis_m_i = R_i.T @ axis
                tilt_i = np.degrees(np.arcsin(min(1.0, max(-1.0, axis_m_i[2]))))
                tilt_list.append(tilt_i)
                
                vec_c_to_mi = T_i[:3, 3] - center
                radial_vec = vec_c_to_mi - np.dot(vec_c_to_mi, axis) * axis
                ideal_tangent = np.cross(axis, radial_vec)
                norm_ideal = np.linalg.norm(ideal_tangent)
                if norm_ideal > 1e-6:
                    ideal_tangent /= norm_ideal
                
                marker_x = R_i[:, 0]
                marker_x_plane = marker_x - np.dot(marker_x, axis) * axis
                norm_mx = np.linalg.norm(marker_x_plane)
                if norm_mx > 1e-6:
                    marker_x_plane /= norm_mx
                
                twist_cos = np.dot(marker_x_plane, ideal_tangent)
                twist_angle = np.degrees(np.arccos(min(1.0, max(-1.0, twist_cos))))
                if np.dot(np.cross(ideal_tangent, marker_x_plane), axis) < 0:
                    twist_angle = -twist_angle
                
                if twist_angle > 90: twist_angle -= 180
                if twist_angle < -90: twist_angle += 180
                yaw_list.append(twist_angle)

            robust_tilt = np.median(tilt_list)
            robust_yaw = np.median(yaw_list)
            
            result_dict.update({
                'tilt': robust_tilt,
                'yaw': robust_yaw,
                'tilt_list': tilt_list
            })
            return result_dict
        
        return None
