import sys
import os
import cv2
import numpy as np
import time
import argparse
import logging
import rby1_sdk as rby
from scipy.optimize import least_squares

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QCheckBox, QLineEdit, QDialog, QMessageBox, QTabWidget)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap
import matplotlib.pyplot as plt

# --- Configuration ---
# 6-Axis: ±20, 4 steps | 5-Axis: ±10, 2 steps
# ---------------------

def initialize_robot(address, model, power=".*", servo=".*"):
    robot = rby.create_robot(address, model)
    if not robot.connect():
        logging.error(f"Failed to connect robot {address}")
        sys.exit(1)
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            logging.error(f"Failed to turn power ({power}) on")
            sys.exit(1)
    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            logging.error(f"Failed to servo ({servo}) on")
            sys.exit(1)
    if robot.get_control_manager_state().state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        if not robot.reset_fault_control_manager():
            logging.error(f"Failed to reset control manager")
            sys.exit(1)
    if not robot.enable_control_manager():
        logging.error(f"Failed to enable control manager")
        sys.exit(1)
    return robot

def movej(robot, torso=None, right_arm=None, left_arm=None, minimum_time=0):
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

# Add parent directory to access marker_detection
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from marker_detection import Marker_Transform
except ImportError:
    print("Cannot find marker_detection.py in parent directory.")
    sys.exit(1)

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

# --- Custom UI Widgets ---
class IndicatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.is_detected = False
    
    def set_detected(self, detected):
        self.is_detected = detected
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(0, 255, 0) if self.is_detected else QColor(255, 0, 0)
        painter.setBrush(color)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(2, 2, 26, 26)

class MoveCenterWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, marker_st, robot, arm_side):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side

    def run(self):
        if not self.robot:
            self.log_signal.emit("[ERROR] Robot not connected.")
            self.finished_signal.emit()
            return

        self.log_signal.emit("\n" + "="*40)
        self.log_signal.emit("   STARTING MOVE TO CENTER [0, 0, 180]")
        self.log_signal.emit("="*40)

        for attempt in range(3):
            self.log_signal.emit(f"[Attempt {attempt + 1}/3] Capturing marker...")
            time.sleep(1.0)
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=self.arm_side)
            if not res:
                self.log_signal.emit("  [ERROR] Marker not visible.")
                break
            
            if isinstance(res, list):
                pose = np.array(res[0]).reshape(4, 4)
            else:
                pose = np.array(list(res.values())[0]).reshape(4, 4)
                
            cam_x, cam_y, cam_z = pose[:3, 3] * 1000.0
            err_x = 0.0 - cam_x
            err_y = 0.0 - cam_y
            err_z = 180.0 - cam_z
            
            dist = np.sqrt(err_x**2 + err_y**2 + err_z**2)
            self.log_signal.emit(f"  Current: X={cam_x:.1f}, Y={cam_y:.1f}, Z={cam_z:.1f} mm")
            self.log_signal.emit(f"  Error: dX={err_x:.1f}, dY={err_y:.1f}, dZ={err_z:.1f} (Dist: {dist:.2f} mm)")

            if abs(err_x) <= 0.5 and abs(err_y) <= 0.5 and abs(err_z) <= 0.5:
                self.log_signal.emit("  [SUCCESS] Reached target center (all axes error <= 0.5mm)!")
                break
                
            self.log_signal.emit("  Moving robot to correct error...")
            # Camera (X, Y, Z) -> Robot (-Y, -Z, X)
            dx_rob = err_z / 1000.0
            dy_rob = -err_x / 1000.0
            dz_rob = -err_y / 1000.0
            
            model = self.robot.model()
            dyn_robot = self.robot.get_dynamics()
            ee_name = f"ee_{self.arm_side}"
            dyn_state = dyn_robot.make_state(["base", ee_name], model.robot_joint_names)
            dyn_state.set_q(self.robot.get_state().position)
            dyn_robot.compute_forward_kinematics(dyn_state)
            T_ref = dyn_robot.compute_transformation(dyn_state, 0, 1)
            
            T_target = T_ref.copy()
            T_target[0, 3] += dx_rob
            T_target[1, 3] += dy_rob
            T_target[2, 3] += dz_rob
            
            cb = rby.CartesianCommandBuilder().set_minimum_time(3.0)
            cb.add_target("base", ee_name, T_target, 1.0, 1.0, 1.0)
            
            rc = rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(cb)
            )
            
            rv = self.robot.send_command(rc, 4.0).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                self.log_signal.emit("  [ERROR] Failed to move Cartesian.")
                break
                
            time.sleep(0.5)

        self.log_signal.emit("Move to Center finished.\n")
        self.finished_signal.emit()

# --- Calibration Worker Thread ---
class CalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)
    
    def __init__(self, marker_st, robot, arm_side, axis_mode):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        self.axis_mode = axis_mode # 6 or 5
        
    def run(self):
        try:
            self._calibrate()
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            self.finished_signal.emit(None)

    def _calibrate(self):
        self.log_signal.emit("\n" + "="*40)
        self.log_signal.emit(f"   STARTING {self.axis_mode}-AXIS CALIBRATION SWEEP")
        self.log_signal.emit("="*40)

        # Pre-check Marker Presence
        self.log_signal.emit("  - Checking if marker is visible before starting...")
        initial_check = self.marker_st.get_marker_transform(sampling_time=1.0, side=self.arm_side)
        if not initial_check:
            self.log_signal.emit("\n[ERROR] 마커가 위치해 있지 않습니다. 시작할 수 없습니다.")
            self.status_signal.emit(False)
            self.finished_signal.emit(None)
            return
        
        self.status_signal.emit(True)
        captured_poses = []
        captured_angles = []

        if not self.robot:
            self.log_signal.emit("[ERROR] Robot not connected. Cannot perform automated sweep.")
            self.finished_signal.emit(None)
            return

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if self.arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])
        
        # Configure Sweep based on Axis Mode
        if self.axis_mode == 6:
            max_points = 11
            start_deg = -20
            step_deg = 4
            joint_i = 6
        else:
            max_points = 11
            start_deg = -10
            step_deg = 2
            joint_i = 5

        self.log_signal.emit(f"[INFO] Initial Joint Pose: {np.round(initial_joint_pos, 2)}")
        
        for i in range(max_points):
            self.log_signal.emit(f"\n[STEP {i + 1}/{max_points}]")
            
            target_offset_deg = start_deg + (i * step_deg)
            target_joint_pos = list(initial_joint_pos)
            target_joint_pos[joint_i] = initial_joint_pos[joint_i] + np.radians(target_offset_deg)
            
            if target_offset_deg == 0:
                self.log_signal.emit(f"  - Moving to Center Pose (0 deg)...")
            else:
                self.log_signal.emit(f"  - Moving axis {self.axis_mode} to {target_offset_deg:.1f} deg offset...")
            
            if self.arm_side == "left":
                move_status = movej(self.robot, left_arm=target_joint_pos, minimum_time=1.5)
            else:
                move_status = movej(self.robot, right_arm=target_joint_pos, minimum_time=1.5)

            if move_status:
                time.sleep(1.0) # Settling time
            else:
                self.log_signal.emit(f"  [ERROR] Arm movement failed.")
                break

            self.log_signal.emit(f"  - Capturing {self.arm_side} marker with LPF (2.0s)...")
            lpf_results = self.marker_st.get_marker_transform(sampling_time=2.0, side=self.arm_side)
            
            captured_pose = None
            if lpf_results and len(lpf_results) > 0:
                self.status_signal.emit(True)
                if isinstance(lpf_results, list):
                    captured_pose = np.array(lpf_results[0]).reshape(4, 4)
                elif isinstance(lpf_results, dict):
                    first_key = list(lpf_results.keys())[0]
                    captured_pose = np.array(lpf_results[first_key]).reshape(4, 4)
            else:
                self.status_signal.emit(False)

            if captured_pose is not None:
                captured_pose[:3, 3] *= 1000.0 # m to mm
                captured_poses.append(captured_pose)
                captured_angles.append(target_offset_deg)
                self.log_signal.emit(f"  - Pose Saved: Pos={np.round(captured_pose[:3, 3], 2)}")
            else:
                self.log_signal.emit("  [ERROR] Marker lost during sweep. Aborting.")
                break 
        
        # Return to Initial Pose
        self.log_signal.emit("\n[INFO] Sweep complete. Returning to initial pose...")
        if self.arm_side == "left":
            movej(self.robot, left_arm=initial_joint_pos, minimum_time=2.0)
        else:
            movej(self.robot, right_arm=initial_joint_pos, minimum_time=2.0)

        # Math Review: Median Averaging
        result_dict = None
        if len(captured_poses) >= max_points:
            self.log_signal.emit("\n" + "="*40)
            self.log_signal.emit(f"   {self.axis_mode}-AXIS BRACKET ANALYSIS (MEDIAN AVERAGED)")
            self.log_signal.emit("="*40)
            
            T_cam_ref = captured_poses[0]
            T_ref_cam = np.linalg.inv(T_cam_ref)
            relative_poses = [T_ref_cam @ T for T in captured_poses]
            points = [T[:3, 3] for T in relative_poses]
            
            center, axis, radius, rmse, pts_2d, uc_opt, vc_opt = fit_circle_kinematic(points, captured_angles, return_plot_data=True)
            fitting_score = max(0.0, 100.0 * (1.0 - rmse / 4.0))
            
            tilt_list = []
            yaw_list = []
            
            for T_i in relative_poses:
                R_i = T_i[:3, :3]
                
                # Main Tilt (Roll for 6-axis, Pitch for 5-axis)
                axis_m_i = R_i.T @ axis
                tilt_i = np.degrees(np.arcsin(min(1.0, max(-1.0, axis_m_i[2]))))
                tilt_list.append(tilt_i)
                
                # Twist/Yaw
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

            self.log_signal.emit(f"  [1] Geometric Tracking Stability:")
            self.log_signal.emit(f"      Radius (Center-to-Axis): {radius:.2f} mm")
            self.log_signal.emit(f"      Quality Score (FIT): {fitting_score:.1f}%")
            self.log_signal.emit(f"      Jitter (StdDev): {np.std(tilt_list):.2f} deg")
            self.log_signal.emit("-" * 30)
            
            self.log_signal.emit(f"  [2] Robust Axis Alignment (Median):")
            if self.axis_mode == 6:
                self.log_signal.emit(f"      Roll  (상하 기울기): {robust_tilt:.2f} deg")
                self.log_signal.emit(f"      Yaw  (비틀림): {robust_yaw:.2f} deg")
            else:
                pass # Pitch (좌우 기울기)는 해석이 어려워 숨김 처리: self.log_signal.emit(f"      Pitch (좌우 기울기): {robust_tilt:.2f} deg")
            self.log_signal.emit("="*40)
            self.log_signal.emit("\n[SWEEP SUCCESSFUL]\n")
            
            # Plotting and saving
            plt.figure(figsize=(6, 6))
            plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='b', label='Captured Points')
            circle = plt.Circle((uc_opt, vc_opt), radius, color='r', fill=False, label='Fitted Circle')
            plt.gca().add_patch(circle)
            plt.plot(uc_opt, vc_opt, 'rx', label='Center')
            
            # 확대(Zoom)을 위해 데이터 기준 x, y 축 리밋 설정
            x_min, x_max = pts_2d[:, 0].min(), pts_2d[:, 0].max()
            y_min, y_max = pts_2d[:, 1].min(), pts_2d[:, 1].max()
            margin_x = max(1.0, (x_max - x_min) * 0.5)
            margin_y = max(1.0, (y_max - y_min) * 0.5)
            plt.xlim(x_min - margin_x, x_max + margin_x)
            plt.ylim(y_min - margin_y, y_max + margin_y)
            plt.gca().set_aspect('equal', adjustable='datalim')

            plt.grid(True)
            plt.title(f"Axis {self.axis_mode} Sweep (RMSE: {rmse:.2f} px)")
            plt.legend()
            plot_path = os.path.join(os.path.dirname(__file__), f"circle_fit_axis_{self.axis_mode}.png")
            plt.savefig(plot_path)
            plt.close()
            
            result_dict = {
                'axis_mode': self.axis_mode,
                'radius': radius,
                'tilt': robust_tilt,
                'yaw': robust_yaw,
                'plot_path': plot_path
            }
        
        self.finished_signal.emit(result_dict)


class CalibrationApp(QWidget):
    def __init__(self, marker_st, robot, arm_side):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        
        # Unified tracking Data
        self.data_5 = None
        self.data_6 = None
        
        self.setWindowTitle("Unified 5/6 Axis Bracket Calibration")
        self.resize(900, 600)
        
        self.init_ui()
        
        # Timer for polling marker and temp
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera_status)
        self.poll_timer.start(200) # 5 Hz
        
        self.worker = None

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Main Tab
        main_tab = QWidget()
        main_tab_layout = QHBoxLayout()
        
        # Left Panel
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)
        
        # Connection
        conn_box = QGroupBox("Robot Connection")
        conn_layout = QVBoxLayout()
        self.ip_input = QLineEdit("192.168.30.1:50051")
        self.model_input = QComboBox()
        self.model_input.addItems(["a", "m"])
        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #ff9900; color: black; font-weight: bold;")
        self.btn_connect.clicked.connect(self.connect_robot)
        conn_layout.addWidget(QLabel("IP:"))
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(QLabel("Model:"))
        conn_layout.addWidget(self.model_input)
        conn_layout.addWidget(self.btn_connect)
        conn_box.setLayout(conn_layout)
        left_panel.addWidget(conn_box)
        
        status_box = QGroupBox("Camera & Marker Status")
        status_layout = QVBoxLayout()
        
        # Indicator
        ind_layout = QHBoxLayout()
        self.indicator = IndicatorWidget()
        ind_layout.addWidget(self.indicator)
        self.status_label = QLabel("Not Detected")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        ind_layout.addWidget(self.status_label)
        ind_layout.addStretch()
        status_layout.addLayout(ind_layout)
        
        # Monitoring Toggle
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        self.btn_monitor.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.btn_monitor)
        
        # Temp
        self.temp_label = QLabel("Camera Temp: -- °C")
        self.temp_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.temp_label)
        status_box.setLayout(status_layout)
        
        # Controls
        controls_box = QGroupBox("Calibration Controls")
        controls_layout = QVBoxLayout()
        
        self.arm_side_sel = QComboBox()
        self.arm_side_sel.addItems(["Right Arm", "Left Arm"])
        if getattr(self, 'arm_side', 'right') == "left":
            self.arm_side_sel.setCurrentIndex(1)
        self.arm_side_sel.currentTextChanged.connect(self.on_arm_side_changed)
        controls_layout.addWidget(self.arm_side_sel)

        self.axis_sel = QComboBox()
        self.axis_sel.addItems(["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"])
        controls_layout.addWidget(self.axis_sel)
        
        self.btn_center = QPushButton("MOVE TO CENTER")
        self.btn_center.setMinimumHeight(40)
        self.btn_center.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold;")
        self.btn_center.clicked.connect(self.move_to_center)

        self.btn_start = QPushButton("START SWEEP")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_calibration)
        
        self.btn_result = QPushButton("UNIFIED RESULT")
        self.btn_result.setMinimumHeight(40)
        self.btn_result.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_result.clicked.connect(self.show_unified_result)
        
        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setMinimumHeight(30)
        self.btn_quit.setStyleSheet("background-color: #dc3545; color: white;")
        self.btn_quit.clicked.connect(self.close)
        
        controls_layout.addWidget(self.btn_center)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_result)
        controls_layout.addWidget(self.btn_quit)
        controls_box.setLayout(controls_layout)
        
        left_panel.addWidget(status_box)
        left_panel.addWidget(controls_box)
        left_panel.addStretch()
        
        # Right Panel (Logs)
        right_panel = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        right_panel.addWidget(self.log_text)
        
        main_tab_layout.addLayout(left_panel, 1)
        main_tab_layout.addLayout(right_panel, 3)
        main_tab.setLayout(main_tab_layout)
        self.tabs.addTab(main_tab, "Main Calibration")
        
        # Plot Tab
        self.plot_tab = QWidget()
        plot_layout = QHBoxLayout()
        
        self.plot_label_6 = QLabel("6-Axis Plot will appear here")
        self.plot_label_6.setAlignment(Qt.AlignCenter)
        self.plot_label_6.setStyleSheet("background-color: #333333; color: white;")
        
        self.plot_label_5 = QLabel("5-Axis Plot will appear here")
        self.plot_label_5.setAlignment(Qt.AlignCenter)
        self.plot_label_5.setStyleSheet("background-color: #333333; color: white;")
        
        plot_layout.addWidget(self.plot_label_6)
        plot_layout.addWidget(self.plot_label_5)
        self.plot_tab.setLayout(plot_layout)
        self.tabs.addTab(self.plot_tab, "Plot Viewer")
        
        self.setLayout(main_layout)
        
        self.log_msg("Unified 5/6 Axis Calibration App Ready.\nCenter the marker physically before starting sweeps.")

    def log_msg(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def connect_robot(self):
        try:
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip()
            self.log_msg(f"[INFO] Connecting to robot at {addr} ({model})...")
            self.robot = initialize_robot(addr, model)
            self.log_msg("[INFO] Robot successfully connected and activated.")
            self.btn_connect.setText("CONNECTED")
            self.btn_connect.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
            self.btn_connect.setEnabled(False)
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to connect: {e}")

    def move_to_center(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        if not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not detected!\nPlease use the teaching button to make the camera recognize the marker.")
            return
            
        self.btn_center.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.worker_mc = MoveCenterWorker(self.marker_st, self.robot, self.arm_side)
        self.worker_mc.log_signal.connect(self.log_msg)
        self.worker_mc.finished_signal.connect(self.on_move_center_finished)
        self.worker_mc.start()

    def on_move_center_finished(self):
        self.btn_center.setEnabled(True)
        self.btn_start.setEnabled(True)

    def on_arm_side_changed(self, text):
        if "Left" in text:
            self.arm_side = "left"
        else:
            self.arm_side = "right"
        self.log_msg(f"[INFO] Target arm changed to {self.arm_side.upper()} Arm. (Prior sweep data cleared)")
        self.data_5 = None
        self.data_6 = None

    def on_monitor_toggled(self, checked):
        if checked:
            self.btn_monitor.setText("Marker Monitor: ON")
            self.btn_monitor.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold;")
        else:
            self.btn_monitor.setText("Marker Monitor: OFF")
            self.btn_monitor.setStyleSheet("font-weight: bold;")

    def update_marker_indicator(self, detected):
        self.indicator.set_detected(detected)
        if detected:
            self.status_label.setText("Detected")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText("Not Detected")
            self.status_label.setStyleSheet("color: red;")

    def poll_camera_status(self):
        try:
            results = self.marker_st.get_marker_transform(sampling_time=0, side=self.arm_side)
            detected = bool(results and len(results) > 0)
            self.update_marker_indicator(detected)
            
            if self.btn_monitor.isChecked() and detected:
                if isinstance(results, list):
                    pose = np.array(results[0]).reshape(4, 4)
                elif isinstance(results, dict):
                    k = list(results.keys())[0]
                    pose = np.array(results[k]).reshape(4, 4)
                # Print real-time coordinate to log
                x, y, z = pose[:3, 3] * 1000.0
                self.log_msg(f"[LIVE] Marker X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm")
            
            # Flush pipeline to keep it fresh
            self.marker_st.camera.get_color_image()
            
            # Poll temp
            temp = self.marker_st.camera.get_camera_temperature()
            if temp:
                self.temp_label.setText(f"Camera Temp: {temp:.1f} °C")
        except Exception as e:
            pass

    def start_calibration(self):
        if not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not detected!\nPlease use the teaching button to make the camera recognize the marker.")
            return

        axis_mode = 6 if "6" in self.axis_sel.currentText() else 5
        self.btn_start.setEnabled(False)
        self.btn_result.setEnabled(False)
        self.poll_timer.stop()
        
        self.log_text.clear()
        
        self.worker = CalibrationWorker(self.marker_st, self.robot, self.arm_side, axis_mode)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.status_signal.connect(self.update_marker_indicator)
        self.worker.finished_signal.connect(self.on_calibration_finished)
        self.worker.start()

    def on_calibration_finished(self, result_dict):
        self.btn_start.setEnabled(True)
        self.btn_result.setEnabled(True)
        self.poll_timer.start(200)
        
        if result_dict:
            if result_dict['axis_mode'] == 6:
                self.data_6 = result_dict
            else:
                self.data_5 = result_dict
            
            # Show Plot in Plot Tab
            if 'plot_path' in result_dict and os.path.exists(result_dict['plot_path']):
                pixmap = QPixmap(result_dict['plot_path'])
                scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                if result_dict['axis_mode'] == 6:
                    self.plot_label_6.setPixmap(scaled_pixmap)
                else:
                    self.plot_label_5.setPixmap(scaled_pixmap)
                
                # Switch to plot tab automatically
                self.tabs.setCurrentIndex(1)

    def get_link_length(self):
        if not self.robot: return 0.0
        try:
            dyn_model = self.robot.get_dynamics()
            names = self.robot.model().robot_joint_names
            # Request kinematics from 5th arm link down to EE
            state = dyn_model.make_state(
                [f"link_{self.arm_side}_arm_5", f"ee_{self.arm_side}"],
                names
            )
            state.set_q(self.robot.get_state().position)
            dyn_model.compute_forward_kinematics(state)
            T = dyn_model.compute_transformation(state, 0, 1)
            return np.linalg.norm(T[:3, 3]) * 1000.0 # Convert to mm
        except Exception as e:
            self.log_msg(f"[WARN] Failed to get kinematics L_5_ee: {e}")
            return 0.0

    def show_unified_result(self):
        self.log_text.clear()
        self.log_msg("\n" + "#"*50)
        self.log_msg("       UNIFIED BRACKET CALIBRATION RESULT")
        self.log_msg("#"*50)
        
        if not self.data_5 or not self.data_6:
            self.log_msg("\n[ERROR] Missing Dataset!")
            if not self.data_6: self.log_msg(" -> Axis 6 Sweep (Yaw) data is missing. Please run it.")
            if not self.data_5: self.log_msg(" -> Axis 5 Sweep (Pitch) data is missing. Please run it.")
            self.log_msg("\nBoth sequences must be completed before unified calculation.")
            return
            
        L_5_ee = self.get_link_length()
        
        z_offset = self.data_6['radius']
        y_offset = self.data_5['radius'] - L_5_ee 
        
        roll = self.data_6['tilt']
        yaw = self.data_6['yaw']
        pitch = self.data_5['tilt']
        
        self.log_msg("\n[1] Cartesian Offset (Translations)")
        self.log_msg(f"    - Z-Offset (6축 회전중심과의 거리): {z_offset:.2f} mm")
        if L_5_ee > 0:
            self.log_msg(f"    - Y-Offset (EE로부터 마커까지의 연장 길이): {y_offset:.2f} mm")
            self.log_msg(f"       * (참고 5축반지름: {self.data_5['radius']:.1f} / Kinematics 링크길이: {L_5_ee:.1f})")
        else:
            self.log_msg("    - Y-Offset: N/A (Failed to get kinematic link length)")
            
        self.log_msg("\n[2] Angular Misalignment (Rotations)")
        self.log_msg(f"    - Roll  (상하 틀어짐 from 6축): {roll:.2f} deg")
        # self.log_msg(f"    - Pitch (좌우 틀어짐 from 5축): {pitch:.2f} deg") # 숨김 처리
        self.log_msg(f"    - Yaw   (마커 비틀림 from 6축): {yaw:.2f} deg")
        
        self.log_msg("\n[3] setting.yaml 복사 양식")
        
        # Y and Z values swapped as requested. 
        # y_offset is from Axis 6 radius, z_offset is from Axis 5 radius - Link Length.
        y_offset_m = self.data_6['radius'] / 1000.0
        z_offset_m = (self.data_5['radius'] - L_5_ee) / 1000.0
        
        if self.arm_side == "left":
            self.log_msg(f"  Tf_to_marker_left:  [0.0, {y_offset_m:.5f}, {z_offset_m:.5f}, {90.0 + roll:.2f}, 0.00, {0.0 + yaw:.2f}]")
        else:
            self.log_msg(f"  Tf_to_marker_right: [0.0, {-y_offset_m:.5f}, {z_offset_m:.5f}, {90.0 + roll:.2f}, 0.00, {180.0 + yaw:.2f}]")
        
        self.log_msg("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Unified Marker Bracket Calibration GUI")
    parser.add_argument("--address", type=str, help="Robot IP address (optional for manual mode)")
    parser.add_argument("--model", type=str, default="a", help="Robot model (default: rby1_a)")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"], help="Side of the arm/marker")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    print("[INFO] Robot connection is now managed via GUI.")
    robot = None

    print("[INFO] Initializing Camera System...")
    marker_st = Marker_Transform()
    marker_st.marker_detection.set_marker_type("plate")
    
    gui = CalibrationApp(marker_st, robot, args.side)
    gui.show()
    
    try:
        sys.exit(app.exec())
    finally:
        marker_st.camera.stream_off()
        print("Camera resource released.")

if __name__ == "__main__":
    main()
