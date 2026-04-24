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
                             QPushButton, QTextEdit, QLabel, QGroupBox, QFrame)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont

# --- Configuration ---
MAX_POINTS = 11 # Total points for -20 to +20 sweep (4 deg steps)
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

def fit_circle_3d_robust(points):
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

    A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
    b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc_init, vc_init, offset_init = res[0], res[1], res[2]
    
    radius_init = np.sqrt(max(0, offset_init + uc_init**2 + vc_init**2))
    initial_guess = [uc_init, vc_init, radius_init]

    def residuals(params, xy):
        uc, vc, R = params
        distances = np.sqrt((xy[:, 0] - uc)**2 + (xy[:, 1] - vc)**2)
        return distances - R

    opt_result = least_squares(residuals, initial_guess, args=(pts_2d,), loss='huber')
    uc_opt, vc_opt, radius_opt = opt_result.x

    center_3d = centroid + uc_opt * ex + vc_opt * ey
    final_residuals = residuals(opt_result.x, pts_2d)
    rmse = np.sqrt(np.mean(final_residuals**2))
    
    return center_3d, normal, radius_opt, rmse

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

# --- Calibration Worker Thread ---
class CalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal()
    
    def __init__(self, marker_st, robot, arm_side):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        
    def run(self):
        try:
            self._calibrate()
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
        finally:
            self.finished_signal.emit()

    def _calibrate(self):
        self.log_signal.emit("\n" + "="*40)
        self.log_signal.emit("   STARTING AUTOMATED CALIBRATION")
        self.log_signal.emit("="*40)

        # Pre-check Marker Presence
        self.log_signal.emit("  - Checking if marker is visible before starting...")
        initial_check = self.marker_st.get_marker_transform(sampling_time=1.0, side=self.arm_side)
        if not initial_check:
            self.log_signal.emit("\n[ERROR] 마커가 위치해 있지 않습니다. 시작할 수 없습니다.")
            self.status_signal.emit(False)
            return
        
        self.status_signal.emit(True)
        captured_poses = []

        if not self.robot:
            self.log_signal.emit("[ERROR] Robot not connected. Cannot perform automated sweep.")
            return

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if self.arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])
        self.log_signal.emit(f"[INFO] Recorded Center Joint Pose: {np.round(initial_joint_pos, 2)}")

        for i in range(MAX_POINTS):
            self.log_signal.emit(f"\n[STEP {i + 1}/{MAX_POINTS}]")
            
            target_offset_deg = -20 + (i * 4)
            target_joint_pos = list(initial_joint_pos)
            target_joint_pos[6] = initial_joint_pos[6] + np.radians(target_offset_deg)
            
            if target_offset_deg == 0:
                self.log_signal.emit(f"  - Moving to Center Pose (0 deg)...")
            else:
                self.log_signal.emit(f"  - Moving arm to {target_offset_deg:.1f} deg offset...")
            
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
        if len(captured_poses) >= MAX_POINTS:
            self.log_signal.emit("\n" + "="*40)
            self.log_signal.emit("   BRACKET ALIGNMENT ANALYSIS (FINAL - MEDIAN AVERAGED)")
            self.log_signal.emit("="*40)
            
            T_cam_ref = captured_poses[0] # The -20 deg point
            T_ref_cam = np.linalg.inv(T_cam_ref)
            relative_poses = [T_ref_cam @ T for T in captured_poses]
            points = [T[:3, 3] for T in relative_poses]
            
            center, axis, radius, rmse = fit_circle_3d_robust(points)
            fitting_score = max(0.0, 100.0 * (1.0 - rmse / 4.0))
            
            roll_list = []
            yaw_list = []
            
            for T_i in relative_poses:
                R_i = T_i[:3, :3]
                
                # Roll
                axis_m_i = R_i.T @ axis
                roll_i = np.degrees(np.arcsin(min(1.0, max(-1.0, axis_m_i[2]))))
                roll_list.append(roll_i)
                
                # Yaw
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

            robust_roll = np.median(roll_list)
            robust_yaw = np.median(yaw_list)

            self.log_signal.emit(f"  [1] Geometric Tracking Stability:")
            self.log_signal.emit(f"      Radius (Center-to-Axis): {radius:.2f} mm")
            self.log_signal.emit(f"      Quality Score (FIT): {fitting_score:.1f}%")
            self.log_signal.emit(f"      Roll Jitter (StdDev): {np.std(roll_list):.2f} deg")
            self.log_signal.emit("-" * 30)
            self.log_signal.emit(f"  [2] Robust Bracket Misalignment (Median of 11 pts):")
            self.log_signal.emit(f"      Roll (상하 기울기): {robust_roll:.2f} deg")
            self.log_signal.emit(f"      Yaw  (좌우 비틀림): {robust_yaw:.2f} deg")
            self.log_signal.emit("="*40)
            self.log_signal.emit("\n[CALIBRATION SUCCESSFUL]")

class CalibrationApp(QWidget):
    def __init__(self, marker_st, robot, arm_side):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        
        self.setWindowTitle("RBY1 Marker Bracket Calibration")
        self.resize(800, 500)
        
        self.init_ui()
        
        # Timer for polling marker and temp
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera_status)
        self.poll_timer.start(200) # 5 Hz
        
        self.worker = None

    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # Left Panel
        left_panel = QVBoxLayout()
        
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
        
        # Temp
        self.temp_label = QLabel("Camera Temp: -- °C")
        self.temp_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.temp_label)
        status_box.setLayout(status_layout)
        
        # Controls
        controls_box = QGroupBox("Calibration Controls")
        controls_layout = QVBoxLayout()
        
        self.btn_start = QPushButton("START CALIBRATION")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_calibration)
        
        self.btn_retry = QPushButton("RETRY")
        self.btn_retry.setMinimumHeight(30)
        self.btn_retry.clicked.connect(self.start_calibration)
        
        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setMinimumHeight(30)
        self.btn_quit.setStyleSheet("background-color: #dc3545; color: white;")
        self.btn_quit.clicked.connect(self.close)
        
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_retry)
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
        
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)
        self.setLayout(main_layout)
        
        self.log_msg("Calibration App Ready.\nCheck marker status and click START.")

    def log_msg(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

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
            # Poll marker quickly
            results = self.marker_st.get_marker_transform(sampling_time=0, side="all")
            detected = bool(results and len(results) > 0)
            self.update_marker_indicator(detected)
            
            # Flush pipeline to keep it fresh
            self.marker_st.camera.get_color_image()
            
            # Poll temp
            temp = self.marker_st.camera.get_camera_temperature()
            if temp:
                self.temp_label.setText(f"Camera Temp: {temp:.1f} °C")
        except Exception as e:
            pass

    def start_calibration(self):
        self.btn_start.setEnabled(False)
        self.btn_retry.setEnabled(False)
        self.poll_timer.stop() # Stop UI polling during heavy worker ops
        
        self.log_text.clear()
        
        self.worker = CalibrationWorker(self.marker_st, self.robot, self.arm_side)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.status_signal.connect(self.update_marker_indicator)
        self.worker.finished_signal.connect(self.on_calibration_finished)
        self.worker.start()

    def on_calibration_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_retry.setEnabled(True)
        self.poll_timer.start(200) # Resume UI polling

def main():
    parser = argparse.ArgumentParser(description="Marker Bracket Calibration GUI")
    parser.add_argument("--address", type=str, help="Robot IP address (optional for manual mode)")
    parser.add_argument("--model", type=str, default="a", help="Robot model (default: rby1_a)")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"], help="Side of the arm/marker")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    print("[INFO] Connecting to robot...")
    if args.address:
        robot = initialize_robot(args.address, args.model)
        print("[INFO] Robot initialized.")
    else:
        robot = None
        print("[WARN] No Robot IP provided.")

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
