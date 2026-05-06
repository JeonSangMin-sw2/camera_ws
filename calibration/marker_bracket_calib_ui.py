import sys
import os
import cv2
import numpy as np
import time
import argparse
import logging
import rby1_sdk as rby

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QCheckBox, QLineEdit, QDialog, QMessageBox, QTabWidget)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy

# Import custom calibrator logic
from MarkerCalibrator import MarkerCalibrator

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

    def __init__(self, calibrator, arm_side):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side

    def run(self):
        self.calibrator.perform_move_to_center(self.arm_side, log_callback=self.log_signal.emit)
        self.finished_signal.emit()

# --- Calibration Worker Thread ---
class CalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)
    
    def __init__(self, calibrator, arm_side, axis_mode):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.axis_mode = axis_mode # 6 or 5
        
    def run(self):
        try:
            res = self.calibrator.perform_calibration_sweep(
                self.arm_side, self.axis_mode, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit
            )
            
            if res:
                # Add plot and summary logic back to UI thread side if needed, 
                # or handle plotting here.
                fitting_score = max(0.0, 100.0 * (1.0 - res['rmse'] / 4.0))
                
                self.log_signal.emit(f"  [1] Geometric Tracking Stability:")
                self.log_signal.emit(f"      Radius (Center-to-Axis): {res['radius']:.2f} mm")
                self.log_signal.emit(f"      Quality Score (FIT): {fitting_score:.1f}%")
                self.log_signal.emit(f"      Jitter (StdDev): {np.std(res['tilt_list']):.2f} deg")
                self.log_signal.emit("-" * 30)
                
                self.log_signal.emit(f"  [2] Robust Axis Alignment (Median):")
                if self.axis_mode == 6:
                    self.log_signal.emit(f"      Roll  (상하 기울기): {res['tilt']:.2f} deg")
                    self.log_signal.emit(f"      Yaw  (비틀림): {res['yaw']:.2f} deg")
                else:
                    pass
                self.log_signal.emit("="*40)
                self.log_signal.emit("\n[SWEEP SUCCESSFUL]\n")
                
                # Plotting
                plt.figure(figsize=(6, 6))
                plt.scatter(res['pts_2d'][:, 0], res['pts_2d'][:, 1], c='b', label='Captured Points')
                circle = plt.Circle((res['uc_opt'], res['vc_opt']), res['radius'], color='r', fill=False, label='Fitted Circle')
                plt.gca().add_patch(circle)
                plt.plot(res['uc_opt'], res['vc_opt'], 'rx', label='Center')
                
                x_min, x_max = res['pts_2d'][:, 0].min(), res['pts_2d'][:, 0].max()
                y_min, y_max = res['pts_2d'][:, 1].min(), res['pts_2d'][:, 1].max()
                span = max(x_max - x_min, y_max - y_min)
                margin = max(1.0, span * 0.5)
                cx = (x_max + x_min) / 2
                cy = (y_max + y_min) / 2
                plt.xlim(cx - span/2 - margin, cx + span/2 + margin)
                plt.ylim(cy - span/2 - margin, cy + span/2 + margin)
                plt.gca().set_aspect('equal')
                plt.grid(True)
                plt.title(f"Axis {self.axis_mode} Sweep (RMSE: {res['rmse']:.2f} px)")
                plt.legend()
                
                plot_path = os.path.join(os.path.dirname(__file__), f"circle_fit_axis_{self.axis_mode}.png")
                plt.savefig(plot_path)
                plt.close()
                
                res['plot_path'] = plot_path
                self.finished_signal.emit(res)
            else:
                self.finished_signal.emit(None)
                
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            self.finished_signal.emit(None)


class CalibrationApp(QWidget):
    def __init__(self, marker_st, robot, arm_side, ui_only=False):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        self.ui_only = ui_only
        self.calibrator = MarkerCalibrator(marker_st, robot)
        
        # Unified tracking Data
        self.data_5 = None
        self.data_6 = None
        
        self.setWindowTitle("Unified 5/6 Axis Bracket Calibration")
        self.resize(900, 600)
        
        self.init_ui()
        
        if not self.ui_only:
            # Timer for polling marker and temp
            self.poll_timer = QTimer(self)
            self.poll_timer.timeout.connect(self.poll_camera_status)
            self.poll_timer.start(200) # 5 Hz
        else:
            self.log_msg("[DEBUG] UI Only Mode: Hardware polling disabled.")
        
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
        if self.robot:
            # Disconnect
            self.log_msg("[INFO] Disconnecting from robot...")
            MarkerCalibrator.terminate_robot(self.robot)
            self.robot = None
            self.calibrator.robot = None
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #ff9900; color: black; font-weight: bold;")
            self.log_msg("[INFO] Robot disconnected.")
            return

        if self.ui_only:
            self.log_msg("[DEBUG] UI Only Mode: Mocking robot connection.")
            self.btn_connect.setText("DISCONNECT (MOCK)")
            self.btn_connect.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold;")
            self.robot = "MOCK_ROBOT"
            return

        try:
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip()
            self.log_msg(f"[INFO] Connecting to robot at {addr} ({model})...")
            self.robot = MarkerCalibrator.initialize_robot(addr, model)
            if self.robot:
                self.calibrator.robot = self.robot
                self.log_msg("[INFO] Robot successfully connected and activated.")
                self.btn_connect.setText("DISCONNECT")
                self.btn_connect.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold;")
            else:
                self.log_msg("[ERROR] Robot initialization failed. Please check the IP address and robot status.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to connect: {e}")

    def move_to_center(self):
        if self.ui_only:
            self.log_msg("[DEBUG] UI Only Mode: Move to Center skipped.")
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        if not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not detected!\nPlease use the teaching button to make the camera recognize the marker.")
            return
            
        self.btn_center.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.worker_mc = MoveCenterWorker(self.calibrator, self.arm_side)
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
        if self.ui_only:
            self.log_msg("[DEBUG] UI Only Mode: Start Sweep skipped.")
            return

        if not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not detected!\nPlease use the teaching button to make the camera recognize the marker.")
            return

        axis_mode = 6 if "6" in self.axis_sel.currentText() else 5
        self.btn_start.setEnabled(False)
        self.btn_result.setEnabled(False)
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        
        self.log_text.clear()
        
        self.worker = CalibrationWorker(self.calibrator, self.arm_side, axis_mode)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.status_signal.connect(self.update_marker_indicator)
        self.worker.finished_signal.connect(self.on_calibration_finished)
        self.worker.start()

    def on_calibration_finished(self, result_dict):
        self.btn_start.setEnabled(True)
        self.btn_result.setEnabled(True)
        if hasattr(self, 'poll_timer'):
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
            
        L_5_ee = self.get_link_length() # mm
        if L_5_ee <= 0:
            self.log_msg("[WARN] Could not retrieve link length. Using default 0.0.")
            L_5_ee = 0.0

        # --- Vector Math based on Axis Measurements ---
        z_e_in_m = self.data_6['axis']
        y_e_in_m = self.data_5['axis']
        
        y_e_in_m = y_e_in_m - np.dot(y_e_in_m, z_e_in_m) * z_e_in_m
        y_e_in_m /= np.linalg.norm(y_e_in_m)
        x_e_in_m = np.cross(y_e_in_m, z_e_in_m)
        
        R_E_M_mat = np.column_stack((x_e_in_m, y_e_in_m, z_e_in_m))
        euler_deg = R_scipy.from_matrix(R_E_M_mat).as_euler('zyx', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        
        radius_6 = self.data_6['radius'] 
        radius_5 = self.data_5['radius'] 
        
        x_e = 0.0
        if self.arm_side == "left":
            y_e = radius_6
        else:
            y_e = -radius_6
            
        z_e = - (L_5_ee - radius_5)
        
        self.log_msg("\n[1] Cartesian Offset (EE Link Frame)")
        self.log_msg(f"    - X-Offset: {x_e:.2f} mm")
        self.log_msg(f"    - Y-Offset: {y_e:.2f} mm")
        self.log_msg(f"    - Z-Offset: {z_e:.2f} mm")
        self.log_msg(f"       * (L_5_ee: {L_5_ee:.1f} mm, R6: {radius_6:.2f} mm, R5: {radius_5:.2f} mm)")
            
        self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
        self.log_msg(f"    - Roll : {roll_e:.2f} deg")
        self.log_msg(f"    - Pitch: {pitch_e:.2f} deg")
        self.log_msg(f"    - Yaw  : {yaw_e:.2f} deg")
        
        self.log_msg("\n[3] setting.yaml Format (meters, degrees)")
        
        x_m = x_e / 1000.0
        y_m = y_e / 1000.0
        z_m = z_e / 1000.0
        
        if self.arm_side == "left":
            self.log_msg(f"  Tf_to_marker_left:  [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {roll_e:.2f}, {pitch_e:.2f}, {yaw_e:.2f}]")
        else:
            self.log_msg(f"  Tf_to_marker_right: [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {roll_e:.2f}, {pitch_e:.2f}, {yaw_e:.2f}]")
        
        # --- [4] Calibration Confidence & Verification ---
        # 1. Orthogonality: Axis 5 and Axis 6 should be 90 deg apart.
        dot_val = np.dot(z_e_in_m, y_e_in_m)
        angle_between = np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0)))
        ortho_error = abs(90.0 - angle_between)
        
        # 2. Axis-Marker Alignment: Check if Axis 6 is Normal (~90deg) and Axis 5 is Parallel (~0deg) to marker plane.
        tilt_6 = self.data_6['tilt']
        tilt_5 = self.data_5['tilt']
        
        # Deviation from ideal alignment (Z-axis normal, Y-axis parallel)
        dev_6 = abs(90.0 - abs(tilt_6))
        dev_5 = abs(tilt_5)
        
        self.log_msg("\n[4] Calibration Confidence Report")
        self.log_msg(f"    - Axis Orthogonality Error  : {ortho_error:.3f} deg (Target: 0.0)")
        self.log_msg(f"    - A6-to-Marker Normal Error : {dev_6:.3f} deg (Tilt: {tilt_6:.2f})")
        self.log_msg(f"    - A5-to-Marker Plane Error  : {dev_5:.3f} deg (Tilt: {tilt_5:.2f})")
        
        # Determine pass/fail based on orthogonality and individual axis fit quality
        if ortho_error < 1.0 and dev_6 < 10.0 and dev_5 < 10.0:
            self.log_msg("    - Status: [PASS] (Errors within acceptable range)")
        else:
            self.log_msg("    - Status: [FAIL] (High inconsistency detected)")
            self.log_msg("\n" + "!"*60)
            self.log_msg(" [WARNING] High Calibration Inconsistency Detected!")
            self.log_msg(" 1. The camera may not be recognizing the marker correctly. \n    Please run 'camera_intrinsic_calib_ui.py' to calibrate internal parameters.")
            self.log_msg(" 2. The marker may not be properly attached or stable. \n    Please ensure the marker is securely fixed and not vibrating.")
            self.log_msg("!"*60 + "\n")

        self.log_msg("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Unified Marker Bracket Calibration GUI")
    parser.add_argument("--ui", action="store_true", help="Start only UI for debugging")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    robot = None
    marker_st = None

    if not args.ui:
        print("[INFO] Robot connection is now managed via GUI.")
        print("[INFO] Initializing Camera System...")
        
        try:
            from marker_detection import Marker_Transform
            marker_st = Marker_Transform()
            marker_st.marker_detection.set_marker_type("plate")
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera system: {e}")
            # Optional: fallback to UI mode or exit
            # sys.exit(1)
    else:
        print("[INFO] Starting in UI-only mode.")

    gui = CalibrationApp(marker_st, robot, "right", ui_only=args.ui)
    gui.show()
    
    try:
        sys.exit(app.exec())
    finally:
        if marker_st:
            marker_st.camera.stream_off()
            print("Camera resource released.")

if __name__ == "__main__":
    main()
