import sys
import os
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
from PitchHeadCalibrator import PitchHeadCalibrator

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

    def __init__(self, calibrator, arm_side, stop_event, target_dist=300.0):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.stop_event = stop_event
        self.target_dist = target_dist

    def run(self):
        self.calibrator.perform_move_to_center(
            self.arm_side, 
            log_callback=self.log_signal.emit, 
            stop_event=self.stop_event,
            target_dist=self.target_dist
        )
        self.finished_signal.emit()

class MoveToReadyWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, calibrator, arm_side, mode):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode

    def run(self):
        self.calibrator.perform_move_to_ready_pose(self.arm_side, self.mode, log_callback=self.log_signal.emit)
        self.finished_signal.emit()

# --- Calibration Worker Thread ---
class CalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)
    
    def __init__(self, calibrator, arm_side, mode, ui_only=False):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode
        self.ui_only = ui_only
        
    def run(self):
        try:
            if self.ui_only:
                # Mock Mode Execution
                time.sleep(1.5)
                self.log_signal.emit("[MOCK] Starting Sweep Steps...")
                for i in range(5):
                    self.log_signal.emit(f"  [STEP {i+1}/5] Mock calibration step executed.")
                    time.sleep(0.2)
                
                # Mock results
                if self.mode == "head":
                    res = {
                        'mode': 'head',
                        'opt_yaw': 0.35,
                        'opt_pitch': -0.48,
                        'rmse': 0.14,
                        'meas_pts_yaw': np.array([[y, y*0.01 + 0.1, y*0.02 + 0.2, y*0.03 + 0.3] for y in range(-10, 11)]),
                        'pred_pts_yaw': np.array([[y, y*0.01 + 0.11, y*0.02 + 0.21, y*0.03 + 0.31] for y in range(-10, 11)]),
                        'meas_pts_pitch': np.array([[p, p*0.01 + 0.1, p*0.02 + 0.2, p*0.03 + 0.3] for p in range(-10, 11)]),
                        'pred_pts_pitch': np.array([[p, p*0.01 + 0.11, p*0.02 + 0.21, p*0.03 + 0.31] for p in range(-10, 11)])
                    }
                else:
                    theta = np.linspace(-np.pi, np.pi, 20)
                    pts_2d_A = np.column_stack((np.cos(theta)*45 + 1.0, np.sin(theta)*45 + 2.0))
                    pts_2d_B = np.column_stack((np.cos(theta)*43 - 0.5, np.sin(theta)*43 + 1.5))
                    res = {
                        'mode': self.mode,
                        'optimal_offset': -0.45,
                        'offsets': [-4, -2, 0, 2, 4],
                        'errors': [2.1, 1.0, 0.3, 1.4, 2.9],
                        'pts_2d_A': pts_2d_A,
                        'uc_A': 1.0,
                        'vc_A': 2.0,
                        'r_A': 45.0,
                        'pts_2d_B': pts_2d_B,
                        'uc_B': -0.5,
                        'vc_B': 1.5,
                        'r_B': 43.0,
                        'rmse_A': 0.09,
                        'rmse_B': 0.12
                    }
            else:
                if self.mode == "head":
                    res = self.calibrator.perform_head_calibration_sweep(
                        self.arm_side, 
                        log_callback=self.log_signal.emit, 
                        status_callback=self.status_signal.emit
                    )
                else:
                    res = self.calibrator.perform_calibration_sweep_5_or_3(
                        self.arm_side, self.mode,
                        log_callback=self.log_signal.emit, 
                        status_callback=self.status_signal.emit
                    )

            if res:
                self.log_signal.emit("-" * 30)
                self.log_signal.emit(f"  [1] Calibration Target: {self.mode}")
                if self.mode == "head":
                    self.log_signal.emit(f"      Yaw Offset Estimation  : {res['opt_yaw']:.3f} deg")
                    self.log_signal.emit(f"      Pitch Offset Estimation: {res['opt_pitch']:.3f} deg")
                    self.log_signal.emit(f"      Fit Quality (RMSE)     : {res['rmse']:.3f} mm")
                else:
                    self.log_signal.emit(f"      Estimated Optimal Offset: {res['optimal_offset']:.3f} deg")
                    self.log_signal.emit(f"      Sweep RMSE (A/B)       : {res['rmse_A']:.3f} / {res['rmse_B']:.3f}")
                self.log_signal.emit("-" * 30)
                self.log_signal.emit("\n[CALIBRATION COMPLETE]\n")
                
                # Plotting
                plot_path_left = os.path.join(os.path.dirname(__file__), f"fit_left_{self.mode}.png")
                plot_path_right = os.path.join(os.path.dirname(__file__), f"fit_right_{self.mode}.png")
                
                if self.mode == "head":
                    # Left: Yaw sweep plot
                    plt.figure(figsize=(5, 5))
                    plt.plot(res['meas_pts_yaw'][:, 0], res['meas_pts_yaw'][:, 3]*1000, 'ro', label='Measured')
                    plt.plot(res['pred_pts_yaw'][:, 0], res['pred_pts_yaw'][:, 3]*1000, 'r-', label='Calibrated')
                    plt.title("Head Yaw Sweep (Z Axis)")
                    plt.xlabel("Yaw Angle (deg)")
                    plt.ylabel("Z (mm)")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path_left)
                    plt.close()
                    
                    # Right: Pitch sweep plot
                    plt.figure(figsize=(5, 5))
                    plt.plot(res['meas_pts_pitch'][:, 0], res['meas_pts_pitch'][:, 3]*1000, 'bo', label='Measured')
                    plt.plot(res['pred_pts_pitch'][:, 0], res['pred_pts_pitch'][:, 3]*1000, 'b-', label='Calibrated')
                    plt.title("Head Pitch Sweep (Z Axis)")
                    plt.xlabel("Pitch Angle (deg)")
                    plt.ylabel("Z (mm)")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path_right)
                    plt.close()
                else:
                    # Left: Circle A
                    plt.figure(figsize=(5, 5))
                    plt.scatter(res['pts_2d_A'][:, 0], res['pts_2d_A'][:, 1], c='r', label='Sweep Axis A')
                    circle_A = plt.Circle((res['uc_A'], res['vc_A']), res['r_A'], color='r', fill=False, label='Fit A')
                    plt.gca().add_patch(circle_A)
                    plt.plot(res['uc_A'], res['vc_A'], 'rx')
                    plt.title(f"Axis A Circle Fit (RMSE: {res['rmse_A']:.3f})")
                    plt.gca().set_aspect('equal')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path_left)
                    plt.close()
                    
                    # Right: Circle B
                    plt.figure(figsize=(5, 5))
                    plt.scatter(res['pts_2d_B'][:, 0], res['pts_2d_B'][:, 1], c='b', label='Sweep Axis B')
                    circle_B = plt.Circle((res['uc_B'], res['vc_B']), res['r_B'], color='b', fill=False, label='Fit B')
                    plt.gca().add_patch(circle_B)
                    plt.plot(res['uc_B'], res['vc_B'], 'bx')
                    plt.title(f"Axis B Circle Fit (RMSE: {res['rmse_B']:.3f})")
                    plt.gca().set_aspect('equal')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path_right)
                    plt.close()
                
                res['plot_path_left'] = plot_path_left
                res['plot_path_right'] = plot_path_right
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
        self.calibrator = PitchHeadCalibrator(marker_st, robot)
        
        # Datasets
        self.sweep_data = None
        
        self.setWindowTitle("Robot Pitch & Head Joint Offset Calibration")
        self.resize(900, 600)
        
        self.init_ui()
        
        if not self.ui_only:
            # Timer for polling marker and temp
            self.poll_timer = QTimer(self)
            self.poll_timer.timeout.connect(self.poll_camera_status)
            self.poll_timer.start(200) # 5 Hz
        else:
            self.log_msg("[DEBUG] UI Only Mode: Camera status updates are simulated.")
            self.indicator.set_detected(True)
            self.status_label.setText("Simulated (Detected)")
            self.status_label.setStyleSheet("color: green;")
            
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
        
        # Connection Box
        conn_box = QGroupBox("Robot Connection")
        conn_layout = QVBoxLayout()
        self.ip_input = QLineEdit("192.168.30.1:50051")
        if self.ui_only:
            self.ip_input.setText("127.0.0.1:50051")
            self.ip_input.setReadOnly(True)
            self.ip_input.setStyleSheet("background-color: #e9ecef; color: #495057;")
            
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
        
        # Camera Box
        status_box = QGroupBox("Camera & Marker Status")
        status_layout = QVBoxLayout()
        
        ind_layout = QHBoxLayout()
        self.indicator = IndicatorWidget()
        ind_layout.addWidget(self.indicator)
        self.status_label = QLabel("Not Detected")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        ind_layout.addWidget(self.status_label)
        ind_layout.addStretch()
        status_layout.addLayout(ind_layout)
        
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        self.btn_monitor.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.btn_monitor)
        
        self.temp_label = QLabel("Camera Temp: -- °C")
        self.temp_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.temp_label)
        status_box.setLayout(status_layout)
        
        # Controls Box (Matches marker_bracket_calib_ui structure 100%)
        controls_box = QGroupBox("Calibration Controls")
        controls_layout = QVBoxLayout()
        
        self.arm_side_sel = QComboBox()
        self.arm_side_sel.addItems(["Right Arm", "Left Arm"])
        if self.arm_side == "left":
            self.arm_side_sel.setCurrentIndex(1)
        self.arm_side_sel.currentTextChanged.connect(self.on_arm_side_changed)
        controls_layout.addWidget(self.arm_side_sel)

        self.mode_sel = QComboBox()
        self.mode_sel.addItems(["wrist_pitch (5-Axis Sweep)", "elbow (3-Axis Sweep)", "head (Yaw/Pitch Sweep)"])
        controls_layout.addWidget(self.mode_sel)
        
        self.btn_ready = QPushButton("MOVE TO READY")
        self.btn_ready.setMinimumHeight(40)
        self.btn_ready.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold;")
        self.btn_ready.clicked.connect(self.move_to_ready_pose)
        
        self.btn_center_head = QPushButton("CENTER HEAD")
        self.btn_center_head.setMinimumHeight(40)
        self.btn_center_head.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold;")
        self.btn_center_head.clicked.connect(self.move_head_to_zero)
        
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
        
        controls_layout.addWidget(self.btn_ready)
        controls_layout.addWidget(self.btn_center_head)
        controls_layout.addWidget(self.btn_center)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_result)
        controls_layout.addWidget(self.btn_quit)
        controls_box.setLayout(controls_layout)
        
        left_panel.addWidget(conn_box)
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
        
        # Plot Tab (Left / Right split matching marker_bracket_calib_ui)
        self.plot_tab = QWidget()
        plot_layout = QHBoxLayout()
        
        self.plot_label_left = QLabel("Sweep Plot (Axis A / Yaw) will appear here")
        self.plot_label_left.setAlignment(Qt.AlignCenter)
        self.plot_label_left.setStyleSheet("background-color: #333333; color: white;")
        
        self.plot_label_right = QLabel("Sweep Plot (Axis B / Pitch) will appear here")
        self.plot_label_right.setAlignment(Qt.AlignCenter)
        self.plot_label_right.setStyleSheet("background-color: #333333; color: white;")
        
        plot_layout.addWidget(self.plot_label_left)
        plot_layout.addWidget(self.plot_label_right)
        self.plot_tab.setLayout(plot_layout)
        self.tabs.addTab(self.plot_tab, "Plot Viewer")
        
        self.setLayout(main_layout)
        self.log_msg("Robot Pitch & Head Joint Offset Calibration App Ready.\nReady the target arm and choose sweep target.")

    def log_msg(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def connect_robot(self):
        if self.robot:
            self.log_msg("[INFO] Disconnecting from robot...")
            self.robot.disconnect()
            self.robot = None
            self.calibrator.robot = None
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #ff9900; color: black; font-weight: bold;")
            self.log_msg("[INFO] Robot disconnected.")
            return

        try:
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip()
            self.log_msg(f"[INFO] Connecting to robot at {addr} ({model})...")
            
            robot = rby.create_robot(addr, model)
            if robot.connect():
                self.log_msg("[INFO] Connection established. Powering on and enabling servos...")
                
                # 1. Power ON
                if not robot.is_power_on(".*"):
                    self.log_msg("  - Turning power on...")
                    robot.power_on(".*")
                    time.sleep(1.0)
                
                # 2. Servo ON
                if not robot.is_servo_on(".*"):
                    self.log_msg("  - Enabling servos...")
                    robot.servo_on(".*")
                    time.sleep(1.0)

                # 3. Reset Fault Control Manager
                cm_state = robot.get_control_manager_state()
                if cm_state.state in [rby.ControlManagerState.State.MinorFault, rby.ControlManagerState.State.MajorFault]:
                    self.log_msg(f"  - Control manager in fault state ({cm_state.state}). Resetting...")
                    robot.reset_fault_control_manager()
                    time.sleep(1.0)
                
                # 4. Enable Control Manager
                cm_state = robot.get_control_manager_state()
                if cm_state.state != rby.ControlManagerState.State.Enabled:
                    self.log_msg("  - Enabling control manager...")
                    if robot.enable_control_manager():
                        self.log_msg("  - Control manager enabled successfully.")
                    else:
                        self.log_msg("[WARNING] Failed to enable control manager. Please check manual state.")
                else:
                    self.log_msg("  - Control manager is already enabled.")
                
                self.robot = robot
                self.calibrator.robot = robot
                self.log_msg("[INFO] Robot successfully connected and fully activated.")
                self.btn_connect.setText("DISCONNECT")
                self.btn_connect.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold;")
            else:
                self.log_msg("[ERROR] Robot connection failed.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to connect: {e}")

    def on_arm_side_changed(self, text):
        self.arm_side = "left" if "Left" in text else "right"
        self.log_msg(f"[INFO] Target arm changed to {self.arm_side.upper()} Arm. (Prior data cleared)")
        self.sweep_data = None

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
            res = self.marker_st.get_marker_transform(sampling_time=0, side=self.arm_side)
            detected = bool(res and len(res) > 0)
            self.update_marker_indicator(detected)
            
            if detected:
                if self.btn_monitor.isChecked():
                    pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                    x, y, z = pose[:3, 3] * 1000.0
                    self.log_msg(f"[LIVE] Marker X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm")
            
            self.marker_st.camera.get_color_image()
            temp = self.marker_st.camera.get_camera_temperature()
            if temp:
                self.temp_label.setText(f"Camera Temp: {temp:.1f} °C")
        except Exception:
            pass

    def move_to_ready_pose(self):
        if self.ui_only:
            self.log_msg("[DEBUG] UI Only Mode: Move to Ready skipped.")
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        mode_str = self.mode_sel.currentText()
        mode = "wrist_pitch" if "wrist_pitch" in mode_str else ("elbow" if "elbow" in mode_str else "head")
        
        self.btn_ready.setEnabled(False)
        self.btn_center_head.setEnabled(False)
        self.btn_center.setEnabled(False)
        self.btn_start.setEnabled(False)
        
        self.ready_worker = MoveToReadyWorker(self.calibrator, self.arm_side, mode)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_action_finished)
        self.ready_worker.start()

    def move_head_to_zero(self):
        if self.ui_only:
            self.log_msg("[DEBUG] UI Only Mode: Center Head skipped.")
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        self.log_msg("[INFO] Centering head to zero rad...")
        try:
            head_zero = np.zeros(len(self.robot.model().head_idx))
            self.btn_ready.setEnabled(False)
            self.btn_center_head.setEnabled(False)
            self.btn_center.setEnabled(False)
            self.btn_start.setEnabled(False)
            
            # Non-blocking head centering inside QThread or direct command
            class CenterHeadWorker(QThread):
                log_sig = Signal(str)
                done_sig = Signal()
                def __init__(self, calibrator, robot, q_zero):
                    super().__init__()
                    self.calibrator = calibrator
                    self.robot = robot
                    self.q_zero = q_zero
                def run(self):
                    self.calibrator.movej(self.robot, head=self.q_zero, minimum_time=3.0)
                    self.done_sig.emit()
            
            self.head_worker = CenterHeadWorker(self.calibrator, self.robot, head_zero)
            self.head_worker.log_sig.connect(self.log_msg)
            self.head_worker.done_sig.connect(self.on_action_finished)
            self.head_worker.start()
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to center head: {e}")
            self.on_action_finished()

    def move_to_center(self):
        if self.ui_only:
            self.log_msg("[DEBUG] UI Only Mode: Move to Center skipped.")
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        # If already running, cancel it
        if hasattr(self, 'worker_mc') and self.worker_mc.isRunning():
            self.log_msg("[INFO] Cancelling Move to Center...")
            self.stop_event_mc.set()
            return
            
        if not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not detected!\nPlease teach or adjust target pose.")
            return
            
        mode_str = self.mode_sel.currentText()
        target_dist = 200.0 if "head" in mode_str else 300.0
        self.log_msg(f"[INFO] Move to Center triggered for {mode_str}. Target distance: {target_dist} mm")

        self.btn_center.setText("CANCEL")
        self.btn_center.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
        self.btn_ready.setEnabled(False)
        self.btn_center_head.setEnabled(False)
        self.btn_start.setEnabled(False)
        
        import threading
        self.stop_event_mc = threading.Event()
        self.worker_mc = MoveCenterWorker(self.calibrator, self.arm_side, self.stop_event_mc, target_dist=target_dist)
        self.worker_mc.log_signal.connect(self.log_msg)
        self.worker_mc.finished_signal.connect(self.on_move_center_finished)
        self.worker_mc.start()

    def on_move_center_finished(self):
        self.btn_center.setText("MOVE TO CENTER")
        self.btn_center.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold;")
        self.on_action_finished()
        if hasattr(self, 'stop_event_mc'):
            self.stop_event_mc.clear()

    def on_action_finished(self):
        self.btn_ready.setEnabled(True)
        self.btn_center_head.setEnabled(True)
        self.btn_center.setEnabled(True)
        self.btn_start.setEnabled(True)

    def start_calibration(self):
        mode_str = self.mode_sel.currentText()
        mode = "wrist_pitch" if "wrist_pitch" in mode_str else ("elbow" if "elbow" in mode_str else "head")
        
        if not self.ui_only and not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not detected!\nPlease teach or adjust target pose.")
            return

        self.btn_start.setEnabled(False)
        self.btn_ready.setEnabled(False)
        self.btn_center.setEnabled(False)
        self.btn_center_head.setEnabled(False)
        self.btn_result.setEnabled(False)
        
        if not self.ui_only and hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
            
        self.log_text.clear()
        
        self.worker = CalibrationWorker(self.calibrator, self.arm_side, mode, ui_only=self.ui_only)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.status_signal.connect(self.update_marker_indicator)
        self.worker.finished_signal.connect(self.on_calibration_finished)
        self.worker.start()

    def on_calibration_finished(self, result_dict):
        self.on_action_finished()
        self.btn_result.setEnabled(True)
        
        if not self.ui_only and hasattr(self, 'poll_timer'):
            self.poll_timer.start(200)
            
        if result_dict:
            self.sweep_data = result_dict
            
            # Show Plots in Plot Tab
            if 'plot_path_left' in result_dict and os.path.exists(result_dict['plot_path_left']):
                pixmap_l = QPixmap(result_dict['plot_path_left'])
                scaled_l = pixmap_l.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.plot_label_left.setPixmap(scaled_l)
                
            if 'plot_path_right' in result_dict and os.path.exists(result_dict['plot_path_right']):
                pixmap_r = QPixmap(result_dict['plot_path_right'])
                scaled_r = pixmap_r.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.plot_label_right.setPixmap(scaled_r)
                
            # Switch to plot viewer automatically
            self.tabs.setCurrentIndex(1)

    def show_unified_result(self):
        self.log_text.clear()
        self.log_msg("\n" + "#"*50)
        self.log_msg("       CALIBRATION SWEEP ESTIMATED OFFSETS")
        self.log_msg("#"*50)
        
        if not self.sweep_data:
            self.log_msg("\n[ERROR] Sweep dataset is empty!")
            self.log_msg("Please select a target and press 'START SWEEP' first.")
            return

        mode = self.sweep_data['mode']
        self.log_msg(f"\n[1] Calibration Target: {mode}")
        
        if mode == "head":
            self.log_msg(f"    - Estimated Head Yaw Offset  : {self.sweep_data['opt_yaw']:.4f} deg")
            self.log_msg(f"    - Estimated Head Pitch Offset: {self.sweep_data['opt_pitch']:.4f} deg")
            self.log_msg(f"    - Residual Alignment RMSE    : {self.sweep_data['rmse']:.3f} mm")
            
            self.log_msg("\n[2] setting.yaml Head Config update suggestions:")
            self.log_msg("  (Apply these offsets to your physical zero calibration offsets)")
            self.log_msg(f"  Head Yaw Calibration Change  : {self.sweep_data['opt_yaw']:.4f} deg")
            self.log_msg(f"  Head Pitch Calibration Change: {self.sweep_data['opt_pitch']:.4f} deg")
        else:
            self.log_msg(f"    - Sweep Target Joint       : {'Joint 5' if mode == 'wrist_pitch' else 'Joint 3'}")
            self.log_msg(f"    - Estimated Optimal Offset : {self.sweep_data['optimal_offset']:.4f} deg")
            self.log_msg(f"    - Circle A Fitting RMSE     : {self.sweep_data['rmse_A']:.4f} mm")
            self.log_msg(f"    - Circle B Fitting RMSE     : {self.sweep_data['rmse_B']:.4f} mm")
            
            self.log_msg("\n[2] Joint Calibration suggestions:")
            self.log_msg(f"  Suggest applying offset: {self.sweep_data['optimal_offset']:.4f} deg to your joint calibration parameters.")

        self.log_msg("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Pitch & Head Joint Calibration GUI")
    parser.add_argument("--ui", action="store_true", help="Start only UI for debugging")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    robot = None
    marker_st = None

    if not args.ui:
        try:
            from marker_detection import Marker_Transform
            marker_st = Marker_Transform()
            marker_st.marker_detection.set_marker_type("plate")
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera system: {e}")

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
