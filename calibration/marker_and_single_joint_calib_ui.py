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
from MarkerCalibrator import MarkerCalibrator
from PitchHeadCalibrator import PitchHeadCalibrator

# --- Premium Dark CSS Stylesheet ---
DARK_STYLESHEET = """
QWidget {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Malgun Gothic', Arial, sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 2px solid #2d2d2d;
    border-radius: 8px;
    margin-top: 15px;
    font-weight: bold;
    font-size: 13px;
    color: #2979ff;
    padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 5px;
}
QPushButton {
    background-color: #1e1e1e;
    color: #ffffff;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2c2c2c;
    border: 1px solid #2979ff;
}
QPushButton:pressed {
    background-color: #121212;
}
QPushButton:disabled {
    background-color: #1a1a1a;
    border: 1px solid #242424;
    color: #555555;
}
QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
    min-width: 120px;
}
QComboBox::drop-down {
    border: none;
}
QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
}
QTabWidget::pane {
    border: 1px solid #2d2d2d;
    background: #121212;
    border-radius: 6px;
}
QTabBar::tab {
    background: #1a1a1a;
    border: 1px solid #2d2d2d;
    border-bottom: none;
    padding: 8px 16px;
    font-weight: bold;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #888888;
}
QTabBar::tab:selected {
    background: #121212;
    color: #2979ff;
    border-bottom: 2px solid #2979ff;
}
QTabBar::tab:hover:!selected {
    background: #252525;
    color: #e0e0e0;
}
QCheckBox {
    spacing: 8px;
    font-weight: bold;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    background-color: #1e1e1e;
}
QCheckBox::indicator:checked {
    background-color: #2979ff;
    border: 1px solid #2979ff;
}
"""

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
        color = QColor(0, 230, 118) if self.is_detected else QColor(255, 23, 68)
        painter.setBrush(color)
        painter.setPen(QPen(Qt.black, 1.5))
        painter.drawEllipse(2, 2, 26, 26)

# --- Common Worker Threads ---
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

    def __init__(self, calibrator, arm_side, mode=None):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode

    def run(self):
        if self.mode is not None:
            # Joint Calibrator
            self.calibrator.perform_move_to_ready_pose(self.arm_side, self.mode, log_callback=self.log_signal.emit)
        else:
            # Marker Calibrator
            self.calibrator.perform_move_to_ready_pose(self.arm_side, log_callback=self.log_signal.emit)
        self.finished_signal.emit()

# --- Specialized Calibration Workers ---
class MarkerCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)
    
    def __init__(self, calibrator, arm_side, axis_mode, use_head_tracking=True):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.axis_mode = axis_mode  # 6 or 5
        self.use_head_tracking = use_head_tracking
        
    def run(self):
        try:
            res = self.calibrator.perform_calibration_sweep(
                self.arm_side, self.axis_mode, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking
            )
            
            if res:
                fitting_score = max(0.0, 100.0 * (1.0 - res['rmse'] / 4.0))
                
                self.log_signal.emit(f"  [1] Geometric Tracking Stability:")
                self.log_signal.emit(f"      Radius (Center-to-Axis): {res['radius']:.2f} mm")
                self.log_signal.emit(f"      Quality Score (FIT): {fitting_score:.1f}%")
                self.log_signal.emit(f"      Jitter (StdDev): {np.std(res['tilt_list']):.2f} deg")
                self.log_signal.emit("-" * 30)
                
                self.log_signal.emit(f"  [2] Robust Axis Alignment (Median):")
                if self.axis_mode == 6:
                    self.log_signal.emit(f"      Roll  (상하 기울기): {res['tilt']:.2f} deg")
                    self.log_signal.emit(f"      Yaw   (비틀림): {res['yaw']:.2f} deg")
                else:
                    self.log_signal.emit(f"      Tilt  (기울기): {res['tilt']:.2f} deg")
                    self.log_signal.emit(f"      Yaw   (비틀림): {res['yaw']:.2f} deg")
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
                plt.title(f"Axis {self.axis_mode} Sweep (RMSE: {res['rmse']:.3f})")
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

class JointCalibrationWorker(QThread):
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
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Starting Sweep Steps...")
                for i in range(5):
                    self.log_signal.emit(f"  [STEP {i+1}/5] Mock calibration step executed.")
                    time.sleep(0.2)
                
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
                    # Yaw Sweep
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
                    
                    # Pitch Sweep
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
                    # Circle A
                    plt.figure(figsize=(5, 5))
                    plt.scatter(res['pts_2d_A'][:, 0], res['pts_2d_A'][:, 1], c='r', label='Sweep Axis A')
                    circle_A = plt.Circle((res['uc_A'], res['vc_A']), res['r_A'], color='r', fill=False, label='Fit A')
                    plt.gca().add_patch(circle_A)
                    plt.plot(res['uc_A'], res['vc_A'], 'rx')
                    plt.title(f"Axis A Circle (RMSE: {res['rmse_A']:.3f})")
                    plt.gca().set_aspect('equal')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path_left)
                    plt.close()
                    
                    # Circle B
                    plt.figure(figsize=(5, 5))
                    plt.scatter(res['pts_2d_B'][:, 0], res['pts_2d_B'][:, 1], c='b', label='Sweep Axis B')
                    circle_B = plt.Circle((res['uc_B'], res['vc_B']), res['r_B'], color='b', fill=False, label='Fit B')
                    plt.gca().add_patch(circle_B)
                    plt.plot(res['uc_B'], res['vc_B'], 'bx')
                    plt.title(f"Axis B Circle (RMSE: {res['rmse_B']:.3f})")
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


# --- Unified Calibration App ---
class UnifiedCalibrationApp(QWidget):
    def __init__(self, marker_st, robot, arm_side="right", ui_only=False):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        self.ui_only = ui_only
        
        # Core Calibrator Instances
        self.marker_calibrator = MarkerCalibrator(marker_st, robot)
        self.joint_calibrator = PitchHeadCalibrator(marker_st, robot)
        
        # Saved Calibration Results
        self.marker_data_5 = None
        self.marker_data_6 = None
        self.joint_sweep_data = None
        
        self.setWindowTitle("Unified Robot Joint & Marker Bracket Calibration")
        self.resize(1100, 750)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.init_ui()
        
        # Poll indicators
        if not self.ui_only:
            self.poll_timer = QTimer(self)
            self.poll_timer.timeout.connect(self.poll_camera_status)
            self.poll_timer.start(200) # 5 Hz
        else:
            self.log_msg("[DEBUG] UI-Only Simulation Mode Enabled.")
            self.update_marker_indicator(True)
            self.temp_label.setText("Camera Temp: 34.2 °C (Simulated)")
            
        self.active_worker = None

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # --- Tab 1: Calibration Panel ---
        calib_tab = QWidget()
        calib_layout = QHBoxLayout()
        
        # Left Panel (Controls)
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(5, 5, 5, 5)
        
        # Group 1: Connection Box
        conn_box = QGroupBox("Robot Connection")
        conn_layout = QVBoxLayout()
        self.ip_input = QLineEdit("192.168.30.1:50051")
        if self.ui_only:
            self.ip_input.setText("127.0.0.1:50051")
            self.ip_input.setReadOnly(True)
            self.ip_input.setStyleSheet("background-color: #1a1a1a; color: #888888; border: 1px solid #2d2d2d;")
            
        self.model_input = QComboBox()
        self.model_input.addItems(["a", "m"])
        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold;")
        self.btn_connect.clicked.connect(self.connect_robot)
        
        conn_layout.addWidget(QLabel("IP / Port:"))
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(QLabel("Robot Model:"))
        conn_layout.addWidget(self.model_input)
        conn_layout.addWidget(self.btn_connect)
        conn_box.setLayout(conn_layout)
        left_panel.addWidget(conn_box)
        
        # Group 2: Status Indicator
        status_box = QGroupBox("Camera & Marker Status")
        status_layout = QVBoxLayout()
        
        ind_layout = QHBoxLayout()
        self.indicator = IndicatorWidget()
        ind_layout.addWidget(self.indicator)
        self.status_label = QLabel("Not Detected")
        self.status_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.status_label.setStyleSheet("color: #ff1744;")
        ind_layout.addWidget(self.status_label)
        ind_layout.addStretch()
        
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        
        self.temp_label = QLabel("Camera Temp: -- °C")
        
        status_layout.addLayout(ind_layout)
        status_layout.addWidget(self.btn_monitor)
        status_layout.addWidget(self.temp_label)
        status_box.setLayout(status_layout)
        left_panel.addWidget(status_box)
        
        # Group 3: Nested Calibration Workflow Selector Tabs
        workflow_box = QGroupBox("Calibration Workflows")
        workflow_layout = QVBoxLayout()
        self.workflow_tabs = QTabWidget()
        
        # Sub-tab A: Single Joint Calibration
        joint_subtab = QWidget()
        joint_sublayout = QVBoxLayout()
        
        self.joint_arm_sel = QComboBox()
        self.joint_arm_sel.addItems(["Right Arm", "Left Arm"])
        self.joint_arm_sel.currentTextChanged.connect(self.on_arm_side_changed)
        
        self.joint_mode_sel = QComboBox()
        self.joint_mode_sel.addItems(["wrist_pitch (5-Axis Sweep)", "elbow (3-Axis Sweep)", "head (Yaw/Pitch Sweep)"])
        
        self.btn_joint_ready = QPushButton("MOVE TO READY")
        self.btn_joint_ready.setStyleSheet("background-color: #6a1b9a; color: white;")
        self.btn_joint_ready.clicked.connect(self.move_to_ready_pose_joint)
        
        self.btn_joint_center_head = QPushButton("CENTER HEAD")
        self.btn_joint_center_head.setStyleSheet("background-color: #424242; color: white;")
        self.btn_joint_center_head.clicked.connect(self.move_head_to_zero)
        
        self.btn_joint_center = QPushButton("MOVE TO CENTER")
        self.btn_joint_center.setStyleSheet("background-color: #00838f; color: white;")
        self.btn_joint_center.clicked.connect(self.move_to_center_joint)
        
        self.btn_joint_start = QPushButton("START SWEEP")
        self.btn_joint_start.setStyleSheet("background-color: #1565c0; color: white;")
        self.btn_joint_start.clicked.connect(self.start_calibration_joint)
        
        self.btn_joint_result = QPushButton("SHOW RESULT")
        self.btn_joint_result.setStyleSheet("background-color: #2e7d32; color: white;")
        self.btn_joint_result.clicked.connect(self.show_result_joint)
        
        joint_sublayout.addWidget(QLabel("Target Arm:"))
        joint_sublayout.addWidget(self.joint_arm_sel)
        joint_sublayout.addWidget(QLabel("Calibration Mode:"))
        joint_sublayout.addWidget(self.joint_mode_sel)
        joint_sublayout.addWidget(self.btn_joint_ready)
        joint_sublayout.addWidget(self.btn_joint_center_head)
        joint_sublayout.addWidget(self.btn_joint_center)
        joint_sublayout.addWidget(self.btn_joint_start)
        joint_sublayout.addWidget(self.btn_joint_result)
        joint_subtab.setLayout(joint_sublayout)
        self.workflow_tabs.addTab(joint_subtab, "1. Joint Calib")
        
        # Sub-tab B: Marker Bracket Calibration
        marker_subtab = QWidget()
        marker_sublayout = QVBoxLayout()
        
        self.marker_arm_sel = QComboBox()
        self.marker_arm_sel.addItems(["Right Arm", "Left Arm"])
        self.marker_arm_sel.currentTextChanged.connect(self.on_arm_side_changed)
        
        self.marker_axis_sel = QComboBox()
        self.marker_axis_sel.addItems(["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"])
        
        tol_lay = QHBoxLayout()
        tol_lay.addWidget(QLabel("Tolerance (deg):"))
        self.tolerance_input = QLineEdit("0.5")
        self.tolerance_input.setFixedWidth(50)
        tol_lay.addWidget(self.tolerance_input)
        
        self.cb_head_tracking = QCheckBox("Active Head Tracking")
        self.cb_head_tracking.setChecked(True)
        
        self.btn_marker_ready = QPushButton("MOVE TO READY")
        self.btn_marker_ready.setStyleSheet("background-color: #6a1b9a; color: white;")
        self.btn_marker_ready.clicked.connect(self.move_to_ready_pose_marker)
        
        self.btn_marker_center_head = QPushButton("CENTER HEAD")
        self.btn_marker_center_head.setStyleSheet("background-color: #424242; color: white;")
        self.btn_marker_center_head.clicked.connect(self.move_head_to_zero)
        
        self.btn_marker_center = QPushButton("MOVE TO CENTER")
        self.btn_marker_center.setStyleSheet("background-color: #00838f; color: white;")
        self.btn_marker_center.clicked.connect(self.move_to_center_marker)
        
        self.btn_marker_start = QPushButton("START SWEEP")
        self.btn_marker_start.setStyleSheet("background-color: #1565c0; color: white;")
        self.btn_marker_start.clicked.connect(self.start_calibration_marker)
        
        self.btn_marker_result = QPushButton("UNIFIED RESULT")
        self.btn_marker_result.setStyleSheet("background-color: #2e7d32; color: white;")
        self.btn_marker_result.clicked.connect(self.show_unified_result_marker)
        
        marker_sublayout.addWidget(QLabel("Target Arm:"))
        marker_sublayout.addWidget(self.marker_arm_sel)
        marker_sublayout.addWidget(QLabel("Sweep Target:"))
        marker_sublayout.addWidget(self.marker_axis_sel)
        marker_sublayout.addLayout(tol_lay)
        marker_sublayout.addWidget(self.cb_head_tracking)
        marker_sublayout.addWidget(self.btn_marker_ready)
        marker_sublayout.addWidget(self.btn_marker_center_head)
        marker_sublayout.addWidget(self.btn_marker_center)
        marker_sublayout.addWidget(self.btn_marker_start)
        marker_sublayout.addWidget(self.btn_marker_result)
        marker_subtab.setLayout(marker_sublayout)
        self.workflow_tabs.addTab(marker_subtab, "2. Marker Calib")
        
        workflow_layout.addWidget(self.workflow_tabs)
        workflow_box.setLayout(workflow_layout)
        left_panel.addWidget(workflow_box)
        
        # Quit
        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setStyleSheet("background-color: #b71c1c; color: white;")
        self.btn_quit.clicked.connect(self.close)
        left_panel.addWidget(self.btn_quit)
        left_panel.addStretch()
        
        # Right Panel (Console Console Log)
        right_panel = QVBoxLayout()
        console_title = QLabel("System Log / Execution Console")
        console_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        console_title.setStyleSheet("color: #2979ff; margin-bottom: 2px;")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("background-color: #0e0e0e; color: #00e676; border: 2px solid #2d2d2d; border-radius: 6px;")
        right_panel.addWidget(console_title)
        right_panel.addWidget(self.log_text)
        
        calib_layout.addLayout(left_panel, 1)
        calib_layout.addLayout(right_panel, 2)
        calib_tab.setLayout(calib_layout)
        self.tabs.addTab(calib_tab, "Main Calibration Panel")
        
        # --- Tab 2: Plot Viewer ---
        self.plot_tab = QWidget()
        plot_layout = QHBoxLayout()
        
        self.plot_label_left = QLabel("Circle Fit / Yaw Sweep Plot")
        self.plot_label_left.setAlignment(Qt.AlignCenter)
        self.plot_label_left.setStyleSheet("background-color: #1a1a1a; color: #888888; border: 2px solid #2d2d2d; border-radius: 8px;")
        
        self.plot_label_right = QLabel("Circle Fit / Pitch Sweep Plot")
        self.plot_label_right.setAlignment(Qt.AlignCenter)
        self.plot_label_right.setStyleSheet("background-color: #1a1a1a; color: #888888; border: 2px solid #2d2d2d; border-radius: 8px;")
        
        plot_layout.addWidget(self.plot_label_left)
        plot_layout.addWidget(self.plot_label_right)
        self.plot_tab.setLayout(plot_layout)
        self.tabs.addTab(self.plot_tab, "Interactive Plot Viewer")
        
        self.setLayout(main_layout)
        
        # Startup info
        self.log_msg("="*60)
        self.log_msg("  UNIFIED CALIBRATION TOOL LOADED SUCCESSFULLY")
        self.log_msg("="*60)
        self.log_msg("[RECOMMENDED SEQUENCE]")
        self.log_msg("  1. Calibrate joint offsets first using '1. Joint Calib' subtab.")
        self.log_msg("  2. Perform marker bracket sweeps using '2. Marker Calib' subtab.")
        self.log_msg("  3. Calibrate head yaw/pitch joint offsets after marker calibration.")
        self.log_msg("="*60)

    # --- Common Helper Functions ---
    def log_msg(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def connect_robot(self):
        if self.robot:
            self.log_msg("[INFO] Disconnecting from robot...")
            if not self.ui_only:
                MarkerCalibrator.terminate_robot(self.robot)
            self.robot = None
            self.marker_calibrator.robot = None
            self.joint_calibrator.robot = None
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold;")
            self.log_msg("[INFO] Robot disconnected.")
            return

        try:
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip()
            self.log_msg(f"[INFO] Connecting to robot at {addr} ({model})...")
            
            if self.ui_only:
                self.robot = "mock_robot"
                self.log_msg("[MOCK] Connected successfully to mock robot.")
            else:
                self.robot = MarkerCalibrator.initialize_robot(addr, model)
                
            if self.robot:
                self.marker_calibrator.robot = self.robot
                self.joint_calibrator.robot = self.robot
                self.log_msg("[INFO] Robot successfully connected and initialized.")
                self.btn_connect.setText("DISCONNECT")
                self.btn_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold;")
            else:
                self.log_msg("[ERROR] Robot initialization failed. Check IP.")
        except Exception as e:
            self.log_msg(f"[ERROR] Connection failure: {e}")

    def on_arm_side_changed(self, text):
        new_side = "left" if "Left" in text else "right"
        if self.arm_side != new_side:
            self.arm_side = new_side
            self.log_msg(f"[INFO] Changed active arm to {self.arm_side.upper()}. Cleared loaded datasets.")
            self.marker_data_5 = None
            self.marker_data_6 = None
            self.joint_sweep_data = None
            
            # Sync dropdown indexes between workflow tabs
            self.joint_arm_sel.blockSignals(True)
            self.marker_arm_sel.blockSignals(True)
            idx = 1 if self.arm_side == "left" else 0
            self.joint_arm_sel.setCurrentIndex(idx)
            self.marker_arm_sel.setCurrentIndex(idx)
            self.joint_arm_sel.blockSignals(False)
            self.marker_arm_sel.blockSignals(False)

    def on_monitor_toggled(self, checked):
        if checked:
            self.btn_monitor.setText("Marker Monitor: ON")
            self.btn_monitor.setStyleSheet("background-color: #ffeb3b; color: black; font-weight: bold;")
        else:
            self.btn_monitor.setText("Marker Monitor: OFF")
            self.btn_monitor.setStyleSheet("")

    def update_marker_indicator(self, detected):
        self.indicator.set_detected(detected)
        if detected:
            self.status_label.setText("Detected")
            self.status_label.setStyleSheet("color: #00e676;")
        else:
            self.status_label.setText("Not Detected")
            self.status_label.setStyleSheet("color: #ff1744;")

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

    def move_head_to_zero(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        self.log_msg("[INFO] Centering head to zero...")
        if self.ui_only:
            self.log_msg("[MOCK] Head centered to zero.")
            return

        try:
            head_zero = np.zeros(len(self.robot.model().head_idx))
            self.set_controls_enabled(False)
            
            class CenterHeadWorker(QThread):
                done_sig = Signal()
                def __init__(self, calibrator, robot, q_zero):
                    super().__init__()
                    self.calibrator = calibrator
                    self.robot = robot
                    self.q_zero = q_zero
                def run(self):
                    self.calibrator.movej(self.robot, head=self.q_zero, minimum_time=3.0)
                    self.done_sig.emit()
            
            self.head_worker = CenterHeadWorker(self.joint_calibrator, self.robot, head_zero)
            self.head_worker.done_sig.connect(self.on_action_finished)
            self.head_worker.start()
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to center head: {e}")
            self.on_action_finished()

    def set_controls_enabled(self, enabled):
        self.btn_joint_ready.setEnabled(enabled)
        self.btn_joint_center_head.setEnabled(enabled)
        self.btn_joint_center.setEnabled(enabled)
        self.btn_joint_start.setEnabled(enabled)
        self.btn_joint_result.setEnabled(enabled)
        
        self.btn_marker_ready.setEnabled(enabled)
        self.btn_marker_center_head.setEnabled(enabled)
        self.btn_marker_center.setEnabled(enabled)
        self.btn_marker_start.setEnabled(enabled)
        self.btn_marker_result.setEnabled(enabled)

    def on_action_finished(self):
        self.set_controls_enabled(True)
        if hasattr(self, 'stop_event_mc'):
            self.stop_event_mc.clear()

    # --- Joint Calibration Workflows ---
    def move_to_ready_pose_joint(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if self.ui_only:
            self.log_msg("[MOCK] Moved to joint ready pose.")
            return

        mode_str = self.joint_mode_sel.currentText()
        mode = "wrist_pitch" if "wrist_pitch" in mode_str else ("elbow" if "elbow" in mode_str else "head")
        
        self.set_controls_enabled(False)
        self.ready_worker = MoveToReadyWorker(self.joint_calibrator, self.arm_side, mode)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_action_finished)
        self.ready_worker.start()

    def move_to_center_joint(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if self.ui_only:
            self.log_msg("[MOCK] Moved to center (Joint mode).")
            return

        if hasattr(self, 'active_worker') and self.active_worker and self.active_worker.isRunning():
            self.log_msg("[INFO] Cancelling Move to Center...")
            self.stop_event_mc.set()
            return

        if not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not visible. Teach target position first.")
            return

        mode_str = self.joint_mode_sel.currentText()
        target_dist = 200.0 if "head" in mode_str else 300.0
        self.log_msg(f"[INFO] Move to Center (Joint Calibration) -> target: {target_dist} mm")

        self.btn_joint_center.setText("CANCEL")
        self.btn_joint_center.setStyleSheet("background-color: #b71c1c; color: white;")
        self.set_controls_enabled(False)
        self.btn_joint_center.setEnabled(True)

        import threading
        self.stop_event_mc = threading.Event()
        self.active_worker = MoveCenterWorker(self.joint_calibrator, self.arm_side, self.stop_event_mc, target_dist=target_dist)
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_move_center_finished_joint)
        self.active_worker.start()

    def on_move_center_finished_joint(self):
        self.btn_joint_center.setText("MOVE TO CENTER")
        self.btn_joint_center.setStyleSheet("background-color: #00838f; color: white;")
        self.on_action_finished()

    def start_calibration_joint(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if not self.ui_only and not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not visible.")
            return

        mode_str = self.joint_mode_sel.currentText()
        mode = "wrist_pitch" if "wrist_pitch" in mode_str else ("elbow" if "elbow" in mode_str else "head")

        self.set_controls_enabled(False)
        if not self.ui_only and hasattr(self, 'poll_timer'):
            self.poll_timer.stop()

        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Joint Sweep: {mode.upper()}")
        
        self.active_worker = JointCalibrationWorker(self.joint_calibrator, self.arm_side, mode, ui_only=self.ui_only)
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.finished_signal.connect(self.on_calibration_finished_joint)
        self.active_worker.start()

    def on_calibration_finished_joint(self, res):
        self.on_action_finished()
        if not self.ui_only and hasattr(self, 'poll_timer'):
            self.poll_timer.start(200)

        if res:
            self.joint_sweep_data = res
            
            # Update Plot viewer
            if 'plot_path_left' in res and os.path.exists(res['plot_path_left']):
                pix_l = QPixmap(res['plot_path_left']).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.plot_label_left.setPixmap(pix_l)
            if 'plot_path_right' in res and os.path.exists(res['plot_path_right']):
                pix_r = QPixmap(res['plot_path_right']).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.plot_label_right.setPixmap(pix_r)
                
            self.tabs.setCurrentIndex(1) # Auto swap to plot tab
        else:
            self.log_msg("[ERROR] Joint sweep failed or aborted.")

    def show_result_joint(self):
        self.log_text.clear()
        self.log_msg("\n" + "="*50)
        self.log_msg("       JOINT CALIBRATION ESTIMATED RESULTS")
        self.log_msg("="*50)
        
        if not self.joint_sweep_data:
            self.log_msg("\n[ERROR] No joint sweep data loaded! Perform a sweep first.")
            return

        mode = self.joint_sweep_data['mode']
        self.log_msg(f"\n[1] Calibration Target: {mode}")
        
        if mode == "head":
            self.log_msg(f"    - Estimated Head Yaw Offset  : {self.joint_sweep_data['opt_yaw']:.4f} deg")
            self.log_msg(f"    - Estimated Head Pitch Offset: {self.joint_sweep_data['opt_pitch']:.4f} deg")
            self.log_msg(f"    - Alignment Residual RMSE    : {self.joint_sweep_data['rmse']:.3f} mm")
            
            self.log_msg("\n[2] setting.yaml Head Update suggestion:")
            self.log_msg("  (Sum these offsets with your current physical zero offsets)")
            self.log_msg(f"  Head Yaw Calibration Change  : {self.joint_sweep_data['opt_yaw']:.4f} deg")
            self.log_msg(f"  Head Pitch Calibration Change: {self.joint_sweep_data['opt_pitch']:.4f} deg")
        else:
            self.log_msg(f"    - Target Swept Joint       : {'Joint 5' if mode == 'wrist_pitch' else 'Joint 3'}")
            self.log_msg(f"    - Estimated Optimal Offset : {self.joint_sweep_data['optimal_offset']:.4f} deg")
            self.log_msg(f"    - Circle A Fitting RMSE     : {self.joint_sweep_data['rmse_A']:.4f} mm")
            self.log_msg(f"    - Circle B Fitting RMSE     : {self.joint_sweep_data['rmse_B']:.4f} mm")
            
            self.log_msg("\n[2] Sugested Joint Home Offset update:")
            self.log_msg(f"  Add offset: {self.joint_sweep_data['optimal_offset']:.4f} deg to calibration config.")
        self.log_msg("="*50)

    # --- Marker Bracket Calibration Workflows ---
    def move_to_ready_pose_marker(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if self.ui_only:
            self.log_msg("[MOCK] Moved to marker ready pose.")
            return

        self.set_controls_enabled(False)
        self.ready_worker = MoveToReadyWorker(self.marker_calibrator, self.arm_side)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_action_finished)
        self.ready_worker.start()

    def move_to_center_marker(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if self.ui_only:
            self.log_msg("[MOCK] Moved to center (Marker mode).")
            return

        if hasattr(self, 'active_worker') and self.active_worker and self.active_worker.isRunning():
            self.log_msg("[INFO] Cancelling Move to Center...")
            self.stop_event_mc.set()
            return

        if not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not visible. Teach target position first.")
            return

        axis_str = self.marker_axis_sel.currentText()
        target_dist = 180.0 if "Axis 6" in axis_str else 200.0
        self.log_msg(f"[INFO] Move to Center (Marker Calibration) -> target: {target_dist} mm")

        self.btn_marker_center.setText("CANCEL")
        self.btn_marker_center.setStyleSheet("background-color: #b71c1c; color: white;")
        self.set_controls_enabled(False)
        self.btn_marker_center.setEnabled(True)

        import threading
        self.stop_event_mc = threading.Event()
        self.active_worker = MoveCenterWorker(self.marker_calibrator, self.arm_side, self.stop_event_mc, target_dist=target_dist)
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_move_center_finished_marker)
        self.active_worker.start()

    def on_move_center_finished_marker(self):
        self.btn_marker_center.setText("MOVE TO CENTER")
        self.btn_marker_center.setStyleSheet("background-color: #00838f; color: white;")
        self.on_action_finished()

    def start_calibration_marker(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if not self.ui_only and not self.indicator.is_detected:
            QMessageBox.warning(self, "Marker Not Detected", "Marker is not visible.")
            return

        axis_mode = 6 if "6" in self.marker_axis_sel.currentText() else 5
        use_head = self.cb_head_tracking.isChecked()

        self.set_controls_enabled(False)
        if not self.ui_only and hasattr(self, 'poll_timer'):
            self.poll_timer.stop()

        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Marker Sweep: Axis {axis_mode} (Head Tracking: {use_head})")

        if self.ui_only:
            # Simulate marker sweep results
            class MockMarkerWorker(QThread):
                log_sig = Signal(str)
                finished_sig = Signal(dict)
                def __init__(self, axis_mode):
                    super().__init__()
                    self.axis_mode = axis_mode
                def run(self):
                    time.sleep(1.0)
                    self.log_sig.emit("[MOCK] Running sweeps...")
                    theta = np.linspace(-np.pi/6, np.pi/6, 11)
                    if self.axis_mode == 6:
                        pts = np.column_stack((np.cos(theta)*74.8 + 0.1, np.sin(theta)*74.8 - 0.2))
                        res = {
                            'axis_mode': 6,
                            'radius': 74.85,
                            'rmse': 0.12,
                            'axis': np.array([0.01, -0.999, 0.02]),
                            'center': np.array([0.5, 1.2, 0.3]),
                            'pts_2d': pts,
                            'uc_opt': 0.1,
                            'vc_opt': -0.2,
                            'tilt': 89.48,
                            'yaw': 1.4,
                            'tilt_list': [89.48]*11
                        }
                    else:
                        pts = np.column_stack((np.cos(theta)*280.2 + 0.5, np.sin(theta)*280.2 + 0.4))
                        res = {
                            'axis_mode': 5,
                            'radius': 280.15,
                            'rmse': 0.18,
                            'axis': np.array([0.02, 0.03, 0.999]),
                            'center': np.array([1.2, 0.8, -0.4]),
                            'pts_2d': pts,
                            'uc_opt': 0.5,
                            'vc_opt': 0.4,
                            'tilt': 0.25,
                            'yaw': -20.68,
                            'tilt_list': [0.25]*11
                        }
                    
                    plt.figure(figsize=(5, 5))
                    plt.scatter(pts[:, 0], pts[:, 1], c='b')
                    plt.gca().add_patch(plt.Circle((res['uc_opt'], res['vc_opt']), res['radius'], color='r', fill=False))
                    plt.title(f"Mock Axis {self.axis_mode}")
                    plt.grid(True)
                    plot_path = os.path.join(os.path.dirname(__file__), f"circle_fit_axis_{self.axis_mode}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    res['plot_path'] = plot_path
                    self.finished_sig.emit(res)
            
            self.active_worker = MockMarkerWorker(axis_mode)
            self.active_worker.log_sig.connect(self.log_msg)
            self.active_worker.finished_sig.connect(self.on_calibration_finished_marker)
            self.active_worker.start()
        else:
            self.active_worker = MarkerCalibrationWorker(self.marker_calibrator, self.arm_side, axis_mode, use_head_tracking=use_head)
            self.active_worker.log_signal.connect(self.log_msg)
            self.active_worker.status_signal.connect(self.update_marker_indicator)
            self.active_worker.finished_signal.connect(self.on_calibration_finished_marker)
            self.active_worker.start()

    def on_calibration_finished_marker(self, res):
        self.on_action_finished()
        if not self.ui_only and hasattr(self, 'poll_timer'):
            self.poll_timer.start(200)

        if res:
            if res['axis_mode'] == 6:
                self.marker_data_6 = res
                if 'plot_path' in res and os.path.exists(res['plot_path']):
                    pix = QPixmap(res['plot_path']).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.plot_label_left.setPixmap(pix)
            else:
                self.marker_data_5 = res
                if 'plot_path' in res and os.path.exists(res['plot_path']):
                    pix = QPixmap(res['plot_path']).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.plot_label_right.setPixmap(pix)
            
            self.tabs.setCurrentIndex(1) # Auto swap to plot tab
        else:
            self.log_msg("[ERROR] Marker sweep failed.")

    def get_link_length(self):
        if self.ui_only:
            return 300.0
        if not self.robot: return 0.0
        try:
            dyn_model = self.robot.get_dynamics()
            names = self.robot.model().robot_joint_names
            state = dyn_model.make_state(
                [f"link_{self.arm_side}_arm_5", f"ee_{self.arm_side}"],
                names
            )
            state.set_q(self.robot.get_state().position)
            dyn_model.compute_forward_kinematics(state)
            T = dyn_model.compute_transformation(state, 0, 1)
            return np.linalg.norm(T[:3, 3]) * 1000.0 # m to mm
        except Exception as e:
            self.log_msg(f"[WARN] Failed to get link kinematics: {e}")
            return 0.0

    def show_unified_result_marker(self):
        self.log_text.clear()
        self.log_msg("\n" + "="*50)
        self.log_msg("       UNIFIED BRACKET CALIBRATION RESULTS")
        self.log_msg("="*50)
        
        if not self.marker_data_5 or not self.marker_data_6:
            self.log_msg("\n[ERROR] Missing dataset!")
            if not self.marker_data_6: self.log_msg(" -> Axis 6 Sweep (Yaw) data is missing.")
            if not self.marker_data_5: self.log_msg(" -> Axis 5 Sweep (Pitch) data is missing.")
            return
            
        L_5_ee = self.get_link_length()
        if L_5_ee <= 0:
            L_5_ee = 300.0 # Fallback

        # Vector Math: Align measured axes with nominal EE axes
        z_e_in_m = self.marker_data_6['axis']
        y_e_in_m = self.marker_data_5['axis']
        
        if self.arm_side == "left":
            ideal_rpy = [90.0, 0.0, 0.0]
        else:
            ideal_rpy = [90.0, 0.0, 180.0]
            
        T_ee_m_ideal = self.marker_calibrator.make_transform([0, 0, 0] + ideal_rpy)
        R_ee_m_ideal = T_ee_m_ideal[:3, :3]
        
        # Check directions
        if np.dot(z_e_in_m, R_ee_m_ideal[:, 2]) < 0: z_e_in_m = -z_e_in_m
        if np.dot(y_e_in_m, R_ee_m_ideal[:, 1]) < 0: y_e_in_m = -y_e_in_m
        
        # Orthogonalize
        y_e_in_m = y_e_in_m - np.dot(y_e_in_m, z_e_in_m) * z_e_in_m
        y_e_in_m /= np.linalg.norm(y_e_in_m)
        x_e_in_m = np.cross(y_e_in_m, z_e_in_m)
        
        R_ee_m_actual = np.column_stack((x_e_in_m, y_e_in_m, z_e_in_m))
        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        
        radius_6 = self.marker_data_6['radius']
        radius_5 = self.marker_data_5['radius']
        
        # Offset translation
        x_e = 0.0
        y_e = radius_6 if self.arm_side == "left" else -radius_6
        z_e = -abs(radius_5 - L_5_ee)
        
        self.log_msg("\n[1] Cartesian Offset (EE Link Frame)")
        self.log_msg(f"    - X-Offset: {x_e:.2f} mm")
        self.log_msg(f"    - Y-Offset: {y_e:.2f} mm")
        self.log_msg(f"    - Z-Offset: {z_e:.2f} mm")
        self.log_msg(f"       * (L_5_ee: {L_5_ee:.1f} mm, R6: {radius_6:.2f} mm, R5: {radius_5:.2f} mm)")
            
        self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
        self.log_msg(f"    - Roll : {roll_e:.2f} deg")
        self.log_msg(f"    - Pitch: {pitch_e:.2f} deg")
        self.log_msg(f"    - Yaw  : {yaw_e:.2f} deg")
        
        self.log_msg("\n[3] setting.yaml Config Update values:")
        x_m, y_m, z_m = x_e/1000.0, y_e/1000.0, z_e/1000.0
        
        if self.arm_side == "left":
            self.log_msg(f"  Tf_to_marker_left:  [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {roll_e:.2f}, {pitch_e:.2f}, {yaw_e:.2f}]")
        else:
            self.log_msg(f"  Tf_to_marker_right: [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {roll_e:.2f}, {pitch_e:.2f}, {yaw_e:.2f}]")
            
        # Analysis Orthogonality / Quality metrics
        dot_val = np.dot(z_e_in_m, y_e_in_m)
        angle_between = np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0)))
        ortho_err = abs(90.0 - angle_between)
        
        rmse_6 = self.marker_data_6['rmse']
        rmse_5 = self.marker_data_5['rmse']
        
        self.log_msg(f"\n[4] Confidence Metrics:")
        self.log_msg(f"    - Orthogonality Error  : {ortho_err:.3f} deg")
        self.log_msg(f"    - Fitting RMSE (A6/A5) : {rmse_6:.3f} / {rmse_5:.3f} mm")
        
        if rmse_6 > 0.5 or rmse_5 > 0.5:
            self.log_msg("\n" + "!"*60)
            self.log_msg(" [WARNING] Fitting RMSE exceeds 0.5 mm!")
            self.log_msg("  The marker coordinates may have high noise. Check hardware.")
            self.log_msg("!"*60)
        self.log_msg("="*50)


def main():
    parser = argparse.ArgumentParser(description="Unified Robot Joint & Marker Bracket Calibration GUI")
    parser.add_argument("--ui", action="store_true", help="Start only UI for debugging/simulation")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    robot = None
    marker_st = None

    if not args.ui:
        print("[INFO] Initializing Camera Marker Transform System...")
        try:
            from marker_detection import Marker_Transform
            marker_st = Marker_Transform()
            marker_st.marker_detection.set_marker_type("plate")
        except Exception as e:
            print(f"[ERROR] Failed to load camera marker system: {e}")
            print("[INFO] Fallback to UI-only mode.")
            args.ui = True
    else:
        print("[INFO] Starting in simulation (UI-only) mode.")

    gui = UnifiedCalibrationApp(marker_st, robot, "right", ui_only=args.ui)
    gui.show()
    
    try:
        sys.exit(app.exec())
    finally:
        if marker_st:
            try:
                marker_st.camera.stream_off()
                print("Camera resource released.")
            except Exception:
                pass

if __name__ == "__main__":
    main()
