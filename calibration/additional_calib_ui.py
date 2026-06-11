import sys
import os
# Ensure local core module is imported first
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
core_dir = os.path.abspath(os.path.join(parent_dir, "core"))
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

import cv2
import numpy as np
import time
import argparse
import logging
import rby1_sdk as rby
import threading

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QCheckBox, 
                             QLineEdit, QDialog, QMessageBox, QTabWidget, QInputDialog, QGridLayout,
                             QTableWidget, QHeaderView, QTableWidgetItem, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy

# Import custom calibrator logic
from Calibrator import MarkerCalibrator, JointCalibrator
from IntrinsicsCalibrator import IntrinsicsCalibrator

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
QTextEdit {
    background-color: #0e0e0e;
    color: #00e676;
    border: 2px solid #2d2d2d;
    border-radius: 6px;
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

class CameraFeedDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Live Feed")
        self.resize(640, 480)
        
        layout = QVBoxLayout(self)
        self.lbl_feed = QLabel("Waiting for camera frame...")
        self.lbl_feed.setAlignment(Qt.AlignCenter)
        self.lbl_feed.setStyleSheet("background-color: black; color: white; border: 1px solid #2d2d2d; border-radius: 4px;")
        self.lbl_feed.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.lbl_feed)
        
    def closeEvent(self, event):
        if self.parent() and hasattr(self.parent(), "on_feed_dialog_closed"):
            self.parent().on_feed_dialog_closed()
        super().closeEvent(event)

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

class ManualHeadWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, calibrator, yaw_rad, pitch_rad):
        super().__init__()
        self.calibrator = calibrator
        self.yaw_rad = yaw_rad
        self.pitch_rad = pitch_rad

    def run(self):
        try:
            ok = self.calibrator.movej(
                self.calibrator.robot, 
                head=np.array([self.yaw_rad, self.pitch_rad]), 
                minimum_time=1.5
            )
            if ok:
                self.log_signal.emit("[MANUAL HEAD] Move head completed successfully.")
            else:
                self.log_signal.emit("[ERROR] Failed manual head move: command rejected by robot.")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Failed manual head move: {e}")
        self.finished_signal.emit()

# --- Specialized Calibration Workers ---
class MarkerCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)
    
    def __init__(self, calibrator, arm_side, use_head_tracking=True, tolerance=0.5):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.use_head_tracking = use_head_tracking
        self.tolerance = tolerance
        
    def run(self):
        try:
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit("   STARTING TWO-STAGE UNIFIED MARKER SWEEP")
            self.log_signal.emit("   [Stage 1/2] Sweeping Axis 6 (Roll)...")
            self.log_signal.emit("="*50 + "\n")
            
            # 1. Axis 6 sweep
            res_6 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 6, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking
            )
            if not res_6:
                self.log_signal.emit("[ERROR] Stage 1 (Axis 6) sweep failed. Aborting.")
                self.finished_signal.emit(None)
                return
                
            res_6['axis_mode'] = 6
            res_6['axis'] = res_6['axis_opt']
            
            if getattr(self.calibrator, 'stop_requested', False):
                self.finished_signal.emit(None)
                return
                
            time.sleep(1.0)
            
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit("   [Stage 2/2] Sweeping Axis 5 (Pitch)...")
            self.log_signal.emit("="*50 + "\n")
            
            # 2. Axis 5 sweep
            res_5 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 5, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking
            )
            if not res_5:
                self.log_signal.emit("[ERROR] Stage 2 (Axis 5) sweep failed. Aborting.")
                self.finished_signal.emit(None)
                return
                
            res_5['axis_mode'] = 5
            res_5['axis'] = res_5['axis_opt']
            
            # 3. Compute unified bracket calibration
            self.log_signal.emit("\n[PROCESSING] Computing unified bracket calibration parameters...")
            unified_res = self.calibrator.compute_unified_bracket_calibration(res_5, res_6, self.arm_side, tolerance=self.tolerance)
            
            unified_res['res_5'] = res_5
            unified_res['res_6'] = res_6
            
            # 4. Generate combined single plot (1 row, 2 columns)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            def plot_single_axis(ax, res, axis_num, color):
                ax.scatter(res['pts_2d'][:, 0], res['pts_2d'][:, 1], c=color, label='Captured Points')
                circle = plt.Circle((res['uc_opt'], res['vc_opt']), res['radius'], color='r', fill=False, label='Fitted Circle')
                ax.add_patch(circle)
                ax.plot(res['uc_opt'], res['vc_opt'], 'rx', label='Center')
                
                x_min, x_max = res['pts_2d'][:, 0].min(), res['pts_2d'][:, 0].max()
                y_min, y_max = res['pts_2d'][:, 1].min(), res['pts_2d'][:, 1].max()
                span = max(x_max - x_min, y_max - y_min)
                margin = max(1.0, span * 0.5)
                cx = (x_max + x_min) / 2
                cy = (y_max + y_min) / 2
                ax.set_xlim(cx - span/2 - margin, cx + span/2 + margin)
                ax.set_ylim(cy - span/2 - margin, cy + span/2 + margin)
                ax.set_aspect('equal')
                ax.grid(True)
                ax.set_title(f"Axis {axis_num} Sweep (Radius: {res['radius']:.2f}mm, RMSE: {res['rmse']:.3f})")
                ax.legend()
            
            plot_single_axis(ax1, res_6, 6, 'blue')
            plot_single_axis(ax2, res_5, 5, 'green')
            
            fig.suptitle(f"Unified Marker Sweep Results ({self.arm_side.upper()} Arm)\n"
                         f"Y-Offset: {unified_res['y_e']:.2f} mm | Z-Offset: {unified_res['z_e']:.2f} mm\n"
                         f"Roll: {unified_res['roll_e']:.2f}° | Pitch: {unified_res['pitch_e']:.2f}°", fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_img")
            os.makedirs(result_dir, exist_ok=True)
            plot_path = os.path.join(result_dir, f"circle_fit_{self.arm_side}_marker_unified.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            unified_res['plot_path_combined'] = plot_path
            self.finished_signal.emit(unified_res)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(None)

class JointCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)

    def __init__(self, calibrator, arm_side, mode, ui_only=False, current_offset_deg=0.0, sweep_duration=15.0):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode
        self.ui_only = ui_only
        self.current_offset_deg = current_offset_deg
        self.sweep_duration = sweep_duration

    def run(self):
        try:
            if self.ui_only:
                # Mock Mode Execution
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Starting Sweep Steps...")
                for i in range(5):
                    self.log_signal.emit(f"  [STEP {i+1}/5] Mock calibration step executed.")
                    time.sleep(0.2)
                
                theta = np.linspace(-np.pi, np.pi, 20)
                pts_2d_A = np.column_stack((np.cos(theta)*45 + 1.0, np.sin(theta)*45 + 2.0))
                pts_2d_B = np.column_stack((np.cos(theta)*43 - 0.5, np.sin(theta)*43 + 1.5))
                opt_offset = -1.5 + self.current_offset_deg
                res = {
                    'mode': self.mode,
                    'optimal_offset': opt_offset,
                    'recommended_joint_offset': opt_offset,
                    'converged': True,
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
                res = self.calibrator.perform_3step_joint_calibration(
                    self.arm_side, self.mode,
                    log_callback=self.log_signal.emit, 
                    status_callback=self.status_signal.emit,
                    current_offset_deg=self.current_offset_deg,
                    sweep_duration=self.sweep_duration
                )

            if res:
                self.log_signal.emit("-" * 30)
                self.log_signal.emit(f"  [1] Calibration Target: {self.mode}")
                recommended = res.get('recommended_joint_offset', res['optimal_offset'])
                self.log_signal.emit(f"      Estimated Optimal Offset: {recommended:.3f} deg")
                # self.log_signal.emit(f"      Sweep RMSE (A/B)       : {res['rmse_A']:.3f} / {res['rmse_B']:.3f}")
                self.log_signal.emit("-" * 30)
                self.log_signal.emit("\n[CALIBRATION COMPLETE]\n")
                
                # result_img 디렉토리 아래에 저장하도록 수정
                result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_img")
                os.makedirs(result_dir, exist_ok=True)
                plot_filename = f"circle_fit_{self.arm_side}_{self.mode}_joint_calib.png"
                plot_path_combined = os.path.join(result_dir, plot_filename)
                
                if 'plot_path_combined' in res and os.path.exists(res['plot_path_combined']):
                    self.log_signal.emit(f"      [INFO] Detailed 2x2 comparison plot already exists: {res['plot_path_combined']}")
                else:
                    plt.figure(figsize=(10, 5))
                    
                    # Trajectory representations based on mode
                    if self.mode == "wrist_pitch":
                        title_A = f"Joint 4 (Wrist Yaw) Sweep [Axis A] (RMSE: {res['rmse_A']:.3f})"
                        title_B = f"Joint 6 (Wrist Roll) Sweep [Axis B] (RMSE: {res['rmse_B']:.3f})"
                    else: # elbow mode
                        title_A = f"Joint 2 (Shoulder Pitch) Sweep [Axis A] (RMSE: {res['rmse_A']:.3f})"
                        title_B = f"Joint 4 (Elbow Yaw) Sweep [Axis B] (RMSE: {res['rmse_B']:.3f})"
                        
                    # Subplot 1: Circle A
                    plt.subplot(1, 2, 1)
                    plt.scatter(res['pts_2d_A'][:, 0], res['pts_2d_A'][:, 1], c='r', label='Sweep Axis A')
                    circle_A = plt.Circle((res['uc_A'], res['vc_A']), res['r_A'], color='r', fill=False, label='Fit A')
                    plt.gca().add_patch(circle_A)
                    plt.plot(res['uc_A'], res['vc_A'], 'rx', markersize=8)
                    plt.title(title_A)
                    plt.xlabel("U coord (mm) [Fitting Local Plane X]")
                    plt.ylabel("V coord (mm) [Fitting Local Plane Y]")
                    plt.gca().set_aspect('equal')
                    plt.grid(True)
                    plt.legend()
                    
                    # Subplot 2: Circle B
                    plt.subplot(1, 2, 2)
                    plt.scatter(res['pts_2d_B'][:, 0], res['pts_2d_B'][:, 1], c='b', label='Sweep Axis B')
                    circle_B = plt.Circle((res['uc_B'], res['vc_B']), res['r_B'], color='b', fill=False, label='Fit B')
                    plt.gca().add_patch(circle_B)
                    plt.plot(res['uc_B'], res['vc_B'], 'bx', markersize=8)
                    plt.title(title_B)
                    plt.xlabel("U coord (mm) [Fitting Local Plane X]")
                    plt.ylabel("V coord (mm) [Fitting Local Plane Y]")
                    plt.gca().set_aspect('equal')
                    plt.grid(True)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(plot_path_combined, dpi=120)
                    plt.close()
                    res['plot_path_combined'] = plot_path_combined
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
        self.joint_calibrator = JointCalibrator(marker_st, robot)
        
        # Intrinsics Calibrator (Tab 3 용)
        self.intrinsics_calibrator = IntrinsicsCalibrator()
        # Default: 8x5 squares, 36mm x 27mm, DICT_5X5_100
        self.intrinsics_calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 36.0, 27.0, "DICT_5X5_100")
        
        try:
            from marker_detection import Marker_Detection
            self.marker_detector = Marker_Detection()
            self.marker_detector.set_marker_type("plate")
        except ImportError:
            self.marker_detector = None
            
        self.monitor_enabled = False
        self.captured_images = []
        self.output_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "camera_intrinsics.yaml"))
        
        # Saved Calibration Results
        self.marker_data_5 = None
        self.marker_data_6 = None
        self.joint_sweep_data = None
        
        # Cumulative Joint Offsets for iterative sweeps
        self.joint_offsets = {"wrist_pitch": 0.0, "elbow": 0.0}
        self.load_offsets_from_yaml()
        
        self.recommended_joint_offset = None
        
        self.marker_calibrator.joint_offsets = self.joint_offsets
        self.joint_calibrator.joint_offsets = self.joint_offsets
        
        self.setWindowTitle("Unified Robot Calibration Suite")
        self.resize(1400, 800)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # 1. 200ms poll timer (탭 1, 2, 4 용)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera_status)
        
        # 2. 33ms video timer (탭 3용)
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        
        self.init_ui()
        self.load_bracket_design_values()
        self.update_applied_offset_label()
        
        # 초기화 시 탭 상태에 맞춰 타이머 활성화
        self.on_left_tab_changed(self.left_tabs.currentIndex())
        
        self.active_worker = None

    def load_offsets_from_yaml(self):
        import yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, "..", "config", "setting.yaml"))
        try:
            if not os.path.exists(config_path):
                self.joint_offsets_store = {
                    "left": {"joint5": 0.0, "joint3": 0.0},
                    "right": {"joint5": 0.0, "joint3": 0.0}
                }
                self.save_offsets_to_yaml()
            else:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict) and "joint_offset" in data:
                    self.joint_offsets_store = data["joint_offset"]
                else:
                    self.joint_offsets_store = {
                        "left": {"joint5": 0.0, "joint3": 0.0},
                        "right": {"joint5": 0.0, "joint3": 0.0}
                    }
                    self.save_offsets_to_yaml()
                    self.log_msg(f"[WARNING] Added default joint_offset to setting.yaml.")
            
            # Sync current offsets with active arm_side from YAML
            self.joint_offsets["wrist_pitch"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint5", 0.0)
            self.joint_offsets["elbow"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint3", 0.0)
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets
            
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to load/create setting.yaml: {e}")
            # Fallback default values
            self.joint_offsets_store = {
                "left": {"joint5": 0.0, "joint3": 0.0},
                "right": {"joint5": 0.0, "joint3": 0.0}
            }
            self.joint_offsets["wrist_pitch"] = 0.0
            self.joint_offsets["elbow"] = 0.0
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets

    def save_offsets_to_yaml(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, "..", "config", "setting.yaml"))
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            lines = []
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    lines = f.readlines()
            
            jo_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("joint_offset:"):
                    jo_idx = i
                    break

            if jo_idx == -1:
                # If joint_offset doesn't exist, we append it to the end of the file
                if lines and not lines[-1].endswith("\n"):
                    lines.append("\n")
                lines.append("joint_offset:\n")
                lines.append("  left:\n")
                lines.append(f"    joint3: {self.joint_offsets_store['left']['joint3']}\n")
                lines.append(f"    joint5: {self.joint_offsets_store['left']['joint5']}\n")
                lines.append("  right:\n")
                lines.append(f"    joint3: {self.joint_offsets_store['right']['joint3']}\n")
                lines.append(f"    joint5: {self.joint_offsets_store['right']['joint5']}\n")
            else:
                curr_arm = None
                i = jo_idx + 1
                while i < len(lines):
                    line = lines[i]
                    stripped = line.strip()
                    if not stripped:
                        i += 1
                        continue
                    # If we hit another top-level key, we stop
                    if not line.startswith(" ") and not line.startswith("\t") and stripped.endswith(":"):
                        break
                    
                    if stripped.startswith("left:"):
                        curr_arm = "left"
                    elif stripped.startswith("right:"):
                        curr_arm = "right"
                    
                    if curr_arm is not None:
                        if stripped.startswith("joint3:"):
                            indent = len(line) - len(line.lstrip())
                            lines[i] = " " * indent + f"joint3: {self.joint_offsets_store[curr_arm]['joint3']}\n"
                        elif stripped.startswith("joint5:"):
                            indent = len(line) - len(line.lstrip())
                            lines[i] = " " * indent + f"joint5: {self.joint_offsets_store[curr_arm]['joint5']}\n"
                    i += 1

            with open(config_path, "w") as f:
                f.writelines(lines)
            self.log_msg(f"[SUCCESS] Saved offsets permanently to setting.yaml!")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save setting.yaml: {e}")

    def on_cell_double_clicked(self, row, col):
        arm = "right" if row == 0 else "left"
        joint_key = "joint5" if col == 0 else "joint3"
        
        current_val = self.joint_offsets_store[arm][joint_key]
        new_val, ok = QInputDialog.getDouble(
            self, 
            "Manual Offset Override", 
            f"Enter manual staged offset for {arm.upper()} Arm {'Joint 5' if col==0 else 'Joint 3'} (degrees):", 
            current_val, -45.0, 45.0, 4
        )
        if ok:
            self.joint_offsets_store[arm][joint_key] = new_val
            self.update_applied_offset_label()
            self.log_msg(f"[MANUAL OVERRIDE] Staged {arm.upper()} Arm {'Joint 5' if col==0 else 'Joint 3'} offset manually to {new_val:.4f}°. (Not saved to disk yet. Click APPLY OFFSET to save)")

    def init_ui(self):
        # Main horizontal split layout
        main_layout = QHBoxLayout()
        
        # --- Left Panel (TabWidget based!) ---
        self.left_tabs = QTabWidget()
        self.left_tabs.currentChanged.connect(self.on_left_tab_changed)
        
        # ==========================================
        # 1. Main Tab (로봇 동작 및 캘리브레이션 모듈)
        # ==========================================
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout()
        main_tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # Connection Box
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
        
        self.btn_stop_motion = QPushButton("STOP MOTION")
        self.btn_stop_motion.setStyleSheet("background-color: #ff1744; color: #ffffff; font-weight: bold;")
        self.btn_stop_motion.clicked.connect(self.stop_motion)
        
        conn_layout.addWidget(QLabel("IP / Port:"))
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(QLabel("Robot Model:"))
        conn_layout.addWidget(self.model_input)
        conn_layout.addWidget(self.btn_connect)
        conn_layout.addWidget(self.btn_stop_motion)
        conn_box.setLayout(conn_layout)
        main_tab_layout.addWidget(conn_box)
        
        # Target Arm Selection (Moved to Top-Level Control!)
        arm_box = QGroupBox("Target Arm Selection")
        arm_layout = QHBoxLayout()
        arm_layout.addWidget(QLabel("Active Arm Side:"))
        self.arm_sel = QComboBox()
        self.arm_sel.addItems(["Right Arm", "Left Arm"])
        idx = 1 if self.arm_side == "left" else 0
        self.arm_sel.setCurrentIndex(idx)
        self.arm_sel.currentTextChanged.connect(self.on_arm_side_changed)
        arm_layout.addWidget(self.arm_sel)
        arm_box.setLayout(arm_layout)
        main_tab_layout.addWidget(arm_box)
        
        # Bind legacy individual combo variables to the single top-level control to prevent attribute errors
        self.joint_arm_sel = self.arm_sel
        self.marker_arm_sel = self.arm_sel
        
        # Status Indicator
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
        
        btn_layout = QHBoxLayout()
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        
        self.btn_camera_feed = QPushButton("Camera Feed")
        self.btn_camera_feed.clicked.connect(self.toggle_camera_feed_dialog)
        
        btn_layout.addWidget(self.btn_monitor)
        btn_layout.addWidget(self.btn_camera_feed)
        
        self.temp_label = QLabel("Camera Temp: -- °C")
        
        status_layout.addLayout(ind_layout)
        status_layout.addLayout(btn_layout)
        status_layout.addWidget(self.temp_label)
        status_box.setLayout(status_layout)
        main_tab_layout.addWidget(status_box)
        
        # Sub-Workflow Selector Tabs
        workflow_box = QGroupBox("Calibration Workflows")
        workflow_layout = QVBoxLayout()
        self.workflow_tabs = QTabWidget()
        
        # Sub-tab 1: Joint Calibration
        joint_subtab = QWidget()
        joint_sublayout = QVBoxLayout()
        
        self.joint_mode_sel = QComboBox()
        self.joint_mode_sel.addItems(["wrist_pitch (5-Axis Sweep)", "elbow (3-Axis Sweep)"])
        self.joint_mode_sel.currentIndexChanged.connect(self.update_applied_offset_label)
        
        self.btn_joint_ready = QPushButton("MOVE TO READY")
        self.btn_joint_ready.setStyleSheet("background-color: #6a1b9a; color: white;")
        self.btn_joint_ready.clicked.connect(self.move_to_ready_pose_joint)
        
        self.btn_joint_start = QPushButton("START SWEEP")
        self.btn_joint_start.setStyleSheet("background-color: #1565c0; color: white;")
        self.btn_joint_start.clicked.connect(self.start_calibration_joint)
        
        joint_sublayout.addWidget(QLabel("Calibration Mode:"))
        joint_sublayout.addWidget(self.joint_mode_sel)
        joint_sublayout.addWidget(self.btn_joint_ready)
        joint_sublayout.addWidget(self.btn_joint_start)
        joint_subtab.setLayout(joint_sublayout)
        self.workflow_tabs.addTab(joint_subtab, "1. Joint Calib")
        
        # Sub-tab 2: Marker Bracket Calibration
        marker_subtab = QWidget()
        marker_sublayout = QVBoxLayout()
        
        self.marker_axis_sel = QComboBox()
        self.marker_axis_sel.addItems(["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"])
        
        tol_lay = QHBoxLayout()
        tol_lay.addWidget(QLabel("Tolerance (deg):"))
        self.tolerance_input = QLineEdit("0.5")
        self.tolerance_input.setFixedWidth(50)
        tol_lay.addWidget(self.tolerance_input)
        
        # self.cb_head_tracking = QCheckBox("Active Head Tracking")
        # self.cb_head_tracking.setChecked(True)
        
        self.btn_marker_ready = QPushButton("MOVE TO READY")
        self.btn_marker_ready.setStyleSheet("background-color: #6a1b9a; color: white;")
        self.btn_marker_ready.clicked.connect(self.move_to_ready_pose_marker)
        
        self.btn_marker_center = QPushButton("MOVE TO CENTER")
        self.btn_marker_center.setStyleSheet("background-color: #00838f; color: white;")
        self.btn_marker_center.clicked.connect(self.move_to_center_marker)
        
        self.btn_marker_start = QPushButton("START SWEEP")
        self.btn_marker_start.setStyleSheet("background-color: #1565c0; color: white;")
        self.btn_marker_start.clicked.connect(self.start_calibration_marker)
        
        self.btn_marker_result = QPushButton("UNIFIED RESULT")
        self.btn_marker_result.setStyleSheet("background-color: #2e7d32; color: white;")
        self.btn_marker_result.clicked.connect(self.show_unified_result_marker)
        
        marker_sublayout.addLayout(tol_lay)
        # marker_sublayout.addWidget(self.cb_head_tracking)
        marker_sublayout.addWidget(self.btn_marker_ready)
        marker_sublayout.addWidget(self.btn_marker_center)
        marker_sublayout.addWidget(self.btn_marker_start)
        marker_subtab.setLayout(marker_sublayout)
        self.workflow_tabs.addTab(marker_subtab, "2. Marker Calib")
        
        # Sub-tab 3: Head Control (원래 4번 탭, 이제 3번 탭)
        head_subtab = QWidget()
        head_sublayout = QVBoxLayout()
        
        head_group = QGroupBox("Manual Head Control")
        head_layout = QVBoxLayout()
        
        yaw_layout = QHBoxLayout()
        yaw_layout.addWidget(QLabel("Yaw (deg):"))
        self.txt_head_yaw = QLineEdit("0.0")
        self.txt_head_yaw.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 4px;")
        yaw_layout.addWidget(self.txt_head_yaw)
        
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch (deg):"))
        self.txt_head_pitch = QLineEdit("0.0")
        self.txt_head_pitch.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 4px;")
        pitch_layout.addWidget(self.txt_head_pitch)
        
        self.btn_move_head = QPushButton("MOVE HEAD")
        self.btn_move_head.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold;")
        self.btn_move_head.clicked.connect(self.move_head_manually)
        
        head_layout.addLayout(yaw_layout)
        head_layout.addLayout(pitch_layout)
        head_layout.addWidget(self.btn_move_head)
        head_group.setLayout(head_layout)
        
        head_sublayout.addWidget(head_group)
        head_sublayout.addStretch()
        head_subtab.setLayout(head_sublayout)
        self.workflow_tabs.addTab(head_subtab, "3. Head Control")
        
        workflow_layout.addWidget(self.workflow_tabs)
        workflow_box.setLayout(workflow_layout)
        main_tab_layout.addWidget(workflow_box)
        
        # Quit Button
        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setStyleSheet("background-color: #b71c1c; color: white;")
        self.btn_quit.clicked.connect(self.close)
        main_tab_layout.addWidget(self.btn_quit)
        main_tab_layout.addStretch()
        
        main_tab.setLayout(main_tab_layout)
        self.left_tabs.addTab(main_tab, "1. Main")
        
        # ==========================================
        # 2. Camera Tab (카메라 내부 파라미터 보정 전용)
        # ==========================================
        camera_tab = QWidget()
        camera_tab_layout = QHBoxLayout()
        
        int_left = QVBoxLayout()
        self.video_label = QLabel("Camera Feed Loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        int_left.addWidget(self.video_label, 3)
        
        instr_box = QGroupBox("Calibration Guidelines")
        instr_box.setStyleSheet("QGroupBox::title { color: #ff1744; font-weight: bold; }")
        instr_layout = QVBoxLayout()
        instructions = [
            "1. Ensure the calibration board is recognized correctly (green overlay).",
            "2. Tilt the board at various angles while capturing.",
            "3. Acquire data covering the entire camera field of view.",
            "4. Keep the board as steady as possible during each capture."
        ]
        for text in instructions:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #ff5252; font-weight: bold;")
            instr_layout.addWidget(lbl)
        instr_box.setLayout(instr_layout)
        int_left.addWidget(instr_box, 1)

        stats_box = QGroupBox("Capture Stats")
        stats_layout = QHBoxLayout()
        self.lbl_captured = QLabel("Captured Frames: 0")
        self.lbl_captured.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #2979ff;")
        
        self.lbl_temp = QLabel("Camera Temp: -- °C")
        self.lbl_temp.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #ff5500;")
        
        stats_layout.addWidget(self.lbl_captured)
        stats_layout.addStretch()
        stats_layout.addWidget(self.lbl_temp)
        stats_box.setLayout(stats_layout)
        int_left.addWidget(stats_box, 1)
        
        int_right = QVBoxLayout()
        
        monitor_box = QGroupBox("Marker 3D Monitoring")
        monitor_layout = QVBoxLayout()
        
        side_layout = QHBoxLayout()
        side_layout.addWidget(QLabel("Arm Side:"))
        self.int_side_sel = QComboBox()
        self.int_side_sel.addItems(["Left Arm", "Right Arm"])
        idx = 1 if self.arm_side == "left" else 0
        self.int_side_sel.setCurrentIndex(idx)
        self.int_side_sel.currentTextChanged.connect(self.on_arm_side_changed)
        side_layout.addWidget(self.int_side_sel)
        monitor_layout.addLayout(side_layout)
        
        self.btn_int_monitor = QPushButton("ENABLE MONITORING")
        self.btn_int_monitor.setCheckable(True)
        self.btn_int_monitor.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.btn_int_monitor.toggled.connect(self.toggle_intrinsics_monitoring)
        monitor_layout.addWidget(self.btn_int_monitor)
        
        # Relocated permanently to the Right Panel
        
        monitor_box.setLayout(monitor_layout)
        int_right.addWidget(monitor_box)

        controls_box = QGroupBox("Calibration Controls")
        controls_layout = QVBoxLayout()
        
        self.btn_int_capture = QPushButton("CAPTURE FRAME (C)")
        self.btn_int_capture.setMinimumHeight(45)
        self.btn_int_capture.setStyleSheet("background-color: #1565c0; color: white; font-size: 13px;")
        self.btn_int_capture.clicked.connect(self.capture_intrinsics_frame)
        controls_layout.addWidget(self.btn_int_capture)
        
        self.btn_int_calibrate = QPushButton("RUN CALIBRATION")
        self.btn_int_calibrate.setMinimumHeight(45)
        self.btn_int_calibrate.setStyleSheet("background-color: #2e7d32; color: white; font-size: 13px;")
        self.btn_int_calibrate.clicked.connect(self.run_intrinsics_calibration)
        controls_layout.addWidget(self.btn_int_calibrate)
        
        self.btn_int_save = QPushButton("SAVE PARAMETERS")
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #e65100; color: white; font-size: 13px;")
        self.btn_int_save.clicked.connect(self.save_intrinsics_calibration)
        self.btn_int_save.setEnabled(False)
        controls_layout.addWidget(self.btn_int_save)
        
        self.btn_int_reset = QPushButton("RESET CAPTURES")
        self.btn_int_reset.setMinimumHeight(30)
        self.btn_int_reset.setStyleSheet("background-color: #37474f; color: white;")
        self.btn_int_reset.clicked.connect(self.reset_intrinsics_captures)
        controls_layout.addWidget(self.btn_int_reset)
        
        controls_box.setLayout(controls_layout)
        int_right.addWidget(controls_box)
        int_right.addStretch()
        
        camera_tab_layout.addLayout(int_left, 2)
        camera_tab_layout.addLayout(int_right, 1)
        camera_tab.setLayout(camera_tab_layout)
        self.left_tabs.addTab(camera_tab, "2. Camera")
        
        # --- Right Panel ---
        right_panel_container = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_container)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Calibration Status & Monitoring Dashboard
        dash_box = QGroupBox("Calibration Status & Monitoring")
        dash_layout = QVBoxLayout()
        dash_layout.setSpacing(3)
        dash_layout.setContentsMargins(8, 4, 8, 4)
        
        # Monitoring Table
        self.tbl_offset_monitor = QTableWidget(2, 2)
        self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 5 (Wrist)", "Joint 3 (Elbow)"])
        self.tbl_offset_monitor.setVerticalHeaderLabels(["Right Arm", "Left Arm"])
        self.tbl_offset_monitor.setFixedHeight(75)
        self.tbl_offset_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_offset_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.tbl_offset_monitor.setStyleSheet("""
            QTableWidget {
                background-color: #121212;
                color: #00e5ff;
                gridline-color: #2d2d2d;
                font-weight: bold;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #888888;
                font-weight: bold;
                padding: 2px;
                border: 1px solid #2d2d2d;
            }
        """)
        dash_layout.addWidget(self.tbl_offset_monitor)
        
        # Marker Bracket Design Offset UI GroupBox
        bracket_box = QGroupBox("Marker Bracket Design Offset (Tf_to_marker)")
        bracket_layout = QVBoxLayout()
        bracket_layout.setSpacing(4)
        bracket_layout.setContentsMargins(6, 6, 6, 6)
        
        input_style = "background-color: #1c1c1c; color: #00e5ff; border: 1px solid #3d3d3d; border-radius: 3px; padding: 2px;"
        
        grid = QGridLayout()
        grid.setSpacing(6)
        
        # Row 0: Translation X, Y, Z
        grid.addWidget(QLabel("X (m):"), 0, 0)
        self.txt_bracket_x = QLineEdit()
        self.txt_bracket_x.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_x, 0, 1)
        
        grid.addWidget(QLabel("Y (m):"), 0, 2)
        self.txt_bracket_y = QLineEdit()
        self.txt_bracket_y.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_y, 0, 3)
        
        grid.addWidget(QLabel("Z (m):"), 0, 4)
        self.txt_bracket_z = QLineEdit()
        self.txt_bracket_z.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_z, 0, 5)
        
        # Row 1: Rotation Roll, Pitch, Yaw
        grid.addWidget(QLabel("Roll (deg):"), 1, 0)
        self.txt_bracket_roll = QLineEdit()
        self.txt_bracket_roll.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_roll, 1, 1)
        
        grid.addWidget(QLabel("Pitch (deg):"), 1, 2)
        self.txt_bracket_pitch = QLineEdit()
        self.txt_bracket_pitch.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_pitch, 1, 3)
        
        grid.addWidget(QLabel("Yaw (deg):"), 1, 4)
        self.txt_bracket_yaw = QLineEdit()
        self.txt_bracket_yaw.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_yaw, 1, 5)
        
        bracket_layout.addLayout(grid)
        
        self.btn_apply_bracket = QPushButton("APPLY BRACKET")
        self.btn_apply_bracket.setStyleSheet("background-color: #2979ff; color: white; font-weight: bold; font-size: 11px;")
        self.btn_apply_bracket.setFixedHeight(24)
        self.btn_apply_bracket.clicked.connect(self.apply_bracket_design_values)
        bracket_layout.addWidget(self.btn_apply_bracket)
        
        bracket_box.setLayout(bracket_layout)
        dash_layout.addWidget(bracket_box)
        
        # Apply & Clear buttons
        btn_joint_layout = QHBoxLayout()
        btn_joint_layout.setSpacing(6)
        
        self.btn_joint_apply = QPushButton("APPLY OFFSET")
        self.btn_joint_apply.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; font-size: 11px;")
        self.btn_joint_apply.clicked.connect(self.apply_joint_offset)
        self.btn_joint_apply.setFixedHeight(24)
        
        self.btn_joint_clear = QPushButton("CLEAR OFFSET")
        self.btn_joint_clear.setStyleSheet("background-color: #555555; color: white; font-weight: bold; font-size: 11px;")
        self.btn_joint_clear.clicked.connect(self.clear_joint_offset)
        self.btn_joint_clear.setFixedHeight(24)
        
        btn_joint_layout.addWidget(self.btn_joint_apply)
        btn_joint_layout.addWidget(self.btn_joint_clear)
        
        dash_layout.addLayout(btn_joint_layout)
        
        dash_box.setLayout(dash_layout)
        right_panel_layout.addWidget(dash_box)
        
        # 2. Right Tabs
        self.right_tabs = QTabWidget()
        
        # Tab 1: System Log
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        console_title = QLabel("System Log / Execution Console")
        console_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        console_title.setStyleSheet("color: #2979ff; margin-bottom: 2px;")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        
        log_layout.addWidget(console_title)
        log_layout.addWidget(self.log_text)
        log_tab.setLayout(log_layout)
        
        # Tab 2: Interactive Plot Viewer (Plot & Img)
        self.plot_tab = QWidget()
        plot_layout = QVBoxLayout()
        
        self.plot_label_combined = QLabel("Joint Sweep Calibration Visualizations")
        self.plot_label_combined.setAlignment(Qt.AlignCenter)
        self.plot_label_combined.setStyleSheet("background-color: #1a1a1a; color: #888888; border: 2px solid #2d2d2d; border-radius: 8px;")
        
        plot_layout.addWidget(self.plot_label_combined)
        self.plot_tab.setLayout(plot_layout)
        
        self.right_tabs.addTab(log_tab, "System Log")
        self.right_tabs.addTab(self.plot_tab, "Plot / Img")
        
        right_panel_layout.addWidget(self.right_tabs)
        
        # Assemble side-by-side (Stretch factor 3:7)
        main_layout.addWidget(self.left_tabs, 3)
        main_layout.addWidget(right_panel_container, 7)
        
        self.setLayout(main_layout)
        
        # Startup info
        self.log_msg("="*60)
        self.log_msg("  UNIFIED ROBOT CALIBRATION SUITE LOADED")
        self.log_msg("="*60)
        self.log_msg("[RECOMMENDED SEQUENCE]")
        self.log_msg("  1. Calibrate camera intrinsics first if needed ('2. Camera' tab).")
        self.log_msg("  2. Calibrate joint offsets using '1. Joint Calib' subtab.")
        self.log_msg("  3. Perform marker bracket sweeps using '2. Marker Calib' subtab.")
        self.log_msg("  4. Control head and verify offsets as a final check.")
        self.log_msg("="*60)

    # --- Common Helper Functions ---
    def log_msg(self, msg):
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(msg)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        else:
            print(msg)

    def connect_robot(self):
        if self.robot:
            self.log_msg("[INFO] Disconnecting from robot...")
            if self.robot != "mock_robot":
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
                try:
                    self.log_msg(f"[INFO] (UI Mode) Trying to connect to actual robot at {addr} ({model})...")
                    self.robot = MarkerCalibrator.initialize_robot(addr, model)
                except Exception as e:
                    self.log_msg(f"[INFO] Actual robot connection raised exception: {e}.")
                    self.robot = None
                
                if not self.robot:
                    self.log_msg("[INFO] Actual robot connection failed. Fallback to mock.")
                    self.robot = "mock_robot"
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
            
            # Sync current offsets with active arm_side from memory store (do not reload from yaml disk)
            self.joint_offsets["wrist_pitch"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint5", 0.0)
            self.joint_offsets["elbow"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint3", 0.0)
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets
            self.update_applied_offset_label()
            
            self.load_bracket_design_values()
            
            # Sync dropdown indexes between controls (blocking signals to avoid cycles)
            self.arm_sel.blockSignals(True)
            self.int_side_sel.blockSignals(True)
            
            idx = 1 if self.arm_side == "left" else 0
            self.arm_sel.setCurrentIndex(idx)
            self.int_side_sel.setCurrentIndex(idx)
            
            self.arm_sel.blockSignals(False)
            self.int_side_sel.blockSignals(False)

    def on_monitor_toggled(self, checked):
        if checked:
            self.btn_monitor.setText("Marker Monitor: ON")
            self.btn_monitor.setStyleSheet("background-color: #ffeb3b; color: black; font-weight: bold;")
        else:
            self.btn_monitor.setText("Marker Monitor: OFF")
            self.btn_monitor.setStyleSheet("")

    def toggle_camera_feed_dialog(self):
        if hasattr(self, 'feed_dialog') and self.feed_dialog is not None:
            self.feed_dialog.close()
            return
            
        self.feed_dialog = CameraFeedDialog(self)
        self.feed_dialog.show()
        self.on_left_tab_changed(self.left_tabs.currentIndex())

    def on_feed_dialog_closed(self):
        self.feed_dialog = None
        self.on_left_tab_changed(self.left_tabs.currentIndex())

    def update_marker_indicator(self, detected):
        self.indicator.set_detected(detected)
        if detected:
            self.status_label.setText("Detected")
            self.status_label.setStyleSheet("color: #00e676;")
        else:
            self.status_label.setText("Not Detected")
            self.status_label.setStyleSheet("color: #ff1744;")

    def poll_camera_status(self):
        # 탭 3(Camera Intrinsics)이 켜져있을 때는 poll_camera_status 생략 (update_video_frame이 처리함)
        if self.workflow_tabs.currentIndex() == 2:
            return
            
        try:
            if not self.btn_monitor.isChecked():
                self.update_marker_indicator(False)
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText("Position: Monitor Off")
                return
                
            res = self.marker_st.get_marker_transform(sampling_time=0, side=self.arm_side)
            detected = bool(res and len(res) > 0)
            self.update_marker_indicator(detected)
            
            if detected:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                x, y, z = pose[:3, 3] * 1000.0
                
                self.log_msg(f"[LIVE] Marker X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm")
                
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText(f"Position: X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f} mm")
            else:
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText("Position: Marker Not Detected")
            
            self.marker_st.camera.get_color_image()
            temp = self.marker_st.camera.get_camera_temperature()
            if temp:
                self.temp_label.setText(f"Camera Temp: {temp:.1f} °C")
        except Exception:
            pass

    def on_left_tab_changed(self, index):
        # 방어적 코드: 타이머 객체가 아직 미생성된 상태이면 처리를 생략
        if not hasattr(self, 'poll_timer') or not hasattr(self, 'video_timer'):
            return

        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        if index == 1 or dialog_visible: # Camera Tab or popup feed dialog is open
            # poll_timer 끄고 video_timer 가동
            if self.poll_timer.isActive():
                self.poll_timer.stop()
            self.video_timer.start(50)
        else: # Main Tab (no popup feed dialog open)
            # video_timer 끄고 poll_timer 기동
            if self.video_timer.isActive():
                self.video_timer.stop()
            if not self.ui_only and self.marker_st is not None:
                self.poll_timer.start(200)

    def update_applied_offset_label(self):
        if not hasattr(self, 'tbl_offset_monitor') or not hasattr(self, 'btn_joint_apply'):
            return
        
        for row_idx, arm in enumerate(["right", "left"]):
            for col_idx, joint_key in enumerate(["joint5", "joint3"]):
                val = self.joint_offsets_store.get(arm, {}).get(joint_key, 0.0)
                item = QTableWidgetItem(f"{val:.4f}°")
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl_offset_monitor.setItem(row_idx, col_idx, item)
        
        # Keeps the apply button permanently visible on the Right Panel dashboard

    def apply_joint_offset(self):
        self.joint_offsets["wrist_pitch"] = self.joint_offsets_store[self.arm_side]["joint5"]
        self.joint_offsets["elbow"] = self.joint_offsets_store[self.arm_side]["joint3"]
        self.joint_calibrator.joint_offsets = self.joint_offsets
        self.marker_calibrator.joint_offsets = self.joint_offsets
        
        self.save_offsets_to_yaml()
        self.update_applied_offset_label()
        
        self.log_msg(f"\n" + "="*50)
        self.log_msg(f"[APPLY] Applied current staged offsets for {self.arm_side.upper()} Arm to active parameters:")
        self.log_msg(f"  - Joint 5 (wrist_pitch): {self.joint_offsets['wrist_pitch']:.4f}°")
        self.log_msg(f"  - Joint 3 (elbow)      : {self.joint_offsets['elbow']:.4f}°")
        self.log_msg("[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!")
        self.log_msg("="*50 + "\n")

    def stop_motion(self):
        self.log_msg("[STOP] Stop requested by user.")
        self.joint_calibrator.stop_requested = True
        self.marker_calibrator.stop_requested = True
        if hasattr(self, 'stop_event_mc') and self.stop_event_mc:
            self.stop_event_mc.set()
        if self.robot:
            self.log_msg("[STOP] Sending cancel_control to robot!")
            if self.robot != "mock_robot":
                self.robot.cancel_control()
            else:
                self.log_msg("[STOP] (Mock Mode) Cancel control called.")
        else:
            self.log_msg("[STOP] No robot connected to cancel control.")

    def clear_joint_offset(self):
        reply = QMessageBox.question(
            self, 
            "Clear Joint Offset", 
            f"Are you sure you want to reset all staged/saved joint offsets for {self.arm_side.upper()} arm to 0.0?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.joint_offsets_store[self.arm_side]["joint5"] = 0.0
            self.joint_offsets_store[self.arm_side]["joint3"] = 0.0
            
            self.joint_offsets["wrist_pitch"] = 0.0
            self.joint_offsets["elbow"] = 0.0
            self.joint_calibrator.joint_offsets = self.joint_offsets
            self.marker_calibrator.joint_offsets = self.joint_offsets
            
            self.save_offsets_to_yaml()
            self.update_applied_offset_label()
            
            self.log_msg(f"[CLEAR] Staged and saved offsets cleared to 0.0 for {self.arm_side.upper()} Arm.")

    def load_bracket_design_values(self):
        import yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, "..", "config", "setting.yaml"))
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                if data and "camera" in data:
                    key = f"Tf_to_marker_{self.arm_side}"
                    val_list = data["camera"].get(key, None)
                    if val_list and len(val_list) == 6:
                        self.txt_bracket_x.setText(f"{val_list[0]:.5f}")
                        self.txt_bracket_y.setText(f"{val_list[1]:.5f}")
                        self.txt_bracket_z.setText(f"{val_list[2]:.5f}")
                        self.txt_bracket_roll.setText(f"{val_list[3]:.2f}")
                        self.txt_bracket_pitch.setText(f"{val_list[4]:.2f}")
                        self.txt_bracket_yaw.setText(f"{val_list[5]:.2f}")
                        self.log_msg(f"[INFO] Loaded {key} from setting.yaml: {val_list}")
                        return
            self.log_msg(f"[WARNING] Could not load bracket design values for {self.arm_side} from setting.yaml.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to load setting.yaml: {e}")

    def apply_bracket_design_values(self):
        import yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, "..", "config", "setting.yaml"))
        try:
            try:
                x = float(self.txt_bracket_x.text())
                y = float(self.txt_bracket_y.text())
                z = float(self.txt_bracket_z.text())
                roll = float(self.txt_bracket_roll.text())
                pitch = float(self.txt_bracket_pitch.text())
                yaw = float(self.txt_bracket_yaw.text())
            except ValueError:
                QMessageBox.critical(self, "Invalid Inputs", "Please enter valid numeric values for all bracket design fields.")
                return
            
            lines = []
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    lines = f.readlines()
            
            camera_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("camera:"):
                    camera_idx = i
                    break
            
            key = f"Tf_to_marker_{self.arm_side}"
            new_vals = [x, y, z, roll, pitch, yaw]
            new_val_str = f"[{x:.5f}, {y:.5f}, {z:.5f}, {roll:.2f}, {pitch:.2f}, {yaw:.2f}]"
            
            key_found = False
            if camera_idx != -1:
                i = camera_idx + 1
                while i < len(lines):
                    line = lines[i]
                    stripped = line.strip()
                    if not stripped:
                        i += 1
                        continue
                    # If we hit another top-level key, we stop
                    if not line.startswith(" ") and not line.startswith("\t") and stripped.endswith(":"):
                        break
                    
                    if stripped.startswith(f"{key}:"):
                        comment = ""
                        if "#" in line:
                            comment_idx = line.find("#")
                            comment = " " + line[comment_idx:].rstrip()
                        
                        indent = len(line) - len(line.lstrip())
                        lines[i] = " " * indent + f"{key}: {new_val_str}{comment}\n"
                        key_found = True
                        break
                    i += 1
            
            if not key_found:
                if camera_idx == -1:
                    lines.append("camera:\n")
                    lines.append(f"  {key}: {new_val_str}\n")
                else:
                    lines.insert(camera_idx + 1, f"  {key}: {new_val_str}\n")
            
            with open(config_path, "w") as f:
                f.writelines(lines)
                
            self.log_msg(f"[SUCCESS] Saved {key} to setting.yaml: {new_vals}")
            QMessageBox.information(self, "Success", f"Bracket design values saved for {self.arm_side.upper()} arm!")
            
            if not self.ui_only and self.marker_st is not None:
                detector = self.marker_st.marker_detection
                if hasattr(detector, 'camera_config'):
                    detector.camera_config[key] = new_vals
                    tf_vec_l = detector.camera_config.get("Tf_to_marker_left", [0.0, 0.0775, -0.06677, 90.0, 0.0, 0.0])
                    tf_vec_r = detector.camera_config.get("Tf_to_marker_right", [0.0, -0.0775, -0.06677, 90.0, 0.0, 180.0])
                    detector.Tf_to_marker_tf_left = detector.make_transform(tf_vec_l)
                    detector.Tf_to_marker_tf_right = detector.make_transform(tf_vec_r)
                    self.log_msg("[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save bracket values: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save bracket values: {e}")

    def set_controls_enabled(self, enabled):
        self.btn_joint_ready.setEnabled(enabled)
        self.btn_joint_start.setEnabled(enabled)
        self.btn_joint_apply.setEnabled(enabled)
        if hasattr(self, 'btn_joint_clear'):
            self.btn_joint_clear.setEnabled(enabled)
        if hasattr(self, 'btn_apply_bracket'):
            self.btn_apply_bracket.setEnabled(enabled)
            
        if hasattr(self, 'txt_bracket_x'):
            self.txt_bracket_x.setEnabled(enabled)
            self.txt_bracket_y.setEnabled(enabled)
            self.txt_bracket_z.setEnabled(enabled)
            self.txt_bracket_roll.setEnabled(enabled)
            self.txt_bracket_pitch.setEnabled(enabled)
            self.txt_bracket_yaw.setEnabled(enabled)
        
        self.btn_marker_ready.setEnabled(enabled)
        self.btn_marker_center.setEnabled(enabled)
        self.btn_marker_start.setEnabled(enabled)
        if hasattr(self, 'btn_marker_result'):
            self.btn_marker_result.setEnabled(enabled)
        
        self.btn_int_capture.setEnabled(enabled)
        self.btn_int_calibrate.setEnabled(enabled)
        self.btn_int_reset.setEnabled(enabled)
        
        if hasattr(self, 'btn_move_head'):
            self.btn_move_head.setEnabled(enabled)
        if hasattr(self, 'txt_head_yaw'):
            self.txt_head_yaw.setEnabled(enabled)
        if hasattr(self, 'txt_head_pitch'):
            self.txt_head_pitch.setEnabled(enabled)
            
        self.btn_connect.setEnabled(enabled)
        self.model_input.setEnabled(enabled)
        self.workflow_tabs.setEnabled(enabled)
        self.arm_sel.setEnabled(enabled)
        self.joint_mode_sel.setEnabled(enabled)
        self.int_side_sel.setEnabled(enabled)
        if hasattr(self, 'marker_axis_sel'):
            self.marker_axis_sel.setEnabled(enabled)
        if hasattr(self, 'btn_camera_feed'):
            self.btn_camera_feed.setEnabled(True) # Keep camera feed button enabled always!

    def on_action_finished(self):
        self.set_controls_enabled(True)
        if hasattr(self, 'stop_event_mc'):
            self.stop_event_mc.clear()
        
        # 탭 상태에 맞춰 타이머 활성화
        self.on_left_tab_changed(self.left_tabs.currentIndex())

    # --- Joint Calibration Workflows ---
    def move_to_ready_pose_joint(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if self.robot == "mock_robot":
            self.log_msg("[MOCK] Moved to joint ready pose.")
            return

        mode_str = self.joint_mode_sel.currentText()
        mode = "wrist_pitch" if "wrist_pitch" in mode_str else "elbow"
        
        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.ready_worker = MoveToReadyWorker(self.joint_calibrator, self.arm_side, mode)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_action_finished)
        self.ready_worker.start()

    def start_calibration_joint(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        mode_str = self.joint_mode_sel.currentText()
        mode = "wrist_pitch" if "wrist_pitch" in mode_str else "elbow"

        self.original_joint_offset = self.joint_offsets.get(mode, 0.0)
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Joint Sweep: {mode.upper()}")
        
        curr_offset = self.joint_offsets.get(mode, 0.0)
        self.active_worker = JointCalibrationWorker(
            self.joint_calibrator, self.arm_side, mode, 
            ui_only=self.ui_only, 
            current_offset_deg=curr_offset,
            sweep_duration=15.0
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.finished_signal.connect(self.on_calibration_finished_joint)
        self.active_worker.start()

    def on_calibration_finished_joint(self, res):
        if not res:
            self.on_action_finished()
            return

        mode = res['mode']
        recommended = res.get('recommended_joint_offset', res['optimal_offset'])
        self.recommended_joint_offset = recommended
        self.finalize_joint_calibration_run(mode, res, converged=res.get('converged', True))

    def finalize_joint_calibration_run(self, mode, res, converged=True):
        self.on_action_finished()
        self.joint_sweep_data = res

        # Update Plot viewer if plots exist
        if 'plot_path_combined' in res and os.path.exists(res['plot_path_combined']):
            pix = QPixmap(res['plot_path_combined']).scaled(900, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.plot_label_combined.setPixmap(pix)
            
        self.right_tabs.setCurrentIndex(1) # Auto swap to plot tab

        joint_key = "joint5" if mode == "wrist_pitch" else "joint3"
        self.joint_offsets_store[self.arm_side][joint_key] = float(self.recommended_joint_offset)

        # Revert active offsets to nominal original values in model (until user clicks APPLY)
        self.joint_offsets[mode] = self.original_joint_offset
        self.joint_calibrator.joint_offsets[mode] = self.original_joint_offset
        self.marker_calibrator.joint_offsets[mode] = self.original_joint_offset
        self.update_applied_offset_label()

        self.log_msg(f"\n" + "="*50)
        if converged:
            self.log_msg(f"   [SUCCESS] 3-STEP POLARITY CALIBRATION CONVERGED SUCCESSFULLY!")
        else:
            self.log_msg(f"   [INFO] 3-STEP CALIBRATION COMPLETED")
        self.log_msg(f"   * Recommended Absolute Offset : {self.recommended_joint_offset:.4f}°")
        self.log_msg(f"   * Current Active Offset       : {self.original_joint_offset:.4f}° (REVERTED)")
        self.log_msg(f"   --> Click 'APPLY OFFSET' on the UI panel to apply this new calibration.")
        self.log_msg("="*50 + "\n")
        
        self.show_result_joint()

    def show_result_joint(self):
        self.log_msg("\n" + "="*50)
        self.log_msg("       JOINT CALIBRATION ESTIMATED RESULTS")
        self.log_msg("="*50)
        
        if not self.joint_sweep_data:
            self.log_msg("\n[ERROR] No joint sweep data loaded! Perform a sweep first.")
            return

        mode = self.joint_sweep_data['mode']
        self.log_msg(f"\n[1] Calibration Target: {mode}")
        
        recommended = self.joint_sweep_data.get('recommended_joint_offset', self.joint_sweep_data['optimal_offset'])
        self.log_msg(f"    - Target Swept Joint       : {'Joint 5' if mode == 'wrist_pitch' else 'Joint 3'}")
        self.log_msg(f"    - Estimated Optimal Offset : {recommended:.4f} deg")
        # self.log_msg(f"    - Circle A Fitting RMSE     : {self.joint_sweep_data['rmse_A']:.4f} mm")
        # self.log_msg(f"    - Circle B Fitting RMSE     : {self.joint_sweep_data['rmse_B']:.4f} mm")
        
        self.log_msg("\n[2] Suggested Joint Home Offset update:")
        self.log_msg(f"  Add offset: {recommended:.4f} deg to calibration config.")
        self.log_msg("="*50)


    # --- Marker Bracket Calibration Workflows ---
    def move_to_ready_pose_marker(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if self.robot == "mock_robot":
            self.log_msg("[MOCK] Moved to marker ready pose.")
            return

        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.ready_worker = MoveToReadyWorker(self.marker_calibrator, self.arm_side)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_action_finished)
        self.ready_worker.start()

    def move_to_center_marker(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        if self.robot == "mock_robot":
            self.log_msg("[MOCK] Moved to center (Marker mode).")
            return

        if hasattr(self, 'active_worker') and self.active_worker and self.active_worker.isRunning():
            self.log_msg("[INFO] Cancelling Move to Center...")
            self.stop_event_mc.set()
            return

        axis_str = self.marker_axis_sel.currentText()
        target_dist = 190.0 if "Axis 6" in axis_str else 200.0
        self.log_msg(f"[INFO] Move to Center (Marker Calibration) -> target: {target_dist} mm")

        self.btn_marker_center.setText("CANCEL")
        self.btn_marker_center.setStyleSheet("background-color: #b71c1c; color: white;")
        self.set_controls_enabled(False)
        self.btn_marker_center.setEnabled(True)
        
        if self.poll_timer.isActive(): self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
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

        # use_head = self.cb_head_tracking.isChecked()
        use_head = False
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Unified Marker Sweep (Axis 6 & 5) (Head Tracking: {use_head})")

        if self.ui_only:
            # Simulate unified marker sweep results
            class MockMarkerWorker(QThread):
                log_sig = Signal(str)
                finished_sig = Signal(dict)
                def __init__(self, arm_side):
                    super().__init__()
                    self.arm_side = arm_side
                def run(self):
                    time.sleep(0.5)
                    self.log_sig.emit("[MOCK] Starting Stage 1 (Axis 6 sweep)...")
                    time.sleep(1.0)
                    self.log_sig.emit("[MOCK] Stage 1 finished. Starting Stage 2 (Axis 5 sweep)...")
                    time.sleep(1.0)
                    self.log_sig.emit("[MOCK] Stage 2 finished. Calculating unified results...")
                    
                    theta = np.linspace(-np.pi/6, np.pi/6, 11)
                    res_6 = {
                        'axis_mode': 6,
                        'radius': 74.85,
                        'rmse': 0.12,
                        'axis': np.array([0.01, -0.999, 0.02]),
                        'center': np.array([0.5, 1.2, 0.3]),
                        'pts_2d': np.column_stack((np.cos(theta)*74.8 + 0.1, np.sin(theta)*74.8 - 0.2)),
                        'uc_opt': 0.1,
                        'vc_opt': -0.2,
                        'tilt': 89.48,
                        'yaw': 1.4,
                        'tilt_list': [89.48]*11
                    }
                    res_5 = {
                        'axis_mode': 5,
                        'radius': 280.15,
                        'rmse': 0.18,
                        'axis': np.array([0.02, 0.03, 0.999]),
                        'center': np.array([1.2, 0.8, -0.4]),
                        'pts_2d': np.column_stack((np.cos(theta)*280.2 + 0.5, np.sin(theta)*280.2 + 0.4)),
                        'uc_opt': 0.5,
                        'vc_opt': 0.4,
                        'tilt': 0.25,
                        'yaw': -20.68,
                        'tilt_list': [0.25]*11
                    }
                    
                    unified_res = {
                        'x_e': 0.0,
                        'y_e': 74.85 if self.arm_side == "left" else -74.85,
                        'z_e': -50.15,
                        'roll_e': 0.15,
                        'pitch_e': -0.25,
                        'yaw_e': 0.0 if self.arm_side == "left" else 180.0,
                        'L_5_ee': 330.0,
                        'radius_6': 74.85,
                        'radius_5': 280.15,
                        'ortho_err': 0.1,
                        'rmse_6': 0.12,
                        'rmse_5': 0.18,
                        'rot_err_deg': 0.2,
                        'tilt_diff': 0.05
                    }
                    unified_res['res_5'] = res_5
                    unified_res['res_6'] = res_6
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    def plot_single(ax, res, axis_num, color):
                        ax.scatter(res['pts_2d'][:, 0], res['pts_2d'][:, 1], c=color, label='Captured')
                        circle = plt.Circle((res['uc_opt'], res['vc_opt']), res['radius'], color='r', fill=False, label='Fit')
                        ax.add_patch(circle)
                        ax.plot(res['uc_opt'], res['vc_opt'], 'rx', label='Center')
                        ax.set_aspect('equal')
                        ax.grid(True)
                        ax.set_title(f"Axis {axis_num} Sweep (MOCK)")
                        ax.legend()
                        
                    plot_single(ax1, res_6, 6, 'blue')
                    plot_single(ax2, res_5, 5, 'green')
                    
                    fig.suptitle(f"Unified Marker Sweep Results ({self.arm_side.upper()} Arm) - MOCK")
                    plt.tight_layout()
                    
                    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_img")
                    os.makedirs(result_dir, exist_ok=True)
                    plot_path = os.path.join(result_dir, f"circle_fit_{self.arm_side}_marker_unified.png")
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    
                    unified_res['plot_path_combined'] = plot_path
                    self.finished_sig.emit(unified_res)
            
            self.active_worker = MockMarkerWorker(self.arm_side)
            self.active_worker.log_sig.connect(self.log_msg)
            self.active_worker.finished_sig.connect(self.on_calibration_finished_marker)
            self.active_worker.start()
        else:
            try:
                tolerance = float(self.tolerance_input.text())
            except ValueError:
                tolerance = 0.5
            self.active_worker = MarkerCalibrationWorker(self.marker_calibrator, self.arm_side, use_head_tracking=use_head, tolerance=tolerance)
            self.active_worker.log_signal.connect(self.log_msg)
            self.active_worker.status_signal.connect(self.update_marker_indicator)
            self.active_worker.finished_signal.connect(self.on_calibration_finished_marker)
            self.active_worker.start()

    def on_calibration_finished_marker(self, res):
        self.on_action_finished()

        if res:
            self.marker_data_unified = res
            self.marker_data_5 = res['res_5']
            self.marker_data_6 = res['res_6']
                
            if 'plot_path_combined' in res and os.path.exists(res['plot_path_combined']):
                pix = QPixmap(res['plot_path_combined']).scaled(900, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.plot_label_combined.setPixmap(pix)
            
            self.right_tabs.setCurrentIndex(1) # Auto swap to plot tab
            self.show_unified_result_marker_direct(res)
        else:
            self.log_msg("[ERROR] Marker sweep failed.")

    def show_unified_result_marker_direct(self, res):
        self.log_msg("\n" + "="*50)
        self.log_msg("       UNIFIED BRACKET CALIBRATION RESULTS")
        self.log_msg("="*50)
        
        self.log_msg("\n[1] Cartesian Offset (EE Link Frame)")
        self.log_msg(f"    - X-Offset: {res['x_e']:.2f} mm")
        self.log_msg(f"    - Y-Offset: {res['y_e']:.2f} mm")
        self.log_msg(f"    - Z-Offset: {res['z_e']:.2f} mm")
        self.log_msg(f"       * (L_5_ee: {res['L_5_ee']:.1f} mm, R6: {res['radius_6']:.2f} mm, R5: {res['radius_5']:.2f} mm)")
            
        self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
        self.log_msg(f"    - Roll : {res['roll_e']:.2f} deg")
        self.log_msg(f"    - Pitch: {res['pitch_e']:.2f} deg")
        self.log_msg(f"    - Yaw  : {res['yaw_e']:.2f} deg")
        
        self.log_msg("\n[3] setting.yaml Config Update values:")
        x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0
        
        if self.arm_side == "left":
            self.log_msg(f"  Tf_to_marker_left:  [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
        else:
            self.log_msg(f"  Tf_to_marker_right: [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
            
        self.log_msg(f"\n[4] Confidence Metrics:")
        self.log_msg(f"    - Orthogonality Error  : {res['ortho_err']:.3f} deg")
        
        if res['rmse_6'] > 0.5 or res['rmse_5'] > 0.5:
            self.log_msg("\n" + "!"*60)
            self.log_msg(" [WARNING] Fitting RMSE exceeds 0.5 mm!")
            self.log_msg("  The marker coordinates may have high noise. Check hardware.")
            self.log_msg("!"*60)
        self.log_msg("="*50)

    def show_unified_result_marker(self):
        self.log_msg("\n" + "="*50)
        self.log_msg("       UNIFIED BRACKET CALIBRATION RESULTS")
        self.log_msg("="*50)
        
        if not self.marker_data_5 or not self.marker_data_6:
            self.log_msg("\n[ERROR] Missing dataset!")
            if not self.marker_data_6: self.log_msg(" -> Axis 6 Sweep (Yaw) data is missing.")
            if not self.marker_data_5: self.log_msg(" -> Axis 5 Sweep (Pitch) data is missing.")
            return

        try:
            try:
                tolerance = float(self.tolerance_input.text())
            except ValueError:
                tolerance = 0.5
            res = self.marker_calibrator.compute_unified_bracket_calibration(
                self.marker_data_5, self.marker_data_6, self.arm_side, tolerance=tolerance
            )
            
            self.log_msg("\n[1] Cartesian Offset (EE Link Frame)")
            self.log_msg(f"    - X-Offset: {res['x_e']:.2f} mm")
            self.log_msg(f"    - Y-Offset: {res['y_e']:.2f} mm")
            self.log_msg(f"    - Z-Offset: {res['z_e']:.2f} mm")
            self.log_msg(f"       * (L_5_ee: {res['L_5_ee']:.1f} mm, R6: {res['radius_6']:.2f} mm, R5: {res['radius_5']:.2f} mm)")
                
            self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
            self.log_msg(f"    - Roll : {res['roll_e']:.2f} deg")
            self.log_msg(f"    - Pitch: {res['pitch_e']:.2f} deg")
            self.log_msg(f"    - Yaw  : {res['yaw_e']:.2f} deg")
            
            self.log_msg("\n[3] setting.yaml Config Update values:")
            x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0
            
            if self.arm_side == "left":
                self.log_msg(f"  Tf_to_marker_left:  [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
            else:
                self.log_msg(f"  Tf_to_marker_right: [{x_m:.5f}, {y_m:.5f}, {z_m:.5f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
                
            self.log_msg(f"\n[4] Confidence Metrics:")
            self.log_msg(f"    - Orthogonality Error  : {res['ortho_err']:.3f} deg")
            #self.log_msg(f"    - Fitting RMSE (A6/A5) : {res['rmse_6']:.3f} / {res['rmse_5']:.3f} mm")
            
            if res['rmse_6'] > 0.5 or res['rmse_5'] > 0.5:
                self.log_msg("\n" + "!"*60)
                self.log_msg(" [WARNING] Fitting RMSE exceeds 0.5 mm!")
                self.log_msg("  The marker coordinates may have high noise. Check hardware.")
                self.log_msg("!"*60)
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to calculate bracket calibration: {e}")
            
        self.log_msg("="*50)

    # --- Head Control and Manual Operations ---
    def move_head_manually(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        try:
            yaw = float(self.txt_head_yaw.text())
            pitch = float(self.txt_head_pitch.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Inputs", "Head angles must be valid numbers.")
            return
            
        self.log_msg(f"[MANUAL HEAD] Commands sent - Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°")
        if self.ui_only:
            self.log_msg("[MOCK] Moved head to target angles.")
            return
            
        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.head_worker = ManualHeadWorker(self.joint_calibrator, np.radians(yaw), np.radians(pitch))
        self.head_worker.log_signal.connect(self.log_msg)
        self.head_worker.finished_signal.connect(self.on_action_finished)
        self.head_worker.start()

    # --- Camera Intrinsics Calibration Workflows (Tab 3) ---
    def toggle_intrinsics_monitoring(self, checked):
        self.monitor_enabled = checked
        if checked:
            self.btn_int_monitor.setText("STOP MONITORING")
            self.btn_int_monitor.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold;")
        else:
            self.btn_int_monitor.setText("ENABLE MONITORING")
            self.btn_int_monitor.setStyleSheet("background-color: #1e1e1e; color: white;")
            if hasattr(self, 'lbl_marker_pos'):
                self.lbl_marker_pos.setText("Position: X: 0.0, Y: 0.0, Z: 0.0 mm")

    def update_video_frame(self):
        # 왼쪽 Camera 탭(인덱스 1)이 활성화되어 있거나, Camera Feed 대화상자가 열려있을 때 업데이트
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        if self.left_tabs.currentIndex() != 1 and not dialog_visible:
            return

        if not self.ui_only and self.marker_st is not None:
            self.marker_st.camera.capture_image()
            img = self.marker_st.camera.get_color_image()
        else:
            # Mock image
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "UI-ONLY MODE", (440, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 100, 100), 3)

        if img is None:
            return
            
        self.current_frame = img.copy()
        display_img = img.copy()
        
        # Convert to QImage and display
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        if self.left_tabs.currentIndex() == 1:
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if dialog_visible:
            w_lbl = max(20, self.feed_dialog.lbl_feed.width())
            h_lbl = max(20, self.feed_dialog.lbl_feed.height())
            self.feed_dialog.lbl_feed.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C and self.workflow_tabs.currentIndex() == 2:
            self.capture_intrinsics_frame()
        super().keyPressEvent(event)

    def capture_intrinsics_frame(self):
        if hasattr(self, 'current_frame'):
            self.captured_images.append(self.current_frame.copy())
            self.lbl_captured.setText(f"Captured Frames: {len(self.captured_images)}")
            self.log_msg(f"[INTRINSICS] Frame {len(self.captured_images)} captured.")

    def reset_intrinsics_captures(self):
        self.captured_images.clear()
        self.lbl_captured.setText(f"Captured Frames: 0")
        self.btn_int_save.setEnabled(False)
        self.log_msg("[INTRINSICS] Capture memory cleared.")

    def run_intrinsics_calibration(self):
        if len(self.captured_images) < 5:
            self.log_msg("[ERROR] Need at least 5 frames to run calibration!")
            return
            
        self.log_msg(f"\n[INTRINSICS] Running calibration on {len(self.captured_images)} images. Please wait...")
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        success = self.intrinsics_calibrator.run_calibration_with_images(self.captured_images, None)
        
        QApplication.restoreOverrideCursor()
        
        if success:
            self.log_msg(f"[SUCCESS] Calibration complete! RMS Error: {self.intrinsics_calibrator.rms_error:.4f}")
            self.log_msg("[INTRINSICS] Click 'SAVE PARAMETERS' to apply changes.")
            self.btn_int_save.setEnabled(True)
            self.show_intrinsics_verification()
        else:
            self.log_msg("[ERROR] Calibration failed. Check images and board settings.")

    def save_intrinsics_calibration(self):
        if self.intrinsics_calibrator.cameraMatrix is None:
            self.log_msg("[ERROR] No calibration data to save!")
            return
            
        import yaml
        try:
            data = {
                "camera_matrix": self.intrinsics_calibrator.cameraMatrix.tolist(),
                "dist_coeffs": self.intrinsics_calibrator.distCoeffs.tolist(),
                "rms_error": float(self.intrinsics_calibrator.rms_error),
                "width": int(self.captured_images[0].shape[1]),
                "height": int(self.captured_images[0].shape[0])
            }
            os.makedirs(os.path.dirname(self.output_yaml), exist_ok=True)
            with open(self.output_yaml, "w") as f:
                yaml.dump(data, f)
            self.log_msg(f"[SUCCESS] Intrinsic parameters saved to: {self.output_yaml}")
            
            # Sync with the local marker detector instances
            if self.marker_detector is not None:
                self.marker_detector.fx = self.intrinsics_calibrator.cameraMatrix[0, 0]
                self.marker_detector.fy = self.intrinsics_calibrator.cameraMatrix[1, 1]
                self.marker_detector.principal_point = [self.intrinsics_calibrator.cameraMatrix[0, 2], self.intrinsics_calibrator.cameraMatrix[1, 2]]
                self.marker_detector.dist_coeffs = self.intrinsics_calibrator.distCoeffs
        except Exception as e:
            self.log_msg(f"[ERROR] Save failed: {e}")

    def show_intrinsics_verification(self):
        if len(self.captured_images) == 0:
            return
            
        test_img = self.captured_images[-1]
        h, w = test_img.shape[:2]
        
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics_calibrator.cameraMatrix, self.intrinsics_calibrator.distCoeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(test_img, self.intrinsics_calibrator.cameraMatrix, self.intrinsics_calibrator.distCoeffs, None, new_mtx)
        
        combined_res = np.hstack((test_img, undistorted))
        h_res, w_res = combined_res.shape[:2]

        grid_size = 60
        for y in range(0, h_res, grid_size):
            cv2.line(combined_res, (0, y), (w_res, y), (0, 255, 0), 1)
        for x in range(0, w_res, grid_size):
            cv2.line(combined_res, (x, 0), (x, h_res), (0, 255, 0), 1)

        cv2.putText(combined_res, "Original", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(combined_res, "Undistorted", (w + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(combined_res, f"RMS Error: {self.intrinsics_calibrator.rms_error:.4f}", (w + 30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # Save to result_img folder
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_img")
        os.makedirs(result_dir, exist_ok=True)
        save_path = os.path.join(result_dir, "camera_intrinsics_verification.png")
        cv2.imwrite(save_path, combined_res)
        
        # Load inside Plot & Img tab
        pix = QPixmap(save_path).scaled(900, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.plot_label_combined.setPixmap(pix)
        self.right_tabs.setCurrentIndex(1) # Switch to Plot / Img tab
        
        self.log_msg(f"[INTRINSICS] Verification image loaded to Plot / Img tab and saved to: {save_path}")

    def closeEvent(self, event):
        if hasattr(self, 'feed_dialog') and self.feed_dialog is not None:
            try:
                self.feed_dialog.close()
            except Exception:
                pass
        if self.video_timer.isActive():
            self.video_timer.stop()
        if self.poll_timer.isActive():
            self.poll_timer.stop()
            
        if not self.ui_only and self.marker_st is not None:
            try:
                self.marker_st.camera.stream_off()
                print("Camera stream closed.")
            except Exception:
                pass
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Unified Robot Calibration Suite GUI")
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
