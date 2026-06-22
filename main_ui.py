import sys
import os
import time
import json
import threading
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy

import rby1_sdk as rby

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QCheckBox, 
                             QLineEdit, QDialog, QMessageBox, QTabWidget, QGridLayout,
                             QTableWidget, QHeaderView, QTableWidgetItem, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage

# Ensure local core module is imported first
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.abspath(os.path.join(current_dir, "core"))
calibration_dir = os.path.abspath(os.path.join(core_dir, "calibration"))
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)
if calibration_dir not in sys.path:
    sys.path.insert(1, calibration_dir)

# Import calibrators and core scripts
from Calibrator import MarkerCalibrator, JointCalibrator, BaseCalibrator
from IntrinsicsCalibrator import IntrinsicsCalibrator
from homeoffset_core import (
    reset_current_pose_home_offsets,
    save_home_reset_baseline_json,
    move_to_offset_candidate_from_json,
    load_offset_from_json
)
from core.calibration_core import (
    create_robot,
    create_live_marker_transform,
    capture_one_sample as capture_robot_sample,
    get_arm_config,
    get_both_arm_config,
    get_head_config,
    load_npz_dataset,
    save_npz_dataset,
    split_arm_offsets,
    validate_dataset,
    generate_sim_measurements,
    check_calibration_state
)
from core.calibration_optimizer import (
    DEFAULT_LAMBDA_CAM_POS,
    DEFAULT_LAMBDA_CAM_ROT,
    CalibrationOptimizer,
    QPCalibrationOptimizer
)
from core.robot_motion import (
    AutoCollectionConfig,
    build_incremental_motion_plan,
    move_to_auto_ready_pose,
    execute_auto_motion_step,
    compute_fk,
    reset_motion_state
)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
WARNING_POSE_PATH = BASE_DIR / "warning_pose.png"

# --- Premium Modern Dark CSS Stylesheet ---
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
    padding: 6px 10px;
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
    padding: 4px;
    color: #ffffff;
    min-width: 100px;
}
QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 4px;
    color: #ffffff;
}
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    color: #a5d6a7;
    font-family: 'Consolas', 'Courier New', monospace;
}
QTabWidget::pane {
    border: 1px solid #2d2d2d;
    border-radius: 6px;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #1a1a1a;
    border: 1px solid #2d2d2d;
    border-bottom: none;
    padding: 6px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    color: #888888;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #2979ff;
    font-weight: bold;
    border-bottom: 2px solid #2979ff;
}
QCheckBox {
    spacing: 5px;
    font-weight: bold;
}
"""

class IndicatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.detected = False

    def set_detected(self, detected):
        self.detected = detected
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor("#00e676") if self.detected else QColor("#ff1744")
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 16, 16)

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

class ApplyHomeOffsetDialog(QDialog):
    def __init__(self, parent, result_path, baseline_path, arm, include_head):
        super().__init__(parent)
        self.parent_app = parent
        self.result_path = result_path
        self.baseline_path = baseline_path
        self.arm = arm
        self.include_head = include_head
        
        self.setWindowTitle("Apply Home Offset")
        self.resize(780, 520)
        self.setStyleSheet(parent.styleSheet())
        
        layout = QVBoxLayout(self)
        
        msg = (
            "Compare the original baseline zero and the optimized zero before applying.\n\n"
            "1. Move to Baseline Zero to inspect the zero pose before calibration reset.\n"
            "2. Move to Optimized Zero to inspect the computed calibration zero.\n"
            "3. When the robot is at the pose you want to keep, click Apply Current Pose.\n\n"
            "Make sure the workspace is clear before each move."
        )
        layout.addWidget(QLabel(msg))
        
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setFont(QFont("Consolas", 10))
        layout.addWidget(self.summary_box)
        
        btn_frame = QHBoxLayout()
        self.btn_baseline = QPushButton("Move to Baseline Zero")
        self.btn_optimized = QPushButton("Move to Optimized Zero")
        self.btn_apply = QPushButton("Apply Current Pose")
        self.btn_apply.setStyleSheet("background-color: #e65100; color: white; font-weight: bold;")
        self.btn_close = QPushButton("Close")
        
        btn_frame.addWidget(self.btn_baseline)
        btn_frame.addWidget(self.btn_optimized)
        btn_frame.addWidget(self.btn_apply)
        btn_frame.addStretch()
        btn_frame.addWidget(self.btn_close)
        layout.addLayout(btn_frame)
        
        self.btn_baseline.clicked.connect(self.move_baseline)
        self.btn_optimized.clicked.connect(self.move_optimized)
        self.btn_apply.clicked.connect(self.apply_current)
        self.btn_close.clicked.connect(self.close)
        
        if not self.baseline_path or not Path(self.baseline_path).exists():
            self.btn_baseline.setEnabled(False)
            
        self.update_summary()
        self.active_worker = None

    def update_summary(self):
        lines = [
            "Preview moves use the same convention as Apply Home Offset:",
            "the robot moves to zero pose first, then to -joint_offset.",
            "",
            f"Optimized result: {self.result_path}",
        ]
        if not self.baseline_path or not Path(self.baseline_path).exists():
            lines.append("Baseline reset: not found")
        else:
            lines.append(f"Baseline reset: {self.baseline_path}")
            try:
                opt_arm, opt_head = load_offset_from_json(str(self.result_path))
                base_arm, base_head = load_offset_from_json(str(self.baseline_path))
                if len(opt_arm) == len(base_arm):
                    diff_arm_deg = np.rad2deg(base_arm - opt_arm)
                    lines.append("")
                    lines.append("Baseline - Optimized arm diff (deg):")
                    lines.append(np.array2string(np.round(diff_arm_deg, 4), separator=", "))
                else:
                    lines.append("")
                    lines.append(
                        f"Arm diff unavailable: optimized has {len(opt_arm)} values, baseline has {len(base_arm)}."
                    )

                if opt_head is not None and base_head is not None:
                    if len(opt_head) == len(base_head):
                        diff_head_deg = np.rad2deg(base_head - opt_head)
                        lines.append("")
                        lines.append("Baseline - Optimized head diff (deg):")
                        lines.append(np.array2string(np.round(diff_head_deg, 4), separator=", "))
                    else:
                        lines.append("")
                        lines.append(
                            f"Head diff unavailable: optimized has {len(opt_head)} values, baseline has {len(base_head)}."
                        )
            except Exception as e:
                lines.append("")
                lines.append(f"Failed to compute diff: {e}")
        self.summary_box.setPlainText("\n".join(lines))

    def set_buttons_enabled(self, enabled):
        self.btn_baseline.setEnabled(enabled and (self.baseline_path and Path(self.baseline_path).exists()))
        self.btn_optimized.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        self.btn_close.setEnabled(enabled)

    def move_baseline(self):
        if not self.parent_app.robot: return
        self.set_buttons_enabled(False)
        self.parent_app.log_msg("[ApplyHome] Moving to Baseline Zero...")
        self.active_worker = MoveHomeOffsetWorker(
            self.parent_app.robot, self.parent_app.model, self.arm, self.baseline_path, self.include_head, "Baseline"
        )
        self.active_worker.finished_signal.connect(self.on_move_finished)
        self.active_worker.start()

    def move_optimized(self):
        if not self.parent_app.robot: return
        self.set_buttons_enabled(False)
        self.parent_app.log_msg("[ApplyHome] Moving to Optimized Zero...")
        self.active_worker = MoveHomeOffsetWorker(
            self.parent_app.robot, self.parent_app.model, self.arm, self.result_path, self.include_head, "Optimized"
        )
        self.active_worker.finished_signal.connect(self.on_move_finished)
        self.active_worker.start()

    def on_move_finished(self, success):
        self.set_buttons_enabled(True)
        if success:
            self.parent_app.log_msg("[ApplyHome] Move completed successfully.")
        else:
            self.parent_app.log_msg("[ApplyHome] Move failed.")

    def apply_current(self):
        if not self.parent_app.robot: return
        
        reply = QMessageBox.warning(
            self,
            "Confirm Apply",
            "This will apply the CURRENT physical joint pose as the new Zero/Home offsets.\n"
            "Make sure the robot has completed its motion and you have verified its alignment.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
            
        self.set_buttons_enabled(False)
        self.parent_app.log_msg("[ApplyHome] Resetting current pose home offsets...")
        self.active_worker = ApplyCurrentPoseWorker(
            self.parent_app.robot, self.parent_app.model, self.arm, self.include_head
        )
        self.active_worker.finished_signal.connect(self.on_apply_finished)
        self.active_worker.start()

    def on_apply_finished(self, result):
        self.set_buttons_enabled(True)
        if result and result.get("success"):
            self.parent_app.log_msg("[SUCCESS] Home offsets updated to current pose successfully.")
            QMessageBox.information(self, "Success", "Home offsets updated successfully!")
            self.accept()
        else:
            err = result.get("error", "Unknown error") if result else "Thread failed"
            self.parent_app.log_msg(f"[ERROR] Failed to apply home offsets: {err}")
            QMessageBox.critical(self, "Error", f"Failed to apply offsets:\n{err}")

# --- Background Worker Threads ---
class MoveHomeOffsetWorker(QThread):
    finished_signal = Signal(bool)
    
    def __init__(self, robot, model, arm, json_path, include_head, label):
        super().__init__()
        self.robot = robot
        self.model = model
        self.arm = arm
        self.json_path = json_path
        self.include_head = include_head
        self.label = label
        
    def run(self):
        try:
            print(f"[Worker] Moving to {self.label} Zero Pose using candidate path: {self.json_path}...")
            # Read offsets
            offset_arm, offset_head = load_offset_from_json(str(self.json_path))
            
            # Lock manager check
            self.robot.enable_control_manager(False)
            time.sleep(0.5)
            
            # Send movement command
            move_to_offset_candidate_from_json(
                self.robot,
                self.model,
                self.arm,
                offset_arm,
                offset_head if self.include_head else None,
                minimum_time=6.0
            )
            self.finished_signal.emit(True)
        except Exception as e:
            print(f"[Worker] Zero preview motion error: {e}")
            self.finished_signal.emit(False)

class ApplyCurrentPoseWorker(QThread):
    finished_signal = Signal(dict)
    
    def __init__(self, robot, model, arm, include_head):
        super().__init__()
        self.robot = robot
        self.model = model
        self.arm = arm
        self.include_head = include_head
        
    def run(self):
        try:
            # Send the reset command
            result = reset_current_pose_home_offsets(
                self.robot,
                self.model,
                arm=self.arm,
                include_head=self.include_head,
                log_cb=print
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.finished_signal.emit({"success": False, "error": str(e)})

class MoveCenterWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool)
    
    def __init__(self, calibrator, arm_side, stop_event, target_dist=300.0):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.stop_event = stop_event
        self.target_dist = target_dist
        
    def run(self):
        try:
            self.log_signal.emit("[INFO] Running Move to Center...")
            success = self.calibrator.move_center_3d(self.arm_side, self.stop_event, target_distance_mm=self.target_dist)
            self.finished_signal.emit(success)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Move Center failed: {e}")
            self.finished_signal.emit(False)

class MoveToReadyWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool)
    
    def __init__(self, calibrator, arm_side, mode=None):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode
        
    def run(self):
        try:
            self.log_signal.emit("[INFO] Moving to Sweep Ready Pose...")
            success = self.calibrator.go_to_ready_pose(self.arm_side, self.mode)
            self.finished_signal.emit(success)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Move to Ready failed: {e}")
            self.finished_signal.emit(False)

class ManualHeadWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool)
    
    def __init__(self, calibrator, yaw_rad, pitch_rad):
        super().__init__()
        self.calibrator = calibrator
        self.yaw_rad = yaw_rad
        self.pitch_rad = pitch_rad
        
    def run(self):
        try:
            self.log_signal.emit(f"[INFO] Moving head manually to Yaw={np.degrees(self.yaw_rad):.2f}°, Pitch={np.degrees(self.pitch_rad):.2f}°...")
            success = self.calibrator.move_head(self.calibrator.robot, self.yaw_rad, self.pitch_rad, minimum_time=2.0)
            self.finished_signal.emit(success)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Manual head control failed: {e}")
            self.finished_signal.emit(False)

class HomeOffsetResetWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(dict)
    
    def __init__(self, robot, model, model_name, include_head):
        super().__init__()
        self.robot = robot
        self.model = model
        self.model_name = model_name
        self.include_head = include_head
        
    def run(self):
        try:
            baseline_path, _ = save_home_reset_baseline_json(
                self.robot,
                self.model,
                BASE_DIR / "config",
                model_name=self.model_name,
                include_head=self.include_head
            )
            self.log_signal.emit(f"[INFO] Home reset baseline saved to: {baseline_path}")

            result = reset_current_pose_home_offsets(
                self.robot,
                self.model,
                arm="both",
                include_head=self.include_head,
                log_cb=self.log_signal.emit
            )
            result["baseline_path"] = baseline_path
            self.finished_signal.emit(result)
        except Exception as e:
            self.finished_signal.emit({"success": False, "error": str(e)})

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
            version_num = self.calibrator.get_robot_version()
            is_v13 = (abs(version_num - 1.3) < 0.05)
            
            is_mock_run = (not self.calibrator.robot or self.calibrator.robot == "mock_robot")
            if not is_mock_run:
                state = self.calibrator.robot.get_state()
                model = self.calibrator.robot.model()
                arm_idx = model.left_arm_idx if self.arm_side == "left" else model.right_arm_idx
                first_starting_pose = list(state.position[arm_idx])
            else:
                first_starting_pose = [0.0]*7
            
            res_4 = None
            if is_v13:
                # Stage 1 Axis 4 sweep starts immediately from the initial/current pose
                if getattr(self.calibrator, 'stop_requested', False):
                    self.finished_signal.emit(None)
                    return
                
                self.log_signal.emit("\n" + "="*50)
                self.log_signal.emit("   [Stage 1/3] Sweeping Axis 4 (Wrist Yaw)...")
                self.log_signal.emit("="*50 + "\n")
                res_4 = self.calibrator.perform_calibration_sweep(
                    self.arm_side, 4,
                    log_callback=self.log_signal.emit,
                    status_callback=self.status_signal.emit,
                    use_head_tracking=self.use_head_tracking
                )
                if not res_4:
                    self.log_signal.emit("[ERROR] Stage 1 (Axis 4) sweep failed. Aborting.")
                    self.finished_signal.emit(None)
                    return
                res_4['axis_mode'] = 4
                res_4['axis'] = res_4['axis_opt']
                
                if getattr(self.calibrator, 'stop_requested', False):
                    self.finished_signal.emit(None)
                    return
                
                time.sleep(1.0)
                
                # Move back to initial starting pose
                if not is_mock_run:
                    self.log_signal.emit("\n" + "="*50)
                    self.log_signal.emit("   [Stage 2/3] Returning to Initial Starting Pose...")
                    self.log_signal.emit("="*50 + "\n")
                    
                    if self.arm_side == "right":
                        success_other = self.calibrator.movej(self.calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                    else:
                        success_other = self.calibrator.movej(self.calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                    
                    success_move = self.calibrator.movej(self.calibrator.robot, **{f"{self.arm_side}_arm": first_starting_pose}, minimum_time=8.0, apply_offsets=False)
                    if not success_move:
                        self.log_signal.emit("[ERROR] Failed to return to initial pose. Aborting.")
                        self.finished_signal.emit(None)
                        return
                    time.sleep(1.5)
            
            # Stage 2/3: Axis 6
            if getattr(self.calibrator, 'stop_requested', False):
                self.finished_signal.emit(None)
                return
                
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit("   [Stage 2/3] Sweeping Axis 6 (Roll)..." if is_v13 else "   [Stage 1/2] Sweeping Axis 6 (Roll)...")
            self.log_signal.emit("="*50 + "\n")
            res_6 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 6, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking
            )
            if not res_6:
                self.log_signal.emit("[ERROR] Stage 6 sweep failed. Aborting.")
                self.finished_signal.emit(None)
                return
            res_6['axis_mode'] = 6
            res_6['axis'] = res_6['axis_opt']
            
            if getattr(self.calibrator, 'stop_requested', False):
                self.finished_signal.emit(None)
                return
            
            time.sleep(1.0)
            
            # Stage 3/3: Axis 5
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit("   [Stage 3/3] Sweeping Axis 5 (Pitch)..." if is_v13 else "   [Stage 2/2] Sweeping Axis 5 (Pitch)...")
            self.log_signal.emit("="*50 + "\n")
            
            res_5 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 5, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking
            )
            if not res_5:
                self.log_signal.emit("[ERROR] Stage 5 sweep failed. Aborting.")
                self.finished_signal.emit(None)
                return
                
            res_5['axis_mode'] = 5
            res_5['axis'] = res_5['axis_opt']
            
            # Compute unified bracket calibration
            self.log_signal.emit("\n[PROCESSING] Computing unified bracket calibration parameters...")
            unified_res = self.calibrator.compute_unified_bracket_calibration(
                res_5, res_6, self.arm_side, tolerance=self.tolerance, marker_data_4=res_4
            )
            
            unified_res['res_5'] = res_5
            unified_res['res_6'] = res_6
            if is_v13 and res_4 is not None:
                unified_res['res_4'] = res_4

            # Helper plot function
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
            
            # Plot results
            if is_v13 and res_4 is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                plot_single_axis(ax1, res_6, 6, 'blue')
                plot_single_axis(ax2, res_5, 5, 'green')
                plot_single_axis(ax3, res_4, 4, 'purple')
                fig.suptitle(f"Unified Marker Sweep Results ({self.arm_side.upper()} Arm)\n"
                             f"Y-Offset: {unified_res['y_e']:.2f} mm | Z-Offset: {unified_res['z_e']:.2f} mm\n"
                             f"Roll: {unified_res['roll_e']:.2f}° | Pitch: {unified_res['pitch_e']:.2f}° | Yaw: {unified_res['yaw_e']:.2f}°\n"
                             f"Opt d5: {unified_res.get('opt_delta_5', 0.0):.3f}° | Opt d6: {unified_res.get('opt_delta_6', 0.0):.3f}° | Min Radius: {unified_res.get('min_radius', 0.0):.2f} mm", fontsize=12, fontweight='bold')
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                plot_single_axis(ax1, res_6, 6, 'blue')
                plot_single_axis(ax2, res_5, 5, 'green')
                fig.suptitle(f"Unified Marker Sweep Results ({self.arm_side.upper()} Arm)\n"
                             f"Y-Offset: {unified_res['y_e']:.2f} mm | Z-Offset: {unified_res['z_e']:.2f} mm\n"
                             f"Roll: {unified_res['roll_e']:.2f}° | Pitch: {unified_res['pitch_e']:.2f}° | Yaw: {unified_res['yaw_e']:.2f}°", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            result_dir = os.path.join(current_dir, "core", "calibration", "result_img")
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
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit(f"   STARTING JOINT SWEEP ({self.mode.upper()})")
            self.log_signal.emit("="*50 + "\n")
            
            res = self.calibrator.perform_calibration_sweep(
                self.arm_side, 
                self.mode, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                sweep_duration=self.sweep_duration
            )
            
            if not res:
                self.log_signal.emit("[ERROR] Sweep failed. Aborting.")
                self.finished_signal.emit(None)
                return
                
            self.log_signal.emit("\n[PROCESSING] Computing joint calibration offset...")
            calib_res = self.calibrator.compute_calibration_offset(res, self.current_offset_deg)
            
            # Helper single axis plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(res['pts_2d'][:, 0], res['pts_2d'][:, 1], c='blue', label='Captured Points')
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
            ax.set_title(f"Joint Sweep ({self.mode.upper()}) Result\nRadius: {res['radius']:.2f}mm, RMSE: {res['rmse']:.3f}")
            ax.legend()
            plt.tight_layout()
            
            result_dir = os.path.join(current_dir, "core", "calibration", "result_img")
            os.makedirs(result_dir, exist_ok=True)
            plot_path = os.path.join(result_dir, f"circle_fit_{self.arm_side}_joint_{self.mode}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            calib_res['plot_path'] = plot_path
            calib_res['axis_mode'] = self.mode
            calib_res['radius'] = res['radius']
            calib_res['rmse'] = res['rmse']
            calib_res['pts_2d'] = res['pts_2d']
            calib_res['uc_opt'] = res['uc_opt']
            calib_res['vc_opt'] = res['vc_opt']
            
            self.finished_signal.emit(calib_res)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(None)

class AutoCollectionWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int) # (current, total)
    message_signal = Signal(str, str) # (title, msg)
    finished_signal = Signal(bool)
    
    def __init__(self, app, mode):
        super().__init__()
        self.app = app
        self.mode = mode # "init_pose" or "auto_motion"
        
    def run(self):
        try:
            if not self.app.robot:
                raise RuntimeError("Robot is not connected.")
                
            active_arms = self.app.get_active_arms()
            
            if self.mode == "init_pose":
                self.log_signal.emit(f"[INFO] Moving to auto init pose (active_arms: {active_arms})...")
                move_to_auto_ready_pose(
                    robot=self.app.robot,
                    active_arms=active_arms,
                    minimum_time=10.0,
                    priority=self.app.auto_config.priority
                )
                self.app.auto_ready_done = True
                self.app.auto_base_head_q = None
                
                # Fetch default steps if headless
                if len(self.app.model.head_idx) == 0:
                    self.app.auto_config.angle_step_deg = 4.0
                    self.app.auto_config.position_step_m = 0.02
                else:
                    self.app.auto_config.angle_step_deg = float(self.app.txt_angle_step.text())
                    self.app.auto_config.position_step_m = float(self.app.txt_pos_step.text())
                    
                self.app.auto_motion_plan = None
                
                if self.app.chk_head_on.isChecked():
                    head_cfg = get_head_config(self.app.model)
                    if head_cfg["head_idx"] is not None:
                        self.app.auto_base_head_q = self.app.robot.get_state().position[head_cfg["head_idx"]].copy()
                        self.log_signal.emit(f"[INFO] Auto base head pose: {np.round(np.rad2deg(self.app.auto_base_head_q), 3)}")
                    else:
                        self.app.auto_base_head_q = None
                        
                self.app.head_move_count = 0
                self.message_signal.emit(
                    "Teaching Required",
                    "Robot has moved to the initial pose.\n\n"
                    "Please use the Teaching button to adjust the robot's pose so that the marker is clearly visible to the camera.\n"
                    "Once adjusted, press 'Auto Motion' to start the sequence."
                )
                self.finished_signal.emit(True)
                
            elif self.mode == "auto_motion":
                if not self.app.auto_ready_done:
                    raise RuntimeError("Please move to Init Pose first.")
                    
                if self.app.auto_motion_plan is None or len(self.app.auto_motion_plan) == 0:
                    self.log_signal.emit("[INFO] Re-building auto motion plan based on current pose...")
                    self.app.auto_config.angle_step_deg = float(self.app.txt_angle_step.text())
                    self.app.auto_config.position_step_m = float(self.app.txt_pos_step.text())
                    self.app.auto_motion_plan = build_incremental_motion_plan(
                        self.app.robot, self.app.dyn_model, self.app.auto_config, active_arms
                    )
                
                total_steps = len(self.app.auto_motion_plan)
                self.log_signal.emit(f"[INFO] Auto collection loop started. Total steps = {total_steps}")
                
                while self.app.head_move_count < total_steps:
                    if self.app.auto_stop_requested:
                        self.log_signal.emit("[INFO] Auto Motion sequence stopped by user.")
                        break
                        
                    self.progress_signal.emit(self.app.head_move_count, total_steps)
                    step = self.app.auto_motion_plan[self.app.head_move_count]
                    
                    motion_info = execute_auto_motion_step(
                        robot=self.app.robot,
                        config=self.app.auto_config,
                        motion_plan_step=step,
                        active_arms=active_arms,
                        include_head_motion=self.app.chk_head_on.isChecked()
                    )
                    self.log_signal.emit(f"[INFO] Step {self.app.head_move_count + 1}/{total_steps}: {step['desc']}")
                    
                    if self.app.auto_stop_requested:
                        break
                        
                    # Capture sample
                    q_arm, q_head, T_meas = self.app.capture_one_sample()
                    if q_arm is not None:
                        self.app.shared_arm_q_list.append(q_arm)
                        if q_head is not None:
                            self.app.shared_head_q_list.append(q_head)
                        self.app.shared_T_list.append(T_meas)
                        self.log_signal.emit(f"[INFO] Capture success. Total samples = {len(self.app.shared_arm_q_list)}")
                    else:
                        self.log_signal.emit("[WARNING] Capture failed. Skipping pose.")
                        
                    self.app.head_move_count += 1
                    time.sleep(0.3)
                else:
                    self.log_signal.emit("[INFO] Auto collection loop completed successfully!")
                
                self.finished_signal.emit(True)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Auto Collection Error: {e}")
            self.finished_signal.emit(False)

class IntrinsicsCalibrationDialog(QDialog):
    log_signal = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Camera Intrinsics Calibration")
        self.resize(800, 600)
        self.setStyleSheet(parent.styleSheet())
        
        main_layout = QHBoxLayout(self)
        
        # Left Panel (Controls)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        ctrl_box = QGroupBox("Calibration Settings")
        ctrl_lay = QVBoxLayout()
        
        self.btn_int_monitor = QPushButton("ENABLE MONITORING")
        self.btn_int_monitor.clicked.connect(self.toggle_monitoring)
        ctrl_lay.addWidget(self.btn_int_monitor)
        
        self.lbl_captured = QLabel("Captured Frames: 0")
        self.lbl_captured.setFont(QFont("Segoe UI", 11, QFont.Bold))
        ctrl_lay.addWidget(self.lbl_captured)
        
        self.btn_int_capture = QPushButton("CAPTURE FRAME (C)")
        self.btn_int_capture.clicked.connect(self.capture_frame)
        ctrl_lay.addWidget(self.btn_int_capture)
        
        self.btn_int_reset = QPushButton("RESET CAPTURES")
        self.btn_int_reset.clicked.connect(self.reset_captures)
        ctrl_lay.addWidget(self.btn_int_reset)
        
        self.btn_int_run = QPushButton("RUN CALIBRATION")
        self.btn_int_run.setStyleSheet("background-color: #2979ff; color: white;")
        self.btn_int_run.clicked.connect(self.run_calibration)
        ctrl_lay.addWidget(self.btn_int_run)
        
        self.btn_int_save = QPushButton("SAVE PARAMETERS")
        self.btn_int_save.setStyleSheet("background-color: #4caf50; color: white;")
        self.btn_int_save.setEnabled(False)
        self.btn_int_save.clicked.connect(self.save_parameters)
        ctrl_lay.addWidget(self.btn_int_save)
        
        ctrl_box.setLayout(ctrl_lay)
        left_layout.addWidget(ctrl_box)
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box)
        
        main_layout.addLayout(left_layout, 1)
        
        # Right Panel (Feed)
        right_layout = QVBoxLayout()
        self.video_feed_label = QLabel("Feed Inactive")
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        self.video_feed_label.setStyleSheet("background-color: black; border: 1px solid #2d2d2d; border-radius: 6px;")
        right_layout.addWidget(self.video_feed_label, 1)
        
        main_layout.addLayout(right_layout, 2)
        
        # Log signal handler
        self.log_signal.connect(self.append_log)
        
        # Setup board config
        self.calibrator = parent.intrinsics_calibrator
        self.captured_images = []
        self.output_yaml = parent.output_yaml
        self.monitoring = False

    def toggle_monitoring(self):
        self.monitoring = not self.monitoring
        if self.monitoring:
            self.btn_int_monitor.setText("STOP MONITORING")
            self.btn_int_monitor.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold;")
            self.append_log("[INTRINSICS] Monitoring enabled.")
        else:
            self.btn_int_monitor.setText("ENABLE MONITORING")
            self.btn_int_monitor.setStyleSheet("background-color: #1e1e1e; color: white;")
            self.append_log("[INTRINSICS] Monitoring disabled.")
            self.video_feed_label.setText("Feed Inactive")

    def capture_frame(self):
        if not hasattr(self.parent_app, 'current_frame') or self.parent_app.current_frame is None:
            self.append_log("[ERROR] No camera frame available to capture!")
            return
        
        self.captured_images.append(self.parent_app.current_frame.copy())
        self.lbl_captured.setText(f"Captured Frames: {len(self.captured_images)}")
        self.append_log(f"[INTRINSICS] Frame {len(self.captured_images)} captured.")

    def reset_captures(self):
        self.captured_images.clear()
        self.lbl_captured.setText("Captured Frames: 0")
        self.btn_int_save.setEnabled(False)
        self.append_log("[INTRINSICS] Capture memory cleared.")

    def run_calibration(self):
        if len(self.captured_images) < 5:
            self.append_log("[ERROR] Need at least 5 frames to run calibration!")
            return
            
        self.append_log(f"[INTRINSICS] Running calibration on {len(self.captured_images)} images. Please wait...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        success = self.calibrator.run_calibration_with_images(self.captured_images, None)
        QApplication.restoreOverrideCursor()
        
        if success:
            self.append_log(f"[SUCCESS] Calibration complete! RMS Error: {self.calibrator.rms_error:.4f}")
            self.btn_int_save.setEnabled(True)
            self.show_verification()
        else:
            self.append_log("[ERROR] Calibration failed. Check images and board settings.")

    def save_parameters(self):
        if self.calibrator.cameraMatrix is None:
            self.append_log("[ERROR] No calibration data to save!")
            return
            
        try:
            data = {
                "camera_matrix": self.calibrator.cameraMatrix.tolist(),
                "dist_coeffs": self.calibrator.distCoeffs.tolist(),
                "rms_error": float(self.calibrator.rms_error),
                "width": int(self.captured_images[0].shape[1]),
                "height": int(self.captured_images[0].shape[0])
            }
            os.makedirs(os.path.dirname(self.output_yaml), exist_ok=True)
            with open(self.output_yaml, "w") as f:
                yaml.dump(data, f)
            self.append_log(f"[SUCCESS] Calibration saved successfully to: {self.output_yaml}")
            
            # Sync back to detector if present
            if self.parent_app.marker_detector is not None:
                self.parent_app.marker_detector.fx = self.calibrator.cameraMatrix[0, 0]
                self.parent_app.marker_detector.fy = self.calibrator.cameraMatrix[1, 1]
                self.parent_app.marker_detector.principal_point = [self.calibrator.cameraMatrix[0, 2], self.calibrator.cameraMatrix[1, 2]]
                self.parent_app.marker_detector.dist_coeffs = self.calibrator.distCoeffs
        except Exception as e:
            self.append_log(f"[ERROR] Save failed: {e}")

    def show_verification(self):
        if len(self.captured_images) == 0: return
        test_img = self.captured_images[-1]
        h, w = test_img.shape[:2]
        
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.calibrator.cameraMatrix, self.calibrator.distCoeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(test_img, self.calibrator.cameraMatrix, self.calibrator.distCoeffs, None, new_mtx)
        
        combined_res = np.hstack((test_img, undistorted))
        h_res, w_res = combined_res.shape[:2]
        
        grid_size = 60
        for y in range(0, h_res, grid_size):
            cv2.line(combined_res, (0, y), (w_res, y), (0, 255, 0), 1)
        for x in range(0, w_res, grid_size):
            cv2.line(combined_res, (x, 0), (x, h_res), (0, 255, 0), 1)
            
        cv2.putText(combined_res, "Original", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.text_w = w
        cv2.putText(combined_res, "Undistorted", (w + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(combined_res, f"RMS: {self.calibrator.rms_error:.4f}", (w + 30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        result_dir = os.path.join(current_dir, "core", "calibration", "result_img")
        os.makedirs(result_dir, exist_ok=True)
        save_path = os.path.join(result_dir, "camera_intrinsics_verification.png")
        cv2.imwrite(save_path, combined_res)
        
        pix = QPixmap(save_path).scaled(self.video_feed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_feed_label.setPixmap(pix)

    def append_log(self, text):
        self.log_box.append(text)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.capture_frame()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.monitoring = False
        super().closeEvent(event)

class MainWindow(QWidget):
    log_signal = Signal(str)
    
    def __init__(self, marker_st, robot, arm_side="right", ui_only=False):
        super().__init__()
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        self.ui_only = ui_only
        self.dyn_model = None
        self.model = None
        
        # Core Sweep Calibrator Instances
        self.marker_calibrator = MarkerCalibrator(marker_st, robot)
        self.joint_calibrator = JointCalibrator(marker_st, robot)
        
        # Intrinsics Calibrator
        self.intrinsics_calibrator = IntrinsicsCalibrator()
        self.intrinsics_calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 36.0, 27.0, "DICT_5X5_100")
        
        # Marker detector
        try:
            from marker_detection import Marker_Detection
            self.marker_detector = Marker_Detection()
            self.marker_detector.set_marker_type("plate")
        except ImportError:
            self.marker_detector = None
            
        self.monitor_enabled = False
        self.output_yaml = os.path.abspath(os.path.join(current_dir, "config", "camera_intrinsics.yaml"))
        
        # Auto Motion state data
        self.shared_arm_q_list = []
        self.shared_head_q_list = []
        self.shared_T_list = []
        self.head_move_count = 0
        self.auto_config = AutoCollectionConfig()
        self.auto_motion_plan = None
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.auto_motion_running = False
        self.auto_stop_requested = False
        self.last_result_path = None
        self.last_home_reset_path = None
        self.last_dataset_path = None
        self.dataset_saved_in_session = False
        
        self.joint_offsets = {"wrist_pitch": 0.0, "elbow": 0.0}
        self.joint_offsets_store = {
            "left": {"joint3": 0.0, "joint5": 0.0},
            "right": {"joint3": 0.0, "joint5": 0.0}
        }
        self.camera_config = {"mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]}
        self.load_offsets_from_yaml()
        
        self.setWindowTitle("Unified Robot Calibration Workspace")
        self.resize(1350, 800)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # Background threads
        self.active_worker = None
        self.auto_worker = None
        
        # Frame capture
        self.current_frame = None
        self.feed_dialog = None
        
        # Init UI elements
        self.init_ui()
        
        # Video timer & Poll timer
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_timer.start(33)
        
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera_status)
        self.poll_timer.start(200)

        # Log signal connection
        self.log_signal.connect(self.log_msg)

    def load_offsets_from_yaml(self):
        config_path = os.path.abspath(os.path.join(current_dir, "config", "setting.yaml"))
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                if data:
                    if "joint_offset" in data:
                        self.joint_offsets_store = data["joint_offset"]
                    if "camera" in data:
                        self.camera_config = data["camera"]
            
            # Sync current sweep offsets
            self.joint_offsets["wrist_pitch"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint5", 0.0)
            self.joint_offsets["elbow"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint3", 0.0)
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets
        except Exception as e:
            print(f"[ERROR] Failed to load setting.yaml: {e}")

    def save_offsets_to_yaml(self):
        config_path = os.path.abspath(os.path.join(current_dir, "config", "setting.yaml"))
        try:
            lines = []
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    lines = f.readlines()
            
            # Update joint_offset lines
            jo_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("joint_offset:"):
                    jo_idx = i
                    break

            if jo_idx == -1:
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

            # Update marker_offset lines in setting.yaml
            left_marker_str = f"[{self.txt_l_mx.text()}, {self.txt_l_my.text()}, {self.txt_l_mz.text()}, {self.txt_l_mr.text()}, {self.txt_l_mp.text()}, {self.txt_l_myaw.text()}]"
            right_marker_str = f"[{self.txt_r_mx.text()}, {self.txt_r_my.text()}, {self.txt_r_mz.text()}, {self.txt_r_mr.text()}, {self.txt_r_mp.text()}, {self.txt_r_myaw.text()}]"

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("Tf_to_marker_left:"):
                    indent = len(line) - len(line.lstrip())
                    comment = line.split("#")[1] if "#" in line else ""
                    comment_str = f" # {comment.strip()}" if comment else ""
                    lines[i] = " " * indent + f"Tf_to_marker_left: {left_marker_str}{comment_str}\n"
                elif stripped.startswith("Tf_to_marker_right:"):
                    indent = len(line) - len(line.lstrip())
                    comment = line.split("#")[1] if "#" in line else ""
                    comment_str = f" # {comment.strip()}" if comment else ""
                    lines[i] = " " * indent + f"Tf_to_marker_right: {right_marker_str}{comment_str}\n"

            with open(config_path, "w") as f:
                f.writelines(lines)
            self.log_msg("[SUCCESS] Offsets saved permanently to setting.yaml!")
            QMessageBox.information(self, "Success", "Offsets applied and saved to setting.yaml successfully.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save setting.yaml: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save setting.yaml:\n{e}")

    def init_ui(self):
        # Top-level main layout
        window_layout = QHBoxLayout(self)
        window_layout.setContentsMargins(10, 10, 10, 10)
        window_layout.setSpacing(10)
        
        # ==========================================
        # --- LEFT PANEL (Controls & Workflows) ---
        # ==========================================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)
        
        # 1. Robot Connection GroupBox
        conn_box = QGroupBox("Robot Connection")
        conn_grid = QGridLayout()
        conn_grid.setVerticalSpacing(4)
        
        conn_grid.addWidget(QLabel("IP / Port:"), 0, 0)
        self.ip_input = QLineEdit("192.168.30.1:50051")
        if self.ui_only:
            self.ip_input.setText("127.0.0.1:50051")
        conn_grid.addWidget(self.ip_input, 0, 1)
        
        conn_grid.addWidget(QLabel("Model:"), 0, 2)
        self.model_input = QComboBox()
        self.model_input.addItems(["a", "m"])
        conn_grid.addWidget(self.model_input, 0, 3)
        
        conn_grid.addWidget(QLabel("Mode:"), 1, 0)
        self.mode_input = QComboBox()
        self.mode_input.addItems(["live", "sim"])
        conn_grid.addWidget(self.mode_input, 1, 1)
        
        self.chk_servo_on = QCheckBox("Servo On")
        self.chk_servo_on.setChecked(True)
        conn_grid.addWidget(self.chk_servo_on, 1, 2)
        
        self.chk_head_on = QCheckBox("Head On")
        self.chk_head_on.setChecked(True)
        conn_grid.addWidget(self.chk_head_on, 1, 3)
        
        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold;")
        self.btn_connect.clicked.connect(self.connect_robot)
        conn_grid.addWidget(self.btn_connect, 2, 0, 1, 4)
        
        conn_box.setLayout(conn_grid)
        left_panel.addWidget(conn_box)
        
        # 2. Camera Status GroupBox
        cam_box = QGroupBox("Camera & Marker Status")
        cam_lay = QHBoxLayout()
        
        self.indicator = IndicatorWidget()
        cam_lay.addWidget(self.indicator)
        
        self.status_label = QLabel("Monitor Off")
        self.status_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.status_label.setStyleSheet("color: #ff1744;")
        cam_lay.addWidget(self.status_label)
        
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        cam_lay.addWidget(self.btn_monitor)
        
        self.btn_camera_feed = QPushButton("Camera Feed")
        self.btn_camera_feed.clicked.connect(self.toggle_camera_feed_dialog)
        cam_lay.addWidget(self.btn_camera_feed)
        
        self.temp_label = QLabel("Camera Temp: -- °C")
        cam_lay.addWidget(self.temp_label)
        
        cam_box.setLayout(cam_lay)
        left_panel.addWidget(cam_box)
        
        # 3. Bottom TabWidget (Main Calibration vs Sub Calibration)
        self.bottom_tabs = QTabWidget()
        
        # --- TAB 1: 메인 calibration ---
        main_tab = QWidget()
        main_tab_lay = QVBoxLayout()
        main_tab_lay.setContentsMargins(5, 5, 5, 5)
        
        # Actions
        act_box = QGroupBox("Actions")
        act_grid = QGridLayout()
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("background-color: #d50000; color: white;")
        self.btn_stop.clicked.connect(self.stop_all_motions)
        act_grid.addWidget(self.btn_stop, 0, 0)
        
        self.btn_clear = QPushButton("Clear Samples")
        self.btn_clear.clicked.connect(self.clear_samples)
        act_grid.addWidget(self.btn_clear, 0, 1)
        
        self.btn_zero = QPushButton("Zero")
        self.btn_zero.clicked.connect(self.show_zero_pose_check)
        act_grid.addWidget(self.btn_zero, 0, 2)
        
        self.btn_home_reset = QPushButton("Home Offset Reset")
        self.btn_home_reset.setStyleSheet("background-color: #e65100; color: white;")
        self.btn_home_reset.clicked.connect(self.dev_home_offset_reset)
        act_grid.addWidget(self.btn_home_reset, 0, 3)
        
        self.btn_step_init = QPushButton("1. init pose")
        self.btn_step_init.clicked.connect(self.run_init_pose)
        act_grid.addWidget(self.btn_step_init, 1, 0)
        
        self.btn_step_auto = QPushButton("2. auto motion")
        self.btn_step_auto.clicked.connect(self.run_auto_motion)
        act_grid.addWidget(self.btn_step_auto, 1, 1)
        
        self.btn_step_calc = QPushButton("3. calculate")
        self.btn_step_calc.clicked.connect(self.dev_calculate)
        act_grid.addWidget(self.btn_step_calc, 1, 2)
        
        self.btn_step_apply = QPushButton("4. apply home offset")
        self.btn_step_apply.clicked.connect(self.dev_apply_home_offset)
        act_grid.addWidget(self.btn_step_apply, 1, 3)
        
        self.btn_step_check = QPushButton("5. check calibration state")
        self.btn_step_check.clicked.connect(self.dev_check_calibration_state)
        act_grid.addWidget(self.btn_step_check, 2, 0, 1, 4)
        
        act_box.setLayout(act_grid)
        main_tab_lay.addWidget(act_box)
        
        # Options
        opt_box = QGroupBox("Options")
        opt_grid = QGridLayout()
        
        opt_grid.addWidget(QLabel("Part:"), 0, 0)
        self.cb_part = QComboBox()
        self.cb_part.addItems(["right_arm", "left_arm", "both_arm"])
        opt_grid.addWidget(self.cb_part, 0, 1)
        
        opt_grid.addWidget(QLabel("Auto Step Angle (deg):"), 0, 2)
        self.txt_angle_step = QLineEdit("5.0")
        self.txt_angle_step.setFixedWidth(50)
        opt_grid.addWidget(self.txt_angle_step, 0, 3)
        
        opt_grid.addWidget(QLabel("Auto Step Pose (m):"), 1, 2)
        self.txt_pos_step = QLineEdit("0.03")
        self.txt_pos_step.setFixedWidth(50)
        opt_grid.addWidget(self.txt_pos_step, 1, 3)
        
        self.btn_capture = QPushButton("self_capture_sample")
        self.btn_capture.clicked.connect(self.dev_record)
        opt_grid.addWidget(self.btn_capture, 1, 0, 1, 2)
        
        opt_grid.addWidget(QLabel("NPZ Path:"), 2, 0)
        self.npz_path_input = QLineEdit("result/dataset_latest.npz")
        opt_grid.addWidget(self.npz_path_input, 2, 1, 1, 2)
        
        self.btn_cal_npz = QPushButton("cal use npz")
        self.btn_cal_npz.setStyleSheet("background-color: #2e7d32; color: white;")
        self.btn_cal_npz.clicked.connect(self.dev_calculate_npz)
        opt_grid.addWidget(self.btn_cal_npz, 2, 3)
        
        opt_box.setLayout(opt_grid)
        main_tab_lay.addWidget(opt_box)
        
        main_tab.setLayout(main_tab_lay)
        self.bottom_tabs.addTab(main_tab, "메인 calibration")
        
        # --- TAB 2: 서브 calibration ---
        sub_tab = QWidget()
        sub_tab_lay = QVBoxLayout()
        sub_tab_lay.setContentsMargins(5, 5, 5, 5)
        
        # Camera Intrinsics and Manual Head Row
        row_lay = QHBoxLayout()
        
        self.btn_intrinsics = QPushButton("camera 내부파라미터 보정기능")
        self.btn_intrinsics.setStyleSheet("background-color: #1e88e5; color: white; padding: 10px;")
        self.btn_intrinsics.clicked.connect(self.open_intrinsics_dialog)
        row_lay.addWidget(self.btn_intrinsics)
        
        head_box = QGroupBox("Manual Head Control")
        head_lay = QGridLayout()
        head_lay.addWidget(QLabel("Yaw (deg):"), 0, 0)
        self.txt_head_yaw = QLineEdit("0.0")
        self.txt_head_yaw.setFixedWidth(50)
        head_lay.addWidget(self.txt_head_yaw, 0, 1)
        
        head_lay.addWidget(QLabel("Pitch (deg):"), 1, 0)
        self.txt_head_pitch = QLineEdit("0.0")
        self.txt_head_pitch.setFixedWidth(50)
        head_lay.addWidget(self.txt_head_pitch, 1, 1)
        
        self.btn_move_head = QPushButton("MOVE HEAD")
        self.btn_move_head.clicked.connect(self.move_head_manually)
        head_lay.addWidget(self.btn_move_head, 0, 2, 2, 1)
        head_box.setLayout(head_lay)
        row_lay.addWidget(head_box)
        
        sub_tab_lay.addLayout(row_lay)
        
        # Calibration Workflows (Joint and Marker Sweeps)
        work_box = QGroupBox("Sweep Workflows")
        work_lay = QGridLayout()
        
        work_lay.addWidget(QLabel("Workflow Side:"), 0, 0)
        self.cb_workflow_side = QComboBox()
        self.cb_workflow_side.addItems(["Right Arm", "Left Arm"])
        self.cb_workflow_side.currentTextChanged.connect(self.on_workflow_side_changed)
        work_lay.addWidget(self.cb_workflow_side, 0, 1)
        
        work_lay.addWidget(QLabel("Joint Calibration Mode:"), 1, 0)
        self.cb_joint_sweep_mode = QComboBox()
        self.cb_joint_sweep_mode.addItems(["wrist_pitch (5-Axis)", "elbow (3-Axis)"])
        work_lay.addWidget(self.cb_joint_sweep_mode, 1, 1)
        
        self.btn_joint_ready = QPushButton("MOVE TO READY (JOINT)")
        self.btn_joint_ready.clicked.connect(self.move_to_ready_pose_joint)
        work_lay.addWidget(self.btn_joint_ready, 1, 2)
        
        self.btn_joint_start = QPushButton("START JOINT SWEEP")
        self.btn_joint_start.clicked.connect(self.start_calibration_joint)
        work_lay.addWidget(self.btn_joint_start, 1, 3)
        
        work_lay.addWidget(QLabel("Marker Calibration Mode:"), 2, 0)
        self.cb_marker_sweep_mode = QComboBox()
        self.cb_marker_sweep_mode.addItems(["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"])
        work_lay.addWidget(self.cb_marker_sweep_mode, 2, 1)
        
        self.btn_marker_ready = QPushButton("MOVE TO READY (MARKER)")
        self.btn_marker_ready.clicked.connect(self.move_to_ready_pose_marker)
        work_lay.addWidget(self.btn_marker_ready, 2, 2)
        
        self.btn_marker_center = QPushButton("MOVE TO CENTER")
        self.btn_marker_center.clicked.connect(self.move_to_center_marker)
        work_lay.addWidget(self.btn_marker_center, 2, 3)
        
        self.btn_marker_start = QPushButton("START MARKER SWEEP (UNIFIED)")
        self.btn_marker_start.setStyleSheet("background-color: #3f51b5; color: white;")
        self.btn_marker_start.clicked.connect(self.start_calibration_marker)
        work_lay.addWidget(self.btn_marker_start, 3, 0, 1, 4)
        
        work_box.setLayout(work_lay)
        sub_tab_lay.addWidget(work_box)
        
        sub_tab.setLayout(sub_tab_lay)
        self.bottom_tabs.addTab(sub_tab, "서브 calibration")
        
        left_panel.addWidget(self.bottom_tabs)
        window_layout.addLayout(left_panel, 1)
        
        # ==========================================
        # --- RIGHT PANEL (Config & Output/Plots) ---
        # ==========================================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)
        
        # 1. Config GroupBox (Joint & Marker offsets)
        conf_box = QGroupBox("Configuration Offsets")
        conf_grid = QGridLayout()
        conf_grid.setVerticalSpacing(4)
        
        # Joint Offsets
        conf_grid.addWidget(QLabel("LEFT Arm Joint Offsets (deg):"), 0, 0)
        conf_grid.addWidget(QLabel("Joint 3:"), 0, 1)
        self.txt_l_j3 = QLineEdit("0.0")
        self.txt_l_j3.setFixedWidth(60)
        conf_grid.addWidget(self.txt_l_j3, 0, 2)
        
        conf_grid.addWidget(QLabel("Joint 5:"), 0, 3)
        self.txt_l_j5 = QLineEdit("0.0")
        self.txt_l_j5.setFixedWidth(60)
        conf_grid.addWidget(self.txt_l_j5, 0, 4)
        
        conf_grid.addWidget(QLabel("RIGHT Arm Joint Offsets (deg):"), 1, 0)
        conf_grid.addWidget(QLabel("Joint 3:"), 1, 1)
        self.txt_r_j3 = QLineEdit("0.0")
        self.txt_r_j3.setFixedWidth(60)
        conf_grid.addWidget(self.txt_r_j3, 1, 2)
        
        conf_grid.addWidget(QLabel("Joint 5:"), 1, 3)
        self.txt_r_j5 = QLineEdit("0.0")
        self.txt_r_j5.setFixedWidth(60)
        conf_grid.addWidget(self.txt_r_j5, 1, 4)
        
        # Marker Offsets
        conf_grid.addWidget(QLabel("LEFT Marker Offset (X,Y,Z / R,P,Y):"), 2, 0)
        self.txt_l_mx = QLineEdit("0.0"); self.txt_l_mx.setFixedWidth(50)
        self.txt_l_my = QLineEdit("0.0"); self.txt_l_my.setFixedWidth(50)
        self.txt_l_mz = QLineEdit("0.0"); self.txt_l_mz.setFixedWidth(50)
        self.txt_l_mr = QLineEdit("0.0"); self.txt_l_mr.setFixedWidth(50)
        self.txt_l_mp = QLineEdit("0.0"); self.txt_l_mp.setFixedWidth(50)
        self.txt_l_myaw = QLineEdit("0.0"); self.txt_l_myaw.setFixedWidth(50)
        
        m_lay_l = QHBoxLayout()
        m_lay_l.addWidget(self.txt_l_mx); m_lay_l.addWidget(self.txt_l_my); m_lay_l.addWidget(self.txt_l_mz)
        m_lay_l.addWidget(self.txt_l_mr); m_lay_l.addWidget(self.txt_l_mp); m_lay_l.addWidget(self.txt_l_myaw)
        conf_grid.addLayout(m_lay_l, 2, 1, 1, 4)
        
        conf_grid.addWidget(QLabel("RIGHT Marker Offset (X,Y,Z / R,P,Y):"), 3, 0)
        self.txt_r_mx = QLineEdit("0.0"); self.txt_r_mx.setFixedWidth(50)
        self.txt_r_my = QLineEdit("0.0"); self.txt_r_my.setFixedWidth(50)
        self.txt_r_mz = QLineEdit("0.0"); self.txt_r_mz.setFixedWidth(50)
        self.txt_r_mr = QLineEdit("0.0"); self.txt_r_mr.setFixedWidth(50)
        self.txt_r_mp = QLineEdit("0.0"); self.txt_r_mp.setFixedWidth(50)
        self.txt_r_myaw = QLineEdit("0.0"); self.txt_r_myaw.setFixedWidth(50)
        
        m_lay_r = QHBoxLayout()
        m_lay_r.addWidget(self.txt_r_mx); m_lay_r.addWidget(self.txt_r_my); m_lay_r.addWidget(self.txt_r_mz)
        m_lay_r.addWidget(self.txt_r_mr); m_lay_r.addWidget(self.txt_r_mp); m_lay_r.addWidget(self.txt_r_myaw)
        conf_grid.addLayout(m_lay_r, 3, 1, 1, 4)
        
        self.btn_apply_offsets = QPushButton("Apply Offsets")
        self.btn_apply_offsets.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold;")
        self.btn_apply_offsets.clicked.connect(self.save_offsets_to_yaml)
        conf_grid.addWidget(self.btn_apply_offsets, 4, 0, 1, 5)
        
        conf_box.setLayout(conf_grid)
        right_panel.addWidget(conf_box)
        
        # Load values into config text fields
        self.load_offsets_to_ui()
        
        # 2. TabWidget for Logs vs Plots
        self.right_tabs = QTabWidget()
        
        # Tab 1: Terminal Log
        log_tab = QWidget()
        log_lay = QVBoxLayout()
        log_lay.setContentsMargins(5, 5, 5, 5)
        self.terminal_log = QTextEdit()
        self.terminal_log.setReadOnly(True)
        log_lay.addWidget(self.terminal_log)
        log_tab.setLayout(log_lay)
        self.right_tabs.addTab(log_tab, "Terminal Log & Results")
        
        # Tab 2: Plots / Images
        plot_tab = QWidget()
        plot_lay = QVBoxLayout()
        plot_lay.setContentsMargins(5, 5, 5, 5)
        self.plot_display = QLabel("No Fitting Plots Available")
        self.plot_display.setAlignment(Qt.AlignCenter)
        self.plot_display.setStyleSheet("background-color: black; border: 1px solid #2d2d2d; border-radius: 6px;")
        plot_lay.addWidget(self.plot_display)
        plot_tab.setLayout(plot_lay)
        self.right_tabs.addTab(plot_tab, "Plots & Circle Fitting")
        
        right_panel.addWidget(self.right_tabs, 1)
        window_layout.addLayout(right_panel, 2)

    def load_offsets_to_ui(self):
        # Populate joint offsets
        self.txt_l_j3.setText(f"{self.joint_offsets_store.get('left', {}).get('joint3', 0.0):.6f}")
        self.txt_l_j5.setText(f"{self.joint_offsets_store.get('left', {}).get('joint5', 0.0):.6f}")
        self.txt_r_j3.setText(f"{self.joint_offsets_store.get('right', {}).get('joint3', 0.0):.6f}")
        self.txt_r_j5.setText(f"{self.joint_offsets_store.get('right', {}).get('joint5', 0.0):.6f}")
        
        # Populate marker offsets
        lm = self.camera_config.get("Tf_to_marker_left", [0.0]*6)
        self.txt_l_mx.setText(f"{lm[0]:.5f}")
        self.txt_l_my.setText(f"{lm[1]:.5f}")
        self.txt_l_mz.setText(f"{lm[2]:.5f}")
        self.txt_l_mr.setText(f"{lm[3]:.2f}")
        self.txt_l_mp.setText(f"{lm[4]:.2f}")
        self.txt_l_myaw.setText(f"{lm[5]:.2f}")
        
        rm = self.camera_config.get("Tf_to_marker_right", [0.0]*6)
        self.txt_r_mx.setText(f"{rm[0]:.5f}")
        self.txt_r_my.setText(f"{rm[1]:.5f}")
        self.txt_r_mz.setText(f"{rm[2]:.5f}")
        self.txt_r_mr.setText(f"{rm[3]:.2f}")
        self.txt_r_mp.setText(f"{rm[4]:.2f}")
        self.txt_r_myaw.setText(f"{rm[5]:.2f}")

    def on_workflow_side_changed(self, text):
        self.arm_side = "left" if text == "Left Arm" else "right"
        self.load_offsets_from_yaml()
        self.log_msg(f"[INFO] Workflow Active Arm side changed to: {self.arm_side}")

    def log_msg(self, msg):
        self.terminal_log.append(msg)

    # Robot Connection Logic
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
                    self.robot = BaseCalibrator.initialize_robot(addr, model)
                except Exception as e:
                    self.log_msg(f"[INFO] Actual robot connection raised exception: {e}.")
                    self.robot = None
                
                if not self.robot:
                    self.log_msg("[INFO] Actual robot connection failed. Fallback to mock.")
                    self.robot = "mock_robot"
            else:
                self.robot = BaseCalibrator.initialize_robot(addr, model)
                
            if self.robot:
                self.marker_calibrator.robot = self.robot
                self.joint_calibrator.robot = self.robot
                
                # Determine version classification automatically
                detected_version = 1.2
                if self.robot != "mock_robot":
                    self.dyn_model = self.robot.get_dynamics()
                    self.model = self.robot.model()
                    
                    try:
                        robot_info = self.robot.get_robot_info()
                        raw_version = robot_info.robot_model_version
                        self.log_msg(f"[INFO] Connected robot model version: '{raw_version}'")
                        if "1.3" in raw_version:
                            detected_version = 1.3
                    except Exception as e:
                        self.log_msg(f"[WARNING] Failed to query version from robot: {e}")
                else:
                    self.dyn_model = None
                    self.model = "mock"
                    detected_version = 1.3 # Mock default
                
                self.marker_calibrator.robot_version = detected_version
                self.joint_calibrator.robot_version = detected_version
                
                # Auto step defaults based on head presence
                if self.model and self.model != "mock":
                    if len(self.model.head_idx) == 0:
                        self.txt_angle_step.setText("4.0")
                        self.txt_pos_step.setText("0.02")
                        self.log_msg("[INFO] Headless robot detected. Setting auto motion defaults: angle=4.0°, pose=0.02m.")
                
                # Setup SimulatedMarkerTransform if simulator is connected in UI Mode
                if self.ui_only and self.robot != "mock_robot":
                    self.marker_st = SimulatedMarkerTransform(self.robot, self.marker_calibrator.camera_config)
                    self.marker_calibrator.marker_st = self.marker_st
                    self.joint_calibrator.marker_st = self.marker_st
                    self.log_msg("[INFO] Configured SimulatedMarkerTransform.")
                
                self.btn_connect.setText("DISCONNECT")
                self.btn_connect.setStyleSheet("background-color: #e57373; color: black; font-weight: bold;")
                self.log_msg(f"[SUCCESS] Robot connected! (model version classified: {detected_version:.1f})")
                
                # Perform Servo-On automatically if checked
                if self.chk_servo_on.isChecked() and self.robot != "mock_robot":
                    try:
                        parts = ["mobile_.*|torso_.*|right_arm_.*|left_arm_.*"]
                        if self.chk_head_on.isChecked():
                            parts.append("head_.*")
                        regex = "|".join(parts)
                        self.robot.power_on(regex)
                        self.robot.servo_on(regex)
                        self.log_msg("[INFO] Servo-on completed successfully.")
                    except Exception as e:
                        self.log_msg(f"[WARNING] Failed to servo-on: {e}")
        except Exception as e:
            self.log_msg(f"[ERROR] Connection failed: {e}")
            QMessageBox.critical(self, "Connection Error", f"Failed to connect:\n{e}")

    # Camera Monitor Loop
    def on_monitor_toggled(self, checked):
        self.monitor_enabled = checked
        if checked:
            self.btn_monitor.setText("Marker Monitor: ON")
            self.btn_monitor.setStyleSheet("background-color: #2e7d32; color: white;")
            self.log_msg("[INFO] Live tracking monitor enabled.")
        else:
            self.btn_monitor.setText("Marker Monitor: OFF")
            self.btn_monitor.setStyleSheet("background-color: #1e1e1e; color: white;")
            self.log_msg("[INFO] Live tracking monitor disabled.")
            self.update_marker_indicator(False)

    def update_marker_indicator(self, detected):
        self.indicator.set_detected(detected)
        if detected:
            self.status_label.setText("Detected")
            self.status_label.setStyleSheet("color: #00e676;")
        else:
            self.status_label.setText("Not Detected" if self.monitor_enabled else "Monitor Off")
            self.status_label.setStyleSheet("color: #ff1744;" if self.monitor_enabled else "color: #888888;")

    def toggle_camera_feed_dialog(self):
        if self.feed_dialog is None:
            self.feed_dialog = CameraFeedDialog(self)
            self.feed_dialog.show()
            self.log_msg("[INFO] Opened Live camera feed dialog.")
        else:
            self.feed_dialog.close()
            self.feed_dialog = None

    def on_feed_dialog_closed(self):
        self.feed_dialog = None
        self.log_msg("[INFO] Live camera feed dialog closed.")

    def poll_camera_status(self):
        # Monitor off check
        if not self.monitor_enabled:
            return
            
        try:
            if not self.ui_only and self.marker_st is not None:
                res = self.marker_st.get_marker_transform(sampling_time=0, side=self.arm_side)
                detected = bool(res and len(res) > 0)
                self.update_marker_indicator(detected)
                
                if detected:
                    pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                    x, y, z = pose[:3, 3] * 1000.0
                    self.log_msg(f"[LIVE] Marker X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm")
                    
                self.marker_st.camera.get_color_image()
                temp = self.marker_st.camera.get_camera_temperature()
                if temp:
                    self.temp_label.setText(f"Camera Temp: {temp:.1f} °C")
            else:
                # Mock detection in UI Mode
                self.update_marker_indicator(True)
                self.temp_label.setText("Camera Temp: 34.5 °C")
        except Exception:
            pass

    def update_video_frame(self):
        # Update intrinsics calibration dialog or live popup if open
        feed_dialog_open = (self.feed_dialog is not None and self.feed_dialog.isVisible())
        intrinsics_open = hasattr(self, 'int_cal_dialog') and self.int_cal_dialog is not None and self.int_cal_dialog.isVisible() and self.int_cal_dialog.monitoring
        
        if not feed_dialog_open and not intrinsics_open:
            return

        if not self.ui_only and self.marker_st is not None:
            self.marker_st.camera.capture_image()
            img = self.marker_st.camera.get_color_image()
        else:
            # Generate mock checkerboard or text image
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "SIMULATION / UI-ONLY FEED", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (120, 120, 120), 3)
            
        if img is None: return
        self.current_frame = img.copy()
        
        # Convert OpenCV frame to QPixmap
        h, w, ch = img.shape
        bytes_per_line = ch * w
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        if feed_dialog_open:
            w_lbl = max(20, self.feed_dialog.lbl_feed.width())
            h_lbl = max(20, self.feed_dialog.lbl_feed.height())
            self.feed_dialog.lbl_feed.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))
            
        if intrinsics_open:
            w_lbl = max(20, self.int_cal_dialog.video_feed_label.width())
            h_lbl = max(20, self.int_cal_dialog.video_feed_label.height())
            self.int_cal_dialog.video_feed_label.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))

    # Intrinsics dialog handler
    def open_intrinsics_dialog(self):
        self.int_cal_dialog = IntrinsicsCalibrationDialog(self)
        self.int_cal_dialog.show()

    # Manual Head Movement
    def move_head_manually(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        try:
            yaw = np.radians(float(self.txt_head_yaw.text()))
            pitch = np.radians(float(self.txt_head_pitch.text()))
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid yaw/pitch angles.")
            return
            
        self.btn_move_head.setEnabled(False)
        self.active_worker = ManualHeadWorker(self.marker_calibrator, yaw, pitch)
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(lambda success: self.btn_move_head.setEnabled(True))
        self.active_worker.start()

    # Joint Sweep Handlers
    def move_to_ready_pose_joint(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
        
        mode = "wrist_pitch" if "wrist_pitch" in self.cb_joint_sweep_mode.currentText() else "elbow"
        self.btn_joint_ready.setEnabled(False)
        self.active_worker = MoveToReadyWorker(self.joint_calibrator, self.arm_side, mode)
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(lambda success: self.btn_joint_ready.setEnabled(True))
        self.active_worker.start()

    def start_calibration_joint(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        mode = "wrist_pitch" if "wrist_pitch" in self.cb_joint_sweep_mode.currentText() else "elbow"
        current_offset = self.joint_offsets.get(mode, 0.0)
        
        self.btn_joint_start.setEnabled(False)
        self.active_worker = JointCalibrationWorker(self.joint_calibrator, self.arm_side, mode, self.ui_only, current_offset)
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_joint_sweep_finished)
        self.active_worker.start()

    def on_joint_sweep_finished(self, res):
        self.btn_joint_start.setEnabled(True)
        if not res:
            self.log_msg("[ERROR] Joint sweep calibration failed.")
            return
            
        mode = res['axis_mode']
        offset_val = res['offset_deg']
        self.log_msg(f"[SUCCESS] Calculated Joint Offset ({mode}): {offset_val:.4f}°")
        
        # Display plot inside Plots tab
        if 'plot_path' in res:
            pix = QPixmap(res['plot_path']).scaled(self.plot_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.plot_display.setPixmap(pix)
            self.right_tabs.setCurrentIndex(1)
            
        # Update config offset fields in UI
        if self.arm_side == "left":
            if mode == "elbow": self.txt_l_j3.setText(f"{offset_val:.6f}")
            else: self.txt_l_j5.setText(f"{offset_val:.6f}")
        else:
            if mode == "elbow": self.txt_r_j3.setText(f"{offset_val:.6f}")
            else: self.txt_r_j5.setText(f"{offset_val:.6f}")
            
        # Sync back to internal memory
        self.joint_offsets_store[self.arm_side][f"joint{3 if mode == 'elbow' else 5}"] = float(offset_val)
        self.load_offsets_from_yaml()

    # Marker Sweep Handlers
    def move_to_ready_pose_marker(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        self.btn_marker_ready.setEnabled(False)
        self.active_worker = MoveToReadyWorker(self.marker_calibrator, self.arm_side, "marker")
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(lambda success: self.btn_marker_ready.setEnabled(True))
        self.active_worker.start()

    def move_to_center_marker(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        self.btn_marker_center.setEnabled(False)
        self.active_worker = MoveCenterWorker(self.marker_calibrator, self.arm_side, threading.Event())
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(lambda success: self.btn_marker_center.setEnabled(True))
        self.active_worker.start()

    def start_calibration_marker(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        self.btn_marker_start.setEnabled(False)
        self.active_worker = MarkerCalibrationWorker(self.marker_calibrator, self.arm_side, use_head_tracking=True)
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_marker_sweep_finished)
        self.active_worker.start()

    def on_marker_sweep_finished(self, res):
        self.btn_marker_start.setEnabled(True)
        if not res:
            self.log_msg("[ERROR] Unified marker calibration failed.")
            return
            
        self.log_msg("\n" + "="*50)
        self.log_msg("       UNIFIED BRACKET CALIBRATION RESULTS")
        self.log_msg("="*50)
        self.log_msg(f"    - X-Offset: {res['x_e']:.2f} mm")
        self.log_msg(f"    - Y-Offset: {res['y_e']:.2f} mm")
        self.log_msg(f"    - Z-Offset: {res['z_e']:.2f} mm")
        self.log_msg(f"    - Roll : {res['roll_e']:.2f} deg")
        self.log_msg(f"    - Pitch: {res['pitch_e']:.2f} deg")
        self.log_msg(f"    - Yaw  : {res['yaw_e']:.2f} deg")
        if 'opt_delta_5' in res:
            self.log_msg(f"    - Opt Delta 5 (5축 오프셋): {res['opt_delta_5']:.3f} deg")
            self.log_msg(f"    - Opt Delta 6 (6축 오프셋): {res['opt_delta_6']:.3f} deg")
            
        # Update config fields in UI
        if self.arm_side == "left":
            self.txt_l_mx.setText(f"{res['x_e']/1000.0:.5f}")
            self.txt_l_my.setText(f"{res['y_e']/1000.0:.5f}")
            self.txt_l_mz.setText(f"{res['z_e']/1000.0:.5f}")
            self.txt_l_mr.setText(f"{res['roll_e']:.2f}")
            self.txt_l_mp.setText(f"{res['pitch_e']:.2f}")
            self.txt_l_myaw.setText(f"{res['yaw_e']:.2f}")
        else:
            self.txt_r_mx.setText(f"{res['x_e']/1000.0:.5f}")
            self.txt_r_my.setText(f"{res['y_e']/1000.0:.5f}")
            self.txt_r_mz.setText(f"{res['z_e']/1000.0:.5f}")
            self.txt_r_mr.setText(f"{res['roll_e']:.2f}")
            self.txt_r_mp.setText(f"{res['pitch_e']:.2f}")
            self.txt_r_myaw.setText(f"{res['yaw_e']:.2f}")
            
        # Load plots
        if 'plot_path_combined' in res:
            pix = QPixmap(res['plot_path_combined']).scaled(self.plot_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.plot_display.setPixmap(pix)
            self.right_tabs.setCurrentIndex(1)

    # Main Calibration Workflows
    def stop_all_motions(self):
        self.auto_stop_requested = True
        if self.robot and self.robot != "mock_robot":
            try:
                self.robot.cancel_control()
                self.log_msg("[INFO] Robot motion cancelled.")
            except Exception as e:
                self.log_msg(f"[ERROR] Cancel control failed: {e}")
        self.auto_motion_running = False

    def clear_samples(self):
        self.shared_arm_q_list.clear()
        self.shared_head_q_list.clear()
        self.shared_T_list.clear()
        self.head_move_count = 0
        self.auto_motion_plan = None
        self.auto_ready_done = False
        self.log_msg("[INFO] All recorded calibration samples cleared.")

    def run_init_pose(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        self.auto_stop_requested = False
        self.btn_step_init.setEnabled(False)
        self.auto_worker = AutoCollectionWorker(self, "init_pose")
        self.auto_worker.log_signal.connect(self.log_msg)
        self.auto_worker.message_signal.connect(lambda title, msg: QMessageBox.information(self, title, msg))
        self.auto_worker.finished_signal.connect(lambda success: self.btn_step_init.setEnabled(True))
        self.auto_worker.start()

    def run_auto_motion(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        self.auto_stop_requested = False
        self.btn_step_auto.setEnabled(False)
        self.auto_worker = AutoCollectionWorker(self, "auto_motion")
        self.auto_worker.log_signal.connect(self.log_msg)
        self.auto_worker.finished_signal.connect(self.on_auto_motion_finished)
        self.auto_worker.start()

    def on_auto_motion_finished(self, success):
        self.btn_step_auto.setEnabled(True)
        if success:
            self.log_msg("[SUCCESS] Auto motion sample collection sequence complete.")
            self.auto_save_current_dataset()
        else:
            self.log_msg("[WARNING] Auto motion sequence failed or stopped.")

    def auto_save_current_dataset(self):
        if len(self.shared_arm_q_list) == 0:
            return
            
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        dataset_path = RESULT_DIR / "dataset_latest.npz"
        
        q_arm_list = np.array(self.shared_arm_q_list)
        q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
        T_meas_list = np.array(self.shared_T_list)
        
        active_arms = self.get_active_arms()
        # Slicing if single arm mode
        if len(active_arms) == 1:
            if q_arm_list.shape[1] == 14:
                if active_arms[0] == "right":
                    q_arm_list = q_arm_list[:, :7]
                else:
                    q_arm_list = q_arm_list[:, 7:]
            if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                if active_arms[0] == "right":
                    T_meas_list = T_meas_list[:, 0]
                else:
                    T_meas_list = T_meas_list[:, 1]
                    
        try:
            save_npz_dataset(str(dataset_path), q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = dataset_path
            self.log_msg(f"[INFO] Auto-saved current session dataset to: {dataset_path}")
        except Exception as e:
            self.log_msg(f"[WARNING] Failed to auto-save dataset: {e}")

    def capture_one_sample(self):
        if not self.ui_only and self.marker_st is None:
            self.marker_st = create_live_marker_transform()
            
        q_arm = None
        q_head = None
        T_meas = None
        
        active_arms = self.get_active_arms()
        
        if self.ui_only:
            # Mock measurements
            q_full = np.zeros(20)
            if self.model and self.model != "mock":
                state = self.robot.get_state()
                q_full = state.position
                
            q_arm = q_full[1:15] if len(active_arms) == 2 else (q_full[1:8] if active_arms[0] == "right" else q_full[8:15])
            if self.chk_head_on.isChecked() and self.model and self.model != "mock":
                head_cfg = get_head_config(self.model)
                q_head = q_full[head_cfg["head_idx"]]
            else:
                q_head = np.zeros(2)
                
            T_meas = np.eye(4) if len(active_arms) == 1 else np.array([np.eye(4), np.eye(4)])
        else:
            try:
                # Actual capture
                res = self.marker_st.get_marker_transform(sampling_time=0.5, side=self.arm_side)
                if not res or len(res) == 0:
                    return None, None, None
                    
                state = self.robot.get_state()
                q_full = state.position
                model = self.robot.model()
                
                if len(active_arms) == 2:
                    q_arm = q_full[model.right_arm_idx[:7] + model.left_arm_idx[:7]]
                    T_meas = np.array([res.get("right", np.eye(4)), res.get("left", np.eye(4))])
                else:
                    arm_idx = model.right_arm_idx if active_arms[0] == "right" else model.left_arm_idx
                    q_arm = q_full[arm_idx[:7]]
                    T_meas = res.get(active_arms[0], np.eye(4))
                    
                if self.chk_head_on.isChecked():
                    head_cfg = get_head_config(model)
                    q_head = q_full[head_cfg["head_idx"]]
            except Exception as e:
                self.log_msg(f"[ERROR] Capture failed: {e}")
                return None, None, None
                
        return q_arm, q_head, T_meas

    def dev_record(self):
        q_arm, q_head, T_meas = self.capture_one_sample()
        if q_arm is None:
            QMessageBox.warning(self, "Capture Failed", "Could not capture camera marker transform sample.")
            return
            
        self.shared_arm_q_list.append(q_arm)
        if q_head is not None:
            self.shared_head_q_list.append(q_head)
        self.shared_T_list.append(T_meas)
        self.log_msg(f"[INFO] Manually captured sample {len(self.shared_arm_q_list)} successfully.")
        self.auto_save_current_dataset()

    def get_active_arms(self):
        part = self.cb_part.currentText()
        if part == "both_arm": return ["right", "left"]
        return [part.replace("_arm", "")]

    def get_target_arm_str(self):
        part = self.cb_part.currentText()
        if part == "both_arm": return "both"
        return "right" if "right" in part else "left"

    def dev_calculate(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        if len(self.shared_arm_q_list) == 0:
            QMessageBox.warning(self, "No Samples", "No recorded samples to calculate.")
            return
            
        q_arm_list = np.array(self.shared_arm_q_list)
        q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
        T_meas_list = np.array(self.shared_T_list)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULT_DIR / f"result_{timestamp}.json"
        
        try:
            self.log_msg("[INFO] Starting optimization on active samples...")
            self.run_optimizer(
                active_arms=self.get_active_arms(),
                optimize_head=self.chk_head_on.isChecked(),
                optimize_camera=False,
                q_arm_list=q_arm_list,
                q_head_list=q_head_list,
                T_meas_list=T_meas_list,
                result_path=result_path
            )
            QMessageBox.information(self, "Calculation Complete", f"Calibration calculation finished. Result saved to:\n{result_path}")
        except Exception as e:
            self.log_msg(f"[ERROR] Calibration calculation failed: {e}")
            QMessageBox.critical(self, "Error", f"Calculation failed:\n{e}")

    def dev_calculate_npz(self):
        npz_raw_path = self.npz_path_input.text().strip()
        npz_path = Path(npz_raw_path)
        if not npz_path.exists():
            # Fallback relative to workspace root
            npz_path = BASE_DIR / npz_raw_path
            
        if not npz_path.exists():
            QMessageBox.critical(self, "File Not Found", f"Specified NPZ file not found at: {npz_raw_path}")
            return
            
        try:
            self.log_msg(f"[INFO] Loading NPZ dataset from: {npz_path}")
            q_arm_list, q_head_list, T_meas_list = load_npz_dataset(str(npz_path))
            
            active_arms = self.get_active_arms()
            if len(active_arms) == 1:
                if q_arm_list.shape[1] == 14:
                    if active_arms[0] == "right":
                        q_arm_list = q_arm_list[:, :7]
                    else:
                        q_arm_list = q_arm_list[:, 7:]
                if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                    if active_arms[0] == "right":
                        T_meas_list = T_meas_list[:, 0]
                    else:
                        T_meas_list = T_meas_list[:, 1]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            result_path = RESULT_DIR / f"result_{timestamp}.json"
            
            self.run_optimizer(
                active_arms=active_arms,
                optimize_head=self.chk_head_on.isChecked(),
                optimize_camera=False,
                q_arm_list=q_arm_list,
                q_head_list=q_head_list,
                T_meas_list=T_meas_list,
                result_path=result_path
            )
            QMessageBox.information(self, "Calculation Complete", f"NPZ Calibration calculation finished. Result saved to:\n{result_path}")
        except Exception as e:
            self.log_msg(f"[ERROR] NPZ Calibration calculation failed: {e}")
            QMessageBox.critical(self, "Error", f"NPZ Calculation failed:\n{e}")

    def run_optimizer(self, active_arms, optimize_head, optimize_camera, q_arm_list, q_head_list, T_meas_list, result_path):
        if self.model is None or self.model == "mock":
            # Mock calculations in simulator mode
            self.log_msg("[INFO] (UI Mode) Mocking calibration optimizer outcomes...")
            self.last_result_path = result_path
            # Write a dummy json
            result_dict = {
                "joint_offset_deg": [0.0]*7 if len(active_arms) == 1 else [0.0]*14,
                "right_arm_joint_offset_deg": [0.0]*7 if "right" in active_arms else None,
                "left_arm_joint_offset_deg": [0.0]*7 if "left" in active_arms else None,
                "head_joint_offset_deg": [0.0, 0.0] if optimize_head else None,
                "xi_cam": [0.0]*6
            }
            with open(result_path, "w") as f:
                json.dump(result_dict, f, indent=4)
            return

        cfg = get_arm_config(self.model, active_arms[0]) if len(active_arms) == 1 else get_both_arm_config(self.model)
        ee_links = {active_arms[0]: cfg["ee_link"]} if len(active_arms) == 1 else cfg["ee_links"]
        ee_to_marker_nom = {active_arms[0]: cfg["ee_to_marker_nom"]} if len(active_arms) == 1 else cfg["ee_to_marker_nom"]
        head_cfg = get_head_config(self.model)
        
        optimizer = CalibrationOptimizer(
            robot=self.robot,
            arm_idx=cfg["arm_idx"],
            ee_links=ee_links,
            mount_to_cam_nom=cfg["mount_to_cam_nom"],
            head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
            ee_to_marker_nom=ee_to_marker_nom,
            active_arms=active_arms,
            optimize_arm=True,
            optimize_head=optimize_head,
            optimize_camera=optimize_camera,
            head_idx=head_cfg["head_idx"],
            use_head_kinematics=optimize_head,
            lambda_cam_pos=1.0,
            lambda_cam_rot=1.0,
            use_sag=False,
            estimate_measurement_noise=True
        )
        
        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
            q_arm_list, q_head_list, T_meas_list
        )
        
        if len(active_arms) == 1:
            right_arm_offset = q_arm_offset if active_arms[0] == "right" else None
            left_arm_offset = q_arm_offset if active_arms[0] == "left" else None
        else:
            right_arm_offset = q_arm_offset[:7]
            left_arm_offset = q_arm_offset[7:]

        # Log results
        self.log_msg("\n===== CALIBRATION RESULT =====")
        if right_arm_offset is not None:
            self.log_msg(f"Right arm offsets (deg): {np.rad2deg(right_arm_offset)}")
        if left_arm_offset is not None:
            self.log_msg(f"Left arm offsets (deg): {np.rad2deg(left_arm_offset)}")
        if q_head_offset is not None:
            self.log_msg(f"Head offsets (deg): {np.rad2deg(q_head_offset)}")
        self.log_msg(f"mount_to_cam xi: {xi_cam}")
        
        result_dict = {
            "joint_offset_deg": np.rad2deg(q_arm_offset).tolist(),
            "right_arm_joint_offset_deg": np.rad2deg(right_arm_offset).tolist() if right_arm_offset is not None else None,
            "left_arm_joint_offset_deg": np.rad2deg(left_arm_offset).tolist() if left_arm_offset is not None else None,
            "head_joint_offset_deg": np.rad2deg(q_head_offset).tolist() if q_head_offset is not None else None,
            "xi_cam": np.array(xi_cam).tolist()
        }
        
        if self.last_home_reset_path is not None and self.last_home_reset_path.exists():
            result_dict["home_reset_baseline_path"] = str(self.last_home_reset_path)
            
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=4)
            
        self.last_result_path = result_path
        self.log_msg(f"[SUCCESS] Calibration JSON saved to {result_path}")

    # Zero Pose Symmetrical Move
    def show_zero_pose_check(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        reply = QMessageBox.warning(
            self,
            "Warning",
            "Zero Pose Check will move BOTH arms and torso.\n"
            "Ensure the workspace is clear of obstacles.\n\n"
            "Do you want to proceed?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
            
        self.log_msg("[INFO] Starting zero pose check movement sequence...")
        
        def run_zero_check():
            try:
                active_arms = self.get_active_arms()
                parts = ["mobile_.*|torso_.*|right_arm_.*|left_arm_.*"]
                if self.chk_head_on.isChecked():
                    parts.append("head_.*")
                regex = "|".join(parts)
                
                check_calibration_state(
                    self.robot,
                    self.model_input.currentText(),
                    active_arms,
                    [0.35, 0.0, 0.2],
                    0.2,
                    log_cb=self.log_msg
                )
                self.log_msg("[SUCCESS] Reached Zero Pose check state.")
            except Exception as e:
                self.log_msg(f"[ERROR] Zero pose check failed: {e}")
                
        t = threading.Thread(target=run_zero_check)
        t.daemon = True
        t.start()

    # Home Offset Reset (Redefining zero offsets)
    def dev_home_offset_reset(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        msg = (
            "Warning: Home Offset Reset will physically redefine the zero offset positions of your robot joints.\n\n"
            "Steps:\n"
            "1. Manually teach/move BOTH arms close to their home pose using direct teaching.\n"
            "2. Ensure the head is also centered/aligned if you want to reset head offsets.\n"
            "3. Click OK to start the process.\n\n"
            "During this, the control manager will disable, 48v power will cycle, and the robot connection will automatically restart."
        )
        reply = QMessageBox.question(self, "Confirm Home Offset Reset", msg, QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return
            
        self.btn_home_reset.setEnabled(False)
        self.active_worker = HomeOffsetResetWorker(self.robot, self.model, self.model_input.currentText(), self.chk_head_on.isChecked())
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_home_reset_finished)
        self.active_worker.start()

    def on_home_reset_finished(self, result):
        self.btn_home_reset.setEnabled(True)
        if result and result.get("success"):
            self.last_home_reset_path = Path(result.get("baseline_path"))
            self.log_msg("[SUCCESS] Home Offset Reset completed successfully!")
            QMessageBox.information(self, "Success", "Home Offset Reset completed successfully!")
            
            # Reconnect robot automatically
            self.robot = None
            self.connect_robot()
        else:
            err = result.get("error", "Unknown error")
            self.log_msg(f"[ERROR] Home Offset Reset failed: {err}")
            QMessageBox.critical(self, "Error", f"Home Offset Reset failed:\n{err}")

    # Compare and Apply Home Offsets Dialog
    def dev_apply_home_offset(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        if self.last_result_path is None or not Path(self.last_result_path).exists():
            # Try to resolve latest result from results dir
            try:
                result_dir = RESULT_DIR
                result_files = sorted(result_dir.glob("result_*.json"), key=os.path.getmtime, reverse=True)
                if result_files:
                    self.last_result_path = result_files[0]
            except Exception:
                pass
                
        if self.last_result_path is None:
            QMessageBox.warning(self, "No Results", "No calibration results JSON found in result directory.")
            return
            
        baseline_path = self.last_home_reset_path
        if baseline_path is None:
            # Fallback check
            path = BASE_DIR / "config" / "home_reset_baseline.json"
            if path.exists():
                baseline_path = path
                
        # Open Comparison dialog
        dialog = ApplyHomeOffsetDialog(
            self,
            self.last_result_path,
            baseline_path,
            self.get_target_arm_str(),
            self.chk_head_on.isChecked()
        )
        dialog.exec()

    # Symmetrical Check Pose & Drawing Square
    def dev_check_calibration_state(self):
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return
            
        # Create verification dialog
        self.check_dialog = QDialog(self)
        self.check_dialog.setWindowTitle("Check Calibration state")
        self.check_dialog.setStyleSheet(self.styleSheet())
        
        lay = QVBoxLayout(self.check_dialog)
        grid = QGridLayout()
        
        grid.addWidget(QLabel("X Position (m):"), 0, 0)
        self.check_x = QLineEdit("0.35"); grid.addWidget(self.check_x, 0, 1)
        grid.addWidget(QLabel("Y Position (m):"), 1, 0)
        self.check_y = QLineEdit("0.0"); grid.addWidget(self.check_y, 1, 1)
        grid.addWidget(QLabel("Z Position (m):"), 2, 0)
        self.check_z = QLineEdit("0.2"); grid.addWidget(self.check_z, 2, 1)
        grid.addWidget(QLabel("Y Offset (m):"), 3, 0)
        self.check_off = QLineEdit("0.175"); grid.addWidget(self.check_off, 3, 1)
        
        lay.addLayout(grid)
        
        btn_box = QHBoxLayout()
        btn_move = QPushButton("Move Symmetrical")
        btn_square = QPushButton("Draw Test Square")
        btn_box.addWidget(btn_move)
        btn_box.addWidget(btn_square)
        lay.addLayout(btn_box)
        
        self.check_status = QLabel("Status: Ready")
        self.check_status.setStyleSheet("color: #2979ff; font-weight: bold;")
        lay.addWidget(self.check_status)
        
        btn_move.clicked.connect(self.move_symmetrical_check)
        btn_square.clicked.connect(self.draw_test_square)
        
        self.check_dialog.exec()

    def move_symmetrical_check(self):
        try:
            x = float(self.check_x.text())
            y = float(self.check_y.text())
            z = float(self.check_z.text())
            offset = float(self.check_off.text())
        except ValueError:
            QMessageBox.warning(self.check_dialog, "Input Error", "Please enter valid floating-point numbers.")
            return
            
        self.check_status.setText("Status: Moving...")
        self.check_status.setStyleSheet("color: #ff9800;")
        
        def run_move():
            try:
                active_arms = self.get_active_arms()
                check_calibration_state(
                    self.robot,
                    self.model_input.currentText(),
                    active_arms,
                    [x, y, z],
                    offset,
                    log_cb=self.log_msg
                )
                self.check_status.setText("Status: Move OK")
                self.check_status.setStyleSheet("color: #00e676;")
            except Exception as e:
                self.check_status.setText("Status: Error")
                self.check_status.setStyleSheet("color: #ff1744;")
                self.log_msg(f"[ERROR] Check calibration symmetrical move failed: {e}")
                
        t = threading.Thread(target=run_move)
        t.daemon = True
        t.start()

    def draw_test_square(self):
        try:
            offset = float(self.check_off.text())
        except ValueError:
            QMessageBox.warning(self.check_dialog, "Input Error", "Please enter a valid Y Offset.")
            return
            
        self.check_status.setText("Status: Drawing Square...")
        self.check_status.setStyleSheet("color: #ff9800;")
        
        def run_draw():
            try:
                active_arms = self.get_active_arms()
                square_points = [
                    [0.35, 0.07, 0.0],
                    [0.35, 0.0, 0.07],
                    [0.35, -0.07, 0.0],
                    [0.35, 0.0, -0.07]
                ]
                
                self.log_msg("[Draw Square] Starting square drawing sequence (2 loops)...")
                for loop_idx in range(2):
                    self.log_msg(f"[Draw Square] Loop {loop_idx + 1} / 2")
                    for pt_idx, pt in enumerate(square_points):
                        self.log_msg(f"[Draw Square] Moving to point {pt_idx + 1}: {pt}")
                        check_calibration_state(
                            self.robot,
                            self.model_input.currentText(),
                            active_arms,
                            pt,
                            offset,
                            log_cb=self.log_msg,
                            skip_ready=True
                        )
                        time.sleep(0.5)
                        
                self.log_msg("[SUCCESS] Square drawing complete.")
                self.check_status.setText("Status: Draw OK")
                self.check_status.setStyleSheet("color: #00e676;")
            except Exception as e:
                self.check_status.setText("Status: Error")
                self.check_status.setStyleSheet("color: #ff1744;")
                self.log_msg(f"[ERROR] Draw square failed: {e}")
                
        t = threading.Thread(target=run_draw)
        t.daemon = True
        t.start()

    def closeEvent(self, event):
        self.poll_timer.stop()
        self.video_timer.stop()
        if self.robot and self.robot != "mock_robot":
            try:
                self.robot.disconnect()
            except Exception:
                pass
        event.accept()

# Simulator / Mock class for marker transform in UI mode
class SimulatedMarkerTransform:
    def __init__(self, robot, camera_config):
        self.robot = robot
        
    class DummyCamera:
        def stream_off(self): pass
        def get_camera_temperature(self): return 34.2
        def capture_image(self): pass
        def get_color_image(self): return None
        
    camera = DummyCamera()
    
    def get_marker_transform(self, sampling_time=0, side="right", use_filter=False):
        return [np.eye(4)]

def main():
    parser = argparse.ArgumentParser(description="Unified Robot Calibration Workspace GUI")
    parser.add_argument("--ui", action="store_true", help="Start UI in dry-run/mock mode")
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
        
    gui = MainWindow(marker_st, robot, "right", ui_only=args.ui)
    gui.show()
    
    try:
        sys.exit(app.exec())
    finally:
        if marker_st:
            try:
                marker_st.camera.stream_off()
                print("Camera stream turned off successfully.")
            except Exception:
                pass

if __name__ == "__main__":
    main()
