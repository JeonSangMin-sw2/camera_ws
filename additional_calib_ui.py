import sys
import os
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
import cv2
import numpy as np
import time
import argparse
import logging
import rby1_sdk as rby
import threading
import yaml
import traceback
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
from pathlib import Path

# Import custom calibrator logic
from marker_detection import Marker_Detection, Marker_Transform
from calibration.Calibrator import MarkerCalibrator, JointCalibrator, BaseCalibrator
from calibration.IntrinsicsCalibrator import IntrinsicsCalibrator
from homeoffset_core import (
    reset_current_pose_home_offsets,
    save_home_reset_baseline_json,
    move_to_offset_candidate_from_json,
    load_offset_from_json
)
current_dir = os.path.dirname(os.path.abspath(__file__))
calibration_dir = os.path.abspath(os.path.join(current_dir,"core","calibration"))
# --- Configuration & Paths ---
CONFIG_PATHS = {
    "setting_yaml": os.path.abspath(os.path.join(current_dir, "config", "setting.yaml")),
    "camera_intrinsics": os.path.abspath(os.path.join(current_dir, "config", "camera_intrinsics.yaml")),
    "result_dir": os.path.abspath(os.path.join(current_dir, "result")),
    "plot_dir": os.path.abspath(os.path.join(calibration_dir, "result_img")),
}

UI_DROPDOWNS = {
    "robot_models": ["a", "m"],
    "arm_sides": ["Right Arm", "Left Arm"],
    "marker_axes": ["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"],
    "joint_modes_v13": ["wrist_roll_v13 (6-Axis Sweep)", "wrist_pitch_v13 (5-Axis Sweep)", "elbow (3-Axis Sweep)"],
    "joint_modes_v12": ["wrist_pitch (5-Axis Sweep)", "elbow (3-Axis Sweep)"]
}

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
            is_mock = (not self.robot or self.robot == "mock_robot" or getattr(self.robot, "is_pure_mock", False) or hasattr(self.robot, "is_pure_mock") or type(self.robot).__name__ in ("PureMockRobot", "OfflineRobot"))
            if is_mock:
                self.log_signal.emit("[MOCK] Home offset baseline save simulated.")
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Home offset reset simulated successfully.")
                self.finished_signal.emit({"success": True})
                return

            config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))
            baseline_path, _ = save_home_reset_baseline_json(
                self.robot,
                self.model,
                config_dir,
                model_name=self.model_name,
                include_head=self.include_head,
            )
            self.log_signal.emit(f"Home reset baseline saved to: {baseline_path}")

            result = reset_current_pose_home_offsets(
                self.robot,
                self.model,
                arm="both",
                include_head=self.include_head,
                log_cb=self.log_signal.emit,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Home Offset Reset worker error: {e}")
            self.finished_signal.emit({"success": False, "error": str(e)})

class MoveHomeOffsetWorker(QThread):
    log_signal = Signal(str)
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
            is_mock = (not self.robot or self.robot == "mock_robot" or getattr(self.robot, "is_pure_mock", False) or hasattr(self.robot, "is_pure_mock") or type(self.robot).__name__ in ("PureMockRobot", "OfflineRobot"))
            if is_mock:
                self.log_signal.emit(f"\n[MOCK] ===== HOME OFFSET PREVIEW: {self.label} =====")
                self.log_signal.emit(f"[MOCK] JSON: {self.json_path}")
                self.log_signal.emit(f"[MOCK] Arm: {self.arm}")
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Preview move complete.")
                self.finished_signal.emit(True)
                return

            self.log_signal.emit(f"\n===== HOME OFFSET PREVIEW: {self.label} =====")
            self.log_signal.emit(f"JSON: {self.json_path}")
            
            result = move_to_offset_candidate_from_json(
                robot=self.robot,
                model=self.model,
                arm=self.arm,
                json_path=str(self.json_path),
                include_head=self.include_head,
                minimum_time=10,
                move_zero_first=True,
            )
            self.log_signal.emit(f"Arm: {result['arm']}")
            if result.get("right_offset_deg") is not None:
                self.log_signal.emit(f"Right move offset (deg): {result['right_offset_deg']}")
            if result.get("left_offset_deg") is not None:
                self.log_signal.emit(f"Left move offset (deg): {result['left_offset_deg']}")
            if result.get("head_offset_deg") is not None:
                self.log_signal.emit(f"Head move offset (deg): {result['head_offset_deg']}")
            self.log_signal.emit("Preview move complete. Inspect the robot pose before applying.")
            self.finished_signal.emit(True)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Preview move failed: {e}")
            self.finished_signal.emit(False)

class ApplyCurrentPoseWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, robot, model, arm, include_head):
        super().__init__()
        self.robot = robot
        self.model = model
        self.arm = arm
        self.include_head = include_head

    def run(self):
        try:
            is_mock = (not self.robot or self.robot == "mock_robot" or getattr(self.robot, "is_pure_mock", False) or hasattr(self.robot, "is_pure_mock") or type(self.robot).__name__ in ("PureMockRobot", "OfflineRobot"))
            if is_mock:
                self.log_signal.emit("[MOCK] Starting Home Offset Reset from current pose...")
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Current pose home offset apply complete.")
                self.finished_signal.emit({"success": True})
                return

            self.log_signal.emit("Starting Home Offset Reset from current pose...")
            result = reset_current_pose_home_offsets(
                self.robot,
                self.model,
                arm=self.arm,
                include_head=self.include_head,
                log_cb=self.log_signal.emit,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Apply current pose failed: {e}")
            self.finished_signal.emit({"success": False, "error": str(e)})

class FullAutoReadyWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, joint_calibrator, marker_calibrator, ui_only=False):
        super().__init__()
        self.joint_calibrator = joint_calibrator
        self.marker_calibrator = marker_calibrator
        self.ui_only = ui_only

    def run(self):
        try:
            self.log_signal.emit("Moving robot arms to Full Auto initial ready poses...")
            version_num = self.marker_calibrator.get_robot_version()
            is_v13 = self.marker_calibrator.is_v13()
            self.log_signal.emit(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")

            is_mock_run = self.ui_only or (not self.joint_calibrator.robot or self.joint_calibrator.robot == "mock_robot" or getattr(self.joint_calibrator.robot, "is_pure_mock", False) or hasattr(self.joint_calibrator.robot, "is_pure_mock") or type(self.joint_calibrator.robot).__name__ in ("PureMockRobot", "OfflineRobot"))
            
            for arm_side in ["right", "left"]:
                self.log_signal.emit(f"Preparing {arm_side.upper()} arm...")
                if is_mock_run:
                    time.sleep(1.0)
                    self.log_signal.emit(f"[MOCK] {arm_side.upper()} arm moved to ready pose.")
                else:
                    if not is_v13:
                        self.log_signal.emit(f"Moving {arm_side} arm to wrist pitch ready pose...")
                        if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch", log_callback=self.log_signal.emit):
                            raise RuntimeError(f"Failed to move {arm_side} arm to wrist pitch ready pose.")
                    else:
                        self.log_signal.emit(f"Moving {arm_side} arm to marker ready pose...")
                        if not self.marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=self.log_signal.emit):
                            raise RuntimeError(f"Failed to move {arm_side} arm to marker ready pose.")
            self.log_signal.emit("All arms moved to initial ready poses successfully.")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Ready pose movement failed: {e}")
        self.finished_signal.emit()

# --- Specialized Calibration Workers ---
class MarkerCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)
    
    def __init__(self, calibrator, arm_side, use_head_tracking=True, tolerance=0.5, save_debug=False):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.use_head_tracking = use_head_tracking
        self.tolerance = tolerance
        self.save_debug = save_debug
        
    def run(self):
        try:
            version_num = self.calibrator.get_robot_version()
            is_v13 = self.calibrator.is_v13()
            
            is_mock_run = (not self.calibrator.robot or self.calibrator.robot == "mock_robot" or getattr(self.calibrator.robot, "is_pure_mock", False) or hasattr(self.calibrator.robot, "is_pure_mock") or type(self.calibrator.robot).__name__ in ("PureMockRobot", "OfflineRobot"))
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
                    use_head_tracking=self.use_head_tracking,
                    save_debug=self.save_debug
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
                    
                    if not success_other:
                        self.log_signal.emit("[ERROR] Failed to move inactive arm to zero pose.")
                        self.finished_signal.emit(None)
                        return
                    
                    success = self.calibrator.movej(
                        self.calibrator.robot,
                        torso=[0.0]*6,
                        right_arm=first_starting_pose if self.arm_side == "right" else None,
                        left_arm=first_starting_pose if self.arm_side == "left" else None,
                        head=[0, 0],
                        minimum_time=5.0
                    )
                    if not success:
                        self.log_signal.emit("[ERROR] Failed to return to initial starting pose. Aborting.")
                        self.finished_signal.emit(None)
                        return
                else:
                    self.log_signal.emit("\n[MOCK] Returning to Initial Starting Pose...")
                    time.sleep(1.0)
                
                if getattr(self.calibrator, 'stop_requested', False):
                    self.finished_signal.emit(None)
                    return
                
                time.sleep(1.0)

            # Stage 2/3 (or 1/2) Axis 6 Sweep
            stage_6_title = "[Stage 2/3] Sweeping Axis 6 (Roll)..." if is_v13 else "[Stage 1/2] Sweeping Axis 6 (Roll)..."
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit(f"   {stage_6_title}")
            self.log_signal.emit("="*50 + "\n")
            
            res_6 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 6, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking,
                save_debug=self.save_debug
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
            
            # Stage 3/3 (or 2/2) Axis 5 Sweep
            stage_5_title = "[Stage 3/3] Sweeping Axis 5 (Pitch)..." if is_v13 else "[Stage 2/2] Sweeping Axis 5 (Pitch)..."
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit(f"   {stage_5_title}")
            self.log_signal.emit("="*50 + "\n")
            
            res_5 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 5, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking,
                save_debug=self.save_debug
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

            # Save plot using the calibrator method
            plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{self.arm_side}_marker_unified.png")
            plot_saved = self.calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, self.arm_side, is_v13, plot_path)
            
            if plot_saved:
                unified_res['plot_path_combined'] = plot_path
            self.finished_signal.emit(unified_res)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(None)

class JointCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)

    def __init__(self, calibrator, arm_side, mode, ui_only=False, current_offset_deg=0.0, sweep_duration=15.0, save_debug=False):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode
        self.ui_only = ui_only
        self.current_offset_deg = current_offset_deg
        self.sweep_duration = sweep_duration
        self.save_debug = save_debug

    def run(self):
        try:
            res = self.calibrator.perform_joint_calibration(
                self.arm_side, self.mode,
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                current_offset_deg=self.current_offset_deg,
                sweep_duration=self.sweep_duration,
                save_debug=self.save_debug
            )

            if res:
                self.log_signal.emit("-" * 30)
                self.log_signal.emit(f"  [1] Calibration Target: {self.mode}")
                recommended = res.get('recommended_joint_offset', res['optimal_offset'])
                self.log_signal.emit(f"      Estimated Optimal Offset: {recommended:.3f} deg")
                self.log_signal.emit("-" * 30)
                self.log_signal.emit("\n[CALIBRATION COMPLETE]\n")
                self.finished_signal.emit(res)
            else:
                self.finished_signal.emit(None)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            self.finished_signal.emit(None)



class SimulatedMarkerTransform:
    def __init__(self, robot, camera_config):
        self.robot = robot
        self.camera_config = camera_config
        
        class DummyCamera:
            def stream_off(self): pass
        self.camera = DummyCamera()

    def get_marker_transform(self, sampling_time=0, side="right", use_filter=False):
        is_mock = (not self.robot or self.robot == "mock_robot" or getattr(self.robot, "is_pure_mock", False) or hasattr(self.robot, "is_pure_mock") or type(self.robot).__name__ in ("PureMockRobot", "OfflineRobot"))
        if is_mock:
            T = np.eye(4)
            T[2, 3] = 0.3
            return [T.tolist()]
        try:
            q = self.robot.get_state().position
            dyn_model = self.robot.get_dynamics()
            
            ee_name = f"ee_{side}"
            
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q, ee_name, "link_torso_5")
            
            T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q, "link_head_2", "link_torso_5")
            mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
            T_head_to_cam = BaseCalibrator.make_transform(mount_to_cam)
            T_t5_to_cam = T_t5_to_head @ T_head_to_cam
            
            if side == "left":
                tf_vec = self.camera_config.get("Tf_to_marker_left", [0.0, 0.0775, -0.06677, 90.0, 0.0, 0.0])
            else:
                tf_vec = self.camera_config.get("Tf_to_marker_right", [0.0, -0.0775, -0.06677, 90.0, 0.0, 180.0])
            T_ee_to_marker = BaseCalibrator.make_transform(tf_vec)
            
            T_cam_to_t5 = np.linalg.inv(T_t5_to_cam)
            T_cam_to_marker = T_cam_to_t5 @ T_t5_to_ee @ T_ee_to_marker
            
            noise_t = np.random.normal(0, 0.0001, 3)
            T_cam_to_marker[:3, 3] += noise_t
            
            return [T_cam_to_marker.tolist()]
        except Exception as e:
            T = np.eye(4)
            T[2, 3] = 0.3
            return [T.tolist()]


class FullAutoWorker(QThread):
    log_msg = Signal(str)
    status_signal = Signal(bool)
    bracket_finished_signal = Signal(dict)
    joint_finished_signal = Signal(dict)
    finished_signal = Signal()

    def __init__(self, joint_calibrator, marker_calibrator, ui_only=False, stop_event=None, joint_offsets_store=None):
        super().__init__()
        self.joint_calibrator = joint_calibrator
        self.marker_calibrator = marker_calibrator
        self.ui_only = ui_only
        self.stop_event = stop_event
        self.joint_offsets_store = joint_offsets_store if joint_offsets_store is not None else {}

    def get_robot_version(self):
        return self.marker_calibrator.get_robot_version()

    def run(self):
        try:
            self.log_msg.emit("Starting FULL AUTO sequential calibration...")
            is_mock_run = self.ui_only and (not self.joint_calibrator.robot or self.joint_calibrator.robot == "mock_robot" or getattr(self.joint_calibrator.robot, "is_pure_mock", False) or hasattr(self.joint_calibrator.robot, "is_pure_mock") or type(self.joint_calibrator.robot).__name__ in ("PureMockRobot", "OfflineRobot"))
            
            for arm_side in ["right", "left"]:
                self.log_msg.emit("\n" + "="*50)
                self.log_msg.emit(f"   STARTING SEQUENTIAL CALIBRATION FOR {arm_side.upper()} ARM")
                self.log_msg.emit("="*50 + "\n")
                
                version_num = self.get_robot_version()
                is_v13 = (version_num == "1.3")
                self.log_msg.emit(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")
                
                # --- Step 1: Marker Bracket Calibration ---
                self.log_msg.emit(f"[FULL AUTO 1/2] Starting Marker Bracket Calibration for {arm_side} arm...")
                
                res_4 = None
                if is_v13:
                    self.log_msg.emit(f"[FULL AUTO] Moving {arm_side} arm to ready pose...")
                    if not self.marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                    if self.stop_event.is_set(): return
                    
                    self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 4...")
                    res_4 = self.marker_calibrator.perform_calibration_sweep(
                        arm_side, 4,
                        log_callback=self.log_msg.emit,
                        status_callback=self.status_signal.emit
                    )
                    if not res_4:
                        raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
                    res_4['axis_mode'] = 4
                    res_4['axis'] = res_4['axis_opt']
                    if self.stop_event.is_set(): return
                    
                    if not is_mock_run:
                        self.log_msg.emit(f"[FULL AUTO] Returning to Initial Starting Pose...")
                        model = self.marker_calibrator.robot.model()
                        state = self.marker_calibrator.robot.get_state()
                        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                        first_starting_pose = list(state.position[arm_idx])
                        if arm_side == "right":
                            self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                            self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=first_starting_pose, head=[0, 0], minimum_time=5.0)
                        else:
                            self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                            self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=first_starting_pose, head=[0, 0], minimum_time=5.0)
                    else:
                        self.log_msg.emit("[FULL AUTO] [MOCK] Returning to Initial Starting Pose...")
                        time.sleep(1.0)
                    if self.stop_event.is_set(): return
                else:
                    self.log_msg.emit(f"[FULL AUTO] Moving {arm_side} arm to ready pose...")
                    if not self.marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                    if self.stop_event.is_set(): return
                
                self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 6...")
                res_6 = self.marker_calibrator.perform_calibration_sweep(
                    arm_side, 6,
                    log_callback=self.log_msg.emit,
                    status_callback=self.status_signal.emit
                )
                if not res_6:
                    raise RuntimeError(f"Axis 6 marker sweep failed on {arm_side} arm")
                res_6['axis_mode'] = 6
                res_6['axis'] = res_6['axis_opt']
                if self.stop_event.is_set(): return
                
                self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 5...")
                res_5 = self.marker_calibrator.perform_calibration_sweep(
                    arm_side, 5,
                    log_callback=self.log_msg.emit,
                    status_callback=self.status_signal.emit
                )
                if not res_5:
                    raise RuntimeError(f"Axis 5 marker sweep failed on {arm_side} arm")
                res_5['axis_mode'] = 5
                res_5['axis'] = res_5['axis_opt']
                if self.stop_event.is_set(): return

                if is_v13:
                    # ----------------- v1.3 Calibration Order -----------------
                    # 1. J6 (Wrist Roll) Calibration first (using Sweep 6 and Sweep 5 data)
                    self.log_msg.emit("\n[FULL AUTO] Calibrating J6 (Wrist Roll) first...")
                    dataset_A_6 = list(zip(res_6['captured_q_full'], res_6['captured_poses']))
                    dataset_B_5 = list(zip(res_5['captured_q_full'], res_5['captured_poses']))
                    model = self.joint_calibrator.robot.model()
                    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                    initial_joint_pos_roll = list(res_6['captured_q_full'][0][arm_idx])
                    
                    joint_res_roll = self.joint_calibrator.compute_calibration_results(
                        arm_side, "wrist_roll_v13", dataset_A_6, dataset_B_5, initial_joint_pos_roll,
                        current_offset_deg=0.0, use_angle_based_fitting=True, log_callback=self.log_msg.emit
                    )
                    if not joint_res_roll:
                        raise RuntimeError(f"J6 wrist roll calibration failed on {arm_side} arm")
                    opt_roll = joint_res_roll["recommended_joint_offset"]
                    self.log_msg.emit(f"[FULL AUTO] Staging J6 (Wrist Roll) offset: {opt_roll:.4f}°")
                    self.joint_offsets_store[arm_side]["joint6"] = opt_roll
                    self.joint_calibrator.joint_offsets["wrist_roll"] = opt_roll
                    self.marker_calibrator.joint_offsets["wrist_roll"] = opt_roll
                    
                    # Emitting the J6 calibration result to the UI plot
                    joint_res_roll['arm_side'] = arm_side
                    joint_res_roll['mode'] = "wrist_roll_v13"
                    self.joint_finished_signal.emit(joint_res_roll)
                    time.sleep(0.5)

                    # 2. Marker Bracket Calibration (with J6 offset locked to opt_roll)
                    self.log_msg.emit("\n[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...")
                    unified_res = self.marker_calibrator.compute_unified_bracket_calibration(
                        res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_deg=opt_roll
                    )
                    
                    unified_res['res_5'] = res_5
                    unified_res['res_6'] = res_6
                    if res_4 is not None:
                        unified_res['res_4'] = res_4
                    unified_res['arm_side'] = arm_side
                    
                    plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                    plot_saved = self.marker_calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, arm_side, is_v13, plot_path)
                    if plot_saved:
                        unified_res['plot_path_combined'] = plot_path
                    
                    x_m, y_m, z_m = unified_res['x_e']/1000.0, unified_res['y_e']/1000.0, unified_res['z_e']/1000.0
                    new_vals = [x_m, y_m, z_m, unified_res['roll_e'], unified_res['pitch_e'], unified_res['yaw_e']]
                    key = f"Tf_to_marker_{arm_side}"
                    self.marker_calibrator.camera_config[key] = new_vals
                    self.joint_calibrator.camera_config[key] = new_vals
                    
                    # Update J5 offset from marker calibration as a first step
                    staged_pitch = unified_res.get('opt_delta_5', 0.0)
                    self.joint_offsets_store[arm_side]["joint5"] = staged_pitch
                    self.joint_calibrator.joint_offsets["wrist_pitch"] = staged_pitch
                    self.marker_calibrator.joint_offsets["wrist_pitch"] = staged_pitch
                    
                    self.bracket_finished_signal.emit(unified_res)
                    time.sleep(0.5)

                else:
                    self.log_msg.emit("[FULL AUTO] Computing unified marker bracket calibration...")
                    unified_res = self.marker_calibrator.compute_unified_bracket_calibration(
                        res_5, res_6, arm_side, marker_data_4=res_4
                    )
                    
                    unified_res['res_5'] = res_5
                    unified_res['res_6'] = res_6
                    unified_res['arm_side'] = arm_side
                    
                    plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                    plot_saved = self.marker_calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, arm_side, is_v13, plot_path)
                    if plot_saved:
                        unified_res['plot_path_combined'] = plot_path
                    
                    x_m, y_m, z_m = unified_res['x_e']/1000.0, unified_res['y_e']/1000.0, unified_res['z_e']/1000.0
                    new_vals = [x_m, y_m, z_m, unified_res['roll_e'], unified_res['pitch_e'], unified_res['yaw_e']]
                    key = f"Tf_to_marker_{arm_side}"
                    self.marker_calibrator.camera_config[key] = new_vals
                    self.joint_calibrator.camera_config[key] = new_vals
                    
                    self.bracket_finished_signal.emit(unified_res)
                    time.sleep(0.5)
                
                # --- Step 2: Joint Calibration ---
                self.log_msg.emit(f"[FULL AUTO 2/2] Starting Joint Calibration for {arm_side} arm...")
                
                if not is_v13:
                    # v1.2 Joint Calibration: Wrist Pitch, then Elbow
                    # 1. Wrist Pitch
                    self.log_msg.emit("[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...")
                    if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch", log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_pitch on {arm_side} arm")
                    if self.stop_event.is_set(): return
                    
                    joint_res_pitch = self.joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_pitch",
                        log_callback=self.log_msg.emit,
                        status_callback=self.status_signal.emit,
                        current_offset_deg=self.joint_offsets_store.get(arm_side, {}).get("joint5", 0.0)
                    )
                    if not joint_res_pitch:
                        raise RuntimeError(f"Wrist pitch joint calibration failed on {arm_side} arm")
                    joint_res_pitch['arm_side'] = arm_side
                    joint_res_pitch['mode'] = "wrist_pitch"
                    
                    self.joint_calibrator.joint_offsets["wrist_pitch"] = joint_res_pitch["recommended_joint_offset"]
                    self.marker_calibrator.joint_offsets["wrist_pitch"] = joint_res_pitch["recommended_joint_offset"]
                    
                    self.joint_finished_signal.emit(joint_res_pitch)
                    time.sleep(0.5)
                    if self.stop_event.is_set(): return
                    
                    # 2. Elbow
                    self.log_msg.emit("[FULL AUTO] Sweeping Elbow (Joint 3)...")
                    if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                    if self.stop_event.is_set(): return
                    
                    joint_res_elbow = self.joint_calibrator.perform_joint_calibration(
                        arm_side, "elbow",
                        log_callback=self.log_msg.emit,
                        status_callback=self.status_signal.emit,
                        current_offset_deg=self.joint_offsets_store.get(arm_side, {}).get("joint3", 0.0)
                    )
                    if not joint_res_elbow:
                        raise RuntimeError(f"Elbow joint calibration failed on {arm_side} arm")
                    joint_res_elbow['arm_side'] = arm_side
                    joint_res_elbow['mode'] = "elbow"
                    
                    self.joint_calibrator.joint_offsets["elbow"] = joint_res_elbow["recommended_joint_offset"]
                    self.marker_calibrator.joint_offsets["elbow"] = joint_res_elbow["recommended_joint_offset"]
                    
                    self.joint_finished_signal.emit(joint_res_elbow)
                    time.sleep(0.5)
                    
                else:
                    # v1.3 Joint Calibration: Wrist Pitch (J5), then Elbow (J3)
                    # 1. Wrist Pitch
                    self.log_msg.emit("[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...")
                    if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch_v13", log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_pitch_v13 on {arm_side} arm")
                    if self.stop_event.is_set(): return
                    
                    joint_res_pitch = self.joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_pitch_v13",
                        log_callback=self.log_msg.emit,
                        status_callback=self.status_signal.emit,
                        current_offset_deg=self.joint_offsets_store.get(arm_side, {}).get("joint5", 0.0)
                    )
                    if not joint_res_pitch:
                        raise RuntimeError(f"Wrist pitch joint calibration failed on {arm_side} arm")
                    joint_res_pitch['arm_side'] = arm_side
                    joint_res_pitch['mode'] = "wrist_pitch_v13"
                    
                    opt_pitch = joint_res_pitch["recommended_joint_offset"]
                    self.joint_calibrator.joint_offsets["wrist_pitch"] = opt_pitch
                    self.marker_calibrator.joint_offsets["wrist_pitch"] = opt_pitch
                    self.joint_offsets_store[arm_side]["joint5"] = opt_pitch
                    
                    self.joint_finished_signal.emit(joint_res_pitch)
                    time.sleep(0.5)
                    if self.stop_event.is_set(): return
                    
                    # 2. Elbow
                    self.log_msg.emit("[FULL AUTO] Sweeping Elbow (Joint 3)...")
                    if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                    if self.stop_event.is_set(): return
                    
                    joint_res_elbow = self.joint_calibrator.perform_joint_calibration(
                        arm_side, "elbow",
                        log_callback=self.log_msg.emit,
                        status_callback=self.status_signal.emit,
                        current_offset_deg=self.joint_offsets_store.get(arm_side, {}).get("joint3", 0.0)
                    )
                    if not joint_res_elbow:
                        raise RuntimeError(f"Elbow joint calibration failed on {arm_side} arm")
                    joint_res_elbow['arm_side'] = arm_side
                    joint_res_elbow['mode'] = "elbow"
                    
                    opt_elbow = joint_res_elbow["recommended_joint_offset"]
                    self.joint_calibrator.joint_offsets["elbow"] = opt_elbow
                    self.marker_calibrator.joint_offsets["elbow"] = opt_elbow
                    self.joint_offsets_store[arm_side]["joint3"] = opt_elbow
                    
                    self.joint_finished_signal.emit(joint_res_elbow)
                    time.sleep(0.5)
                    
                self.log_msg.emit(f"[INFO] {arm_side.upper()} arm sequential calibration completed successfully.")
                if self.stop_event.is_set(): return
                time.sleep(1.0)
                
            self.log_msg.emit("\n" + "="*50)
            self.log_msg.emit("   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!")
            self.log_msg.emit("="*50 + "\n")
        except Exception as e:
            self.log_msg.emit(f"[ERROR] Full Auto sequential calibration failed: {e}")
            import traceback
            self.log_msg.emit(traceback.format_exc())
        finally:
            self.finished_signal.emit()


class ApplyHomeOffsetDialog(QDialog):
    def __init__(self, parent, result_path, baseline_path, arm, include_head):
        super().__init__(parent)
        self.parent = parent
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
        self.btn_baseline.setEnabled(enabled and bool(self.baseline_path and Path(self.baseline_path).exists()))
        self.btn_optimized.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        self.btn_close.setEnabled(enabled)

    def move_baseline(self):
        self.set_buttons_enabled(False)
        self.active_worker = MoveHomeOffsetWorker(
            self.parent.robot,
            self.parent.robot.model() if (self.parent.robot and not self.parent.is_mock) else None,
            self.arm,
            self.baseline_path,
            self.include_head,
            "Baseline Zero"
        )
        self.active_worker.log_signal.connect(self.parent.log_msg)
        self.active_worker.finished_signal.connect(self.on_move_finished)
        self.active_worker.start()

    def move_optimized(self):
        self.set_buttons_enabled(False)
        self.active_worker = MoveHomeOffsetWorker(
            self.parent.robot,
            self.parent.robot.model() if (self.parent.robot and not self.parent.is_mock) else None,
            self.arm,
            self.result_path,
            self.include_head,
            "Optimized Zero"
        )
        self.active_worker.log_signal.connect(self.parent.log_msg)
        self.active_worker.finished_signal.connect(self.on_move_finished)
        self.active_worker.start()

    def on_move_finished(self, success):
        self.set_buttons_enabled(True)
        if success:
            QMessageBox.information(self, "Preview Complete", "Moved to zero candidate pose.")
        else:
            QMessageBox.critical(self, "Error", "Failed to move robot.")

    def apply_current(self):
        msg = (
            "This will redefine the selected joints' home offset using the robot's CURRENT pose.\n\n"
            "Only continue if the robot is currently at the zero pose you want to keep."
        )
        reply = QMessageBox.question(
            self, "Apply Current Pose", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.set_buttons_enabled(False)
            self.active_worker = ApplyCurrentPoseWorker(
                self.parent.robot,
                self.parent.robot.model() if (self.parent.robot and not self.parent.is_mock) else None,
                self.arm,
                self.include_head
            )
            self.active_worker.log_signal.connect(self.parent.log_msg)
            self.active_worker.finished_signal.connect(self.on_apply_finished)
            self.active_worker.start()

    def on_apply_finished(self, result):
        self.set_buttons_enabled(True)
        if result.get("success", False):
            QMessageBox.information(self, "Success", "Home offset applied successfully from current pose.")
            self.parent.log_msg("Re-connecting and initializing robot...")
            self.parent.connect_robot()
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Home offset apply finished, but some joints failed to reset. Check logs.")

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
        self.robot_version = "1.2"
        
        # Intrinsics Calibrator (Tab 3 용)
        self.intrinsics_calibrator = IntrinsicsCalibrator()
        # Default: 8x5 squares, 36mm x 27mm, DICT_5X5_100
        self.intrinsics_calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 36.0, 27.0, "DICT_5X5_100")
        
        try:
            
            self.marker_detector = Marker_Detection()
            self.marker_detector.set_marker_type("plate")
        except ImportError:
            self.marker_detector = None
            
        self.monitor_enabled = False
        self.captured_images = []
        self.output_yaml = CONFIG_PATHS["camera_intrinsics"]
        
        # Saved Calibration Results
        self.marker_data_4 = None
        self.marker_data_5 = None
        self.marker_data_6 = None
        self.joint_sweep_data = None
        
        # Cumulative Joint Offsets for iterative sweeps
        self.joint_offsets = {"wrist_pitch": 0.0, "wrist_roll": 0.0, "elbow": 0.0}
        self.wrist_roll_calibrated = {"right": False, "left": False}
        self.load_offsets_from_yaml()
        
        self.recommended_joint_offset = None
        
        self.marker_calibrator.joint_offsets = self.joint_offsets
        self.joint_calibrator.joint_offsets = self.joint_offsets
        
        self.setWindowTitle("Unified Robot Calibration Suite")
        self.resize(1050, 550)
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
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if not os.path.exists(config_path):
                self.joint_offsets_store = {
                    "left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
                    "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0}
                }
                self.save_offsets_to_yaml()
            else:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict) and "joint_offset" in data:
                    self.joint_offsets_store = data["joint_offset"]
                    for arm in ["left", "right"]:
                        if arm not in self.joint_offsets_store:
                            self.joint_offsets_store[arm] = {}
                        if "joint3" not in self.joint_offsets_store[arm]:
                            self.joint_offsets_store[arm]["joint3"] = 0.0
                        if "joint5" not in self.joint_offsets_store[arm]:
                            self.joint_offsets_store[arm]["joint5"] = 0.0
                        if "joint6" not in self.joint_offsets_store[arm]:
                            self.joint_offsets_store[arm]["joint6"] = 0.0
                else:
                    self.joint_offsets_store = {
                        "left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
                        "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0}
                    }
                    self.save_offsets_to_yaml()
                    self.log_msg(f"[WARNING] Added default joint_offset to setting.yaml.")
            
            # Sync current offsets with active arm_side from YAML
            is_v13 = self.get_robot_version() == "1.3"
            if is_v13:
                self.joint_offsets["wrist_pitch"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint5", 0.0)
                self.joint_offsets["wrist_roll"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint6", 0.0)
            else:
                self.joint_offsets["wrist_pitch"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint5", 0.0)
                self.joint_offsets["wrist_roll"] = 0.0
            self.joint_offsets["elbow"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint3", 0.0)
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets
            
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to load/create setting.yaml: {e}")
            # Fallback default values
            self.joint_offsets_store = {
                "left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
                "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0}
            }
            self.joint_offsets["wrist_pitch"] = 0.0
            self.joint_offsets["wrist_roll"] = 0.0
            self.joint_offsets["elbow"] = 0.0
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets

    def save_offsets_to_yaml(self):
        config_path = CONFIG_PATHS["setting_yaml"]
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
                lines.append(f"    joint5: {self.joint_offsets_store['left'].get('joint5', 0.0)}\n")
                lines.append(f"    joint6: {self.joint_offsets_store['left'].get('joint6', 0.0)}\n")
                lines.append("  right:\n")
                lines.append(f"    joint3: {self.joint_offsets_store['right']['joint3']}\n")
                lines.append(f"    joint5: {self.joint_offsets_store['right'].get('joint5', 0.0)}\n")
                lines.append(f"    joint6: {self.joint_offsets_store['right'].get('joint6', 0.0)}\n")
            else:
                lines = lines[:jo_idx]
                lines.append("joint_offset:\n")
                lines.append("  left:\n")
                lines.append(f"    joint3: {self.joint_offsets_store['left']['joint3']}\n")
                lines.append(f"    joint5: {self.joint_offsets_store['left'].get('joint5', 0.0)}\n")
                lines.append(f"    joint6: {self.joint_offsets_store['left'].get('joint6', 0.0)}\n")
                lines.append("  right:\n")
                lines.append(f"    joint3: {self.joint_offsets_store['right']['joint3']}\n")
                lines.append(f"    joint5: {self.joint_offsets_store['right'].get('joint5', 0.0)}\n")
                lines.append(f"    joint6: {self.joint_offsets_store['right'].get('joint6', 0.0)}\n")

            with open(config_path, "w") as f:
                f.writelines(lines)
            self.log_msg(f"[SUCCESS] Saved offsets permanently to setting.yaml!")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save setting.yaml: {e}")

    def on_cell_double_clicked(self, row, col):
        arm = "right" if row == 0 else "left"
        is_v13 = self.get_robot_version() == "1.3"
        if is_v13:
            if col == 0:
                joint_key = "joint6"
                joint_label = "Joint 6"
            elif col == 1:
                joint_key = "joint5"
                joint_label = "Joint 5"
            else:
                joint_key = "joint3"
                joint_label = "Joint 3"
        else:
            if col == 0:
                joint_key = "joint5"
                joint_label = "Joint 5"
            else:
                joint_key = "joint3"
                joint_label = "Joint 3"
            
        current_val = self.joint_offsets_store[arm][joint_key]
        new_val, ok = QInputDialog.getDouble(
            self, 
            "Manual Offset Override", 
            f"Enter manual staged offset for {arm.upper()} Arm {joint_label} (degrees):", 
            current_val, -45.0, 45.0, 4
        )
        if ok:
            self.joint_offsets_store[arm][joint_key] = new_val
            self.update_applied_offset_label()
            self.log_msg(f"[MANUAL OVERRIDE] Staged {arm.upper()} Arm {joint_label} offset manually to {new_val:.4f}°. (Not saved to disk yet. Click APPLY OFFSET to save)")

    def update_joint_modes(self):
        if not hasattr(self, 'joint_mode_sel'):
            return
        is_v13 = self.get_robot_version() == "1.3"
        self.joint_mode_sel.blockSignals(True)
        self.joint_mode_sel.clear()
        if is_v13:
            self.joint_mode_sel.addItems(UI_DROPDOWNS["joint_modes_v13"])
        else:
            self.joint_mode_sel.addItems(UI_DROPDOWNS["joint_modes_v12"])
        self.joint_mode_sel.blockSignals(False)



    def get_selected_joint_mode(self):
        if not hasattr(self, 'joint_mode_sel'):
            return "wrist_pitch"
        mode_str = self.joint_mode_sel.currentText()
        if "wrist_pitch_v13" in mode_str:
            return "wrist_pitch_v13"
        elif "wrist_roll_v13" in mode_str:
            return "wrist_roll_v13"
        elif "wrist_pitch" in mode_str:
            return "wrist_pitch"
        else:
            return "elbow"

    def get_offset_key_for_mode(self, mode):
        if mode == "wrist_pitch_v13":
            return "wrist_pitch"
        elif mode == "wrist_roll_v13":
            return "wrist_roll"
        else:
            return mode

    def init_ui(self):
        # Main horizontal layout
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
        
        # --- COLUMN 1 (Robot Connection, Head & Home, Workflows) ---
        col1_layout = QVBoxLayout()
        
        # Connection Box (Robot Model next to IP/Port)
        conn_box = QGroupBox("Robot Connection & Control")
        conn_layout = QVBoxLayout()
        conn_layout.setSpacing(4)
        conn_layout.setContentsMargins(6, 6, 6, 6)
        
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP/Port:"))
        self.ip_input = QLineEdit("192.168.30.1:50051")
        if self.ui_only:
            self.ip_input.setText("127.0.0.1:50051")
        ip_row.addWidget(self.ip_input)
        
        ip_row.addWidget(QLabel("Model:"))
        self.model_input = QComboBox()
        self.model_input.addItems(UI_DROPDOWNS["robot_models"])
        ip_row.addWidget(self.model_input)
        conn_layout.addLayout(ip_row)
        
        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold;")
        self.btn_connect.clicked.connect(self.connect_robot)
        self.btn_connect.setFixedHeight(24)
        conn_layout.addWidget(self.btn_connect)
        
        conn_box.setLayout(conn_layout)

        # Manual Head Control Box
        head_box = QGroupBox("Manual Head Control")
        head_layout = QVBoxLayout()
        head_layout.setSpacing(4)
        head_layout.setContentsMargins(6, 6, 6, 6)
        
        inputs_row = QHBoxLayout()
        inputs_row.addWidget(QLabel("Yaw:"))
        self.txt_head_yaw = QLineEdit("0.0")
        self.txt_head_yaw.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        inputs_row.addWidget(self.txt_head_yaw)
        
        inputs_row.addWidget(QLabel("Pitch:"))
        self.txt_head_pitch = QLineEdit("0.0")
        self.txt_head_pitch.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        inputs_row.addWidget(self.txt_head_pitch)
        head_layout.addLayout(inputs_row)
        
        self.btn_move_head = QPushButton("MOVE HEAD")
        self.btn_move_head.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold;")
        self.btn_move_head.clicked.connect(self.move_head_manually)
        self.btn_move_head.setFixedHeight(24)
        head_layout.addWidget(self.btn_move_head)
        
        head_box.setLayout(head_layout)

        # Standalone Robot Home Offset Reset Box (Relocated to the right of Head Control)
        home_offset_box = QGroupBox("Robot Home Offset")
        home_offset_layout = QVBoxLayout()
        home_offset_layout.setSpacing(4)
        home_offset_layout.setContentsMargins(6, 6, 6, 6)
        
        home_offset_layout.addWidget(QLabel("Reset joint offsets:"))
        self.btn_home_reset = QPushButton("Home Offset Reset")
        self.btn_home_reset.setStyleSheet("background-color: #d84315; color: white; font-weight: bold;")
        self.btn_home_reset.clicked.connect(self.home_offset_reset)
        self.btn_home_reset.setFixedHeight(24)
        home_offset_layout.addWidget(self.btn_home_reset)
        
        home_offset_box.setLayout(home_offset_layout)

        # Calibration Workflows Box
        workflow_box = QGroupBox("Calibration Workflows")
        workflow_layout = QVBoxLayout()
        
        self.btn_stop_motion = QPushButton("STOP MOTION")
        self.btn_stop_motion.setStyleSheet("background-color: #ff1744; color: #ffffff; font-weight: bold;")
        self.btn_stop_motion.clicked.connect(self.stop_motion)

        # Target Arm Selection
        self.arm_sel = QComboBox()
        self.arm_sel.addItems(UI_DROPDOWNS["arm_sides"])
        idx = 1 if self.arm_side == "left" else 0
        self.arm_sel.setCurrentIndex(idx)
        self.arm_sel.currentTextChanged.connect(self.on_arm_side_changed)

        self.joint_arm_sel = self.arm_sel
        self.marker_arm_sel = self.arm_sel

        workflow_header = QHBoxLayout()
        arm_side_layout = QHBoxLayout()
        arm_side_layout.addWidget(QLabel("Active Arm Side:"))
        arm_side_layout.addWidget(self.arm_sel)
        workflow_header.addLayout(arm_side_layout)
        workflow_header.addStretch()
        workflow_header.addWidget(self.btn_stop_motion)
        
        workflow_layout.addLayout(workflow_header)
        
        self.workflow_tabs = QTabWidget()
        
        # Sub-tab 1: Joint Calibration
        joint_subtab = QWidget()
        joint_sublayout = QVBoxLayout()
        
        self.joint_mode_sel = QComboBox()
        self.update_joint_modes()
        self.joint_mode_sel.currentIndexChanged.connect(self.update_applied_offset_label)
        
        self.btn_joint_ready = QPushButton("MOVE TO READY")
        self.btn_joint_ready.setStyleSheet("background-color: #6a1b9a; color: white;")
        self.btn_joint_ready.clicked.connect(self.move_to_ready_pose_joint)
        
        self.btn_joint_start = QPushButton("START SWEEP")
        self.btn_joint_start.setStyleSheet("background-color: #1565c0; color: white;")
        self.btn_joint_start.clicked.connect(self.start_calibration_joint)
        
        joint_sublayout.addWidget(QLabel("Joint Sweeps for Polarities & Kinematics:"))
        joint_sublayout.addWidget(self.joint_mode_sel)
        joint_sublayout.addWidget(self.btn_joint_ready)
        joint_sublayout.addWidget(self.btn_joint_start)
        joint_subtab.setLayout(joint_sublayout)
        
        # Sub-tab 2: Marker Bracket Calibration
        marker_subtab = QWidget()
        marker_sublayout = QVBoxLayout()
        
        self.marker_axis_sel = QComboBox()
        self.marker_axis_sel.addItems(UI_DROPDOWNS["marker_axes"])
        
        self.tolerance_input = QLineEdit("0.5")
        
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
        
        marker_sublayout.addWidget(QLabel("Marker Bracket Alignment Sweeps:"))
        marker_sublayout.addWidget(self.marker_axis_sel)
        marker_sublayout.addWidget(self.btn_marker_ready)
        marker_sublayout.addWidget(self.btn_marker_center)
        marker_sublayout.addWidget(self.btn_marker_start)
        marker_subtab.setLayout(marker_sublayout)
        
        # Sub-tab 3: Full Auto Calibration
        full_auto_subtab = QWidget()
        full_auto_sublayout = QVBoxLayout()
        
        self.btn_full_auto_ready = QPushButton("MOVE TO READY")
        self.btn_full_auto_ready.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        self.btn_full_auto_ready.clicked.connect(self.move_to_ready_full_auto)
        
        self.btn_full_auto_start = QPushButton("START FULL AUTO")
        self.btn_full_auto_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_full_auto_start.clicked.connect(self.start_full_auto)
        
        self.btn_full_auto_stop = QPushButton("STOP")
        self.btn_full_auto_stop.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold;")
        self.btn_full_auto_stop.clicked.connect(self.stop_full_auto)
        
        full_auto_sublayout.addWidget(QLabel("Full Auto Sequential Calibration:"))
        full_auto_sublayout.addWidget(self.btn_full_auto_ready)
        full_auto_sublayout.addWidget(self.btn_full_auto_start)
        full_auto_sublayout.addWidget(self.btn_full_auto_stop)
        full_auto_sublayout.addStretch()
        full_auto_subtab.setLayout(full_auto_sublayout)
        
        # Add workflow subtabs in order of: Full Auto, Marker Calib, Joint Calib
        self.workflow_tabs.addTab(full_auto_subtab, "1. Full Auto")
        self.workflow_tabs.addTab(marker_subtab, "2. Marker Calib")
        self.workflow_tabs.addTab(joint_subtab, "3. Joint Calib")
        
        workflow_layout.addWidget(self.workflow_tabs)
        workflow_box.setLayout(workflow_layout)

        # Assemble Column 1
        col1_layout.addWidget(conn_box)
        col1_layout.addWidget(head_box)
        col1_layout.addWidget(workflow_box)
        col1_layout.addStretch()

        # --- COLUMN 2 (Calibration Status & Monitoring) ---
        col2_layout = QVBoxLayout()

        dash_box = QGroupBox("Calibration Status & Monitoring")
        dash_layout = QVBoxLayout()
        dash_layout.setSpacing(4)
        dash_layout.setContentsMargins(8, 4, 8, 4)
        
        # Monitoring Table
        self.tbl_offset_monitor = QTableWidget(2, 2)
        self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 5 (Wrist)", "Joint 3 (Elbow)"])
        self.tbl_offset_monitor.setVerticalHeaderLabels(["Right Arm", "Left Arm"])
        self.tbl_offset_monitor.setFixedHeight(100)
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
        
        # Marker Bracket Design Offset UI GroupBox (Nested inside dash_box)
        bracket_box = QGroupBox("Marker Bracket Design Offset (Tf_to_marker)")
        bracket_layout = QVBoxLayout()
        bracket_layout.setSpacing(4)
        bracket_layout.setContentsMargins(6, 6, 6, 6)
        
        input_style = "background-color: #1c1c1c; color: #00e5ff; border: 1px solid #3d3d3d; border-radius: 3px; padding: 2px;"
        
        grid = QGridLayout()
        grid.setSpacing(6)
        
        # Column headers
        grid.addWidget(QLabel("Parameter"), 0, 0)
        grid.addWidget(QLabel("Left Arm"), 0, 1)
        grid.addWidget(QLabel("Right Arm"), 0, 2)
        
        # Row 1: X
        grid.addWidget(QLabel("X (m):"), 1, 0)
        self.txt_bracket_l_x = QLineEdit()
        self.txt_bracket_l_x.setStyleSheet(input_style)
        self.txt_bracket_r_x = QLineEdit()
        self.txt_bracket_r_x.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_x, 1, 1)
        grid.addWidget(self.txt_bracket_r_x, 1, 2)
        
        # Row 2: Y
        grid.addWidget(QLabel("Y (m):"), 2, 0)
        self.txt_bracket_l_y = QLineEdit()
        self.txt_bracket_l_y.setStyleSheet(input_style)
        self.txt_bracket_r_y = QLineEdit()
        self.txt_bracket_r_y.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_y, 2, 1)
        grid.addWidget(self.txt_bracket_r_y, 2, 2)
        
        # Row 3: Z
        grid.addWidget(QLabel("Z (m):"), 3, 0)
        self.txt_bracket_l_z = QLineEdit()
        self.txt_bracket_l_z.setStyleSheet(input_style)
        self.txt_bracket_r_z = QLineEdit()
        self.txt_bracket_r_z.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_z, 3, 1)
        grid.addWidget(self.txt_bracket_r_z, 3, 2)
        
        # Row 4: Roll
        grid.addWidget(QLabel("Roll (deg):"), 4, 0)
        self.txt_bracket_l_roll = QLineEdit()
        self.txt_bracket_l_roll.setStyleSheet(input_style)
        self.txt_bracket_r_roll = QLineEdit()
        self.txt_bracket_r_roll.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_roll, 4, 1)
        grid.addWidget(self.txt_bracket_r_roll, 4, 2)
        
        # Row 5: Pitch
        grid.addWidget(QLabel("Pitch (deg):"), 5, 0)
        self.txt_bracket_l_pitch = QLineEdit()
        self.txt_bracket_l_pitch.setStyleSheet(input_style)
        self.txt_bracket_r_pitch = QLineEdit()
        self.txt_bracket_r_pitch.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_pitch, 5, 1)
        grid.addWidget(self.txt_bracket_r_pitch, 5, 2)
        
        # Row 6: Yaw
        grid.addWidget(QLabel("Yaw (deg):"), 6, 0)
        self.txt_bracket_l_yaw = QLineEdit()
        self.txt_bracket_l_yaw.setStyleSheet(input_style)
        self.txt_bracket_r_yaw = QLineEdit()
        self.txt_bracket_r_yaw.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_yaw, 6, 1)
        grid.addWidget(self.txt_bracket_r_yaw, 6, 2)
        
        bracket_layout.addLayout(grid)
        
        self.btn_apply_bracket = QPushButton("APPLY BRACKETS")
        self.btn_apply_bracket.setStyleSheet("background-color: #2979ff; color: white; font-weight: bold; font-size: 11px;")
        self.btn_apply_bracket.setFixedHeight(24)
        self.btn_apply_bracket.clicked.connect(self.apply_bracket_design_values)
        bracket_layout.addWidget(self.btn_apply_bracket)
        
        bracket_box.setLayout(bracket_layout)
        dash_layout.addWidget(bracket_box)
        
        # Apply & Clear buttons for Joint offsets
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

        col2_layout.addWidget(home_offset_box)
        col2_layout.addWidget(dash_box)
        col2_layout.addStretch()

        # --- COLUMN 3 (Camera Status & System Log/Plots) ---
        col3_layout = QVBoxLayout()

        # Status Indicator Box (Constructed here for Col 3)
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
        
        self.chk_save_debug = QCheckBox("Save Debug Data")
        self.chk_save_debug.setChecked(True)
        
        status_layout.addLayout(ind_layout)
        status_layout.addLayout(btn_layout)
        status_layout.addWidget(self.temp_label)
        status_layout.addWidget(self.chk_save_debug)
        status_box.setLayout(status_layout)

        # Right Tab Widget
        self.right_tabs = QTabWidget()
        self.right_tabs.setMinimumHeight(320) # Sized down for overall UI height reduction
        
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

        col3_layout.addWidget(status_box)
        col3_layout.addWidget(self.right_tabs, 1)

        # Assemble side-by-side 3 Columns (3:3:4 weight)
        main_tab_columns = QHBoxLayout()
        main_tab_columns.addLayout(col1_layout, 3)
        main_tab_columns.addLayout(col2_layout, 3)
        main_tab_columns.addLayout(col3_layout, 4)
        
        main_tab_layout.addLayout(main_tab_columns)
        
        # Quit Button
        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setStyleSheet("background-color: #b71c1c; color: white;")
        self.btn_quit.clicked.connect(self.close)
        main_tab_layout.addWidget(self.btn_quit)
        
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

        # Move Calibration Controls from left column to right column underneath Stats Box
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
        
        int_right = QVBoxLayout()
        
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
        
        int_right.addWidget(stats_box)
        int_right.addWidget(controls_box) # Placed below stats box!
        int_right.addStretch()
        
        camera_tab_layout.addLayout(int_left, 2)
        camera_tab_layout.addLayout(int_right, 1)
        camera_tab.setLayout(camera_tab_layout)
        self.left_tabs.addTab(camera_tab, "2. Camera")
        
        # Assemble full-width tabs
        main_layout.addWidget(self.left_tabs)
        
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

    @property
    def is_mock(self) -> bool:
        if not self.robot:
            return True
        if self.robot == "mock_robot":
            return True
        if getattr(self.robot, "is_pure_mock", False) or hasattr(self.robot, "is_pure_mock"):
            return True
        if type(self.robot).__name__ in ("PureMockRobot", "OfflineRobot"):
            return True
        return False

    # --- Common Helper Functions ---
    def log_msg(self, msg):
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(msg)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        else:
            print(msg)

    def connect_robot(self):
        from core.calibration.mock_robot import get_mock_robot, pure_mock_compute_fk_impl
        from core.calibration.CalibratorBase import BaseCalibrator

        if self.robot:
            self.log_msg("[INFO] Disconnecting from robot...")
            if not self.is_mock:
                MarkerCalibrator.terminate_robot(self.robot)
            self.robot = None
            self.robot_version = "1.2"
            self.marker_calibrator.robot = None
            self.marker_calibrator.robot_version = "1.2"
            self.joint_calibrator.robot = None
            self.joint_calibrator.robot_version = "1.2"
            self.update_joint_modes()
            self.load_offsets_from_yaml()
            self.update_applied_offset_label()
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
                    self.robot = get_mock_robot(model_name=model)
                    if getattr(self.robot, "is_pure_mock", False):
                        BaseCalibrator.compute_fk = staticmethod(pure_mock_compute_fk_impl)
            else:
                self.robot = BaseCalibrator.initialize_robot(addr, model)
                
            if self.robot:
                self.marker_calibrator.robot = self.robot
                self.joint_calibrator.robot = self.robot
                
                # Determine version classification automatically
                detected_version = "1.2"
                if not self.is_mock:
                    try:
                        robot_info = self.robot.get_robot_info()
                        raw_version = robot_info.robot_model_version
                        self.log_msg(f"[INFO] Connected robot model version string: '{raw_version}'")
                        print(f"[INFO] Connected robot model version string: '{raw_version}'")
                        
                        if "1.3" in raw_version:
                            detected_version = "1.3"
                        else:
                            detected_version = "1.2"
                    except Exception as e:
                        self.log_msg(f"[WARNING] Failed to query version from robot: {e}")
                        detected_version = "1.2"
                else:
                    detected_version = "1.3" if model == "m" else "1.2"
                    self.log_msg(f"[INFO] Connected to mock robot. Using default version: {detected_version}")
                    print(f"[INFO] Connected to mock robot. Using default version: {detected_version}")
                
                # Cache the version classification on the app instance
                self.robot_version = detected_version

                # Configure calibrators version
                self.marker_calibrator.robot_version = detected_version
                self.joint_calibrator.robot_version = detected_version

                # Update UI modes and offsets based on version classification
                self.update_joint_modes()
                self.load_offsets_from_yaml()
                self.update_applied_offset_label()

                # Setup SimulatedMarkerTransform if simulator is connected in UI Mode
                if self.ui_only and not self.is_mock:
                    self.marker_st = SimulatedMarkerTransform(self.robot, self.marker_calibrator.camera_config)
                    self.marker_calibrator.marker_st = self.marker_st
                    self.joint_calibrator.marker_st = self.marker_st
                    self.log_msg("[INFO] Configured SimulatedMarkerTransform for simulation motion.")

                self.log_msg(f"[INFO] Robot successfully connected and initialized (Classified Version: {detected_version}).")
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
            self.marker_data_4 = None
            self.marker_data_5 = None
            self.marker_data_6 = None
            self.joint_sweep_data = None
            
            # Sync current offsets with active arm_side from memory store (do not reload from yaml disk)
            is_v13 = self.get_robot_version() == "1.3"
            self.joint_offsets["wrist_pitch"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint5", 0.0)
            if is_v13:
                self.joint_offsets["wrist_roll"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint6", 0.0)
            else:
                self.joint_offsets["wrist_roll"] = 0.0
            self.joint_offsets["elbow"] = self.joint_offsets_store.get(self.arm_side, {}).get("joint3", 0.0)
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets
            self.update_applied_offset_label()
            
            self.load_bracket_design_values()
            
            # Sync dropdown indexes between controls (blocking signals to avoid cycles)
            self.arm_sel.blockSignals(True)
            idx = 1 if self.arm_side == "left" else 0
            self.arm_sel.setCurrentIndex(idx)
            self.arm_sel.blockSignals(False)

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
        # Camera Tab이 켜져있을 때는 poll_camera_status 생략 (update_video_frame이 처리함)
        if self.left_tabs.currentIndex() == 1:
            return
            
        try:
            # 좌/우 중 하나라도 인식되었는지 확인하기 위해 "all"로 검출 수행
            res_all = self.marker_st.get_marker_transform(sampling_time=0, side="all")
            detected = bool(res_all and len(res_all) > 0)
            self.update_marker_indicator(detected)
            
            if not self.btn_monitor.isChecked():
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText("Position: Monitor Off")
                return
                
            # 모니터가 켜진 경우, 현재 active arm side의 개별 마커 좌표 조회 및 표시
            res = self.marker_st.get_marker_transform(sampling_time=0, side=self.arm_side)
            if res and len(res) > 0:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                x, y, z = pose[:3, 3] * 1000.0
                
                self.log_msg(f"[LIVE] Marker ({self.arm_side}) X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm")
                
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText(f"Position ({self.arm_side}): X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f} mm")
            else:
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText(f"Position ({self.arm_side}): Marker Not Detected")
            
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
        
        is_v13 = self.get_robot_version() == "1.3"
        if is_v13:
            self.tbl_offset_monitor.setColumnCount(3)
            self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Roll)", "Joint 5 (Pitch)", "Joint 3 (Elbow)"])
            for row_idx, arm in enumerate(["right", "left"]):
                for col_idx, joint_key in enumerate(["joint6", "joint5", "joint3"]):
                    val = self.joint_offsets_store.get(arm, {}).get(joint_key, 0.0)
                    item = QTableWidgetItem(f"{val:.4f}°")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.tbl_offset_monitor.setItem(row_idx, col_idx, item)
        else:
            self.tbl_offset_monitor.setColumnCount(2)
            self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 5 (Wrist)", "Joint 3 (Elbow)"])
            for row_idx, arm in enumerate(["right", "left"]):
                for col_idx, joint_key in enumerate(["joint5", "joint3"]):
                    val = self.joint_offsets_store.get(arm, {}).get(joint_key, 0.0)
                    item = QTableWidgetItem(f"{val:.4f}°")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.tbl_offset_monitor.setItem(row_idx, col_idx, item)
        
        # Keeps the apply button permanently visible on the Right Panel dashboard

    def apply_joint_offset(self):
        is_v13 = self.get_robot_version() == "1.3"
        
        if is_v13:
            self.joint_offsets["wrist_pitch"] = self.joint_offsets_store[self.arm_side]["joint5"]
            self.joint_offsets["wrist_roll"] = self.joint_offsets_store[self.arm_side]["joint6"]
        else:
            self.joint_offsets["wrist_pitch"] = self.joint_offsets_store[self.arm_side]["joint5"]
            self.joint_offsets["wrist_roll"] = 0.0
        self.joint_offsets["elbow"] = self.joint_offsets_store[self.arm_side]["joint3"]
        self.joint_calibrator.joint_offsets = self.joint_offsets
        self.marker_calibrator.joint_offsets = self.joint_offsets
        
        self.save_offsets_to_yaml()
        self.update_applied_offset_label()
        
        self.log_msg(f"\n" + "="*50)
        self.log_msg(f"[APPLY] Applied current staged offsets for {self.arm_side.upper()} Arm to active parameters:")
        if is_v13:
            self.log_msg(f"  - Joint 6 (wrist_roll) : {self.joint_offsets['wrist_roll']:.4f}°")
            self.log_msg(f"  - Joint 5 (wrist_pitch): {self.joint_offsets['wrist_pitch']:.4f}°")
        else:
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
            if not self.is_mock:
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
            self.joint_offsets_store[self.arm_side]["joint6"] = 0.0
            self.joint_offsets_store[self.arm_side]["joint3"] = 0.0
            
            self.joint_offsets["wrist_pitch"] = 0.0
            self.joint_offsets["wrist_roll"] = 0.0
            self.joint_offsets["elbow"] = 0.0
            self.joint_calibrator.joint_offsets = self.joint_offsets
            self.marker_calibrator.joint_offsets = self.joint_offsets
            
            self.save_offsets_to_yaml()
            self.update_applied_offset_label()
            
            self.log_msg(f"[CLEAR] Staged and saved offsets cleared to 0.0 for {self.arm_side.upper()} Arm.")

    def load_bracket_design_values(self):
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                if data and "camera" in data:
                    # Load Left Arm values
                    val_left = data["camera"].get("Tf_to_marker_left", None)
                    if val_left and len(val_left) == 6:
                        self.txt_bracket_l_x.setText(f"{val_left[0]:.4f}")
                        self.txt_bracket_l_y.setText(f"{val_left[1]:.4f}")
                        self.txt_bracket_l_z.setText(f"{val_left[2]:.4f}")
                        self.txt_bracket_l_roll.setText(f"{val_left[3]:.2f}")
                        self.txt_bracket_l_pitch.setText(f"{val_left[4]:.2f}")
                        self.txt_bracket_l_yaw.setText(f"{val_left[5]:.2f}")
                    
                    # Load Right Arm values
                    val_right = data["camera"].get("Tf_to_marker_right", None)
                    if val_right and len(val_right) == 6:
                        self.txt_bracket_r_x.setText(f"{val_right[0]:.4f}")
                        self.txt_bracket_r_y.setText(f"{val_right[1]:.4f}")
                        self.txt_bracket_r_z.setText(f"{val_right[2]:.4f}")
                        self.txt_bracket_r_roll.setText(f"{val_right[3]:.2f}")
                        self.txt_bracket_r_pitch.setText(f"{val_right[4]:.2f}")
                        self.txt_bracket_r_yaw.setText(f"{val_right[5]:.2f}")
                    
                    self.log_msg(f"[INFO] Loaded Tf_to_marker values for both arms from setting.yaml")
                    return
            self.log_msg("[WARNING] Could not load bracket design values from setting.yaml.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to load setting.yaml: {e}")

    def apply_bracket_design_values(self, silent=False):
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            try:
                l_x = float(self.txt_bracket_l_x.text())
                l_y = float(self.txt_bracket_l_y.text())
                l_z = float(self.txt_bracket_l_z.text())
                l_roll = float(self.txt_bracket_l_roll.text())
                l_pitch = float(self.txt_bracket_l_pitch.text())
                l_yaw = float(self.txt_bracket_l_yaw.text())
                
                r_x = float(self.txt_bracket_r_x.text())
                r_y = float(self.txt_bracket_r_y.text())
                r_z = float(self.txt_bracket_r_z.text())
                r_roll = float(self.txt_bracket_r_roll.text())
                r_pitch = float(self.txt_bracket_r_pitch.text())
                r_yaw = float(self.txt_bracket_r_yaw.text())
            except ValueError:
                if not silent:
                    QMessageBox.critical(self, "Invalid Inputs", "Please enter valid numeric values for all bracket design fields.")
                return
            
            lines = []
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    lines = f.readlines()
            
            def update_key_in_lines(lines_list, key_str, new_vals_list):
                camera_idx = -1
                for idx, line in enumerate(lines_list):
                    if line.strip().startswith("camera:"):
                        camera_idx = idx
                        break
                
                new_val_str = f"[{new_vals_list[0]:.5f}, {new_vals_list[1]:.5f}, {new_vals_list[2]:.5f}, {new_vals_list[3]:.2f}, {new_vals_list[4]:.2f}, {new_vals_list[5]:.2f}]"
                key_found = False
                if camera_idx != -1:
                    i = camera_idx + 1
                    while i < len(lines_list):
                        line = lines_list[i]
                        stripped = line.strip()
                        if not stripped:
                            i += 1
                            continue
                        if not line.startswith(" ") and not line.startswith("\t") and stripped.endswith(":"):
                            break
                        
                        if stripped.startswith(f"{key_str}:"):
                            comment = ""
                            if "#" in line:
                                comment_idx = line.find("#")
                                comment = " " + line[comment_idx:].rstrip()
                            
                            indent = len(line) - len(line.lstrip())
                            lines_list[i] = " " * indent + f"{key_str}: {new_val_str}{comment}\n"
                            key_found = True
                            break
                        i += 1
                
                if not key_found:
                    if camera_idx == -1:
                        lines_list.append("camera:\n")
                        lines_list.append(f"  {key_str}: {new_val_str}\n")
                    else:
                        lines_list.insert(camera_idx + 1, f"  {key_str}: {new_val_str}\n")
            
            update_key_in_lines(lines, "Tf_to_marker_left", [l_x, l_y, l_z, l_roll, l_pitch, l_yaw])
            update_key_in_lines(lines, "Tf_to_marker_right", [r_x, r_y, r_z, r_roll, r_pitch, r_yaw])
            
            with open(config_path, "w") as f:
                f.writelines(lines)
                
            self.log_msg(f"[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml")
            if not silent:
                QMessageBox.information(self, "Success", "Bracket design values saved for both arms!")
            
            if not self.ui_only and self.marker_st is not None:
                detector = self.marker_st.marker_detection
                if hasattr(detector, 'camera_config'):
                    detector.camera_config["Tf_to_marker_left"] = [l_x, l_y, l_z, l_roll, l_pitch, l_yaw]
                    detector.camera_config["Tf_to_marker_right"] = [r_x, r_y, r_z, r_roll, r_pitch, r_yaw]
                    detector.Tf_to_marker_tf_left = detector.make_transform(detector.camera_config["Tf_to_marker_left"])
                    detector.Tf_to_marker_tf_right = detector.make_transform(detector.camera_config["Tf_to_marker_right"])
                    self.log_msg("[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save bracket values: {e}")
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to save bracket values: {e}")

    def set_controls_enabled(self, enabled):
        if hasattr(self, 'btn_full_auto_start'):
            self.btn_full_auto_start.setEnabled(enabled)
        self.btn_joint_ready.setEnabled(enabled)
        self.btn_joint_start.setEnabled(enabled)
        self.btn_joint_apply.setEnabled(enabled)
        if hasattr(self, 'btn_joint_clear'):
            self.btn_joint_clear.setEnabled(enabled)
        if hasattr(self, 'btn_apply_bracket'):
            self.btn_apply_bracket.setEnabled(enabled)
            
        if hasattr(self, 'txt_bracket_l_x'):
            self.txt_bracket_l_x.setEnabled(enabled)
            self.txt_bracket_l_y.setEnabled(enabled)
            self.txt_bracket_l_z.setEnabled(enabled)
            self.txt_bracket_l_roll.setEnabled(enabled)
            self.txt_bracket_l_pitch.setEnabled(enabled)
            self.txt_bracket_l_yaw.setEnabled(enabled)
            
            self.txt_bracket_r_x.setEnabled(enabled)
            self.txt_bracket_r_y.setEnabled(enabled)
            self.txt_bracket_r_z.setEnabled(enabled)
            self.txt_bracket_r_roll.setEnabled(enabled)
            self.txt_bracket_r_pitch.setEnabled(enabled)
            self.txt_bracket_r_yaw.setEnabled(enabled)
        
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
        if hasattr(self, 'marker_axis_sel'):
            self.marker_axis_sel.setEnabled(enabled)
        if hasattr(self, 'btn_camera_feed'):
            self.btn_camera_feed.setEnabled(True) # Keep camera feed button enabled always!

    def on_action_finished(self):
        self.set_controls_enabled(True)

    def get_robot_version(self) -> str:
        return str(getattr(self, "robot_version", "1.2"))

    def move_to_ready_full_auto(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        self.active_worker = FullAutoReadyWorker(
            self.joint_calibrator,
            self.marker_calibrator,
            ui_only=self.ui_only
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_full_auto_ready_finished)
        self.active_worker.start()

    def on_full_auto_ready_finished(self):
        self.set_controls_enabled(True)
        self.active_worker = None
        # Restart poll_timer if appropriate
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        if self.left_tabs.currentIndex() != 1 and not dialog_visible:
            if not self.poll_timer.isActive():
                self.poll_timer.start(200)

    def get_latest_result_path(self):
        result_dir = Path(CONFIG_PATHS["result_dir"])
        if not result_dir.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
        result_files = sorted(
            result_dir.glob("result_*.json"),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )
        if not result_files:
            raise RuntimeError(f"No calibration result JSON found in {result_dir}")
        return result_files[0]

    def get_latest_home_reset_path(self, required=True):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = Path(os.path.abspath(os.path.join(current_dir, "config", "home_reset_baseline.json")))
        if path.exists():
            return path
        if required:
            raise RuntimeError(f"No home reset baseline JSON found at {path}")
        return None

    def get_home_reset_path_for_result(self, result_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = Path(os.path.abspath(os.path.join(current_dir, "config", "home_reset_baseline.json")))
        if path.exists():
            return path
        return self.get_latest_home_reset_path(required=False)

    def apply_home_offset(self):
        try:
            result_path = self.get_latest_result_path()
            baseline_path = self.get_home_reset_path_for_result(result_path)
            
            # Open Apply Home Offset Dialog
            dialog = ApplyHomeOffsetDialog(
                self,
                result_path,
                baseline_path,
                self.arm_side,
                include_head=True
            )
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Apply Home Offset Error", str(e))
            self.log_msg(f"[ERROR] Apply home offset failed: {e}")

    def home_offset_reset(self):
        if not self.ui_only and not self.robot:
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
        reply = QMessageBox.warning(
            self, "Confirm Home Offset Reset", msg,
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            return

        self.set_controls_enabled(False)
        self.btn_home_reset.setEnabled(False)
        
        # Start worker thread
        self.active_worker = HomeOffsetResetWorker(
            self.robot,
            self.robot.model() if (self.robot and not self.is_mock) else None,
            self.model_input.currentText().strip() if hasattr(self, 'model_input') else "a",
            include_head=True
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_home_offset_reset_finished)
        self.active_worker.start()

    def on_home_offset_reset_finished(self, result):
        self.set_controls_enabled(True)
        self.btn_home_reset.setEnabled(True)
        self.active_worker = None
        
        if result.get("success", False):
            QMessageBox.information(self, "Success", "Home Offset Reset completed successfully!")
            self.log_msg("Re-connecting and initializing robot...")
            self.connect_robot()
            self.log_msg("Home Offset Reset complete!")
        else:
            QMessageBox.warning(self, "Warning", f"Home Offset Reset finished, but some joints failed to reset: {result.get('error', '')}")

    def start_full_auto(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        self.log_text.clear()
        self.log_msg("[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...")
        if self.ui_only:
            self.log_msg("[MOCK GT] Simulated Ground-Truth Offsets:")
            is_v13 = self.get_robot_version() == "1.3"
            for arm in ["right", "left"]:
                if arm == "right":
                    j6_gt = 3.2
                    j5_gt = -2.1 if is_v13 else 1.5
                    j3_gt = 0.8
                    pos_gt = [3.0, -1.0, 4.0] # mm
                    rpy_gt = [1.0, -1.2, 0.8] # deg
                else:
                    j6_gt = -2.5
                    j5_gt = 3.6 if is_v13 else -1.8
                    j3_gt = 0.8
                    pos_gt = [-2.0, 1.0, -3.0] # mm
                    rpy_gt = [-0.8, 1.5, -1.2] # deg
                self.log_msg(f"  --- {arm.upper()} ARM ---")
                self.log_msg(f"  * Bracket Pos: X: {pos_gt[0]:+.1f}, Y: {pos_gt[1]:+.1f}, Z: {pos_gt[2]:+.1f} mm")
                self.log_msg(f"  * Bracket Rot: R: {rpy_gt[0]:+.2f}, P: {rpy_gt[1]:+.2f}, Y: {rpy_gt[2]:+.2f} deg")
                self.log_msg(f"  * Joint Offsets: Joint 6: {j6_gt:+.2f}°, Joint 5: {j5_gt:+.2f}°, Joint 3: {j3_gt:+.2f}°")
        
        self.set_controls_enabled(False)
        self.btn_full_auto_start.setEnabled(False)
        self.btn_full_auto_stop.setEnabled(True)
        
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        
        
        self.full_auto_stop_event = threading.Event()
        
        # update mock robot version on calibrators just in case
        v = self.get_robot_version()
        self.marker_calibrator.robot_version = v
        self.joint_calibrator.robot_version = v
        
        self.active_worker = FullAutoWorker(
            self.joint_calibrator,
            self.marker_calibrator,
            ui_only=self.ui_only,
            stop_event=self.full_auto_stop_event,
            joint_offsets_store=self.joint_offsets_store
        )
        
        self.active_worker.log_msg.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.bracket_finished_signal.connect(self.handle_full_auto_bracket_finished)
        self.active_worker.joint_finished_signal.connect(self.handle_full_auto_joint_finished)
        self.active_worker.finished_signal.connect(self.on_full_auto_finished)
        self.active_worker.start()

    def stop_full_auto(self):
        self.log_msg("[STOP] Stopping Full Auto Calibration...")
        if hasattr(self, 'full_auto_stop_event') and self.full_auto_stop_event:
            self.full_auto_stop_event.set()
        self.joint_calibrator.stop_requested = True
        self.marker_calibrator.stop_requested = True
        if self.robot and not self.is_mock:
            self.robot.cancel_control()

    def on_full_auto_finished(self):
        self.set_controls_enabled(True)
        if hasattr(self, 'btn_full_auto_start'):
            self.btn_full_auto_start.setEnabled(True)
        self.active_worker = None
        self.log_msg("[INFO] Full Auto sequential calibration ended.")
        
        # Restart poll_timer if appropriate (not tab 2 and feed dialog closed)
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        if self.left_tabs.currentIndex() != 1 and not dialog_visible:
            if not self.poll_timer.isActive():
                self.poll_timer.start(200)

    def handle_full_auto_bracket_finished(self, bracket_res):
        arm_side = bracket_res['arm_side']
        
        # Update UI text boxes for corresponding arm
        if arm_side == "left":
            self.txt_bracket_l_x.setText(f"{bracket_res['x_e']/1000.0:.4f}")
            self.txt_bracket_l_y.setText(f"{bracket_res['y_e']/1000.0:.4f}")
            self.txt_bracket_l_z.setText(f"{bracket_res['z_e']/1000.0:.4f}")
            self.txt_bracket_l_roll.setText(f"{bracket_res['roll_e']:.2f}")
            self.txt_bracket_l_pitch.setText(f"{bracket_res['pitch_e']:.2f}")
            self.txt_bracket_l_yaw.setText(f"{bracket_res['yaw_e']:.2f}")
        else:
            self.txt_bracket_r_x.setText(f"{bracket_res['x_e']/1000.0:.4f}")
            self.txt_bracket_r_y.setText(f"{bracket_res['y_e']/1000.0:.4f}")
            self.txt_bracket_r_z.setText(f"{bracket_res['z_e']/1000.0:.4f}")
            self.txt_bracket_r_roll.setText(f"{bracket_res['roll_e']:.2f}")
            self.txt_bracket_r_pitch.setText(f"{bracket_res['pitch_e']:.2f}")
            self.txt_bracket_r_yaw.setText(f"{bracket_res['yaw_e']:.2f}")
            
        # Stage Joint 5 and Joint 6 offsets if solved (v1.3)
        if 'opt_delta_5' in bracket_res:
            self.joint_offsets_store[arm_side]["joint5"] = float(bracket_res['opt_delta_5'])
            self.joint_offsets_store[arm_side]["joint6"] = float(bracket_res['opt_delta_6'])
            self.update_applied_offset_label()
            self.log_msg(f"[INFO] Full Auto: Staged joint offsets for {arm_side.upper()} Arm - Joint 5: {bracket_res['opt_delta_5']:.4f}°, Joint 6: {bracket_res['opt_delta_6']:.4f}°")

        self.log_msg(f"[INFO] Full Auto: Finished bracket calibration for {arm_side.upper()} arm. Values staged in UI (click APPLY BRACKETS to save).")

    def handle_full_auto_joint_finished(self, joint_res):
        arm_side = joint_res['arm_side']
        mode = joint_res.get('mode', 'elbow')
        
        recommended = joint_res['recommended_joint_offset']
        if mode == "wrist_roll_v13":
            joint_key = "joint6"
        elif mode in ("wrist_pitch_v13", "wrist_pitch"):
            joint_key = "joint5"
        else:
            joint_key = "joint3"
            recommended = np.clip(recommended, -3.0, 0.0)
            
        # Update staged offsets store
        self.joint_offsets_store[arm_side][joint_key] = float(recommended)
        
        # Refresh offset monitor table view
        self.update_applied_offset_label()
        
        self.log_msg(f"[INFO] Full Auto: Finished joint calibration for {arm_side.upper()} {mode}. Staged: {recommended:.4f}° (click APPLY OFFSET to save).")
        
        if hasattr(self, 'stop_event_mc'):
            self.stop_event_mc.clear()
        
        # 탭 상태에 맞춰 타이머 활성화
        self.on_left_tab_changed(self.left_tabs.currentIndex())

    # --- Joint Calibration Workflows ---
    def move_to_ready_pose_joint(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return


        mode = self.get_selected_joint_mode()
        
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

        mode = self.get_selected_joint_mode()
        if mode == "wrist_pitch_v13" and not self.wrist_roll_calibrated.get(self.arm_side, False):
            reply = QMessageBox.warning(
                self,
                "Calibration Sequence Warning",
                "Wrist Roll (Joint 6) has not been calibrated yet.\n"
                "It is highly recommended to calibrate Joint 6 (wrist_roll_v13) first.\n"
                "Do you want to proceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        offset_key = self.get_offset_key_for_mode(mode)
        self.original_joint_offset = self.joint_offsets.get(offset_key, 0.0)
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Joint Sweep: {mode.upper()}")
        if self.ui_only:
            is_v13 = self.get_robot_version() == "1.3"
            if self.arm_side == "right":
                j_gt = {"wrist_roll_v13": 3.2, "wrist_pitch_v13": -2.1, "wrist_pitch": 1.5, "elbow": 0.8}
            else:
                j_gt = {"wrist_roll_v13": -2.5, "wrist_pitch_v13": 3.6, "wrist_pitch": -1.8, "elbow": 0.8}
            gt_val = j_gt.get(mode, 0.0)
            self.log_msg(f"[MOCK GT] Simulated Target Joint Offset: {gt_val:+.2f}°")
        
        curr_offset = self.joint_offsets.get(offset_key, 0.0)
        self.active_worker = JointCalibrationWorker(
            self.joint_calibrator, self.arm_side, mode, 
            ui_only=self.ui_only, 
            current_offset_deg=curr_offset,
            sweep_duration=15.0,
            save_debug=self.chk_save_debug.isChecked()
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

        if mode == "wrist_roll_v13":
            joint_key = "joint6"
            if converged:
                self.wrist_roll_calibrated[self.arm_side] = True
        elif mode == "wrist_pitch_v13":
            joint_key = "joint5"
        elif mode == "wrist_pitch":
            joint_key = "joint5"
        else:
            joint_key = "joint3"
        self.joint_offsets_store[self.arm_side][joint_key] = float(self.recommended_joint_offset)

        # Revert active offsets to nominal original values in model (until user clicks APPLY)
        offset_key = self.get_offset_key_for_mode(mode)
        self.joint_offsets[offset_key] = self.original_joint_offset
        self.joint_calibrator.joint_offsets[offset_key] = self.original_joint_offset
        self.marker_calibrator.joint_offsets[offset_key] = self.original_joint_offset
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
        if mode == "wrist_roll_v13":
            joint_name = "Joint 6 (Wrist Roll)"
        elif mode == "wrist_pitch_v13":
            joint_name = "Joint 5 (Wrist Pitch)"
        elif mode == "wrist_pitch":
            joint_name = "Joint 5 (Wrist Pitch)"
        else:
            joint_name = "Joint 3 (Elbow Pitch)"
        self.log_msg(f"    - Target Swept Joint       : {joint_name}")
        self.log_msg(f"    - Estimated Optimal Offset : {recommended:.4f} deg")

        self.log_msg("\n[2] Suggested Joint Home Offset update:")
        self.log_msg(f"  Add offset: {recommended:.4f} deg to calibration config.")

        # v1.3 only: display bracket design verification results
        if mode in ("wrist_pitch_v13", "wrist_roll_v13"):
            d = self.joint_sweep_data
            sweep_axis_label = "Joint 6" if mode == "wrist_roll_v13" else "Joint 5"
            self.log_msg(f"\n[3] Bracket Design Verification (Based on {sweep_axis_label} Axis)")
            perp_b = d.get('perp_dist_before', float('nan'))
            perp_a = d.get('perp_dist_after',  float('nan'))
            self.log_msg(f"    - c_B ~ {sweep_axis_label} axis perp. dist (before) : {perp_b:.4f} mm")
            self.log_msg(f"    - c_B ~ {sweep_axis_label} axis perp. dist (after)  : {perp_a:.4f} mm")
            r_A = d.get('r_A', float('nan'))
            axial  = d.get('axial_offset_mm',   float('nan'))
            lateral = d.get('lateral_offset_mm', float('nan'))
            self.log_msg(f"    - Sweep A fitting radius (r_A, lateral marker offset) : {r_A:.3f} mm")
            self.log_msg(f"    - Axial marker offset (c_B along {sweep_axis_label} axis)  : {axial:.3f} mm")
            self.log_msg(f"    - Lateral marker offset (c_B perp {sweep_axis_label} axis)  : {lateral:.3f} mm")
            axis_dir = "X" if mode == "wrist_roll_v13" else "Y"
            self.log_msg(f"    * Design Reference Offset Axis: {axis_dir}-axis")

        self.log_msg("="*50)


    # --- Marker Bracket Calibration Workflows ---
    def move_to_ready_pose_marker(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
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
        if not self.ui_only and not self.robot:
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
            if self.arm_side == "right":
                pos_gt = [3.0, -1.0, 4.0] # mm
                rpy_gt = [1.0, -1.2, 0.8] # deg
            else:
                pos_gt = [-2.0, 1.0, -3.0] # mm
                rpy_gt = [-0.8, 1.5, -1.2] # deg
            self.log_msg(f"[MOCK GT] Simulated Bracket Offset (Tf_to_marker):")
            self.log_msg(f"  * Pos: X: {pos_gt[0]:+.1f}, Y: {pos_gt[1]:+.1f}, Z: {pos_gt[2]:+.1f} mm")
            self.log_msg(f"  * Rot: R: {rpy_gt[0]:+.2f}, P: {rpy_gt[1]:+.2f}, Y: {rpy_gt[2]:+.2f} deg")

        try:
            tolerance = float(self.tolerance_input.text())
        except ValueError:
            tolerance = 0.5
        self.active_worker = MarkerCalibrationWorker(
            self.marker_calibrator, self.arm_side, 
            use_head_tracking=use_head, tolerance=tolerance, 
            save_debug=self.chk_save_debug.isChecked()
        )
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
            self.marker_data_4 = res.get('res_4', None)
                
            # Stage Joint 5 and Joint 6 offsets if solved (v1.3)
            if 'opt_delta_5' in res:
                self.joint_offsets_store[self.arm_side]["joint5"] = float(res['opt_delta_5'])
                self.joint_offsets_store[self.arm_side]["joint6"] = float(res['opt_delta_6'])
                self.update_applied_offset_label()
                self.log_msg(f"[INFO] Staged joint offsets for {self.arm_side.upper()} Arm - Joint 5: {res['opt_delta_5']:.4f}°, Joint 6: {res['opt_delta_6']:.4f}°")

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
        r4_str = f", R4: {res['radius_4']:.2f} mm" if res.get('radius_4', 0.0) > 0.0 else ""
        self.log_msg(f"       * (L_5_ee: {res['L_5_ee']:.1f} mm, R6: {res['radius_6']:.2f} mm, R5: {res['radius_5']:.2f} mm{r4_str})")
        if 'opt_delta_5' in res:
            self.log_msg(f"    - Opt Delta 5 (5축 오프셋): {res['opt_delta_5']:.3f} deg")
            self.log_msg(f"    - Opt Delta 6 (6축 오프셋): {res['opt_delta_6']:.3f} deg")
            self.log_msg(f"    - Min Circle Fitting Radius (최소 원 피팅 반지름): {res['min_radius']:.2f} mm")
            
        self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
        self.log_msg(f"    - Roll : {res['roll_e']:.2f} deg")
        self.log_msg(f"    - Pitch: {res['pitch_e']:.2f} deg")
        self.log_msg(f"    - Yaw  : {res['yaw_e']:.2f} deg")
        
        self.log_msg("\n[3] setting.yaml Config Update values:")
        x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0
        
        if self.arm_side == "left":
            self.log_msg(f"  Tf_to_marker_left:  [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
        else:
            self.log_msg(f"  Tf_to_marker_right: [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
            
        self.log_msg(f"\n[4] Confidence Metrics:")
        self.log_msg(f"    - Orthogonality Error  : {res['ortho_err']:.3f} deg")
        
        rmse_warn = res['rmse_6'] > 0.5 or res['rmse_5'] > 0.5 or res.get('rmse_4', 0.0) > 0.5
        if rmse_warn:
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
            marker_data_4_val = getattr(self, 'marker_data_4', None)
            res = self.marker_calibrator.compute_unified_bracket_calibration(
                self.marker_data_5, self.marker_data_6, self.arm_side, tolerance=tolerance, marker_data_4=marker_data_4_val
            )
            
            self.log_msg("\n[1] Cartesian Offset (EE Link Frame)")
            self.log_msg(f"    - X-Offset: {res['x_e']:.2f} mm")
            self.log_msg(f"    - Y-Offset: {res['y_e']:.2f} mm")
            self.log_msg(f"    - Z-Offset: {res['z_e']:.2f} mm")
            r4_str = f", R4: {res['radius_4']:.2f} mm" if res.get('radius_4', 0.0) > 0.0 else ""
            self.log_msg(f"       * (L_5_ee: {res['L_5_ee']:.1f} mm, R6: {res['radius_6']:.2f} mm, R5: {res['radius_5']:.2f} mm{r4_str})")
            if 'opt_delta_5' in res:
                self.log_msg(f"    - Opt Delta 5 (5축 오프셋): {res['opt_delta_5']:.3f} deg")
                self.log_msg(f"    - Opt Delta 6 (6축 오프셋): {res['opt_delta_6']:.3f} deg")
                self.log_msg(f"    - Min Circle Fitting Radius (최소 원 피팅 반지름): {res['min_radius']:.2f} mm")
                
            self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
            self.log_msg(f"    - Roll : {res['roll_e']:.2f} deg")
            self.log_msg(f"    - Pitch: {res['pitch_e']:.2f} deg")
            self.log_msg(f"    - Yaw  : {res['yaw_e']:.2f} deg")
            
            self.log_msg("\n[3] setting.yaml Config Update values:")
            x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0
            
            if self.arm_side == "left":
                self.log_msg(f"  Tf_to_marker_left:  [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
            else:
                self.log_msg(f"  Tf_to_marker_right: [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
                
            self.log_msg(f"\n[4] Confidence Metrics:")
            self.log_msg(f"    - Orthogonality Error  : {res['ortho_err']:.3f} deg")
            
            rmse_warn = res['rmse_6'] > 0.5 or res['rmse_5'] > 0.5 or res.get('rmse_4', 0.0) > 0.5
            if rmse_warn:
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
            
            # 백그라운드 마커 검출 및 상태 표시 업데이트
            try:
                res_all = self.marker_st.get_marker_transform(sampling_time=0, side="all")
                detected = bool(res_all and len(res_all) > 0)
                self.update_marker_indicator(detected)
            except Exception:
                pass
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
        if event.key() == Qt.Key_C and self.left_tabs.currentIndex() == 1:
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
        save_path = os.path.join(CONFIG_PATHS["plot_dir"], "camera_intrinsics_verification.png")
        
        # Delegate image generation to IntrinsicsCalibrator
        self.intrinsics_calibrator.generate_verification_image(test_img, save_path)
        
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
