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
import json
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QCheckBox, 
                             QLineEdit, QDialog, QMessageBox, QTabWidget, QInputDialog, QGridLayout,
                             QTableWidget, QHeaderView, QTableWidgetItem, QSizePolicy, QRadioButton, QStackedWidget, QButtonGroup,
                             QSpinBox, QSlider)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QMetaObject
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

"""Compact UI numbers without quantizing unchanged calibration parameters."""


def set_numeric_field(field, value):
    value = float(value)
    rendered = f'{value:.4f}'.rstrip('0').rstrip('.')
    if rendered == '-0':
        rendered = '0'
    field.setProperty('calibration_number', value)
    field.setProperty('calibration_rendered', rendered)
    field.setText(rendered)


def read_numeric_field(field):
    text = field.text().strip()
    if text == field.property('calibration_rendered'):
        return float(field.property('calibration_number'))
    return float(text)

def _joint_result_accepted(result):
    return bool(result and result.get('measurement_accepted', True) and result.get('converged', False))


def _joint_store_key(mode):
    if mode in ('wrist_yaw2', 'wrist_roll_v13'):
        return 'joint6'
    if mode in ('wrist_pitch', 'wrist_pitch_v13'):
        return 'joint5'
    return 'joint3'


D2R = np.pi / 180.0

# Import custom calibrator logic
from core.marker_detection import Marker_Detection, Marker_Transform
from core.calibration.Calibrator import MarkerCalibrator, JointCalibrator, BaseCalibrator, HeadCameraCalibrator
from core.calibration.IntrinsicsCalibrator import IntrinsicsCalibrator
from core.wizard_widget import CalibrationWizardWidget
from core.config_store import Language, Paths, tr, get_asset_path
from core.homeoffset_core import (
    reset_current_pose_home_offsets,
    save_home_reset_baseline_json,
    move_to_offset_candidate_from_json,
    load_offset_from_json,
    move_robot_to_zero_pose
)

from core.calibration_core import (
    CapturedSample,
    FullAutoCalibrationService, OptimizerContext, run_calibration_optimizer,
    capture_calibration_sample, AutoCollectionService, CollectionState,
    calibrate_marker_bracket, build_capture_metadata, prepare_sample_dataset,
    prepare_full_auto_calibration, select_calibration_dataset, prepare_taught_ready_pose,
    capture_one_sample as capture_robot_sample,
    get_arm_config,
    get_both_arm_config,
    get_head_config,
    load_npz_dataset,
    save_npz_dataset,
    validate_dataset,
    check_calibration_state,
)
from core.calibration_optimizer import (
    DEFAULT_LAMBDA_CAM_POS,
    DEFAULT_LAMBDA_CAM_ROT,
    CalibrationOptimizer,
    QPCalibrationOptimizer,
    D2R,
)
from core.robot_motion import (
    AutoCollectionConfig,
    build_incremental_motion_plan,
    move_to_auto_ready_pose,
    execute_auto_motion_step,
    reset_motion_state,
)
# --- Configuration & Paths ---
from core.config_store import CONFIG_PATHS

UI_DROPDOWNS = {
    "robot_models": ["a", "m"],
    "arm_sides": ["Right Arm", "Left Arm"],
    "marker_axes": ["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"],
    "joint_modes_v13": ["wrist_roll_v13 (6-Axis Sweep)", "wrist_pitch_v13 (5-Axis Sweep)", "elbow (3-Axis Sweep)"],
    "joint_modes_v12": ["wrist_yaw2 (6-Axis Sweep)", "wrist_pitch (5-Axis Sweep)", "elbow (3-Axis Sweep)"]
}

# --- Premium Dark CSS Stylesheet (Matched with RBY1 Web UI Dark Theme) ---
DARK_STYLESHEET = """
QWidget {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Noto Sans CJK KR', 'Noto Sans', 'Segoe UI', 'Ubuntu', 'DejaVu Sans', sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #333333;
    border-radius: 8px;
    margin-top: 15px;
    font-weight: bold;
    font-size: 13px;
    color: #e0e0e0;
    background-color: #1a1a1a;
    padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 6px;
    background-color: #121212;
    color: #ffffff;
}
QPushButton {
    background-color: #2c3e50;
    color: #ffffff;
    border: 1px solid #111111;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #34495e;
    border: 1px solid #000000;
}
QPushButton:pressed {
    background-color: #1a252f;
    border: 1px solid #000000;
}
QPushButton:disabled {
    background-color: #262626;
    color: #616161;
    border: 1px solid #1f1f1f;
}
QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
    min-width: 120px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox:hover {
    border: 1px solid #666666;
}
QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
}
QLineEdit:focus {
    border: 1px solid #777777;
    background-color: #262626;
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
    color: #9e9e9e;
}
QTabBar::tab:selected {
    background: #262626;
    color: #ffffff;
    border-bottom: 2px solid #e0e0e0;
}
QTabBar::tab:hover:!selected {
    background: #2a2a2a;
    color: #ffffff;
}
QCheckBox {
    spacing: 8px;
    font-weight: bold;
    color: #e0e0e0;
}
QCheckBox:checked {
    color: #80d8ff;
    font-weight: bold;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border: 2px solid #616161;
    border-radius: 4px;
    background-color: #212121;
}
QCheckBox::indicator:hover {
    border: 2px solid #40c4ff;
}
QCheckBox::indicator:checked {
    background-color: #0091ea;
    border: 2px solid #40c4ff;
}
QCheckBox::indicator:checked:hover {
    background-color: #00b0ff;
    border: 2px solid #80d8ff;
}
QTextEdit {
    background-color: #121212;
    color: #f5f5f5;
    border: 1px solid #333333;
    border-radius: 6px;
    font-family: 'Noto Sans Mono CJK KR', 'Noto Sans Mono', 'DejaVu Sans Mono', 'Consolas', monospace;
    font-size: 11px;
}
QProgressBar {
    background-color: #2a2a2a;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #1e88e5;
    border-radius: 3px;
}
QScrollBar:vertical {
    background: #121212;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #3d3d3d;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #546e7a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QMessageBox {
    background-color: #1e1e1e;
    min-width: 520px;
}
QMessageBox QLabel {
    color: #ffffff;
    font-size: 13px;
    padding: 12px 8px;
    min-height: 85px;
}
QMessageBox QPushButton {
    min-width: 85px;
    padding: 6px 16px;
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

class PlotViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Plot Viewer")
        self.resize(950, 750)
        self.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout(self)
        
        # Navigation layout
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(10, 5, 10, 5)
        nav_layout.setSpacing(10)
        
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedSize(40, 30)
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background-color: #546e7a;
                color: #ffffff;
                border: 1px solid #78909c;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #78909c;
                border-color: #90a4ae;
            }
            QPushButton:pressed {
                background-color: #37474f;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #757575;
                border-color: #3d3d3d;
            }
        """)
        
        self.lbl_title = QLabel("No Plot Loaded")
        self.lbl_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("""
            background-color: #1e1e1e;
            color: #00e5ff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
        """)
        
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedSize(40, 30)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #1e88e5;
                color: #ffffff;
                border: 1px solid #42a5f5;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2196f3;
                border-color: #64b5f6;
            }
            QPushButton:pressed {
                background-color: #1565c0;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #757575;
                border-color: #3d3d3d;
            }
        """)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_title, 1)
        nav_layout.addWidget(self.btn_next)
        
        self.plot_label = QLabel("No plots to display")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setStyleSheet("background-color: #1a1a1a; color: #888888; border: 2px solid #2d2d2d; border-radius: 8px;")
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout.addLayout(nav_layout)
        layout.addWidget(self.plot_label)
        
        if parent:
            self.btn_prev.clicked.connect(parent.show_prev_plot)
            self.btn_next.clicked.connect(parent.show_next_plot)
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent() and hasattr(self.parent(), "display_current_plot"):
            self.parent().display_current_plot()

class ZeroPoseCheckDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zero Pose Check")
        self.resize(760, 800)
        self.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        msg = (
            "The robot has moved to zero pose.\n\n"
            "Please compare the actual robot posture with the reference image.\n\n"
            "- If the posture matches the reference, you can proceed with data collection.\n"
            "- If the posture does not match and the two target joints appear outside the recommended range,\n"
            "  use direct teaching to move the robot to the recommended posture,\n"
            "  perform reset first, and then start data collection."
        )
        msg_lbl = QLabel(msg)
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl)
        
        # Load warning pose check images
        for path_name in ["img/warning_pose_check.png", "img/warning_pose.png", "warning_pose_check.png", "warning_pose.png"]:
            img_path = get_asset_path(path_name)
            if os.path.exists(img_path):
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    lbl = QLabel()
                    lbl.setPixmap(pixmap.scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    lbl.setAlignment(Qt.AlignCenter)
                    layout.addWidget(lbl)
                    
        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        
        self.setLayout(layout)

class MarkerRecognitionProblemDialog(QDialog):
    def __init__(self, parent=None, is_ko=False):
        super().__init__(parent)
        self.is_ko = is_ko
        self.parent_app = parent
        self.setWindowTitle("마커 인식 오류 및 수동 위치 교시 가이드" if is_ko else "Marker Recognition Problem & Teaching Guidance")
        self.resize(1060, 760)
        self.setStyleSheet(DARK_STYLESHEET)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        
        # 1. Title Header
        lbl_title = QLabel("⚠️ 마커 미인식: 수동 위치 교시 및 카메라 시야 확보 안내" if is_ko else "⚠️ Marker Unrecognized: Manual Teaching & View Alignment")
        lbl_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffd700;")
        lbl_title.setWordWrap(True)
        lbl_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl_title)
        
        # 2. Body Layout: Left Guidance Box | Right Live Camera Feed
        body_layout = QHBoxLayout()
        body_layout.setSpacing(15)
        
        # --- Left Box: Guidance Images & Instructions ---
        left_box = QGroupBox("수동 위치 교시 절차" if is_ko else "Manual Teaching Steps")
        left_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 15px; }")
        left_layout = QVBoxLayout(left_box)
        left_layout.setSpacing(10)
        
        img_row = QHBoxLayout()
        img_row.setSpacing(8)
        
        imgs_info = [
            ("img/marker_problem_before.png", "1. 미인식 발생" if is_ko else "1. Unrecognized"),
            ("img/marker_problem_teaching.png", "2. 교시 버튼 누름" if is_ko else "2. Press Teaching"),
            ("img/marker_problem_after.png", "3. 마커 정면 조향" if is_ko else "3. Align Marker")
        ]
        
        for img_path, cap in imgs_info:
            v_box = QVBoxLayout()
            lbl_img = QLabel()
            pix = QPixmap(get_asset_path(img_path))
            if not pix.isNull():
                lbl_img.setPixmap(pix.scaled(150, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                lbl_img.setText(f"[{img_path}]")
            lbl_img.setAlignment(Qt.AlignCenter)
            
            lbl_cap = QLabel(cap)
            lbl_cap.setAlignment(Qt.AlignCenter)
            lbl_cap.setStyleSheet("font-size: 13px; font-weight: bold; color: #43a047;")
            
            v_box.addWidget(lbl_img)
            v_box.addWidget(lbl_cap)
            img_row.addLayout(v_box)
            
        left_layout.addLayout(img_row)
        
        desc_text = (
            "<b>1. 티칭 버튼 누르기:</b> 로봇 팔/헤드의 직접 교시 버튼(Teaching Button)을 누르세요.<br><br>"
            "<b>2. 마커 방향 조정:</b> 우측 실시간 카메라 화면을 보면서 마커가 화각 중앙에 정면으로 오도록 이동시키세요.<br><br>"
            "<b>3. 캘리브레이션 재개:</b> 정렬을 마치면 아래 [티칭 완료] 버튼을 누르세요.<br>"
            "<span style='color: #ffca28; font-size: 13px;'>* 조정한 현재 위치에서 보정 대상 관절만 0°로 정렬 후 스위프를 계속 진행합니다.</span>"
            if is_ko else
            "<b>1. Press Teaching Button:</b> Press the direct teaching button on the robot arm/head.<br><br>"
            "<b>2. Align Marker:</b> Watching the live camera feed on the right, position the marker in the center of the camera view.<br><br>"
            "<b>3. Resume Calibration:</b> Click [Teaching Done] below once aligned.<br>"
            "<span style='color: #ffca28; font-size: 13px;'>* Resumes calibration by aligning only the target calibration joint to 0° from adjusted posture.</span>"
        )
        lbl_desc = QLabel(desc_text)
        lbl_desc.setStyleSheet("font-size: 14px; color: #ffffff; line-height: 1.5;")
        lbl_desc.setWordWrap(True)
        left_layout.addWidget(lbl_desc)
        body_layout.addWidget(left_box, stretch=5)
        
        # --- Right Box: Embedded Real-time Live Feed ---
        right_box = QGroupBox("실시간 카메라 피드 (Live Feed)" if is_ko else "Live Camera Feed")
        right_box.setStyleSheet("QGroupBox::title { color: #43a047; font-weight: bold; font-size: 15px; }")
        right_layout = QVBoxLayout(right_box)
        right_layout.setSpacing(10)
        
        self.lbl_live_feed = QLabel("Camera Feed Loading..." if is_ko else "Camera Feed Loading...")
        self.lbl_live_feed.setAlignment(Qt.AlignCenter)
        self.lbl_live_feed.setStyleSheet("background-color: #000000; color: #888888; border: 1px solid #444444; border-radius: 4px;")
        self.lbl_live_feed.setMinimumSize(420, 320)
        right_layout.addWidget(self.lbl_live_feed)
        
        body_layout.addWidget(right_box, stretch=4)
        main_layout.addLayout(body_layout)
        
        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        btn_done = QPushButton("✅ 티칭 완료 (캘리브레이션 재개)" if is_ko else "✅ Teaching Done (Resume Calibration)")
        btn_done.setMinimumHeight(52)
        btn_done.setStyleSheet("background-color: #43a047; color: #ffffff; font-size: 16px; font-weight: bold; border-radius: 6px;")
        btn_done.clicked.connect(self.accept)
        
        btn_cancel = QPushButton("❌ 캘리브레이션 취소" if is_ko else "❌ Cancel Calibration")
        btn_cancel.setMinimumHeight(52)
        btn_cancel.setStyleSheet("background-color: #e53935; color: #ffffff; font-size: 16px; font-weight: bold; border-radius: 6px;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_done, stretch=2)
        btn_layout.addWidget(btn_cancel, stretch=1)
        main_layout.addLayout(btn_layout)

class ApplyHomeOffsetDialog(QDialog):
    def __init__(self, parent, result_path, baseline_path, arm, include_head, compare_summary=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Home Offset")
        self.resize(900, 600)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.parent_app = parent
        self.result_path = result_path
        self.baseline_path = baseline_path
        self.arm = arm
        self.include_head = include_head
        self._drag_pos = None

        self.current_apply_arm = parent.infer_home_offset_apply_arm(arm, result_path)
        
        if compare_summary is None:
            compare_summary = parent.format_home_offset_compare_summary(result_path, baseline_path)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        msg = (
            "Compare the original baseline zero and the optimized zero before applying.\n\n"
            "1. Select Baseline or Optimized state.\n"
            "2. Move to Zero to inspect the zero pose before calibration reset.\n"
            "3. Move to Check Position to move the robot to the custom check pose.\n"
            "4. Apply the pose you want to keep using Rollback or Apply Optimized Result.\n\n"
            "Make sure the workspace is clear before each move."
        )
        msg_lbl = QLabel(msg)
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl)
        
        # Summary text box
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setText(compare_summary)
        self.summary_box.setFont(QFont("Consolas", 10))
        layout.addWidget(self.summary_box, 1)
        
        # State Switcher (Big Toggle Buttons)
        state_layout = QHBoxLayout()
        self.btn_group = QButtonGroup(self)
        
        self.btn_baseline = QPushButton("BASELINE\n(Rollback)")
        self.btn_opt = QPushButton("OPTIMIZED\n(Apply)")
        
        btn_style = """
        QPushButton {
            font-size: 16px;
            font-weight: bold;
            color: #b0bec5;
            background-color: #2a2a2a;
            border: 2px solid #3d3d3d;
            border-radius: 8px;
        }
        QPushButton:checked {
            color: #ffffff;
            background-color: #1e88e5;
            border: 3px solid #448aff;
        }
        QPushButton:disabled {
            background-color: #222222;
            color: #555555;
            border: 2px solid #333333;
        }
        """
        
        for btn in [self.btn_baseline, self.btn_opt]:
            btn.setCheckable(True)
            btn.setMinimumHeight(60)
            btn.setStyleSheet(btn_style)
            self.btn_group.addButton(btn)
            state_layout.addWidget(btn)
            
        self.btn_baseline.setChecked(True)
        
        if result_path is None or not os.path.exists(result_path):
            self.btn_opt.setEnabled(False)
            
        layout.addLayout(state_layout)
        
        # Movement Buttons
        move_layout = QHBoxLayout()
        self.btn_move_zero = QPushButton("Move to Zero")
        self.btn_move_zero.clicked.connect(self.on_move_zero)
        move_layout.addWidget(self.btn_move_zero)
        
        self.btn_move_check = QPushButton("Move to Check")
        self.btn_move_check.clicked.connect(self.on_move_check)
        move_layout.addWidget(self.btn_move_check)
        layout.addLayout(move_layout)
        
        # Action buttons row
        btn_layout = QHBoxLayout()
        
        self.btn_apply = QPushButton("Apply Selected Offset")
        self.btn_apply.setStyleSheet("background-color: #d84315; color: white; font-weight: bold; font-size: 16px; padding: 10px;")
        self.btn_apply.clicked.connect(self.on_apply_selected)
        
        if result_path is None or not os.path.exists(result_path):
            self.btn_apply.setEnabled(False)
            
        btn_layout.addWidget(self.btn_apply)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self._drag_pos = pos - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, '_drag_pos') and self._drag_pos is not None:
            pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self.move(pos - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)
        
    def on_apply_selected(self):
        state, path = self.get_current_target()
        
        confirm = QMessageBox.question(
            self, 
            "Confirm Apply", 
            f"Are you sure you want to apply the '{state.upper()}' offsets?\n\n"
            f"The robot will first move to the Zero Pose of '{state.upper()}', and then reset/apply the home offset.\n"
            f"Please ensure the workspace around the robot is clear.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            self.on_apply(state)

    def get_current_target(self):
        if self.btn_baseline.isChecked():
            return "baseline", self.baseline_path
        else:
            return "optimized", self.result_path

    def on_move_zero(self):
        state, path = self.get_current_target()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"No {state} JSON found.")
            return
            
        self.set_buttons_enabled(False)
        self.worker = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_zero",
            json_path=path,
            label=f"{state.capitalize()} Zero",
            arm=self.arm,
            include_head=self.include_head
        )
        self.worker.log_signal.connect(self.parent_app.log_msg)
        def on_finished(success, error_msg, res):
            self.set_buttons_enabled(True)
            if success:
                self.current_apply_arm = res["arm"]
                QMessageBox.information(self, "Preview Complete", f"Moved to {state} zero candidate.")
            else:
                QMessageBox.critical(self, "Preview Error", error_msg)
        self.worker.finished_signal.connect(on_finished)
        self.worker.start()

    def on_move_check(self):
        state, path = self.get_current_target()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"No {state} JSON found.")
            return

        self.set_buttons_enabled(False)
        self.worker = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_check",
            json_path=path,
            label=f"{state.capitalize()} Check Position",
            arm=self.arm,
            include_head=self.include_head
        )
        self.worker.log_signal.connect(self.parent_app.log_msg)
        def on_finished(success, error_msg, res):
            self.set_buttons_enabled(True)
            if success:
                self.current_apply_arm = res["arm"]
                QMessageBox.information(self, "Preview Complete", f"Moved to {state} check position candidate.")
            else:
                QMessageBox.critical(self, "Check Preview Error", error_msg)
        self.worker.finished_signal.connect(on_finished)
        self.worker.start()

    def on_apply(self, state):
        state, path = self.get_current_target()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"No {state} JSON found.")
            return

        self.set_buttons_enabled(False)
        self.parent_app.log_msg(f"[INFO] Moving robot to '{state.upper()}' Zero Pose before applying home offset...")
        
        # Step 1: Move to Zero Pose first
        self.worker_move = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_zero",
            json_path=path,
            label=f"{state.capitalize()} Zero",
            arm=self.arm,
            include_head=self.include_head
        )
        self.worker_move.log_signal.connect(self.parent_app.log_msg)

        def on_move_finished(success, error_msg, res):
            if not success:
                self.set_buttons_enabled(True)
                QMessageBox.critical(self, "Zero Pose Move Error", f"Failed to move to zero pose before applying: {error_msg}")
                return
            
            if "arm" in res:
                self.current_apply_arm = res["arm"]

            self.parent_app.log_msg(f"[INFO] Arrived at '{state.upper()}' Zero Pose. Now resetting and applying home offset...")

            # Step 2: Apply Home Offset from the Zero Pose
            self.worker_apply = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "apply",
                arm=self.current_apply_arm,
                include_head=self.include_head,
                json_path=self.result_path if state == "optimized" else None
            )
            self.worker_apply.log_signal.connect(self.parent_app.log_msg)

            def on_apply_finished(app_success, app_error_msg, app_res):
                self.set_buttons_enabled(True)
                if app_success:
                    if app_res.get("needs_reconnect", False):
                        self.parent_app.log_msg("Re-connecting and initializing robot...")
                        if self.parent_app.robot:
                            self.parent_app.connect_robot()
                            from PySide6.QtWidgets import QApplication
                            QApplication.processEvents()
                        self.parent_app.connect_robot()
                        self.parent_app.log_msg("Current pose home offset apply complete.")
                        
                    if app_res.get("success", False) or app_res.get("needs_reconnect", False):
                        # Reset software joint offsets to 0.0 for the applied arm(s) since they are now physically absorbed
                        for arm in ["left", "right"]:
                            if self.current_apply_arm == "both" or self.current_apply_arm == arm:
                                self.parent_app.joint_offsets_store[arm]["joint3"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint5"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint6"] = 0.0
                                
                                self.parent_app.joint_offsets[arm]["wrist_pitch"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_roll"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_yaw2"] = 0.0
                                self.parent_app.joint_offsets[arm]["elbow"] = 0.0

                        # Save zeroed offsets to setting.yaml and update GUI
                        offsets_saved = self.parent_app.save_offsets_to_yaml()
                        if not offsets_saved or app_res.get('camera_save_success') is False:
                            QMessageBox.warning(self, "Settings Save Failed",
                                "Robot home offsets were already changed, but settings were not fully saved. "
                                "Do not repeat the physical reset; resolve the save error before continuing.")
                        self.parent_app.update_applied_offset_label()

                        # Zero out baseline json if it exists to prevent accidental unsafe rollback later
                        if self.baseline_path and os.path.exists(self.baseline_path):
                            try:
                                import json
                                with open(self.baseline_path, "r") as f:
                                    data = json.load(f)
                                
                                if "right_arm_joint_offset_deg" in data and (self.current_apply_arm == "both" or self.current_apply_arm == "right"):
                                    data["right_arm_joint_offset_deg"] = [0.0] * len(data["right_arm_joint_offset_deg"])
                                if "left_arm_joint_offset_deg" in data and (self.current_apply_arm == "both" or self.current_apply_arm == "left"):
                                    data["left_arm_joint_offset_deg"] = [0.0] * len(data["left_arm_joint_offset_deg"])
                                if "head_joint_offset_deg" in data and data["head_joint_offset_deg"] is not None and self.include_head:
                                    data["head_joint_offset_deg"] = [0.0] * len(data["head_joint_offset_deg"])
                                
                                if "right_arm_joint_offset_deg" in data and "left_arm_joint_offset_deg" in data:
                                    data["joint_offset_deg"] = data["right_arm_joint_offset_deg"] + data["left_arm_joint_offset_deg"]
                                elif "joint_offset_deg" in data:
                                    data["joint_offset_deg"] = [0.0] * len(data["joint_offset_deg"])
                                    
                                with open(self.baseline_path, "w") as f:
                                    json.dump(data, f, indent=4)
                                self.parent_app.log_msg(f"[INFO] Zeroed out applied arm offsets in baseline json: {self.baseline_path}")
                            except Exception as e:
                                self.parent_app.log_msg(f"[WARN] Failed to zero out baseline json: {e}")

                        QMessageBox.information(self, "Success", f"Robot moved to Zero Pose and '{state.upper()}' home offset applied successfully. Software joint offsets have been reset to 0.0.")
                        self.accept()
                    else:
                        QMessageBox.warning(self, "Warning", "Home offset apply finished, but some joints failed to reset. Please check the logs.")
                else:
                    QMessageBox.critical(self, "Apply Pose Error", app_error_msg)

            self.worker_apply.finished_signal.connect(on_apply_finished)
            self.worker_apply.start()

        self.worker_move.finished_signal.connect(on_move_finished)
        self.worker_move.start()

    def set_buttons_enabled(self, enabled):
        self.btn_move_zero.setEnabled(enabled)
        self.btn_move_check.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        if self.result_path is not None and os.path.exists(self.result_path):
            self.btn_opt.setEnabled(enabled)
        self.btn_baseline.setEnabled(enabled)

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                                 QLabel, QStackedWidget, QGroupBox, QCheckBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap

class CheckCalibrationStateDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Check Calibration State")
        self.resize(450, 300)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.parent_app = parent
        self.check_state_moved = False
        self._worker = None  # QThread 참조 유지 (GC 방지)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        grid = QGridLayout()
        grid.addWidget(QLabel("X Position (m):"), 0, 0)
        self.x_input = QLineEdit("0.35")
        self.x_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        grid.addWidget(self.x_input, 0, 1)
        
        grid.addWidget(QLabel("Y Position (m):"), 1, 0)
        self.y_input = QLineEdit("0.0")
        self.y_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        grid.addWidget(self.y_input, 1, 1)
        
        grid.addWidget(QLabel("Z Position (m):"), 2, 0)
        self.z_input = QLineEdit("0.0")
        self.z_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        grid.addWidget(self.z_input, 2, 1)
        
        grid.addWidget(QLabel("Y Offset (m):"), 3, 0)
        self.offset_input = QLineEdit("0.175")
        self.offset_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        grid.addWidget(self.offset_input, 3, 1)
        
        layout.addLayout(grid)
        
        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: #2979ff; font-weight: bold;")
        layout.addWidget(self.lbl_status)
        
        btn_layout = QHBoxLayout()
        self.btn_move = QPushButton("Move")
        self.btn_move.clicked.connect(self.on_move)
        btn_layout.addWidget(self.btn_move)
        
        self.btn_draw = QPushButton("Draw Square")
        self.btn_draw.clicked.connect(self.on_draw_square)
        btn_layout.addWidget(self.btn_draw)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _set_buttons_enabled(self, enabled):
        self.btn_move.setEnabled(enabled)
        self.btn_draw.setEnabled(enabled)
        
    def on_move(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            offset = float(self.offset_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid floating-point numbers.")
            return

        if not self.parent_app.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return

        self.lbl_status.setText("Status: Moving...")
        self.lbl_status.setStyleSheet("color: #ff9800;")
        self._set_buttons_enabled(False)

        # threading.Thread 대신 QThread(CheckCalibrationStateWorker) 사용
        # rby1_sdk C++ 라이브러리는 일반 Python 스레드와 호환되지 않아 segfault 발생
        worker = CheckCalibrationStateWorker(
            task_type="move",
            robot=self.parent_app.robot,
            model_name=self.parent_app.model_input.currentText().strip(),
            active_arms=["right", "left"],
            data=[x, y, z],
            offset=offset,
            skip_ready=self.check_state_moved,
        )
        worker.log_signal.connect(lambda msg: self.parent_app.log_msg(f"[Check State] {msg}"))

        def on_move_finished(success, error_msg):
            self._set_buttons_enabled(True)
            self._worker = None
            if success:
                self.check_state_moved = True
                self.parent_app.log_msg("[Check State] Symmetrical move completed successfully.")
                self.lbl_status.setText("Status: Move OK")
                self.lbl_status.setStyleSheet("color: #00e676;")
            else:
                self.parent_app.log_msg(f"[Check State Error] {error_msg}")
                self.lbl_status.setText("Status: Error")
                self.lbl_status.setStyleSheet("color: #ff1744;")

        worker.finished_signal.connect(on_move_finished)
        self._worker = worker
        worker.start()

    def on_draw_square(self):
        if not self.check_state_moved:
            QMessageBox.warning(self, "Error", "Please click 'Move' first to reach the initial check state.")
            return

        if not self.parent_app.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return

        try:
            offset = float(self.offset_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid floating-point numbers for Y Offset.")
            return

        self.lbl_status.setText("Status: Drawing...")
        self.lbl_status.setStyleSheet("color: #ff9800;")
        self._set_buttons_enabled(False)

        # threading.Thread 대신 QThread(CheckCalibrationStateWorker) 사용
        worker = CheckCalibrationStateWorker(
            task_type="draw_square",
            robot=self.parent_app.robot,
            model_name=self.parent_app.model_input.currentText().strip(),
            active_arms=["right", "left"],
            data=[0.35, 0.0, 0.0],  # 기준 포지션 (draw_square 내부에서 포인트 순회)
            offset=offset,
            skip_ready=True,
        )
        worker.log_signal.connect(lambda msg: self.parent_app.log_msg(f"[Draw Square] {msg}"))

        def on_draw_finished(success, error_msg):
            self._set_buttons_enabled(True)
            self._worker = None
            if success:
                self.parent_app.log_msg("[Draw Square] Square drawing sequence completed successfully.")
                self.lbl_status.setText("Status: Draw OK")
                self.lbl_status.setStyleSheet("color: #00e676;")
            else:
                self.parent_app.log_msg(f"[Draw Square Error] {error_msg}")
                self.lbl_status.setText("Status: Draw Error")
                self.lbl_status.setStyleSheet("color: #ff1744;")

        worker.finished_signal.connect(on_draw_finished)
        self._worker = worker
        worker.start()

# --- Common Worker Threads ---

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

class HeadCamReadyWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, calibrator, stop_event=None, parent=None):
        super().__init__(parent)
        self.calibrator = calibrator
        self.stop_event = stop_event

    def run(self):
        try:
            ok = self.calibrator.perform_move_to_ready_pose(
                arm_side="both",
                log_callback=self.log_signal.emit,
                stop_event=self.stop_event
            )
            self.finished_signal.emit(ok, "" if ok else "Failed to reach Head & Camera Ready Pose")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

Step1_5ReadyWorker = HeadCamReadyWorker

class HeadCamSweepWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, dict)

    def __init__(self, calibrator, pan_range=15.0, tilt_range=10.0, num_steps=11, stop_event=None, parent=None):
        super().__init__(parent)
        self.calibrator = calibrator
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.num_steps = num_steps
        self.stop_event = stop_event

    def run(self):
        try:
            results = self.calibrator.perform_head_sweep(
                arm_side="auto",
                pan_range_deg=self.pan_range,
                tilt_range_deg=self.tilt_range,
                num_steps=self.num_steps,
                step_delay=0.6,
                log_callback=self.log_signal.emit,
                stop_event=self.stop_event
            )
            if results and results.get("success"):
                self.finished_signal.emit(True, results)
            else:
                self.finished_signal.emit(False, {})
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Head sweep failed: {e}")
            self.finished_signal.emit(False, {"error": str(e)})

Step1_5HeadSweepWorker = HeadCamSweepWorker

class Step2InitPoseWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)
    head_pose_signal = Signal(object)

    def __init__(self, robot, active_arms, priority, include_head_motion=True, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.active_arms = list(active_arms)
        self.priority = priority
        self.include_head_motion = include_head_motion
        self.robot_version = parent.get_robot_version() if parent is not None else None
        self.marker_transform = getattr(parent, 'marker_st', None)
        model = getattr(parent, 'model', None)
        self.head_idx = get_head_config(model)['head_idx'] if model is not None else None
        self.teaching_callback = getattr(parent, 'prompt_marker_problem_teaching', None)
        if parent is not None:
            self.head_pose_signal.connect(parent.on_capture_head_centered, Qt.QueuedConnection)

    def run(self):
        try:
            from core.robot_motion import prepare_capture_pose
            head_pose = prepare_capture_pose(
                self.robot, self.active_arms, self.priority,
                include_head_motion=self.include_head_motion, robot_version=self.robot_version,
                marker_transform=self.marker_transform, head_idx=self.head_idx,
                teaching_callback=self.teaching_callback, log_callback=self.log_signal.emit)
            if head_pose is not None:
                self.head_pose_signal.emit(head_pose)
            self.finished_signal.emit(True, "")
        except Exception as error:
            self.finished_signal.emit(False, str(error))

class Step2AutoMotionWorker(QThread):
    log_signal = Signal(str)
    sample_signal = Signal(int)
    state_signal = Signal(object)
    captured_signal = Signal(object)
    finished_signal = Signal(bool, str)

    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        self.service.log = self.log_signal.emit
        self.service.progress = self._progress
        self.service.sample_callback = self.captured_signal.emit

    def _progress(self, state):
        self.state_signal.emit(state)
        self.sample_signal.emit(state.pose_index)

    def run(self):
        try:
            completed = self.service.run()
            self.finished_signal.emit(completed, '' if completed else 'Auto Motion stopped by user.')
        except Exception as error:
            self.finished_signal.emit(False, str(error))

class Step2ZeroPoseCheckWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, robot, model, arm, include_head):
        super().__init__()
        self.robot = robot
        self.model = model
        self.arm = arm
        self.include_head = include_head

    def run(self):
        try:
            from core.homeoffset_core import move_connected_robot_to_zero
            self.log_signal.emit("Moving robot to zero pose...")
            move_connected_robot_to_zero(self.robot, self.model, self.include_head)
            self.finished_signal.emit(True, '')
        except Exception as error:
            self.finished_signal.emit(False, str(error))

class Step2ApplyHomeOffsetWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str, dict)

    def __init__(self, app, task_type, **kwargs):
        super().__init__()
        app.ensure_home_offset_robot()
        self.robot = app.robot
        self.model = app.model
        self.task_type = task_type
        self.kwargs = dict(kwargs)
        self.setting_path = CONFIG_PATHS['setting_yaml']

    def run(self):
        try:
            from core.homeoffset_core import run_home_offset_operation
            result = run_home_offset_operation(self.task_type, self.robot, self.model,
                **self.kwargs, setting_path=self.setting_path, log_callback=self.log_signal.emit)
            self.finished_signal.emit(True, '', result)
        except Exception as error:
            self.finished_signal.emit(False, str(error), {})

class CheckCalibrationStateWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, task_type, robot, model_name, active_arms, data, offset, skip_ready=False):
        super().__init__()
        self.task_type = task_type
        self.robot = robot
        self.model_name = model_name
        self.active_arms = active_arms
        self.data = data
        self.offset = offset
        self.skip_ready = skip_ready

    def run(self):
        try:
            from core.robot_motion import check_calibration_state, draw_calibration_square
            if self.task_type == "move":
                check_calibration_state(self.robot, self.model_name, self.active_arms,
                    self.data, self.offset, log_cb=self.log_signal.emit, skip_ready=self.skip_ready)
            elif self.task_type == "draw_square":
                draw_calibration_square(self.robot, self.active_arms, self.offset,
                                        log_callback=self.log_signal.emit)
            self.finished_signal.emit(True, "")
        except Exception as error:
            self.finished_signal.emit(False, str(error))

class Step2CalculateWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, app, active_arms, optimize_head, optimize_camera, q_arm_list, q_head_list, T_meas_list, result_path, lambda_cam_pos, lambda_cam_rot):
        super().__init__()
        self.context = app.optimizer_context()
        self.active_arms = active_arms
        self.optimize_head = optimize_head
        self.optimize_camera = optimize_camera
        self.q_arm_list = q_arm_list
        self.q_head_list = q_head_list
        self.T_meas_list = T_meas_list
        self.result_path = result_path
        self.lambda_cam_pos = lambda_cam_pos
        self.lambda_cam_rot = lambda_cam_rot

    def run(self):
        try:
            self.result = run_calibration_optimizer(
                self.context,
                active_arms=self.active_arms,
                optimize_head=self.optimize_head,
                optimize_camera=self.optimize_camera,
                q_arm_list=self.q_arm_list,
                q_head_list=self.q_head_list,
                T_meas_list=self.T_meas_list,
                result_path=self.result_path,
                lambda_cam_pos=self.lambda_cam_pos,
                lambda_cam_rot=self.lambda_cam_rot,
                solver_type="QP Solver",
                use_sag=False,
                log_callback=self.log_signal.emit,
            )
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class FullAutoReadyWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, joint_calibrator, marker_calibrator):
        super().__init__()
        self.joint_calibrator = joint_calibrator
        self.marker_calibrator = marker_calibrator
        self.error_msg = None

    def run(self):
        self.error_msg = prepare_full_auto_calibration(
            self.joint_calibrator, self.marker_calibrator, self.log_signal.emit)
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
        result = calibrate_marker_bracket(
            self.calibrator, self.arm_side, self.use_head_tracking, self.tolerance,
            self.save_debug, self.log_signal.emit, self.status_signal.emit)
        self.finished_signal.emit(result)

class JointCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)

    def __init__(self, calibrator, arm_side, mode, current_offset_deg=0.0, sweep_duration=None, save_debug=False):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode
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


class FullAutoWorker(QThread):
    log_msg = Signal(str)
    status_signal = Signal(bool)
    bracket_finished_signal = Signal(dict)
    joint_finished_signal = Signal(dict)
    finished_signal = Signal()

    def __init__(self, joint_calibrator, marker_calibrator, stop_event=None, joint_offsets_store=None, save_debug=False,
                 reset_initial_state=False, head_camera_calibrator=None):
        super().__init__()
        self.service = FullAutoCalibrationService(
            joint_calibrator, marker_calibrator, stop_event, joint_offsets_store, save_debug,
            log_callback=self.log_msg.emit, status_callback=self.status_signal.emit,
            bracket_callback=self.bracket_finished_signal.emit,
            joint_callback=self.joint_finished_signal.emit,
            reset_initial_state=reset_initial_state, head_camera_calibrator=head_camera_calibrator)
        self.joint_calibrator = joint_calibrator
        self.marker_calibrator = marker_calibrator
        self.stop_event = self.service.stop_event
        self.joint_offsets_store = self.service.joint_offsets_store

    @property
    def error_msg(self):
        return self.service.error_msg

    @property
    def arm_convergence(self):
        return self.service.arm_convergence

    def run(self):
        try:
            self.service.run()
        finally:
            self.finished_signal.emit()


# --- Unified Calibration App ---
class UnifiedCalibrationApp(QWidget):
    log_signal_safe = Signal(str)
    update_ui_signal_safe = Signal(str)
    marker_problem_signal = Signal(str, object, object)

    def __init__(self, marker_st=None, robot=None, arm_side="right", sim=False):
        super().__init__()
        self.is_ko_ui = False
        self.log_signal_safe.connect(self._log_msg_slot)
        self.update_ui_signal_safe.connect(self._update_ui_slot)
        self.marker_problem_signal.connect(self._on_marker_problem_requested)

        marker_st = marker_st if marker_st is not None else Marker_Transform(sim=sim, robot=robot)
        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        marker_st.set_marker_type('plate')
        marker_st.bind_robot(robot, marker_st.robot_version)
        
        # Core Calibrator Instances
        self.marker_calibrator = MarkerCalibrator(marker_st, robot)
        self.joint_calibrator = JointCalibrator(marker_st, robot)
        self.head_camera_calibrator = HeadCameraCalibrator(marker_st, robot)
        self.marker_calibrator.app = self
        self.joint_calibrator.app = self
        self.head_camera_calibrator.app = self
        self.robot_version = marker_st.robot_version
        for calibrator in (self.marker_calibrator, self.joint_calibrator, self.head_camera_calibrator):
            calibrator.robot_version = self.robot_version
        
        # Intrinsics Calibrator (Tab 3 용)
        self.intrinsics_calibrator = IntrinsicsCalibrator()
        self.intrinsics_calibrator.marker_st = self.marker_st
        # Default: 8x5 squares, 36mm x 27mm, DICT_5X5_100
        self.intrinsics_calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 36.0, 27.0, "DICT_5X5_100")
        
        try:
            
            self.marker_detector = self.marker_st.marker_detection
            self.marker_detector.set_marker_type("plate")
        except ImportError:
            self.marker_detector = None
            
        self.monitor_enabled = False
        self.captured_images = []
        self.current_guide_idx = 0
        self.output_yaml = CONFIG_PATHS["camera_intrinsics"]
        
        # Saved Calibration Results
        self.marker_data_4 = None
        self.marker_data_5 = None
        self.marker_data_6 = None
        self.joint_sweep_data = None
        self.generated_plots = []
        self.current_plot_idx = -1
        
        # Cumulative Joint Offsets for iterative sweeps
        self.joint_offsets = {
            "left": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
            "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
        }
        self.wrist_roll_calibrated = {"right": False, "left": False}
        self.ready_done_joint = False
        self.ready_done_marker = False
        self.load_offsets_from_yaml()
        
        # Step 2 calibration state
        self.apply_joint_offset_flag = False
        self.include_head_motion = True
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
        self.auto_motion_thread = None
        
        self.last_result_path = None
        self.last_home_reset_path = None
        self.last_dataset_path = None
        self.dataset_saved_in_session = False
        self.current_session_dataset_path = None
        
        self.check_state_moved = False
        
        self.model = None
        self.dyn_model = None
        if self.robot and hasattr(self.robot, "model") and hasattr(self.robot, "get_dynamics"):
            try:
                self.model = self.robot.model()
                self.dyn_model = self.robot.get_dynamics()
            except Exception:
                self.model = None
                self.dyn_model = None

        if self.model is None or self.dyn_model is None:
            if hasattr(self, 'joint_calibrator') and getattr(self.joint_calibrator, 'robot', None) is not None:
                try:
                    self.model = self.joint_calibrator.robot.model()
                    self.dyn_model = self.joint_calibrator.robot.get_dynamics()
                except Exception:
                    pass

        self.recommended_joint_offset = None
        
        self.marker_calibrator.joint_offsets = self.joint_offsets
        self.joint_calibrator.joint_offsets = self.joint_offsets
        if self.robot:
            try:
                self.robot.joint_offsets = self.joint_offsets
            except AttributeError:
                pass
        
        self.setWindowTitle("Unified Robot Calibration Suite")
        self.resize(1050, 620)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # 1. 200ms poll timer (탭 1, 2, 4 용)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera_status)
        
        # 2. 33ms video timer (탭 3용)
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        
        # 3. Dedicated Temperature Monitor Timer (runs continuously every 2 seconds)
        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.poll_camera_temperature)
        self.temp_timer.start(2000)
        
        self.init_ui()
        self.load_bracket_design_values()
        self.update_applied_offset_label()
        
        # 초기화 시 탭 상태에 맞춰 타이머 활성화
        self.on_left_tab_changed(self.left_tabs.currentIndex())
        
        self.active_worker = None
        
        if self.marker_st and getattr(self.marker_st, 'intrinsics_mismatch', False):
            calib_device = getattr(self.marker_st, 'calib_device_name', '')
            connected_model = getattr(self.marker_st, 'camera_model', '')
            QMessageBox.warning(
                self,
                tr("dialogs.camera_intrinsics_warning.title"),
                tr("dialogs.camera_intrinsics_warning.text").format(connected_model=connected_model, calib_device=calib_device) + "\n\n" +
                tr("dialogs.camera_intrinsics_warning.informative_text")
            )
        
        self.init_camera_exposure_state()

    def init_camera_exposure_state(self):
        # Default policy: Always start with Auto Exposure enabled
        self.camera_auto_exposure = True
        self.camera_exposure_value = 6000
        self.applied_camera_auto_exposure = True
        self.applied_camera_exposure_value = 6000
        self.saved_camera_auto_exposure = True
        self.saved_camera_exposure_value = 6000

        if self.marker_st is not None and hasattr(self.marker_st, 'set_camera_exposure'):
            self.marker_st.set_camera_exposure(6000, auto_exposure=True)
            self.log_msg("[Camera] Initialized camera exposure to AUTO mode.")

    def set_camera_auto_mode(self):
        self.applied_camera_auto_exposure = True
        if hasattr(self, 'chk_auto_exposure'):
            self.chk_auto_exposure.blockSignals(True)
            self.chk_auto_exposure.setChecked(True)
            self.chk_auto_exposure.blockSignals(False)
        if hasattr(self, 'spin_exposure'):
            self.spin_exposure.setEnabled(False)
        if hasattr(self, 'slider_exposure'):
            self.slider_exposure.setEnabled(False)
            
        if self.marker_st is not None and hasattr(self.marker_st, 'set_camera_exposure'):
            self.marker_st.set_camera_exposure(6000, auto_exposure=True)
        self.log_msg("[Camera] Switched to AUTO exposure mode.")

    def on_auto_exposure_toggled(self, checked):
        if hasattr(self, 'spin_exposure'):
            self.spin_exposure.setEnabled(not checked)
        if hasattr(self, 'slider_exposure'):
            self.slider_exposure.setEnabled(not checked)
        if checked:
            self.set_camera_auto_mode()

    def on_exposure_value_changed(self, value):
        if hasattr(self, 'lbl_exposure_ms'):
            self.lbl_exposure_ms.setText(f"{value / 1000.0:.1f} ms")
        if hasattr(self, 'slider_exposure') and self.slider_exposure.value() != value:
            self.slider_exposure.blockSignals(True)
            self.slider_exposure.setValue(value)
            self.slider_exposure.blockSignals(False)
        if hasattr(self, 'spin_exposure') and self.spin_exposure.value() != value:
            self.spin_exposure.blockSignals(True)
            self.spin_exposure.setValue(value)
            self.spin_exposure.blockSignals(False)

    def apply_camera_exposure(self):
        auto_mode = self.chk_auto_exposure.isChecked() if hasattr(self, 'chk_auto_exposure') else True
        exp_val = self.spin_exposure.value() if hasattr(self, 'spin_exposure') else 6000
        self.applied_camera_auto_exposure = auto_mode
        self.applied_camera_exposure_value = exp_val
        self.saved_camera_auto_exposure = auto_mode
        self.saved_camera_exposure_value = exp_val
        
        if self.marker_st is not None and hasattr(self.marker_st, 'set_camera_exposure'):
            self.marker_st.set_camera_exposure(exp_val, auto_exposure=auto_mode)
            
        if auto_mode:
            self.log_msg("[Camera] Applied AUTO exposure setting.")
        else:
            self.log_msg(f"[Camera] Applied manual exposure: {exp_val} μs ({exp_val/1000.0:.1f} ms)")

    def cancel_camera_exposure(self):
        # Restore previous applied/saved state
        self.applied_camera_auto_exposure = self.saved_camera_auto_exposure
        self.applied_camera_exposure_value = self.saved_camera_exposure_value
        
        if hasattr(self, 'chk_auto_exposure'):
            self.chk_auto_exposure.blockSignals(True)
            self.chk_auto_exposure.setChecked(self.saved_camera_auto_exposure)
            self.chk_auto_exposure.blockSignals(False)
        if hasattr(self, 'spin_exposure'):
            self.spin_exposure.blockSignals(True)
            self.spin_exposure.setValue(self.saved_camera_exposure_value)
            self.spin_exposure.setEnabled(not self.saved_camera_auto_exposure)
            self.spin_exposure.blockSignals(False)
        if hasattr(self, 'slider_exposure'):
            self.slider_exposure.blockSignals(True)
            self.slider_exposure.setValue(self.saved_camera_exposure_value)
            self.slider_exposure.setEnabled(not self.saved_camera_auto_exposure)
            self.slider_exposure.blockSignals(False)
        if hasattr(self, 'lbl_exposure_ms'):
            self.lbl_exposure_ms.setText(f"{self.saved_camera_exposure_value / 1000.0:.1f} ms")
            
        if self.marker_st is not None and hasattr(self.marker_st, 'set_camera_exposure'):
            self.marker_st.set_camera_exposure(self.saved_camera_exposure_value, auto_exposure=self.saved_camera_auto_exposure)
        self.log_msg("[Camera] Exposure changes cancelled. Restored previous setting.")

    def save_camera_exposure(self):
        self.saved_camera_auto_exposure = self.applied_camera_auto_exposure
        self.saved_camera_exposure_value = self.applied_camera_exposure_value
        self.log_msg(f"[SUCCESS] Confirmed and saved exposure for calibration: auto={self.saved_camera_auto_exposure}, exposure={self.saved_camera_exposure_value}μs")

    def reconnect_camera(self, show_dialog=True):
        if self.camera_source_busy():
            self.log_msg('[ERROR] Finish active work and start a new sample session before reconnecting the camera.')
            return False
        self.log_msg("[INFO] Reconnecting RealSense camera...")
        previous = self.marker_st
        try:
            if previous.camera is not None:
                previous.camera.stream_off()
            replacement = Marker_Transform(robot=self.robot, robot_version=self.get_robot_version())
            replacement.set_marker_type('plate')
            self.marker_st = replacement
            self.marker_detector = replacement.marker_detection
            self.intrinsics_calibrator.marker_st = replacement
            for calibrator in (self.marker_calibrator, self.joint_calibrator, self.head_camera_calibrator):
                calibrator.marker_st = replacement
            self.init_camera_exposure_state()
            self.lbl_intrinsics_source.setText('Actual intrinsics: ' + replacement.intrinsics_metadata['file'])
            message = 'Camera unavailable; simulated observations selected.' if replacement.sim else 'RealSense camera connected.'
            self.log_msg('[INFO] ' + message)
            if show_dialog:
                QMessageBox.information(self, 'Camera source', message)
            return True
        except Exception as exc:
            self.log_msg(f'[ERROR] Failed to reconnect camera: {exc}')
            if show_dialog:
                QMessageBox.warning(self, 'Camera Connection Failed', str(exc))
            return False

    def load_offsets_from_yaml(self):
        self.joint_offsets_store = {
            "left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
            "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0}
        }
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                jo = data.get("joint_offset", {})
                for arm in ["left", "right"]:
                    arm_data = jo.get(arm, {})
                    if isinstance(arm_data, dict):
                        self.joint_offsets_store[arm]["joint3"] = float(arm_data.get("joint3", 0.0))
                        self.joint_offsets_store[arm]["joint5"] = float(arm_data.get("joint5", 0.0))
                        self.joint_offsets_store[arm]["joint6"] = float(arm_data.get("joint6", 0.0))
                if "head" in jo and isinstance(jo["head"], dict):
                    self.joint_offsets_store["head"] = {
                        "pan": float(jo["head"].get("pan", 0.0)),
                        "tilt": float(jo["head"].get("tilt", 0.0)),
                    }
                head_str = f" Head[Pan={self.joint_offsets_store['head']['pan']:.4f}°, Tilt={self.joint_offsets_store['head']['tilt']:.4f}°]" if "head" in self.joint_offsets_store else ""
                self.log_msg(f"[INFO] Loaded joint offsets from setting.yaml: "
                             f"R[J3={self.joint_offsets_store['right']['joint3']:.4f}°, "
                             f"J5={self.joint_offsets_store['right']['joint5']:.4f}°, "
                             f"J6={self.joint_offsets_store['right']['joint6']:.4f}°] "
                             f"L[J3={self.joint_offsets_store['left']['joint3']:.4f}°, "
                             f"J5={self.joint_offsets_store['left']['joint5']:.4f}°, "
                             f"J6={self.joint_offsets_store['left']['joint6']:.4f}°]{head_str}")
            else:
                self.log_msg("[INFO] setting.yaml not found. Initialized all joint offsets to 0.0°.")
        except Exception as e:
            self.log_msg(f"[WARNING] Failed to load joint offsets from setting.yaml: {e}. Using 0.0° defaults.")

        is_v13 = self.get_robot_version() == "1.3"
        self.joint_offsets = {
            "left": {
                "wrist_pitch": self.joint_offsets_store["left"]["joint5"],
                "wrist_roll": self.joint_offsets_store["left"]["joint6"] if is_v13 else 0.0,
                "wrist_yaw2": self.joint_offsets_store["left"]["joint6"] if not is_v13 else 0.0,
                "elbow": self.joint_offsets_store["left"]["joint3"]
            },
            "right": {
                "wrist_pitch": self.joint_offsets_store["right"]["joint5"],
                "wrist_roll": self.joint_offsets_store["right"]["joint6"] if is_v13 else 0.0,
                "wrist_yaw2": self.joint_offsets_store["right"]["joint6"] if not is_v13 else 0.0,
                "elbow": self.joint_offsets_store["right"]["joint3"]
            }
        }
        self.marker_calibrator.joint_offsets = self.joint_offsets
        self.joint_calibrator.joint_offsets = self.joint_offsets
        self.marker_calibrator.marker_problem_callback = lambda side: self.prompt_marker_problem_teaching(side, 'marker')
        self.joint_calibrator.marker_problem_callback = lambda side: self.prompt_marker_problem_teaching(
            side, self.joint_calibrator.current_calib_mode)

    def save_offsets_to_yaml(self):
        from core.config_store import update_yaml
        try:
            update_yaml(CONFIG_PATHS["setting_yaml"], self._joint_offset_patch())
            self.log_msg("[SUCCESS] Saved offsets permanently to setting.yaml!")
            return True
        except Exception as exc:
            self.log_msg(f"[ERROR] Failed to save offsets: {exc}")
            return False

    def on_cell_double_clicked(self, row, col):
        arm = "right" if row == 0 else "left"
        if col == 0:
            joint_key = "joint6"
            joint_label = "Joint 6"
        elif col == 1:
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
            for attribute in ('joint_sweep_data', '_failed_joint_result'):
                failed = getattr(self, attribute, None)
                if (failed and not _joint_result_accepted(failed)
                        and failed.get('arm_side', self.arm_side) == arm
                        and _joint_store_key(failed.get('mode', 'elbow')) == joint_key):
                    setattr(self, attribute, None)
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
        elif "wrist_yaw2" in mode_str:
            return "wrist_yaw2"
        elif "wrist_pitch" in mode_str:
            return "wrist_pitch"
        else:
            return "elbow"

    def get_offset_key_for_mode(self, mode):
        if mode == "wrist_pitch_v13":
            return "wrist_pitch"
        elif mode == "wrist_roll_v13":
            return "wrist_roll"
        elif mode == "wrist_yaw2":
            return "wrist_yaw2"
        else:
            return mode

    def init_ui(self):
        # Instantiate the dialog first
        self.plot_dialog = PlotViewerDialog(self)
        # Re-map plot widgets to the floating dialog
        self.lbl_plot_title = self.plot_dialog.lbl_title
        self.plot_label_combined = self.plot_dialog.plot_label
        self.btn_plot_prev = self.plot_dialog.btn_prev
        self.btn_plot_next = self.plot_dialog.btn_next

        # Main horizontal layout
        main_layout = QHBoxLayout()
        
        # --- Top-Level Step Tabs ---
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
        
        # Robot Connection Box (head movement controls removed per user request)
        conn_head_box = QGroupBox("Robot Connection")
        conn_head_box.setFixedHeight(130)
        conn_head_layout = QVBoxLayout()
        conn_head_layout.setSpacing(4)
        conn_head_layout.setContentsMargins(6, 6, 6, 6)
        
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP/Port:"))
        self.ip_input = QLineEdit("192.168.30.1:50051")
        if self.sim:
            self.ip_input.setText("127.0.0.1:50051")
        self.ip_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        ip_row.addWidget(self.ip_input)
        conn_head_layout.addLayout(ip_row)
        
        model_row = QHBoxLayout()
        self.lbl_model_tag = QLabel("Model:")
        model_row.addWidget(self.lbl_model_tag)
        self.model_input = QComboBox()
        self.model_input.addItems(UI_DROPDOWNS["robot_models"])
        model_row.addWidget(self.model_input)
        conn_head_layout.addLayout(model_row)
        
        # Hide model selection UI as it is auto-detected and updated dynamically
        self.lbl_model_tag.hide()
        self.model_input.hide()
        
        connect_head_row = QHBoxLayout()
        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #2b5278; color: #ffffff; font-weight: bold; padding: 4px 8px; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_connect.clicked.connect(self.connect_robot)
        self.btn_connect.setFixedHeight(28)
        connect_head_row.addWidget(self.btn_connect)
        
        # Head checkbox — controls whether head servos are enabled on connect
        self.chk_servo_head = QCheckBox("Head")
        self.chk_servo_head.setChecked(True)
        self.chk_servo_head.setStyleSheet("color: #b0bec5;")
        self.chk_servo_head.toggled.connect(self.on_head_checkbox_changed)
        connect_head_row.addWidget(self.chk_servo_head)
        conn_head_layout.addLayout(connect_head_row)
        
        conn_head_box.setLayout(conn_head_layout)
        
        # Calibration Workflows Box
        workflow_box = QGroupBox("Calibration Workflows")
        workflow_layout = QVBoxLayout()
        
        # Target Arm Selection
        self.arm_sel = QComboBox()
        self.arm_sel.addItems(UI_DROPDOWNS["arm_sides"])
        idx = 1 if self.arm_side == "left" else 0
        self.arm_sel.setCurrentIndex(idx)
        self.arm_sel.currentTextChanged.connect(self.on_arm_side_changed)

        self.joint_arm_sel = self.arm_sel
        self.marker_arm_sel = self.arm_sel

        arm_side_layout = QHBoxLayout()
        arm_side_layout.addWidget(QLabel("Active Arm Side:"))
        arm_side_layout.addWidget(self.arm_sel)
        workflow_layout.addLayout(arm_side_layout)
        
        debug_row = QHBoxLayout()
        self.chk_save_debug = QCheckBox("Save Debug Data")
        self.chk_save_debug.setChecked(True)
        debug_row.addWidget(self.chk_save_debug)
        workflow_layout.addLayout(debug_row)
        
        self.btn_stop_motion = QPushButton("STOP MOTION")
        self.btn_stop_motion.setStyleSheet("background-color: #c0392b; color: #ffffff; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_stop_motion.clicked.connect(self.stop_motion)
        self.btn_stop_motion.setFixedHeight(26)
        workflow_layout.addWidget(self.btn_stop_motion)
        
        self.workflow_tabs = QTabWidget()
        
        # Sub-tab 1: Joint Calibration
        joint_subtab = QWidget()
        joint_sublayout = QVBoxLayout()
        
        self.joint_mode_sel = QComboBox()
        self.update_joint_modes()
        self.joint_mode_sel.currentIndexChanged.connect(self.update_applied_offset_label)
        
        self.btn_joint_ready = QPushButton("MOVE TO READY")
        self.btn_joint_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_ready.clicked.connect(self.move_to_ready_pose_joint)
        
        self.btn_joint_start = QPushButton("START SWEEP")
        self.btn_joint_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_start.clicked.connect(self.start_calibration_joint)
        
        joint_sublayout.addWidget(QLabel("Joint Sweeps for Polarities & Kinematics:"))
        joint_sublayout.addWidget(self.joint_mode_sel)
        joint_sublayout.addWidget(self.btn_joint_ready)
        joint_sublayout.addWidget(self.btn_joint_start)
        joint_subtab.setLayout(joint_sublayout)
        
        # Sub-tab 2: Marker Bracket Calibration
        marker_subtab = QWidget()
        marker_sublayout = QVBoxLayout()
        
        self.lbl_marker_axis = QLabel("Marker Bracket Alignment Sweeps:")
        self.lbl_marker_axis.hide()
        self.marker_axis_sel = QComboBox()
        self.marker_axis_sel.addItems(UI_DROPDOWNS["marker_axes"])
        self.marker_axis_sel.hide()
        
        self.tolerance_input = QLineEdit("0.5")
        
        self.btn_marker_ready = QPushButton("MOVE TO READY")
        self.btn_marker_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_marker_ready.clicked.connect(self.move_to_ready_pose_marker)
        
        self.btn_marker_start = QPushButton("START SWEEP")
        self.btn_marker_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_marker_start.clicked.connect(self.start_calibration_marker)
        
        self.btn_marker_result = QPushButton("UNIFIED RESULT")
        self.btn_marker_result.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_marker_result.clicked.connect(self.show_unified_result_marker)
        
        marker_sublayout.addWidget(self.lbl_marker_axis)
        marker_sublayout.addWidget(self.marker_axis_sel)
        marker_sublayout.addWidget(self.btn_marker_ready)
        marker_sublayout.addWidget(self.btn_marker_start)
        marker_subtab.setLayout(marker_sublayout)
        
        # Sub-tab 3: Full Auto Calibration
        full_auto_subtab = QWidget()
        full_auto_sublayout = QVBoxLayout()
        
        self.btn_full_auto_ready = QPushButton("MOVE TO READY")
        self.btn_full_auto_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_full_auto_ready.clicked.connect(self.move_to_ready_full_auto)
        
        self.btn_full_auto_start = QPushButton("START FULL AUTO")
        self.btn_full_auto_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_full_auto_start.clicked.connect(self.start_full_auto)
        
        self.btn_full_auto_apply = QPushButton("APPLY FULL AUTO RESULTS")
        self.btn_full_auto_apply.setStyleSheet("background-color: #d35400; color: #ffffff; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_full_auto_apply.clicked.connect(self.apply_full_auto_results)
        self.btn_full_auto_apply.setEnabled(False) # Enabled after full auto finishes
        
        full_auto_sublayout.addWidget(QLabel("Full Auto Sequential Calibration:"))
        full_auto_sublayout.addWidget(self.btn_full_auto_ready)
        full_auto_sublayout.addWidget(self.btn_full_auto_start)
        full_auto_sublayout.addWidget(self.btn_full_auto_apply)
        full_auto_sublayout.addStretch()
        full_auto_subtab.setLayout(full_auto_sublayout)

        # Sub-tab 4: Head & Camera Extrinsics Calibration
        head_cam_subtab = QWidget()
        head_cam_sublayout = QVBoxLayout()
        head_cam_sublayout.setSpacing(6)

        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("Pan Range (±deg):"))
        self.step1_5_pan_range = QLineEdit("10.0")
        self.step1_5_pan_range.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        pan_row.addWidget(self.step1_5_pan_range)
        head_cam_sublayout.addLayout(pan_row)

        tilt_row = QHBoxLayout()
        tilt_row.addWidget(QLabel("Tilt Range (±deg):"))
        self.step1_5_tilt_range = QLineEdit("8.0")
        self.step1_5_tilt_range.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        tilt_row.addWidget(self.step1_5_tilt_range)
        head_cam_sublayout.addLayout(tilt_row)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Sweep Steps:"))
        self.step1_5_num_steps = QLineEdit("11")
        self.step1_5_num_steps.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        steps_row.addWidget(self.step1_5_num_steps)
        head_cam_sublayout.addLayout(steps_row)

        self.btn_step1_5_ready = QPushButton("1) MOVE TO READY")
        self.btn_step1_5_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step1_5_ready.setFixedHeight(28)
        self.btn_step1_5_ready.clicked.connect(self.move_to_ready_pose_step1_5)
        head_cam_sublayout.addWidget(self.btn_step1_5_ready)

        self.btn_step1_5_start = QPushButton("2) START HEAD SWEEP")
        self.btn_step1_5_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step1_5_start.setFixedHeight(28)
        self.btn_step1_5_start.clicked.connect(self.start_calibration_step1_5)
        head_cam_sublayout.addWidget(self.btn_step1_5_start)

        self.btn_step1_5_apply = QPushButton("3) APPLY RESULTS")
        self.btn_step1_5_apply.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step1_5_apply.setFixedHeight(28)
        self.btn_step1_5_apply.setEnabled(False)
        self.btn_step1_5_apply.clicked.connect(self.apply_results_step1_5)
        head_cam_sublayout.addWidget(self.btn_step1_5_apply)

        head_cam_sublayout.addStretch()
        head_cam_subtab.setLayout(head_cam_sublayout)
        
        # Add workflow subtabs in order of: Full Auto, Joint Calib, Marker Calib, Head & Cam
        self.workflow_tabs.addTab(full_auto_subtab, "Auto")
        self.workflow_tabs.addTab(joint_subtab, "Joint")
        self.workflow_tabs.addTab(marker_subtab, "Marker")
        self.workflow_tabs.addTab(head_cam_subtab, "Head & Cam")
        self.workflow_tabs.currentChanged.connect(self._on_workflow_tab_changed)
        
        workflow_layout.addWidget(self.workflow_tabs)
        workflow_box.setLayout(workflow_layout)

        # Store shared boxes as instance attributes for reparenting
        self.conn_head_box = conn_head_box
        self.home_offset_box = None  # Will be set below after creation
        self.status_box = None  # Will be set below after creation
        self.log_box = None  # Will be set below after creation

        # Assemble Column 1
        col1_layout.addWidget(conn_head_box)
        col1_layout.addWidget(workflow_box, 1)

        # --- COLUMN 2 (Calibration Status & Monitoring) ---
        col2_layout = QVBoxLayout()

        # Standalone Robot Home Offset Reset Box
        home_offset_box = QGroupBox("Robot Home Offset")
        home_offset_box.setFixedHeight(160)
        home_offset_layout = QVBoxLayout()
        home_offset_layout.setSpacing(6)
        home_offset_layout.setContentsMargins(8, 8, 8, 8)
        
        desc_label = QLabel("Reset joint offsets to zero to restore factory alignment:")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #b0bec5; font-size: 11px;")
        home_offset_layout.addWidget(desc_label)
        
        btn_row = QHBoxLayout()
        self.btn_home_reset = QPushButton("Home Offset Reset")
        self.btn_home_reset.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_home_reset.clicked.connect(lambda: self.home_offset_reset(confirm_dialog=True))
        self.btn_home_reset.setFixedHeight(28)
        btn_row.addWidget(self.btn_home_reset)

        self.btn_step2_zero_pose = QPushButton("Zero Pose")
        self.btn_step2_zero_pose.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_zero_pose.setFixedHeight(28)
        self.btn_step2_zero_pose.clicked.connect(self.step2_zero_pose_check)
        btn_row.addWidget(self.btn_step2_zero_pose)
        
        home_offset_layout.addLayout(btn_row)
        
        hint_label = QLabel("Tip: Double-click any cell in the table below to manually stage individual offsets.")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        home_offset_layout.addWidget(hint_label)
        
        home_offset_box.setLayout(home_offset_layout)
        self.home_offset_box = home_offset_box

        dash_box = QGroupBox("Calibration Status & Monitoring")
        dash_layout = QVBoxLayout()
        dash_layout.setSpacing(4)
        dash_layout.setContentsMargins(8, 4, 8, 4)
        
        # Monitoring Table (Arm Joint Offsets)
        self.tbl_offset_monitor = QTableWidget(2, 3)
        self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Roll/Yaw 2)", "Joint 5 (Wrist Pitch)", "Joint 3 (Elbow)"])
        self.tbl_offset_monitor.setVerticalHeaderLabels(["Right Arm", "Left Arm"])
        self.tbl_offset_monitor.setFixedHeight(110)
        self.tbl_offset_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_offset_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.tbl_offset_monitor.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                gridline-color: #2d2d2d;
                font-weight: bold;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #263238;
                color: #b0bec5;
                font-weight: bold;
                padding: 3px;
                border: 1px solid #2d2d2d;
            }
        """)
        
        # Marker Bracket Design Offset UI GroupBox (Nested inside dash_box)
        bracket_box = QGroupBox("Marker Bracket Design Offset (Tf_to_marker)")
        bracket_layout = QVBoxLayout()
        bracket_layout.setSpacing(4)
        bracket_layout.setContentsMargins(6, 6, 6, 6)
        
        input_style = "background-color: #1c1c1c; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;"
        
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
        self.btn_apply_bracket.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_apply_bracket.setFixedHeight(24)
        self.btn_apply_bracket.clicked.connect(self.apply_bracket_design_values)
        bracket_layout.addWidget(self.btn_apply_bracket)
        
        bracket_box.setLayout(bracket_layout)
        
        # Apply & Clear buttons for Joint offsets
        btn_joint_layout = QHBoxLayout()
        btn_joint_layout.setSpacing(6)
        
        self.btn_joint_apply = QPushButton("APPLY OFFSET")
        self.btn_joint_apply.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_apply.clicked.connect(self.apply_joint_offset)
        self.btn_joint_apply.setFixedHeight(24)
        
        self.btn_joint_clear = QPushButton("CLEAR OFFSET")
        self.btn_joint_clear.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_clear.clicked.connect(self.clear_joint_offset)
        self.btn_joint_clear.setFixedHeight(24)
        
        btn_joint_layout.addWidget(self.btn_joint_apply)
        btn_joint_layout.addWidget(self.btn_joint_clear)

        # Page 0: Arm & Marker Monitoring View
        dash_page_arm_marker = QWidget()
        dash_am_layout = QVBoxLayout()
        dash_am_layout.setContentsMargins(0, 0, 0, 0)
        dash_am_layout.setSpacing(4)
        dash_am_layout.addWidget(self.tbl_offset_monitor)
        dash_am_layout.addWidget(bracket_box)
        dash_am_layout.addLayout(btn_joint_layout)
        dash_page_arm_marker.setLayout(dash_am_layout)

        # Page 1: Head & Camera Parameter Status View
        dash_page_head_cam = QWidget()
        dash_hc_layout = QVBoxLayout()
        dash_hc_layout.setContentsMargins(0, 0, 0, 0)
        dash_hc_layout.setSpacing(6)

        lbl_head_tbl = QLabel("Head Joint Offsets:")
        lbl_head_tbl.setStyleSheet("font-weight: bold; color: #2979ff;")
        dash_hc_layout.addWidget(lbl_head_tbl)

        self.tbl_step1_5_head_monitor = QTableWidget(2, 2)
        self.tbl_step1_5_head_monitor.setHorizontalHeaderLabels(["Nominal / Current", "Calibrated Offset"])
        self.tbl_step1_5_head_monitor.setVerticalHeaderLabels(["Head Pan (Joint 0)", "Head Tilt (Joint 1)"])
        self.tbl_step1_5_head_monitor.setFixedHeight(85)
        self.tbl_step1_5_head_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_step1_5_head_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_step1_5_head_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_step1_5_head_monitor.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                gridline-color: #2d2d2d;
                font-weight: bold;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #263238;
                color: #b0bec5;
                font-weight: bold;
                padding: 3px;
                border: 1px solid #2d2d2d;
            }
        """)
        dash_hc_layout.addWidget(self.tbl_step1_5_head_monitor)

        lbl_cam_tbl = QLabel("Camera Extrinsic Parameters (Mount to Cam):")
        lbl_cam_tbl.setStyleSheet("font-weight: bold; color: #00e676;")
        dash_hc_layout.addWidget(lbl_cam_tbl)

        self.tbl_step1_5_cam_monitor = QTableWidget(6, 2)
        self.tbl_step1_5_cam_monitor.setHorizontalHeaderLabels(["Nominal CAD", "Calibrated Value"])
        self.tbl_step1_5_cam_monitor.setVerticalHeaderLabels(["Roll (°)", "Pitch (°)", "Yaw (°)", "X (mm)", "Y (mm)", "Z (mm)"])
        self.tbl_step1_5_cam_monitor.setFixedHeight(180)
        self.tbl_step1_5_cam_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_step1_5_cam_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_step1_5_cam_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_step1_5_cam_monitor.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                gridline-color: #2d2d2d;
                font-weight: bold;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #263238;
                color: #b0bec5;
                font-weight: bold;
                padding: 3px;
                border: 1px solid #2d2d2d;
            }
        """)
        dash_hc_layout.addWidget(self.tbl_step1_5_cam_monitor)

        step1_5_diag_box = QGroupBox("Fit Quality Diagnostics")
        step1_5_diag_box.setStyleSheet("QGroupBox { border: 1px solid #333333; border-radius: 4px; } QGroupBox::title { color: #b0bec5; font-size: 11px; }")
        diag_layout = QGridLayout()
        diag_layout.setSpacing(4)
        diag_layout.addWidget(QLabel("Tilt Plane RMSE:"), 0, 0)
        self.lbl_step1_5_rmse_tilt = QLabel("-")
        self.lbl_step1_5_rmse_tilt.setStyleSheet("color: #00e5ff; font-weight: bold;")
        diag_layout.addWidget(self.lbl_step1_5_rmse_tilt, 0, 1)

        diag_layout.addWidget(QLabel("Pan Plane RMSE:"), 0, 2)
        self.lbl_step1_5_rmse_pan = QLabel("-")
        self.lbl_step1_5_rmse_pan.setStyleSheet("color: #00e5ff; font-weight: bold;")
        diag_layout.addWidget(self.lbl_step1_5_rmse_pan, 0, 3)

        diag_layout.addWidget(QLabel("Axis Orthogonality Error:"), 1, 0)
        self.lbl_step1_5_ortho_err = QLabel("-")
        self.lbl_step1_5_ortho_err.setStyleSheet("color: #ffca28; font-weight: bold;")
        diag_layout.addWidget(self.lbl_step1_5_ortho_err, 1, 1)

        step1_5_diag_box.setLayout(diag_layout)
        dash_hc_layout.addWidget(step1_5_diag_box)
        dash_hc_layout.addStretch()

        dash_page_head_cam.setLayout(dash_hc_layout)

        # Stack to hold both pages
        self.dash_stack = QStackedWidget()
        self.dash_stack.addWidget(dash_page_arm_marker)
        self.dash_stack.addWidget(dash_page_head_cam)
        dash_layout.addWidget(self.dash_stack)
        dash_box.setLayout(dash_layout)

        col2_layout.addWidget(home_offset_box)
        col2_layout.addWidget(dash_box, 1)

        # --- COLUMN 3 (Camera Status & System Log/Plots) ---
        col3_layout = QVBoxLayout()

        # Status Indicator Box (Constructed here for Col 3)
        status_box = QGroupBox("Camera & Marker Status")
        status_box.setFixedHeight(160)
        status_layout = QVBoxLayout()
        status_layout.setSpacing(6)
        status_layout.setContentsMargins(8, 8, 8, 8)
        
        ind_layout = QHBoxLayout()
        self.indicator = IndicatorWidget()
        ind_layout.addWidget(self.indicator)
        self.status_label = QLabel("Not Detected")
        self.status_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.status_label.setStyleSheet("color: #ff1744;")
        ind_layout.addWidget(self.status_label)
        ind_layout.addStretch()
        status_layout.addLayout(ind_layout)
        
        self.temp_label = QLabel(self.get_temp_label_text())
        self.temp_label.setStyleSheet("color: #fb8c00; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.temp_label)
        
        btn_layout = QHBoxLayout()
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        self.btn_monitor.setFixedHeight(26)
        
        self.btn_camera_feed = QPushButton("Camera Feed")
        self.btn_camera_feed.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_camera_feed.clicked.connect(self.toggle_camera_feed_dialog)
        self.btn_camera_feed.setFixedHeight(26)
        
        btn_layout.addWidget(self.btn_monitor)
        btn_layout.addWidget(self.btn_camera_feed)
        status_layout.addLayout(btn_layout)
        
        status_box.setLayout(status_layout)
        self.status_box = status_box

        # System Log GroupBox
        log_box = QGroupBox("System Log & Control")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(6, 6, 6, 6)
        
        log_header = QHBoxLayout()
        console_title = QLabel("Execution Console Logs")
        console_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        console_title.setStyleSheet("color: #ffffff; margin-bottom: 2px;")
        
        self.btn_show_plot = QPushButton("Show Calibration Plot")
        self.btn_show_plot.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_show_plot.setFixedHeight(24)
        self.btn_show_plot.clicked.connect(self.open_plot_dialog)
        
        log_header.addWidget(console_title)
        log_header.addStretch()
        log_header.addWidget(self.btn_show_plot)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        font_log = QFont("Noto Sans Mono CJK KR", 10)
        font_log.setStyleHint(QFont.Monospace)
        font_log.setStyleStrategy(QFont.PreferAntialias | QFont.PreferQuality)
        self.log_text.setFont(font_log)
        
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_text)
        log_box.setLayout(log_layout)
        self.log_box = log_box

        col3_layout.addWidget(status_box)
        col3_layout.addWidget(log_box, 1)

        # Assemble side-by-side 3 Columns (1:3:3 weight)
        main_tab_columns = QHBoxLayout()
        main_tab_columns.addLayout(col1_layout, 1)
        main_tab_columns.addLayout(col2_layout, 3)
        main_tab_columns.addLayout(col3_layout, 3)
        
        main_tab_layout.addLayout(main_tab_columns)

        
        main_tab.setLayout(main_tab_layout)
        
        # ==========================================
        # Camera Tab (카메라 내부 파라미터 보정 전용)
        # ==========================================
        camera_tab = QWidget()
        camera_tab_layout = QHBoxLayout()
        
        int_left = QVBoxLayout()
        self.video_label = QLabel("Camera Feed Loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 300)
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
        
        self.chk_int_guide = QCheckBox("Show Guide Overlay")
        self.chk_int_guide.setChecked(True)
        self.chk_int_guide.setStyleSheet("color: #00e5ff; font-weight: bold;")
        self.chk_int_guide.stateChanged.connect(self.on_guide_changed)
        controls_layout.addWidget(self.chk_int_guide)
        
        self.btn_int_capture = QPushButton("CAPTURE FRAME (C)")
        self.btn_int_capture.setMinimumHeight(45)
        self.btn_int_capture.setStyleSheet("background-color: #1e88e5; color: white; font-size: 13px; font-weight: bold; border-radius: 4px;")
        self.btn_int_capture.clicked.connect(self.capture_intrinsics_frame)
        controls_layout.addWidget(self.btn_int_capture)
        
        self.btn_int_calibrate = QPushButton("RUN CALIBRATION")
        self.btn_int_calibrate.setMinimumHeight(45)
        self.btn_int_calibrate.setStyleSheet("background-color: #43a047; color: white; font-size: 13px; font-weight: bold; border-radius: 4px;")
        self.btn_int_calibrate.clicked.connect(self.run_intrinsics_calibration)
        controls_layout.addWidget(self.btn_int_calibrate)
        
        self.btn_int_save = QPushButton("SAVE PARAMETERS")
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #fb8c00; color: #000000; font-size: 13px; font-weight: bold; border-radius: 4px;")
        self.btn_int_save.clicked.connect(self.save_intrinsics_calibration)
        self.btn_int_save.setEnabled(False)
        controls_layout.addWidget(self.btn_int_save)
        
        self.btn_int_reset = QPushButton("RESET CAPTURES")
        self.btn_int_reset.setMinimumHeight(30)
        self.btn_int_reset.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;")
        self.btn_int_reset.clicked.connect(self.reset_intrinsics_captures)
        controls_layout.addWidget(self.btn_int_reset)

        self.lbl_intrinsics_source = QLabel('Actual source: ' + str(getattr(self.marker_st, 'intrinsics_metadata', {}).get('source', 'not connected')))
        self.lbl_intrinsics_source.setWordWrap(True)
        controls_layout.addWidget(self.lbl_intrinsics_source)
        
        controls_box.setLayout(controls_layout)
        
        int_right = QVBoxLayout()
        
        stats_box2 = QGroupBox("Capture Stats")
        stats_layout2 = QHBoxLayout()
        self.lbl_captured = QLabel("Captured Frames: 0")
        self.lbl_captured.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #2979ff;")
        
        self.lbl_temp = QLabel(self.get_temp_label_text())
        self.lbl_temp.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #ff5500;")
        
        stats_layout2.addWidget(self.lbl_captured)
        stats_layout2.addStretch()
        stats_layout2.addWidget(self.lbl_temp)
        stats_box2.setLayout(stats_layout2)
        
        self.btn_reconnect_cam = QPushButton("🔄 RECONNECT CAMERA (카메라 재연결)")
        self.btn_reconnect_cam.setMinimumHeight(34)
        self.btn_reconnect_cam.setStyleSheet("background-color: #00838f; color: white; font-size: 11px; font-weight: bold; border-radius: 4px;")
        self.btn_reconnect_cam.clicked.connect(self.reconnect_camera)
        
        # Exposure / Brightness Adjustment Box
        exposure_box = QGroupBox("Camera Exposure / Brightness")
        exposure_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 13px;}")
        exp_layout = QVBoxLayout()
        exp_layout.setSpacing(8)
        
        self.chk_auto_exposure = QCheckBox("Enable Auto Exposure (자동 노출)")
        self.chk_auto_exposure.setChecked(True)
        self.chk_auto_exposure.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.chk_auto_exposure.toggled.connect(self.on_auto_exposure_toggled)
        exp_layout.addWidget(self.chk_auto_exposure)
        
        exp_val_row = QHBoxLayout()
        exp_val_row.addWidget(QLabel("Exposure (μs):"))
        self.spin_exposure = QSpinBox()
        self.spin_exposure.setRange(100, 100000)
        self.spin_exposure.setSingleStep(500)
        self.spin_exposure.setValue(6000)
        self.spin_exposure.setEnabled(False)
        self.spin_exposure.setStyleSheet("background-color: #1e1e1e; color: #00e5ff; font-weight: bold;")
        self.spin_exposure.valueChanged.connect(self.on_exposure_value_changed)
        exp_val_row.addWidget(self.spin_exposure)
        
        self.lbl_exposure_ms = QLabel("6.0 ms")
        self.lbl_exposure_ms.setStyleSheet("color: #ffd700; font-weight: bold; min-width: 50px;")
        exp_val_row.addWidget(self.lbl_exposure_ms)
        exp_layout.addLayout(exp_val_row)
        
        self.slider_exposure = QSlider(Qt.Horizontal)
        self.slider_exposure.setRange(100, 100000)
        self.slider_exposure.setSingleStep(500)
        self.slider_exposure.setPageStep(5000)
        self.slider_exposure.setValue(6000)
        self.slider_exposure.setEnabled(False)
        self.slider_exposure.valueChanged.connect(self.on_exposure_value_changed)
        exp_layout.addWidget(self.slider_exposure)
        
        # Action Buttons: APPLY and CANCEL
        btn_row1 = QHBoxLayout()
        self.btn_apply_exp = QPushButton("APPLY (적용)")
        self.btn_apply_exp.setMinimumHeight(35)
        self.btn_apply_exp.setStyleSheet("background-color: #388e3c; color: white; font-size: 12px; font-weight: bold; border-radius: 4px;")
        self.btn_apply_exp.clicked.connect(self.apply_camera_exposure)
        btn_row1.addWidget(self.btn_apply_exp)
        
        self.btn_cancel_exp = QPushButton("CANCEL (취소)")
        self.btn_cancel_exp.setMinimumHeight(35)
        self.btn_cancel_exp.setStyleSheet("background-color: #546e7a; color: white; font-size: 12px; font-weight: bold; border-radius: 4px;")
        self.btn_cancel_exp.clicked.connect(self.cancel_camera_exposure)
        btn_row1.addWidget(self.btn_cancel_exp)
        exp_layout.addLayout(btn_row1)
        
        exposure_box.setLayout(exp_layout)
        
        int_right.addWidget(stats_box2)
        int_right.addWidget(self.btn_reconnect_cam)
        int_right.addWidget(exposure_box)
        int_right.addWidget(controls_box) # Placed below exposure box!
        int_right.addStretch()
        
        camera_tab_layout.addLayout(int_left, 2)
        camera_tab_layout.addLayout(int_right, 1)
        camera_tab.setLayout(camera_tab_layout)
        
        # ==========================================
        # Overview Tab
        # ==========================================
        overview_tab = QWidget()
        overview_layout = QVBoxLayout()
        overview_layout.setContentsMargins(20, 20, 20, 20)
        
        self.overview_container = QWidget()
        container_layout = QVBoxLayout(self.overview_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)  # Top Stretch to push contents down to center
        
        self.overview_title = QLabel("Calibration Process Overview")
        self.overview_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        self.overview_title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.overview_title)
        
        self.overview_link = QLabel('GitHub Repository: <a href="https://github.com/RainbowRobotics/rby1-calibration" style="color: #00e5ff; font-weight: bold;">https://github.com/RainbowRobotics/rby1-calibration</a>')
        self.overview_link.setStyleSheet("font-size: 15px;")
        self.overview_link.setAlignment(Qt.AlignCenter)
        self.overview_link.setOpenExternalLinks(True)
        container_layout.addWidget(self.overview_link)
        
        self.overview_duration = QLabel("Estimated Execution Time: ~40 minutes")
        self.overview_duration.setStyleSheet("font-size: 16px; font-weight: bold; color: #00e5ff;")
        self.overview_duration.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.overview_duration)
        
        container_layout.addSpacing(25)  # Add spacing before the start button
        
        self.btn_start_wizard = QPushButton(tr("wizard.btn_start_wizard"))
        self.btn_start_wizard.setStyleSheet("background-color: #d84315; color: white; font-weight: bold; font-size: 18px; padding: 10px;")
        self.btn_start_wizard.setFixedWidth(200)
        self.btn_start_wizard.clicked.connect(self.show_wizard_ui)
        container_layout.addWidget(self.btn_start_wizard, alignment=Qt.AlignCenter)
        
        container_layout.addStretch(1)  # Bottom Stretch to push contents up to center
        
        self.overview_img = None
        overview_layout.addWidget(self.overview_container)
        
        self.wizard_widget = CalibrationWizardWidget(self)
        self.wizard_widget.setVisible(False)
        overview_layout.addWidget(self.wizard_widget, stretch=1)
        
        overview_tab.setLayout(overview_layout)

        # ==========================================
        # Step 1 Tab: Contains Main + Camera as sub-tabs
        # ==========================================
        step1_tab = QWidget()
        step1_layout = QVBoxLayout()
        step1_layout.setContentsMargins(0, 0, 0, 0)
        
        self.step1_tabs = QTabWidget()
        self.step1_tabs.currentChanged.connect(self._on_step1_subtab_changed)
        self.step1_tabs.addTab(main_tab, "Main")
        self.step1_tabs.addTab(camera_tab, "Camera")
        
        step1_layout.addWidget(self.step1_tabs)
        step1_tab.setLayout(step1_layout)
        
        self.left_tabs.addTab(overview_tab, "Overview")
        self.left_tabs.addTab(step1_tab, "Step 1")

        # ==========================================
        # Step 2 Tab: Shared widgets + empty Box1
        # ==========================================
        step2_tab = QWidget()
        step2_layout = QVBoxLayout()
        step2_layout.setContentsMargins(5, 5, 5, 5)
        
        # Step 2 Top Row for shared widgets (conn_head_box and home_offset_box)
        self.step2_top_row = QHBoxLayout()
        self.step2_top_row.setSpacing(10)
        step2_layout.addLayout(self.step2_top_row)
        
        # Step 2 columns: Left (Config + Actions), Right (status + log)
        step2_columns = QHBoxLayout()
        
        # Step 2 Left Column
        self.step2_left_col = QVBoxLayout()
        # Placeholders for reparented widgets — Config and Actions boxes go here
        
        # Config Box (replaces Box1 — mirrors calibration_ui Config section)
        config_box = QGroupBox("Config")
        config_box.setStyleSheet("QGroupBox { border: 1px solid #555; border-radius: 4px; } QGroupBox::title { color: #2979ff; font-weight: bold; }")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(4)
        config_layout.setContentsMargins(6, 6, 6, 6)
        
        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.step2_mode_sel = QComboBox()
        self.step2_mode_sel.addItems(["live", "npz"])
        self.step2_mode_sel.setStyleSheet("background-color: #2a2a2a; color: white;")
        mode_row.addWidget(self.step2_mode_sel)
        config_layout.addLayout(mode_row)
        
        # Path input
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self.step2_path_input = QLineEdit("result/result_step2/dataset_YYYYMMDD_HHMMSS.npz")
        self.step2_path_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        path_row.addWidget(self.step2_path_input)
        config_layout.addLayout(path_row)
        
        # Estimated samples label
        self.step2_est_samples_label = QLabel("Est. Samples: 0")
        self.step2_est_samples_label.setStyleSheet("color: #2979ff; font-weight: bold; font-size: 11px;")
        config_layout.addWidget(self.step2_est_samples_label)
        
        # Auto Motion Step parameters
        auto_step_row = QHBoxLayout()
        auto_step_row.addWidget(QLabel("Angle(deg):"))
        self.step2_angle_step = QLineEdit("5.0")
        self.step2_angle_step.setFixedWidth(45)
        self.step2_angle_step.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row.addWidget(self.step2_angle_step)
        
        auto_step_row.addWidget(QLabel("Pos(m):"))
        self.step2_pos_step = QLineEdit("0.03")
        self.step2_pos_step.setFixedWidth(45)
        self.step2_pos_step.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row.addWidget(self.step2_pos_step)
        config_layout.addLayout(auto_step_row)
        
        auto_step_row2 = QHBoxLayout()
        auto_step_row2.addWidget(QLabel("Step(m):"))
        self.step2_step_x = QLineEdit("0.03")
        self.step2_step_x.setFixedWidth(45)
        self.step2_step_x.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row2.addWidget(self.step2_step_x)
        
        auto_step_row2.addWidget(QLabel("Max X(m):"))
        self.step2_max_x = QLineEdit("0.4")
        self.step2_max_x.setFixedWidth(45)
        self.step2_max_x.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row2.addWidget(self.step2_max_x)
        config_layout.addLayout(auto_step_row2)
        
        # Head status label
        self.step2_head_status_label = QLabel("Auto Motion: 0/0")
        self.step2_head_status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        config_layout.addWidget(self.step2_head_status_label)
        
        # Apply Joint Offset checkbox (instead of full joint offset box)
        jo_row = QHBoxLayout()
        self.chk_apply_joint_offset = QCheckBox("Constrain to Step 1 (±0.05° measurement assumption)")
        self.chk_apply_joint_offset.setChecked(False)
        self.chk_apply_joint_offset.setStyleSheet("color: #cccccc; font-weight: bold;")
        self.chk_apply_joint_offset.toggled.connect(self._on_apply_joint_offset_toggled)
        jo_row.addWidget(self.chk_apply_joint_offset)
        
        self.lbl_jo_status = QLabel("ACTIVE")
        self.lbl_jo_status.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px;")
        jo_row.addWidget(self.lbl_jo_status)
        jo_row.addStretch()
        config_layout.addLayout(jo_row)
        
        config_layout.addStretch()
        config_box.setLayout(config_layout)
        self.step2_left_col.addWidget(config_box, 1)
        
        self.step2_angle_step.textChanged.connect(self.update_step2_est_samples)
        self.step2_pos_step.textChanged.connect(self.update_step2_est_samples)
        self.step2_step_x.textChanged.connect(self.update_step2_est_samples)
        self.step2_max_x.textChanged.connect(self.update_step2_est_samples)
        self.step2_mode_sel.currentTextChanged.connect(self.on_step2_mode_changed)
        QTimer.singleShot(100, self.update_step2_est_samples)
        
        # Actions Box (mirrors calibration_ui Actions section)
        actions_box = QGroupBox("Actions")
        actions_box.setStyleSheet("QGroupBox { border: 1px solid #333333; border-radius: 4px; } QGroupBox::title { color: #ffffff; font-weight: bold; }")
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(4)
        actions_layout.setContentsMargins(6, 6, 6, 6)
        
        # Top row: Stop (Zero Pose is located in Calibration Status above)
        top_action_row = QHBoxLayout()
        self.btn_step2_stop = QPushButton("Stop")
        self.btn_step2_stop.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_stop.setFixedHeight(28)
        self.btn_step2_stop.clicked.connect(self.stop_motion)
        top_action_row.addWidget(self.btn_step2_stop)
        actions_layout.addLayout(top_action_row)
        
        # Numbered actions row 1
        act_row1 = QHBoxLayout()
        self.btn_step2_init_pose = QPushButton("1) Init Pose")
        self.btn_step2_init_pose.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_init_pose.setFixedHeight(28)
        self.btn_step2_init_pose.clicked.connect(self.step2_init_pose)
        act_row1.addWidget(self.btn_step2_init_pose)
        
        self.btn_step2_auto_motion = QPushButton("2) Auto Motion")
        self.btn_step2_auto_motion.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_auto_motion.setFixedHeight(28)
        self.btn_step2_auto_motion.clicked.connect(self.step2_auto_motion)
        act_row1.addWidget(self.btn_step2_auto_motion)
        actions_layout.addLayout(act_row1)
        
        # Numbered actions row 2
        act_row2 = QHBoxLayout()
        self.btn_step2_calculate = QPushButton("3) Calculate")
        self.btn_step2_calculate.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_calculate.setFixedHeight(28)
        self.btn_step2_calculate.clicked.connect(self.step2_calculate)
        act_row2.addWidget(self.btn_step2_calculate)
        
        self.btn_step2_clear = QPushButton("4) Clear Samples")
        self.btn_step2_clear.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_clear.setFixedHeight(28)
        self.btn_step2_clear.clicked.connect(self.step2_clear_samples)
        act_row2.addWidget(self.btn_step2_clear)
        actions_layout.addLayout(act_row2)
        
        # Numbered actions row 3
        act_row3 = QHBoxLayout()
        self.btn_step2_apply_home = QPushButton("5) Apply Home Offset")
        self.btn_step2_apply_home.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_apply_home.setFixedHeight(28)
        self.btn_step2_apply_home.clicked.connect(self.step2_apply_home_offset)
        act_row3.addWidget(self.btn_step2_apply_home)
        
        self.btn_step2_check_state = QPushButton("6) Check Calibration State")
        self.btn_step2_check_state.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_check_state.setFixedHeight(28)
        self.btn_step2_check_state.clicked.connect(self.step2_check_calibration_state)
        act_row3.addWidget(self.btn_step2_check_state)
        actions_layout.addLayout(act_row3)
        
        # Sample count label
        self.step2_sample_count_label = QLabel("Shared Samples: 0")
        self.step2_sample_count_label.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 11px;")
        actions_layout.addWidget(self.step2_sample_count_label)
        
        actions_layout.addStretch()
        actions_box.setLayout(actions_layout)
        self.step2_left_col.addWidget(actions_box, 1)
        
        # Step 2 Right Column
        self.step2_right_col = QVBoxLayout()
        # status_box and log_box (without plot button) go here
        
        step2_columns.addLayout(self.step2_left_col, 1)
        step2_columns.addLayout(self.step2_right_col, 1)
        
        step2_layout.addLayout(step2_columns)

        
        step2_tab.setLayout(step2_layout)
        
        self.left_tabs.addTab(step2_tab, "Step 2")
        
        # Keep references for reparenting logic
        # Step 1 Main tab column layout indices for reinserting shared widgets
        self._step1_col1 = col1_layout
        self._step1_col2 = col2_layout
        self._step1_col3 = col3_layout
        
        # Assemble full-width tabs
        main_layout.addWidget(self.left_tabs)
        
        # Create outer vertical layout with Top Language Header
        outer_layout = QVBoxLayout()
        
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(5, 5, 5, 5)
        
        self.lbl_lang_select = QLabel(tr("common.language") + ":")
        self.lbl_lang_select.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffeb3b;")
        
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["English", "한국어 (Korean)"])
        self.combo_lang.setCurrentIndex(0 if Language.instance().current_lang == "en" else 1)
        self.combo_lang.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px 12px; background-color: #2b2b2b; color: #ffffff; border-radius: 4px; min-width: 140px;")
        self.combo_lang.currentIndexChanged.connect(self.on_language_combo_changed)
        
        top_bar.addWidget(self.lbl_lang_select)
        top_bar.addWidget(self.combo_lang)
        top_bar.addStretch()
        
        outer_layout.addLayout(top_bar)
        outer_layout.addLayout(main_layout)
        
        self.btn_quit_global = QPushButton("QUIT")
        self.btn_quit_global.setMinimumHeight(40)
        self.btn_quit_global.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold; font-size: 14px; border-radius: 6px; border: 1px solid #111111;")
        self.btn_quit_global.clicked.connect(self.close)
        outer_layout.addWidget(self.btn_quit_global)
        
        self.setLayout(outer_layout)
        
        # Initial UI translation sync
        self.retranslate_ui()
        
        # Startup info
        self.log_msg("="*60)
        self.log_msg("  UNIFIED ROBOT CALIBRATION SUITE LOADED")
        self.log_msg("="*60)
        self.log_msg("[RECOMMENDED SEQUENCE]")
        self.log_msg("  1. Calibrate camera intrinsics first if needed (Step 1 > Camera tab).")
        self.log_msg("  2. Calibrate joint offsets using Joint subtab.")
        self.log_msg("  3. Perform marker bracket sweeps using Marker subtab.")
        self.log_msg("  4. Control head and verify offsets as a final check.")
        self.log_msg("="*60)

    def is_temp_supported(self):
        if self.sim or self.marker_st is None:
            return False
        return getattr(self.marker_st, 'temp_supported', False)

    def get_temp_label_text(self, temp_val=None):
        is_ko = (Language.instance().current_lang == "ko")
        if not self.is_temp_supported():
            if is_ko:
                return "카메라 온도: 지원하지 않음"
            else:
                return "Camera Temp: Not Supported"
        else:
            if temp_val is not None:
                return f"카메라 온도: {temp_val:.1f} °C" if is_ko else f"Camera Temp: {temp_val:.1f} °C"
            else:
                return "카메라 온도: -- °C" if is_ko else "Camera Temp: -- °C"

    def on_language_combo_changed(self, index):
        lang_code = "en" if index == 0 else "ko"
        Language.instance().set_language(lang_code)
        self.retranslate_ui()

    def retranslate_ui(self):
        if hasattr(self, 'lbl_lang_select'):
            self.lbl_lang_select.setText(tr("common.language") + ":")
        if hasattr(self, 'left_tabs'):
            self.left_tabs.setTabText(0, tr("main_tabs.tab_overview"))
            self.left_tabs.setTabText(1, tr("main_tabs.tab_step1"))
            self.left_tabs.setTabText(2, tr("main_tabs.tab_step2"))
        if hasattr(self, 'btn_start_wizard'):
            self.btn_start_wizard.setText(tr("wizard.btn_start_wizard"))

        # Translate temperature labels
        last_temp = getattr(self, 'last_temp', None)
        temp_text = self.get_temp_label_text(last_temp)
        if hasattr(self, 'temp_label') and self.temp_label is not None:
            self.temp_label.setText(temp_text)
        if hasattr(self, 'lbl_temp') and self.lbl_temp is not None:
            self.lbl_temp.setText(temp_text)
        if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
            if hasattr(self.wizard_widget, 'lbl_temp') and self.wizard_widget.lbl_temp is not None:
                self.wizard_widget.lbl_temp.setText(temp_text)

    @property
    def sim(self) -> bool:
        return self.marker_st.sim

    # --- Common Helper Functions ---
    def _log_msg_slot(self, msg):
        if msg.lstrip().startswith('[SWEEP COMMAND]'):
            try:
                os.makedirs(CONFIG_PATHS['txt_dir'], exist_ok=True)
                path = os.path.join(CONFIG_PATHS['txt_dir'], 'sweep_commands.log')
                with open(path, 'a', encoding='utf-8') as stream:
                    stream.write(msg + '\n')
                return
            except OSError as error:
                msg = f'[WARNING] Failed to save sweep command log: {error}'
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(msg)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        else:
            print(msg)

    def _update_ui_slot(self, action):
        if action == "head_pose":
            self.update_head_pose_status()
        elif action == "samples":
            self.update_step2_est_samples()
        elif action == "sample_counts":
            self.update_sample_counts()

    def show_message_box(self, title, text, icon=QMessageBox.Information, buttons=QMessageBox.Ok):
        from PySide6.QtWidgets import QLabel
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStandardButtons(buttons)
        
        # Enforce minimum size on QMessageBox dialog
        msg_box.setMinimumSize(550, 220)
        
        # Enforce wrap-around and minimum label constraints
        label = msg_box.findChild(QLabel, "qt_msgbox_label")
        if label:
            label.setWordWrap(True)
            label.setMinimumHeight(100)
            label.setStyleSheet("QLabel { font-size: 13px; color: #ffffff; padding: 10px; line-height: 1.4; }")
            
        return msg_box.exec()

    def log_msg(self, msg):
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        if QThread.currentThread() == QApplication.instance().thread():
            self._log_msg_slot(msg)
        else:
            self.log_signal_safe.emit(msg)

    def _write_step2_log(self, msg):
        import os
        log_file = os.path.join(CONFIG_PATHS["txt_dir"], "step2_capture_log.txt")
        try:
            os.makedirs(CONFIG_PATHS["txt_dir"], exist_ok=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            self.log_msg(f"Failed to write step2 log: {e}")

    def safe_cancel_control(self):
        # 동일 gRPC 커넥션에 대한 동시 gRPC 호출로 인한 C++ SDK Segfault 방지
        if self.robot is None:
            return
        try:
            import rby1_sdk as rby
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip().lower()
            temp_robot = rby.create_robot(addr, model)
            if temp_robot.connect():
                temp_robot.cancel_control()
                temp_robot.disconnect()
                self.log_msg("[INFO] Control cancelled safely via temporary connection.")
            else:
                self.robot.cancel_control()
        except Exception as e:
            try:
                self.robot.cancel_control()
            except Exception:
                pass

    def on_head_checkbox_changed(self, checked):
        self.include_head_motion = checked
        if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
            self.joint_calibrator.include_head_motion = checked
        if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
            self.marker_calibrator.include_head_motion = checked
        if hasattr(self, 'head_camera_calibrator') and self.head_camera_calibrator:
            self.head_camera_calibrator.include_head_motion = checked

        if hasattr(self, 'btn_step1_5_ready'):
            self.btn_step1_5_ready.setEnabled(checked)
        if hasattr(self, 'btn_step1_5_start'):
            self.btn_step1_5_start.setEnabled(checked)

        if hasattr(self, 'chk_servo_head'):
            self.chk_servo_head.blockSignals(True)
            self.chk_servo_head.setChecked(checked)
            self.chk_servo_head.blockSignals(False)

        if hasattr(self, 'wizard_widget') and hasattr(self.wizard_widget, 'wizard_chk_head'):
            self.wizard_widget.wizard_chk_head.blockSignals(True)
            self.wizard_widget.wizard_chk_head.setChecked(checked)
            self.wizard_widget.wizard_chk_head.blockSignals(False)

        self.sync_connection_settings('main')
        if hasattr(self, 'wizard_widget') and self.wizard_widget:
            self.wizard_widget.sync_bracket_radio()
        self.log_msg(f"[INFO] Additional camera bracket: {'YES' if not checked else 'NO'} (Head motion: {'ENABLED' if checked else 'DISABLED'})")

    def update_robot_connection_status(self):
        is_connected = self.robot is not None
        if is_connected:
            self.btn_connect.setText("CONNECTED")
            self.btn_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(True)
            if hasattr(self, 'wizard_widget') and self.wizard_widget:
                self.wizard_widget.btn_wizard_connect.setText("CONNECTED")
                self.wizard_widget.btn_wizard_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px;")
                self.wizard_widget.lbl_step2_status.setText("Status: Robot Connected" if not getattr(self, 'is_ko_ui', False) else "상태: 성공 - 로봇 연결 완료")
                self.wizard_widget.lbl_step2_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                self.wizard_widget.mark_step_completed(5, True, "Connected")
        else:
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(True)
            if hasattr(self, 'wizard_widget') and self.wizard_widget:
                self.wizard_widget.btn_wizard_connect.setText("CONNECT (로봇 연결)" if getattr(self, 'is_ko_ui', False) else "CONNECT")
                self.wizard_widget.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
                self.wizard_widget.lbl_step2_status.setText("Status: Disconnected" if not getattr(self, 'is_ko_ui', False) else "상태: 연결 해제됨")
                self.wizard_widget.lbl_step2_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
                self.wizard_widget.mark_step_completed(5, False, "Disconnected")

    def sync_connection_settings(self, source='main'):
        if not hasattr(self, 'wizard_widget') or not self.wizard_widget:
            return
        if source == 'main':
            if hasattr(self, 'ip_input') and hasattr(self.wizard_widget, 'wizard_ip_input'):
                self.wizard_widget.wizard_ip_input.blockSignals(True)
                self.wizard_widget.wizard_ip_input.setText(self.ip_input.text().strip())
                self.wizard_widget.wizard_ip_input.blockSignals(False)
            if hasattr(self, 'chk_servo_head') and hasattr(self.wizard_widget, 'wizard_chk_head'):
                self.wizard_widget.wizard_chk_head.blockSignals(True)
                self.wizard_widget.wizard_chk_head.setChecked(self.chk_servo_head.isChecked())
                self.wizard_widget.wizard_chk_head.blockSignals(False)
        elif source == 'wizard':
            if hasattr(self, 'ip_input') and hasattr(self.wizard_widget, 'wizard_ip_input'):
                self.ip_input.blockSignals(True)
                self.ip_input.setText(self.wizard_widget.wizard_ip_input.text().strip())
                self.ip_input.blockSignals(False)
            if hasattr(self, 'chk_servo_head') and hasattr(self.wizard_widget, 'wizard_chk_head'):
                self.chk_servo_head.blockSignals(True)
                self.chk_servo_head.setChecked(self.wizard_widget.wizard_chk_head.isChecked())
                self.chk_servo_head.blockSignals(False)

    def clear_user_taught_ready_poses(self, arm_side=None, mode=None):
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            if arm_side and mode:
                norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].pop(norm_mode, None)
            elif arm_side:
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].clear()
            else:
                self.user_taught_ready_poses.clear()
        if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
            self.marker_calibrator.clear_user_taught_ready_poses(arm_side, mode)
        if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
            self.joint_calibrator.clear_user_taught_ready_poses(arm_side, mode)

    def _on_marker_problem_requested(self, arm_side, evt, res):
        dlg = MarkerRecognitionProblemDialog(self, is_ko=getattr(self, 'is_ko_ui', False))
        self.marker_problem_dlg = dlg
        self.on_left_tab_changed(self.left_tabs.currentIndex())
        
        is_ko_ui = getattr(self, 'is_ko_ui', False)
        
        while True:
            resolved = (dlg.exec() == QDialog.Accepted)
            if not resolved:
                res['resolved'] = False
                break
                
            if self.robot:
                try:
                    active_mode = "marker"
                    if res.get('active_mode'):
                        active_mode = res['active_mode']
                    elif hasattr(self, 'joint_calibrator') and self.joint_calibrator and getattr(self.joint_calibrator, 'current_calib_mode', None):
                        active_mode = self.joint_calibrator.current_calib_mode
                    elif hasattr(self, 'marker_calibrator') and self.marker_calibrator and getattr(self.marker_calibrator, 'current_calib_mode', None):
                        active_mode = self.marker_calibrator.current_calib_mode
                    elif getattr(self, 'current_calib_mode', None):
                        active_mode = self.current_calib_mode
                    taught_pose, norm_mode, invalid_joints = prepare_taught_ready_pose(
                        self.robot, self.marker_calibrator, arm_side, active_mode, self.log_msg)

                    if len(invalid_joints) > 0:
                        err_lines = []
                        for i, val, low, upp in invalid_joints:
                            if is_ko_ui:
                                err_lines.append(f"- 관절 {i+1}: 현재 {val:.2f}°, 제한 범위 [{low:.1f}°, {upp:.1f}°]")
                            else:
                                err_lines.append(f"- Joint {i+1}: Current {val:.2f}°, Limits [{low:.1f}°, {upp:.1f}°]")
                        
                        if is_ko_ui:
                            title = "관절 가동 범위 초과"
                            msg = "수동으로 조정한 로봇의 자세가 가동 범위를 벗어났습니다. 다시 자세를 조정해주세요.\n\n" + "\n".join(err_lines)
                        else:
                            title = "Joint Limit Exceeded"
                            msg = "The manually adjusted robot posture exceeds the joint limits. Please adjust the posture again.\n\n" + "\n".join(err_lines)
                            
                        self.show_message_box(title, msg, QMessageBox.Warning)
                        continue  # Keep the teaching guide dialog open for readjustment

                    res['resolved'] = True
                    if not hasattr(self, 'user_taught_ready_poses') or not isinstance(self.user_taught_ready_poses, dict):
                        self.user_taught_ready_poses = {}
                    if arm_side not in self.user_taught_ready_poses or not isinstance(self.user_taught_ready_poses[arm_side], dict):
                        self.user_taught_ready_poses[arm_side] = {}

                    self.user_taught_ready_poses[arm_side][norm_mode] = taught_pose

                    if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
                        self.marker_calibrator.user_taught_ready_poses = self.user_taught_ready_poses

                    if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
                        self.joint_calibrator.user_taught_ready_poses = self.user_taught_ready_poses

                    self.log_msg(f"[INFO] Preserved user-taught ready pose for {arm_side} arm ({norm_mode}).")
                    break
                except Exception as e:
                    self.log_msg(f"[WARN] Failed to preserve user-taught ready pose: {e}")
                    res['resolved'] = False
                    break
            else:
                res['resolved'] = True
                break

        self.marker_problem_dlg = None
        self.on_left_tab_changed(self.left_tabs.currentIndex())
        evt.set()

    def prompt_marker_problem_teaching(self, arm_side, active_mode=None):
        import threading
        evt = threading.Event()
        res = {'resolved': False, 'active_mode': active_mode}
        self.marker_problem_signal.emit(arm_side, evt, res)
        evt.wait()
        return res['resolved']

    def connect_robot(self):
        from core.calibration.CalibratorBase import BaseCalibrator
        if self.camera_source_busy():
            self.log_msg('[ERROR] Finish active work and clear the sample session before changing robot connections.')
            return

        if self.robot:
            self.log_msg("[INFO] Disconnecting from robot...")
            MarkerCalibrator.terminate_robot(self.robot)
            self.marker_st.bind_robot(None, self.robot_version)
            self.robot = None
            self.model = None
            self.dyn_model = None
            self.robot_version = "1.2"
            self.marker_calibrator.robot = None
            self.marker_calibrator.robot_version = "1.2"
            self.joint_calibrator.robot = None
            self.joint_calibrator.robot_version = "1.2"
            self.head_camera_calibrator.robot = None
            self.head_camera_calibrator.robot_version = "1.2"
            self.update_joint_modes()
            self.load_offsets_from_yaml()
            self.update_applied_offset_label()
            self.update_robot_connection_status()
            if hasattr(self, 'chk_servo_head'):
                self.chk_servo_head.setEnabled(True)
            self.log_msg("[INFO] Robot disconnected.")
            return

        try:
            self.sync_connection_settings('main')
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip()
            
            # Read head checkbox state (like calibration_ui's servo_head)
            head_enabled = self.chk_servo_head.isChecked() if hasattr(self, 'chk_servo_head') else True
            self.include_head_motion = head_enabled
            if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
                self.marker_calibrator.app = self
                self.marker_calibrator.include_head_motion = head_enabled
            if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
                self.joint_calibrator.app = self
                self.joint_calibrator.include_head_motion = head_enabled
            
            # Update connection button to loading state
            self.btn_connect.setText("CONNECTING...")
            self.btn_connect.setStyleSheet("background-color: #ffb74d; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(False)
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            robot = BaseCalibrator.initialize_robot(
                addr, model, include_head=self.include_head_motion)
            if robot is None:
                raise ConnectionError(f'Failed to initialize robot at {addr}')

            self.robot = robot
                
            if self.robot:
                self.model = self.robot.model()
                self.dyn_model = self.robot.get_dynamics()
                self.marker_calibrator.robot = self.robot
                self.joint_calibrator.robot = self.robot
                self.head_camera_calibrator.robot = self.robot
                
                # Determine version classification automatically
                detected_version = "1.2"
                try:
                    robot_info = self.robot.get_robot_info()
                    actual_model_name = robot_info.robot_model_name
                    if actual_model_name.lower() != model.lower():
                        self.log_msg(f"[INFO] Auto-updating UI model selection to match robot model: '{actual_model_name}'")
                        found = False
                        for i in range(self.model_input.count()):
                            if self.model_input.itemText(i).lower() == actual_model_name.lower():
                                self.model_input.blockSignals(True)
                                self.model_input.setCurrentIndex(i)
                                self.model_input.blockSignals(False)
                                found = True
                                break
                        if not found:
                            self.model_input.blockSignals(True)
                            self.model_input.addItem(actual_model_name)
                            self.model_input.setCurrentIndex(self.model_input.count() - 1)
                            self.model_input.blockSignals(False)
                        
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
                # Cache the version classification on the app instance
                self.robot_version = detected_version
                if self.robot and hasattr(self.robot, "robot_version"):
                    self.robot.robot_version = detected_version
                if self.robot:
                    try:
                        self.robot.joint_offsets = self.joint_offsets
                    except AttributeError:
                        pass

                # Configure calibrators version
                self.marker_calibrator.robot_version = detected_version
                self.joint_calibrator.robot_version = detected_version
                self.head_camera_calibrator.robot_version = detected_version

                # Update UI modes and offsets based on version classification
                self.update_joint_modes()
                self.load_offsets_from_yaml()
                self.update_applied_offset_label()
                self.load_bracket_design_values()

                self.marker_st.bind_robot(self.robot, self.robot_version)
                if self.step2_mode_sel.currentText() != "npz":
                    self.step2_mode_sel.setCurrentText("live")

                self.log_msg(f"[INFO] Robot successfully connected and initialized (Classified Version: {detected_version}).")
                self.update_robot_connection_status()
            else:
                self.log_msg("[ERROR] Robot initialization failed. Check IP.")
                self.update_robot_connection_status()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_msg(f"[ERROR] Connection failure: {e}")
            self.update_robot_connection_status()

    def on_arm_side_changed(self, text):
        new_side = "left" if "Left" in text else "right"
        if self.arm_side != new_side:
            self.arm_side = new_side
            self.ready_done_joint = False
            self.ready_done_marker = False
            self.log_msg(f"[INFO] Changed active arm to {self.arm_side.upper()}. Cleared loaded datasets.")
            self.marker_data_4 = None
            self.marker_data_5 = None
            self.marker_data_6 = None
            self.joint_sweep_data = None
            
            # Sync current offsets with active arm_side from memory store (do not reload from yaml disk)
            is_v13 = self.get_robot_version() == "1.3"
            for arm in ["left", "right"]:
                self.joint_offsets[arm]["wrist_pitch"] = self.joint_offsets_store.get(arm, {}).get("joint5", 0.0)
                if is_v13:
                    self.joint_offsets[arm]["wrist_roll"] = self.joint_offsets_store.get(arm, {}).get("joint6", 0.0)
                    self.joint_offsets[arm]["wrist_yaw2"] = 0.0
                else:
                    self.joint_offsets[arm]["wrist_roll"] = 0.0
                    self.joint_offsets[arm]["wrist_yaw2"] = self.joint_offsets_store.get(arm, {}).get("joint6", 0.0)
                self.joint_offsets[arm]["elbow"] = self.joint_offsets_store.get(arm, {}).get("joint3", 0.0)
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

    def show_wizard_ui(self):
        if hasattr(self, 'overview_container') and self.overview_container:
            self.overview_container.setVisible(False)
        else:
            if hasattr(self, 'overview_title') and self.overview_title:
                self.overview_title.setVisible(False)
            if hasattr(self, 'overview_link') and self.overview_link:
                self.overview_link.setVisible(False)
            if hasattr(self, 'overview_duration') and self.overview_duration:
                self.overview_duration.setVisible(False)
            if self.overview_img:
                self.overview_img.setVisible(False)
            self.btn_start_wizard.setVisible(False)
        self.wizard_widget.setVisible(True)
        self.on_left_tab_changed(self.left_tabs.currentIndex())


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

    def _camera_capture_owned_by_worker(self):
        """Preview reuses sweep frames; teaching resumes GUI acquisition."""
        worker = getattr(self, 'active_worker', None)
        return (worker is not None and worker.isRunning()
                and getattr(self, 'marker_problem_dlg', None) is None)

    def poll_camera_status(self):
        if self.marker_st is None:
            return
        if self._camera_capture_owned_by_worker():
            return
        # Camera Tab이 켜져있을 때는 poll_camera_status 생략 (update_video_frame이 처리함)
        if self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1:
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
        except Exception:
            pass

    def poll_camera_temperature(self):
        if self.sim or self.marker_st is None:
            return
        try:
            if not self.is_temp_supported():
                text = self.get_temp_label_text()
                if hasattr(self, 'temp_label') and self.temp_label is not None:
                    self.temp_label.setText(text)
                if hasattr(self, 'lbl_temp') and self.lbl_temp is not None:
                    self.lbl_temp.setText(text)
                if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                    if hasattr(self.wizard_widget, 'lbl_temp') and self.wizard_widget.lbl_temp is not None:
                        self.wizard_widget.lbl_temp.setText(text)
                return

            if hasattr(self.marker_st, 'camera') and self.marker_st.camera is not None:
                temp = self.marker_st.camera.get_camera_temperature()
                self.last_temp = temp
                if temp is not None:
                    text = self.get_temp_label_text(temp)
                    if hasattr(self, 'temp_label') and self.temp_label is not None:
                        self.temp_label.setText(text)
                    if hasattr(self, 'lbl_temp') and self.lbl_temp is not None:
                        self.lbl_temp.setText(text)
                    if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                        if hasattr(self.wizard_widget, 'lbl_temp') and self.wizard_widget.lbl_temp is not None:
                            self.wizard_widget.lbl_temp.setText(text)
        except Exception:
            pass

    def is_any_camera_dialog_visible(self):
        feed_vis = (hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible())
        prob_vis = (hasattr(self, 'marker_problem_dlg') and self.marker_problem_dlg is not None)
        return feed_vis or prob_vis

    def _on_step1_subtab_changed(self, index):
        """Handle sub-tab switching within Step 1 (Main=0, Camera=1)."""
        if not hasattr(self, 'poll_timer') or not hasattr(self, 'video_timer'):
            return
        # Only act if Step 1 is the active top-level tab
        if self.left_tabs.currentIndex() != 1:
            return
        dialog_visible = self.is_any_camera_dialog_visible()
        if index == 1 or dialog_visible:  # Camera sub-tab
            if self.poll_timer.isActive():
                self.poll_timer.stop()
            self.video_timer.start(50)
        else:  # Main sub-tab
            if self.video_timer.isActive():
                self.video_timer.stop()
            if self.marker_st is not None:
                self.poll_timer.start(200)

    def is_wizard_video_active(self):
        if not hasattr(self, 'wizard_widget') or self.wizard_widget is None:
            return False
        if self.wizard_widget.isHidden():
            return False
        cur_idx = self.wizard_widget.stacked_widget.currentIndex()
        return cur_idx in [0, 2, 4]

    def on_left_tab_changed(self, index):
        # 방어적 코드: 타이머 객체가 아직 미생성된 상태이면 처리를 생략
        if not hasattr(self, 'poll_timer') or not hasattr(self, 'video_timer'):
            return

        if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
            self.wizard_widget.check_pose_init_done = False

        # Reparent shared widgets between Step 1 and Step 2
        self._reparent_shared_widgets(index)

        dialog_visible = self.is_any_camera_dialog_visible()

        if index == 1:  # Step 1 tab
            # Delegate to sub-tab handler
            self._on_step1_subtab_changed(self.step1_tabs.currentIndex())
        elif index == 2:  # Step 2 tab
            # Step 2 has no camera feed — stop video, start poll
            if self.video_timer.isActive():
                self.video_timer.stop()
            if dialog_visible:
                self.video_timer.start(50)
            elif self.marker_st is not None:
                self.poll_timer.start(200)
        else:
            # Overview tab (index 0) or others: check if wizard is on video slides (0, 2, 4)
            wizard_video_active = self.is_wizard_video_active()
            if wizard_video_active or dialog_visible:
                if self.poll_timer.isActive():
                    self.poll_timer.stop()
                self.video_timer.start(50)
            else:
                if self.video_timer.isActive():
                    self.video_timer.stop()
                if self.marker_st is not None:
                    self.poll_timer.start(200)

    def _on_workflow_tab_changed(self, index):
        """Automatically toggle Column 2 Dashboard between Arm/Marker and Head/Camera."""
        if hasattr(self, 'dash_stack') and self.dash_stack is not None:
            if index == 3:  # Head & Cam subtab
                self.dash_stack.setCurrentIndex(1)
            else:
                self.dash_stack.setCurrentIndex(0)

    def _reparent_shared_widgets(self, top_tab_index):
        """Move shared GroupBoxes between Step 1 Main and Step 2 layouts."""
        if not hasattr(self, 'conn_head_box') or self.conn_head_box is None:
            return
        if not hasattr(self, 'step2_left_col'):
            return

        if top_tab_index == 2:  # Switching TO Step 2
            # Move shared widgets into Step 2 layout
            self.step2_top_row.insertWidget(0, self.conn_head_box)
            self.step2_top_row.insertWidget(1, self.home_offset_box)
            self.step2_top_row.insertWidget(2, self.status_box)

            self.step2_right_col.insertWidget(0, self.log_box)
            self.step2_right_col.setStretchFactor(self.log_box, 1)

            if hasattr(self, 'btn_show_plot'):
                self.btn_show_plot.hide()

        else:  # Switching TO Step 1 (or any other tab)
            self._step1_col1.insertWidget(0, self.conn_head_box)
            self._step1_col2.insertWidget(0, self.home_offset_box)
            self._step1_col3.insertWidget(0, self.status_box)
            self._step1_col3.insertWidget(1, self.log_box)
            self._step1_col3.setStretchFactor(self.log_box, 1)

            if hasattr(self, 'btn_show_plot'):
                self.btn_show_plot.show()

    # =============================================
    # Head & Camera Extrinsics Action Handlers
    # =============================================

    def move_to_ready_pose_step1_5(self):
        if not getattr(self, 'include_head_motion', True):
            self.log_msg("[INFO] Headless mode: Robot has no head. Step 1.5 Head & Camera calibration is not required.")
            return

        self.log_msg("\n[Head & Camera] Moving to Ready Pose (Dual-arm Init Pose)...")
        self.btn_step1_5_ready.setEnabled(False)
        self.btn_step1_5_start.setEnabled(False)
        self.stop_requested = False
        if not hasattr(self, 'head_cam_stop_event') or self.head_cam_stop_event is None:
            self.head_cam_stop_event = threading.Event()
        self.head_cam_stop_event.clear()
        
        self.step1_5_ready_worker = HeadCamReadyWorker(self.head_camera_calibrator, self.head_cam_stop_event)
        self.step1_5_ready_worker.log_signal.connect(self.log_msg)
        self.step1_5_ready_worker.finished_signal.connect(self._on_step1_5_ready_finished)
        self.step1_5_ready_worker.start()

    def _on_step1_5_ready_finished(self, success, err_msg):
        self.btn_step1_5_ready.setEnabled(True)
        self.btn_step1_5_start.setEnabled(True)
        if success:
            self.log_msg("[Head & Camera] [SUCCESS] Ready Pose reached. Stationary arm markers verified.")
        else:
            self.log_msg(f"[Head & Camera] [ERROR] Ready pose move failed: {err_msg}")

    def start_calibration_step1_5(self):
        if not getattr(self, 'include_head_motion', True):
            self.log_msg("[INFO] Headless mode: Robot has no head. Step 1.5 Head & Camera calibration is not required.")
            return

        try:
            pan_r = float(self.step1_5_pan_range.text())
            tilt_r = float(self.step1_5_tilt_range.text())
            n_steps = int(self.step1_5_num_steps.text())
        except ValueError:
            self.log_msg("[ERROR] Invalid numeric parameters for Head Sweep.")
            return

        self.btn_step1_5_ready.setEnabled(False)
        self.btn_step1_5_start.setEnabled(False)
        self.btn_step1_5_apply.setEnabled(False)
        self.stop_requested = False
        if not hasattr(self, 'head_cam_stop_event') or self.head_cam_stop_event is None:
            self.head_cam_stop_event = threading.Event()
        self.head_cam_stop_event.clear()

        self.step1_5_sweep_worker = HeadCamSweepWorker(
            self.head_camera_calibrator, pan_r, tilt_r, n_steps, self.head_cam_stop_event
        )
        self.step1_5_sweep_worker.log_signal.connect(self.log_msg)
        self.step1_5_sweep_worker.finished_signal.connect(self._on_step1_5_sweep_finished)
        self.step1_5_sweep_worker.start()

    def _on_step1_5_sweep_finished(self, success, results):
        self.btn_step1_5_ready.setEnabled(True)
        self.btn_step1_5_start.setEnabled(True)
        if not success or not results or not results.get("success"):
            self.log_msg("[Head & Camera] [ERROR] Head Sweep Calibration failed or aborted.")
            return

        self.btn_step1_5_apply.setEnabled(True)
        self.log_msg("[Head & Camera] Calibration complete. Updating monitoring tables...")
        self._update_step1_5_tables(results)

    def _update_step1_5_tables(self, results):
        head_offsets = results.get("head_offsets_deg", {})
        pan_off = head_offsets.get("pan", 0.0)
        tilt_off = head_offsets.get("tilt", 0.0)

        self.tbl_step1_5_head_monitor.setItem(0, 0, QTableWidgetItem("0.000°"))
        self.tbl_step1_5_head_monitor.setItem(0, 1, QTableWidgetItem(f"{pan_off:+.3f}°"))

        self.tbl_step1_5_head_monitor.setItem(1, 0, QTableWidgetItem("0.000°"))
        self.tbl_step1_5_head_monitor.setItem(1, 1, QTableWidgetItem(f"{tilt_off:+.3f}°"))

        nom = results.get("nominal_mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        cal = results.get("calibrated_mount_to_cam", nom)

        rpy_rows = [
            ("Roll (°)", nom[3], cal[3]),
            ("Pitch (°)", nom[4], cal[4]),
            ("Yaw (°)", nom[5], cal[5]),
            ("X (mm)", nom[0]*1000.0, cal[0]*1000.0),
            ("Y (mm)", nom[1]*1000.0, cal[1]*1000.0),
            ("Z (mm)", nom[2]*1000.0, cal[2]*1000.0),
        ]
        for row_idx, (name, val_nom, val_cal) in enumerate(rpy_rows):
            self.tbl_step1_5_cam_monitor.setItem(row_idx, 0, QTableWidgetItem(f"{val_nom:+.3f}"))
            self.tbl_step1_5_cam_monitor.setItem(row_idx, 1, QTableWidgetItem(f"{val_cal:+.3f}"))

        q = results.get("quality", {})
        self.lbl_step1_5_rmse_tilt.setText(f"{q.get('rmse_tilt_plane_mm', 0.0):.3f} mm")
        self.lbl_step1_5_rmse_pan.setText(f"{q.get('rmse_pan_plane_mm', 0.0):.3f} mm")
        self.lbl_step1_5_ortho_err.setText(f"{q.get('ortho_error_deg', 0.0):.3f}°")

    def apply_results_step1_5(self):
        ok = self.head_camera_calibrator.apply_calibration_results(log_callback=self.log_msg)
        if ok:
            self.log_msg("[Head & Camera] [SUCCESS] Applied calibrated parameters to setting.yaml and memory.")
            self.btn_step1_5_apply.setEnabled(False)
        else:
            self.log_msg("[Head & Camera] [ERROR] Failed to apply calibration results.")

    # =============================================
    # Step 2 Action Handlers
    # =============================================

    def on_step2_mode_changed(self, text):
        self.log_msg(f"[INFO] Step 2 Mode changed to: '{text}'")
        self.update_step2_est_samples()

    def update_step2_est_samples(self, *args):
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        if QThread.currentThread() != QApplication.instance().thread():
            self.update_ui_signal_safe.emit("samples")
            return
        try:
            p = float(self.step2_pos_step.text())
            step_x = float(self.step2_step_x.text())
            m = float(self.step2_max_x.text())
            if p <= 0 or step_x <= 0:
                return

            self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
            self.auto_config.position_step_m = p
            self.auto_config.step_x_m = step_x
            self.auto_config.max_x = m
            self.auto_config.max_loops = 1
            from core.robot_motion import estimate_collection_samples
            count, current_x, approximate = estimate_collection_samples(
                self.robot, self.dyn_model, self.auto_config, self.include_head_motion)
            suffix = '(approx)' if approximate else f'(from X={current_x:.3f})'
            self.step2_est_samples_label.setText(f'Est. Samples: {count} {suffix}')
        except:
            pass

    def get_capture_head_idx(self):
        if self.model is None:
            return None
        return get_head_config(self.model)["head_idx"]

    def get_active_arms(self):
        arm_text = self.arm_sel.currentText()
        if "Left" in arm_text:
            return ["left"]
        elif "Right" in arm_text:
            return ["right"]
        return ["right", "left"]

    def get_target_arm_str(self):
        active_arms = self.get_active_arms()
        if len(active_arms) == 1:
            return active_arms[0]
        return "both"

    def ensure_home_offset_robot(self):
        if self.robot is None or self.model is None:
            self.log_msg("[INFO] Robot is not connected. Connecting before home offset operation...")
            self.connect_robot()
        if self.robot is None or self.model is None:
            raise RuntimeError("Robot is not connected.")

    def resolve_input_path(self, raw_path):
        input_path = Path(raw_path).expanduser()
        if input_path.is_absolute():
            return input_path
        return Paths().root / input_path

    def ensure_result_dir(self):
        path = Path(CONFIG_PATHS["result_dir"])
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_output_paths(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.ensure_result_dir()
        dataset_path = result_dir / f"dataset_{timestamp}.npz"
        result_path = result_dir / f"result_{timestamp}.json"
        return dataset_path, result_path

    def format_home_offset_compare_summary(self, result_path, baseline_path):
        lines = [
            "Preview moves use the same convention as Apply Home Offset:",
            "the robot moves to zero pose first, then to -joint_offset.",
            "",
            f"Optimized result: {result_path if result_path else 'None'}",
            f"Baseline reset: {baseline_path if baseline_path else 'None'}",
            ""
        ]

        from core.homeoffset_core import compare_home_offset_files
        comparison = compare_home_offset_files(result_path, baseline_path)

        def format_section(title, opt, base, diff):
            section = [f"--- {title} ---"]
            if base is not None and len(base) > 0:
                section.append(f"  Baseline  : {np.array2string(np.round(base, 4), separator=', ')}")
            else:
                section.append(f"  Baseline  : Unavailable")
                
            if opt is not None and len(opt) > 0:
                section.append(f"  Optimized : {np.array2string(np.round(opt, 4), separator=', ')}")
            else:
                section.append(f"  Optimized : Unavailable")
                
            if base is not None and opt is not None and len(base) == len(opt) and len(base) > 0:
                section.append(f"  Diff (B-O): {np.array2string(np.round(diff, 4), separator=', ')}")
            else:
                section.append(f"  Diff (B-O): Unavailable")
            section.append("")
            return section

        lines.extend(format_section("RIGHT ARM (deg)", comparison['right']['optimized'], comparison['right']['baseline'], comparison['right']['difference']))
        lines.extend(format_section("LEFT ARM (deg)", comparison['left']['optimized'], comparison['left']['baseline'], comparison['left']['difference']))
        lines.extend(format_section("HEAD (deg)", comparison['head']['optimized'], comparison['head']['baseline'], comparison['head']['difference']))

        return "\n".join(lines)
        return "\n".join(lines)

    def infer_home_offset_apply_arm(self, requested_arm, json_path):
        try:
            offset_rad, _ = load_offset_from_json(str(json_path))
        except Exception:
            return requested_arm

        if self.model is not None:
            both_dof = len(self.model.right_arm_idx) + len(self.model.left_arm_idx)
            if len(offset_rad) == both_dof:
                return "both"

        if len(offset_rad) == 14:
            return "both"
        return requested_arm

    def move_home_offset_candidate_path(self, json_path, label, arm, include_head):
        self.ensure_home_offset_robot()
        result = move_to_offset_candidate_from_json(
            robot=self.robot,
            model=self.model,
            arm=arm,
            json_path=str(json_path),
            include_head=include_head,
            minimum_time=10,
            move_zero_first=True,
        )

        self.log_msg(f"\n===== HOME OFFSET PREVIEW: {label} =====")
        self.log_msg(f"JSON: {json_path}")
        self.log_msg(f"Arm: {result['arm']}")
        if result.get("right_offset_deg") is not None:
            self.log_msg(f"Right move offset (deg): {result['right_offset_deg']}")
        if result.get("left_offset_deg") is not None:
            self.log_msg(f"Left move offset (deg): {result['left_offset_deg']}")
        if result.get("head_offset_deg") is not None:
            self.log_msg(f"Head move offset (deg): {result['head_offset_deg']}")
        self.log_msg("Preview move complete. Inspect the robot pose before applying.")
        return result

    def move_to_check_position_candidate_path(self, json_path, label, arm, include_head, skip_init_pose=False):
        self.ensure_home_offset_robot()
        from core.homeoffset_core import move_to_check_position_candidate_path
        return move_to_check_position_candidate_path(self.robot, self.model, json_path,
            label, arm, include_head, skip_init_pose, log_callback=self.log_msg)

    def apply_current_pose_home_offset(self, arm, include_head, json_path=None):
        self.ensure_home_offset_robot()
        from core.homeoffset_core import apply_current_pose_home_offset
        return apply_current_pose_home_offset(self.robot, self.model, arm, include_head,
            json_path=json_path, setting_path=CONFIG_PATHS['setting_yaml'], log_callback=self.log_msg)

    def collection_service(self, log_callback=None, progress_callback=None, sample_callback=None):
        from copy import deepcopy
        if not hasattr(self, '_auto_collection_stop_event'):
            self._auto_collection_stop_event = threading.Event()
        if self.auto_stop_requested:
            self._auto_collection_stop_event.set()
        else:
            self._auto_collection_stop_event.clear()
        state = CollectionState(
            motion_plan=self.auto_motion_plan, pose_index=self.head_move_count,
            ready=self.auto_ready_done, base_head_q=self.auto_base_head_q,
            arm_samples=self.shared_arm_q_list, head_samples=self.shared_head_q_list,
            marker_samples=self.shared_T_list)
        return AutoCollectionService(
            self.robot, self.model, self.dyn_model, self.marker_st, deepcopy(self.auto_config), state,
            robot_version=self.get_robot_version(), include_head_motion=self.include_head_motion,
            capture_head_idx=self.get_capture_head_idx(), stop_callback=self._auto_collection_stop_event.is_set,
            log_callback=log_callback or self.log_msg, progress_callback=progress_callback,
            sample_callback=sample_callback)

    def publish_collection_state(self, state):
        self.auto_motion_plan = state.motion_plan
        self.head_move_count = state.pose_index
        self.auto_base_head_q = state.base_head_q
        self.update_sample_counts()
        self.update_head_pose_status()
        self.update_step2_est_samples()

    def run_auto_motion_step_blocking(self):
        service = self.collection_service(sample_callback=self.log_captured_sample)
        try:
            return service.step()
        finally:
            self.include_head_motion = service.include_head_motion
            self.publish_collection_state(service.state)

    def move_to_all_auto_motions(self):
        if not self.auto_ready_done:
            self.auto_ready_done = True

        if self.step2_mode_sel.currentText() != 'live':
            self.step2_mode_sel.setCurrentText("live")

        try:
            if hasattr(self, 'step2_angle_step'): self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
            if hasattr(self, 'step2_pos_step'): self.auto_config.position_step_m = float(self.step2_pos_step.text())
            if hasattr(self, 'step2_step_x'): self.auto_config.step_x_m = float(self.step2_step_x.text())
            if hasattr(self, 'step2_max_x'): self.auto_config.max_x = float(self.step2_max_x.text())
            self.auto_config.max_loops = 1
        except Exception as e:
            self.log_msg(f"Failed to read auto config: {e}. Using current values.")

        if self.auto_motion_plan is None or len(self.auto_motion_plan) == 0:
            self.log_msg("Motion plan is missing or empty. Re-building...")
            active_arms = ["right", "left"]
            try:
                self.auto_motion_plan = self.collection_service().build_plan()
            except Exception as e:
                self.log_msg(f"[ERROR] Failed to build motion plan: {e}")
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Motion Plan Error")
                msg_box.setText(f"Failed to build motion plan:\n{e}")
                msg_box.setStyleSheet("QLabel { min-height: 70px; font-size: 13px; padding: 12px; } QMessageBox { min-width: 500px; }")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec()
                return

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log_msg("Auto motions have already been executed.")
            return

        if self.auto_motion_running or self.auto_motion_thread is not None:
            self.log_msg("Another robot operation is already running.")
            return

        self.auto_stop_requested = False
        self.auto_motion_running = True
        self.log_msg("Auto Motion started in a background thread. Press Stop to cancel.")

        self.auto_motion_thread = Step2AutoMotionWorker(self.collection_service(), parent=self)
        self.auto_motion_thread.state_signal.connect(self.publish_collection_state, Qt.QueuedConnection)
        self.auto_motion_thread.captured_signal.connect(self.log_captured_sample, Qt.QueuedConnection)
        self.auto_motion_thread.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            self.auto_motion_running = False
            self.auto_motion_thread = None
            self.auto_save_current_dataset()
            if success:
                self.log_msg("Auto motions sequence completed.")
                self.step2_calculate()
            else:
                self.log_msg(f"Auto motion error: {error_msg}")
        self.auto_motion_thread.finished_signal.connect(on_finished)
        self.auto_motion_thread.start()

    def stop_all_auto_motion_internal(self, cancel_robot=False, reset_stop_requested=True):
        if hasattr(self, '_auto_collection_stop_event'):
            self._auto_collection_stop_event.set()
        self.auto_motion_running = False
        if reset_stop_requested:
            self.auto_stop_requested = False

        if cancel_robot:
            self.safe_cancel_control()

        # 백그라운드 스레드가 완전히 종료되어 finished_signal을 통해 on_finished()가 호출될 때까지
        # self.auto_motion_thread를 None으로 설정하지 않고 유지합니다.
        # 이를 통해 이전 작업이 완전히 끝나지 않은 상태에서 새로운 로봇 명령이 병렬로 전송되는 것을 방지합니다.
        pass

        reset_motion_state()

    def request_stop_all_auto_motion(self):
        if hasattr(self, '_auto_collection_stop_event'):
            self._auto_collection_stop_event.set()
        if not self.auto_motion_running and self.auto_motion_thread is None:
            self.log_msg("No Auto Motion sequence is running.")
            self.stop_all_auto_motion_internal(cancel_robot=True)
            return

        self.auto_stop_requested = True
        self.stop_all_auto_motion_internal(cancel_robot=True, reset_stop_requested=False)
        self.log_msg("Stop requested. Sent robot.cancel_control(); the auto motion sequence stops after the current step.")

    def capture_metadata(self):
        return build_capture_metadata(
            self.marker_calibrator.camera_config, self.get_robot_version(), self.include_head_motion,
            source_metadata=self.marker_st.simulation_model.metadata() if self.sim else None,
            intrinsics=getattr(self.marker_st, 'intrinsics_metadata', None))

    def auto_save_current_dataset(self):
        if len(self.shared_arm_q_list) == 0:
            return
        
        try:
            q_arm_list, q_head_list, T_meas_list, active_arms = prepare_sample_dataset(
                self.shared_arm_q_list, self.shared_head_q_list, self.shared_T_list,
                active_arms=self.get_active_arms(), optimize_head=self.include_head_motion)

            if not self.dataset_saved_in_session or self.current_session_dataset_path is None:
                dataset_path, _ = self.build_output_paths()
                self.current_session_dataset_path = dataset_path
                self.dataset_saved_in_session = True
                
            metadata = self.capture_metadata()
            save_npz_dataset(self.current_session_dataset_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list, metadata=metadata)
            self.last_dataset_path = self.current_session_dataset_path
            self.log_msg(f"[Auto-Save] Dataset saved/updated in: {self.current_session_dataset_path}")
        except Exception as e:
            self.log_msg(f"[Auto-Save Error] {e}")

    def update_sample_counts(self):
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        if QThread.currentThread() != QApplication.instance().thread():
            self.update_ui_signal_safe.emit("sample_counts")
            return
        sample_count = len(self.shared_arm_q_list)
        if hasattr(self, 'step2_sample_count_label'):
            self.step2_sample_count_label.setText(f"Shared Samples: {sample_count}")

    def get_auto_pose_target_count(self):
        if self.auto_motion_plan is not None:
            return len(self.auto_motion_plan)
        return 0

    def update_head_pose_status(self):
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        if QThread.currentThread() != QApplication.instance().thread():
            self.update_ui_signal_safe.emit("head_pose")
            return
        pose_target_count = self.get_auto_pose_target_count()
        pose_idx = min(self.head_move_count, pose_target_count)
        label = f"Auto Motion: {pose_idx}/{pose_target_count}"
        if not self.include_head_motion:
            label += " (headless)"
        if hasattr(self, 'step2_head_status_label'):
            self.step2_head_status_label.setText(label)

    def capture_one_sample(self, motion_plan_step=None):
        q_arm, q_head, T_meas = capture_calibration_sample(
            self.robot, self.model, self.marker_st, robot_version=self.get_robot_version(),
            head_idx=self.get_capture_head_idx())
        if T_meas is None:
            self.log_msg("Marker not detected.")
            return None, None, None

        self.log_captured_sample(CapturedSample(
            len(self.shared_arm_q_list) + 1, q_arm, q_head, T_meas))
        return q_arm, q_head, T_meas

    def log_captured_sample(self, sample):
        q_arm, q_head, T_meas = sample.q_arm, sample.q_head, sample.marker
        status_lines = []
        status_lines.append(f"q_arm = {np.round(q_arm, 3)}")
        if q_head is not None:
            status_lines.append(f"q_head = {np.round(q_head, 3)}")
        else:
            status_lines.append("q_head = None")
        status_lines.append(f"marker_right =\n{np.round(T_meas[0], 3)}")
        status_lines.append(f"marker_left =\n{np.round(T_meas[1], 3)}")
        
        status_str = "\n".join(status_lines)
        self.log_msg(f"[Sample {sample.ordinal}] Captured marker: R_pos={np.round(T_meas[0][:3, 3]*1000, 1)}mm, L_pos={np.round(T_meas[1][:3, 3]*1000, 1)}mm")
        self._write_step2_log("--- Captured Sample ---\n" + status_str + "\n")
        

    def optimizer_context(self):
        """Snapshot UI selection and domain values before dispatching background work."""
        from copy import deepcopy
        live_sim = self.sim and self.step2_mode_sel.currentText() == 'live'
        return OptimizerContext(
            robot=self.robot, model=self.model,
            camera_config=deepcopy(self.marker_calibrator.camera_config),
            robot_version=self.get_robot_version(), include_head_motion=self.include_head_motion,
            joint_offsets_store=deepcopy(getattr(self, 'joint_offsets_store', {})),
            apply_joint_offset_limits=getattr(self, 'apply_joint_offset_flag', False),
            head_camera_result=deepcopy(getattr(getattr(self, 'head_camera_calibrator', None), 'calibrated_results', None)),
            capture_metadata=deepcopy(getattr(self, '_loaded_dataset_metadata', None) or self.capture_metadata()),
            home_reset_baseline_path=self.last_home_reset_path,
            comparison_baseline_path=CONFIG_PATHS['home_reset_baseline'],
            simulation_offsets=deepcopy(self.marker_st.simulation_model.config['offsets']) if live_sim else None,
            nominal_brackets=deepcopy(self.marker_calibrator.NOMINAL_BRACKET_TEMPLATES))

    def run_optimizer(self, active_arms, optimize_head, optimize_camera,
                      q_arm_list, q_head_list, T_meas_list, result_path,
                      lambda_cam_pos=1.0, lambda_cam_rot=1e6,
                      solver_type="QP Solver", use_sag=False):
        result = run_calibration_optimizer(
            self.optimizer_context(), active_arms, optimize_head, optimize_camera,
            q_arm_list, q_head_list, T_meas_list, result_path,
            lambda_cam_pos, lambda_cam_rot, solver_type, use_sag,
            log_callback=self.log_msg)
        self.last_result_path = result_path
        return result

    def _on_apply_joint_offset_toggled(self, checked):
        """Toggle apply joint offset flag and update status label."""
        self.apply_joint_offset_flag = checked
        if checked:
            self.lbl_jo_status.setText("ACTIVE")
            self.lbl_jo_status.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_jo_status.setText("INACTIVE")
            self.lbl_jo_status.setStyleSheet("color: #ff1744; font-weight: bold; font-size: 11px;")
        self.log_msg(f"[INFO] Apply Joint Offset: {'ACTIVE' if checked else 'INACTIVE'}")

    def step2_zero_pose_check(self):
        self.log_msg("[Step2] Zero Pose Check requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        
        arm = "both"
        
        self.zero_pose_worker = Step2ZeroPoseCheckWorker(
            self.robot,
            self.model,
            arm,
            self.include_head_motion
        )
        self.zero_pose_worker.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            if success:
                self.log_msg("\n===== ZERO POSE CHECK COMPLETE =====")
                dialog = ZeroPoseCheckDialog(self)
                dialog.exec()
            else:
                self.log_msg(f"Zero pose check failed: {error_msg}")
        self.zero_pose_worker.finished_signal.connect(on_finished)
        self.zero_pose_worker.start()

    def move_to_zero_pose(self):
        self.log_msg("[Wizard 3-1] Move to Zero Pose requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return False
            
        self.zero_pose_worker = Step2ZeroPoseCheckWorker(
            self.robot,
            self.model,
            "both",
            self.include_head_motion
        )
        self.zero_pose_worker.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                self.wizard_widget.set_wizard_busy(False)
                if success:
                    self.wizard_widget.mark_step_completed(6, True, "Moved to Zero Position")
                else:
                    self.wizard_widget.mark_step_completed(6, False, error_msg)

            if success:
                self.log_msg("[Wizard 3-1] Robot moved to zero pose successfully.")
            else:
                self.log_msg(f"[Wizard 3-1] Failed to move to zero pose: {error_msg}")
        self.zero_pose_worker.finished_signal.connect(on_finished)
        self.zero_pose_worker.start()
        return True

    def step2_stop_auto_motion(self):
        self.log_msg("[Step2] Stop Auto Motion requested.")
        self.request_stop_all_auto_motion()
        self.auto_save_current_dataset()

    def on_capture_head_centered(self, head_pose):
        self.auto_base_head_q = head_pose.copy()
        self.auto_motion_plan = None

    def step2_init_pose(self, silent=False):
        self.log_msg("[Step2] Init Pose requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        if self.auto_motion_running or self.auto_motion_thread is not None:
            if not silent:
                QMessageBox.critical(self, "Execution Error", "Another robot operation is currently running.")
            return
            
        self.auto_motion_running = True
        active_arms = ["right", "left"]
        self.auto_motion_thread = Step2InitPoseWorker(
            self.robot,
            active_arms,
            self.auto_config.priority if hasattr(self, 'auto_config') else 0,
            include_head_motion=self.include_head_motion,
            parent=self
        )
        self.auto_motion_thread.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            self.auto_motion_running = False
            self.auto_motion_thread = None
            if success:
                self.auto_ready_done = True
                try:
                    if hasattr(self, 'step2_angle_step'):
                        self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
                    if hasattr(self, 'step2_pos_step'):
                        self.auto_config.position_step_m = float(self.step2_pos_step.text())
                    if hasattr(self, 'step2_step_x'):
                        self.auto_config.step_x_m = float(self.step2_step_x.text())
                    if hasattr(self, 'step2_max_x'):
                        self.auto_config.max_x = float(self.step2_max_x.text())
                    self.auto_config.max_loops = 1
                except Exception as e:
                    self.log_msg(f"Failed to read auto config: {e}. Using default values.")

                self.auto_motion_plan = None
                if self.include_head_motion and self.robot:
                    from core.robot_motion import current_head_pose
                    self.auto_base_head_q = current_head_pose(self.robot, self.model)
                    if self.auto_base_head_q is not None:
                        self.log_msg(f"Auto base head pose (deg): {np.round(np.rad2deg(self.auto_base_head_q), 3)}")

                self.head_move_count = 0
                self.update_head_pose_status()
                self.update_step2_est_samples()
                
                if not silent:
                    self.show_message_box(
                        "Teaching Required",
                        "Robot has moved to the initial pose.\n\n"
                        "Please adjust the robot's pose so that the marker is clearly visible to the camera.\n"
                        "Once adjusted, press '2) Auto Motion' to start the sequence.",
                        QMessageBox.Information
                    )
            else:
                self.log_msg(f"Init pose failed: {error_msg}")
                if not getattr(self, 'auto_stop_requested', False) and not silent:
                    QMessageBox.critical(self, "Init Error", error_msg)
        
        self.auto_motion_thread.finished_signal.connect(on_finished)
        self.auto_motion_thread.start()

    def step2_auto_motion(self):
        self.log_msg("[Step2] Auto Motion requested.")
        mode = self.step2_mode_sel.currentText()
        if mode not in ["live"]:
            self.log_msg("[Step2] Auto motion is only available in live acquisition mode.")
            return
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        try:
            self.move_to_all_auto_motions()
        except Exception as e:
            QMessageBox.critical(self, "Auto Motion Error", str(e))
            self.log_msg(f"Auto motion failed: {e}")

    def step2_calculate(self):
        self.log_msg("[Step2] Calculate requested.")
        
        # Check if calculation is already running
        if hasattr(self, 'calc_worker') and self.calc_worker is not None and self.calc_worker.isRunning():
            self.log_msg("[Step2] [WARNING] Optimization calculation is ALREADY running!")
            QMessageBox.information(
                self,
                "Calculation In Progress",
                "Optimization calculation is already running in the background.\n\nPlease wait for the current calculation to complete."
            )
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        try:
            mode = self.step2_mode_sel.currentText().strip()
            self.log_msg(f"\n[Step2] Calculate requested (Active Mode: '{mode}').")
            active_arms = ["right", "left"]
            camera_cfg = getattr(self.marker_calibrator, "camera_config", {})
            from core.marker_detection import uses_head_camera
            optimize_head = self.include_head_motion and uses_head_camera(camera_cfg, self.model)
            optimize_camera = camera_cfg.get("extrinsic_source") != "independent_measurement"
            if optimize_head and optimize_camera:
                convention = camera_cfg.get('head_zero_convention', 'camera_forward')
                self.log_msg(f"[GAUGE] Head output convention: {convention}. Camera remains estimated; terminal Tilt/camera redistribution preserves the full SE(3) chain. Independent references, if supplied, are retained.")
            elif not optimize_head:
                self.log_msg("[INFO] Head motion/offset estimation disabled. Camera extrinsics are estimated unless independently measured.")
            # No CAD-centred camera penalty: physical bounds protect the solve.
            lambda_cam_pos = 0.0
            lambda_cam_rot = 0.0

            if mode in ["live"]:
                self._loaded_dataset_metadata = None
                if len(self.shared_arm_q_list) == 0:
                    QMessageBox.warning(self, "Warning", "No recorded samples in memory.")
                    return

                self.log_msg(f"[Step2] Using live recorded dataset ({len(self.shared_arm_q_list)} samples in memory).")
                q_arm_list, q_head_list, T_meas_list, active_arms = select_calibration_dataset(
                    self.shared_arm_q_list, self.shared_head_q_list, self.shared_T_list, active_arms)

            elif mode == "npz":
                npz_raw = self.step2_path_input.text().strip()
                npz_path = self.resolve_input_path(npz_raw)
                self.log_msg(f"[Step2] Loading NPZ dataset from: {npz_path}")
                q_arm_list, q_head_list, T_meas_list, self._loaded_dataset_metadata = load_npz_dataset(npz_path, return_metadata=True)
                if self._loaded_dataset_metadata.get('schema_version') == 0:
                    self.log_msg('[WARN] Legacy dataset: original camera/bracket truth and intrinsics provenance are unknown. GT accuracy claims are disabled.')
                self.log_msg(f"[Step2] Loaded {len(q_arm_list)} samples from NPZ dataset.")

                q_arm_list, q_head_list, T_meas_list, active_arms = select_calibration_dataset(
                    q_arm_list, q_head_list, T_meas_list, active_arms)

            dataset_path, result_path = self.build_output_paths()
            
            # Disable calculate button and update UI status
            if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
                self.btn_step2_calculate.setEnabled(False)
                self.btn_step2_calculate.setText("Calculating...")
                self.btn_step2_calculate.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold;")

            self.log_msg("[Step2] Optimization calculation started in background thread...")

            self.calc_worker = Step2CalculateWorker(
                self,
                active_arms,
                optimize_head,
                optimize_camera,
                q_arm_list,
                q_head_list,
                T_meas_list,
                result_path,
                lambda_cam_pos,
                lambda_cam_rot
            )
            self.calc_worker.log_signal.connect(self.log_msg)
            
            def on_finished(success, error_msg):
                if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
                    self.btn_step2_calculate.setEnabled(True)
                    self.btn_step2_calculate.setText("3) Calculate")
                    self.btn_step2_calculate.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")

                if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                    self.wizard_widget.stop_step5(success, error_msg if not success else "")

                if success:
                    self.last_result_path = result_path
                    self.log_msg("Optimization finished successfully.")
                    QMessageBox.information(self, "Step 2 Calculation", "Optimization finished successfully!\nCheck the logs and Result Output for details.")
                else:
                    self.log_msg(f"[Error] Optimization failed: {error_msg}")
                    QMessageBox.warning(self, "Step 2 Calculation Failed", f"Optimization failed:\n{error_msg}")

            self.calc_worker.finished_signal.connect(on_finished)
            self.calc_worker.start()
            
        except Exception as e:
            if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
                self.btn_step2_calculate.setEnabled(True)
                self.btn_step2_calculate.setText("3) Calculate")
                self.btn_step2_calculate.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
            QMessageBox.critical(self, "Calculate Error", str(e))
            self.log_msg(f"Calculate failed: {e}")
            QMessageBox.critical(self, "Calculate Error", str(e))
            self.log_msg(f"Calculate failed: {e}")

    def step2_clear_samples(self):
        self.log_msg("[Step2] Clear Samples requested.")
        self.stop_all_auto_motion_internal(cancel_robot=True)
        reset_motion_state()
        self.shared_arm_q_list.clear()
        self.shared_head_q_list.clear()
        self.shared_T_list.clear()
        self.head_move_count = 0
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.dataset_saved_in_session = False
        self.current_session_dataset_path = None
        self.update_sample_counts()
        self.update_head_pose_status()
        self.log_msg("Shared samples cleared.")

    def step2_apply_home_offset(self):
        self.log_msg("[Step2] Apply Home Offset requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        try:
            result_path = self.get_latest_result_path()
            baseline_path = self.get_home_reset_path_for_result(result_path)
            arm = "both"
            
            compare_summary = self.format_home_offset_compare_summary(result_path, baseline_path)
            
            if hasattr(self, 'apply_offset_dialog') and self.apply_offset_dialog is not None and self.apply_offset_dialog.isVisible():
                self.apply_offset_dialog.raise_()
                self.apply_offset_dialog.activateWindow()
                return

            self.apply_offset_dialog = ApplyHomeOffsetDialog(
                parent=self,
                result_path=result_path,
                baseline_path=baseline_path,
                arm=arm,
                include_head=self.include_head_motion,
                compare_summary=compare_summary
            )
            self.apply_offset_dialog.show()
            self.apply_offset_dialog.raise_()
            self.apply_offset_dialog.activateWindow()
            
        except Exception as e:
            QMessageBox.critical(self, "Apply Home Offset Error", str(e))
            self.log_msg(f"Apply home offset failed: {e}")

    def step2_check_calibration_state(self):
        self.log_msg("[Step2] Check Calibration State requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        dialog = CheckCalibrationStateDialog(self)
        dialog.exec()

    def update_applied_offset_label(self):
        self.ready_done_joint = False
        if not hasattr(self, 'tbl_offset_monitor') or not hasattr(self, 'btn_joint_apply'):
            return
        
        is_v13 = self.get_robot_version() == "1.3"
        self.tbl_offset_monitor.setColumnCount(3)
        if is_v13:
            self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Roll)", "Joint 5 (Pitch)", "Joint 3 (Elbow)"])
        else:
            self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Yaw 2)", "Joint 5 (Wrist Pitch)", "Joint 3 (Elbow)"])
            
        for row_idx, arm in enumerate(["right", "left"]):
            for col_idx, joint_key in enumerate(["joint6", "joint5", "joint3"]):
                val = self.joint_offsets_store.get(arm, {}).get(joint_key, 0.0)
                item = QTableWidgetItem(f"{val:.4f}°")
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl_offset_monitor.setItem(row_idx, col_idx, item)
        
    def apply_joint_offset(self):
        failed = getattr(self, 'joint_sweep_data', None) or getattr(self, '_failed_joint_result', None)
        if failed and not _joint_result_accepted(failed):
            side = failed.get('arm_side', getattr(self, 'arm_side', 'right'))
            key = _joint_store_key(failed.get('mode', 'elbow'))
            candidate = failed.get('recommended_joint_offset', failed.get('optimal_offset'))
            staged = self.joint_offsets_store.get(side, {}).get(key)
            if candidate is not None and staged == candidate:
                self.log_msg('[ERROR] Unaccepted or unconverged calibration candidate cannot be applied.')
                return False
        if not self.save_offsets_to_yaml():
            return False
        self._publish_joint_offsets(self._joint_offset_patch()["joint_offset"])
        self.log_msg("[APPLY] Joint offsets saved and applied for both arms.")
        return True


    def stop_motion(self):
        self.log_msg("[STOP] Stop requested by user.")
        
        # 1. Stop Step 1 sweep calibrations
        self.joint_calibrator.stop_requested = True
        self.marker_calibrator.stop_requested = True
        if hasattr(self, 'stop_event_mc') and self.stop_event_mc:
            self.stop_event_mc.set()
            
        # 2. Stop Step 2 auto collection/motion
        if self.auto_motion_running or self.auto_motion_thread is not None:
            self.request_stop_all_auto_motion()
            self.auto_save_current_dataset()
        else:
            # If not running Step 2 auto motion, we still want to make sure
            # any robot motion is cancelled if robot is connected.
            if self.robot:
                self.log_msg("[STOP] Sending cancel_control to robot!")
                self.safe_cancel_control()
            else:
                self.log_msg("[STOP] No robot connected to cancel control.")
                
        # 3. Stop Full Auto calibration
        if hasattr(self, 'full_auto_stop_event') and self.full_auto_stop_event:
            self.full_auto_stop_event.set()

        # 4. Stop Head & Camera calibration
        if hasattr(self, 'head_cam_stop_event') and self.head_cam_stop_event:
            self.head_cam_stop_event.set()

    def clear_joint_offset(self):
        reply = QMessageBox.question(self, "Clear Joint Offset",
            "Reset all staged/saved joint offsets for BOTH arms and head to 0.0?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return False
        from core.config_store import update_yaml
        updates = {"joint_offset": {
            side: {key: 0.0 for key in ("joint3", "joint5", "joint6")}
            for side in ("left", "right")}}
        updates["joint_offset"]["head"] = {"pan": 0.0, "tilt": 0.0}
        try:
            update_yaml(CONFIG_PATHS["setting_yaml"], updates)
        except Exception as exc:
            self.log_msg(f"[ERROR] Failed to clear saved offsets: {exc}")
            return False
        for side, values in updates["joint_offset"].items():
            self.joint_offsets_store.setdefault(side, {}).update(values)
        self._publish_joint_offsets(updates["joint_offset"])
        if getattr(self, "head_camera_calibrator", None) is not None:
            self.head_camera_calibrator.calibrated_results = None
        self.log_msg("[CLEAR] Staged and saved offsets cleared to 0.0 for BOTH Arms and Head.")
        return True

    def load_bracket_design_values(self):
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                    is_v13 = self.get_robot_version() == "1.3"
                    marker_data = data.get("marker", {})
                    # Load Left Arm values
                    val_left = marker_data.get("Tf_to_marker_left", None)
                    if val_left and len(val_left) == 6:
                        if is_v13 and abs(val_left[0]) < 0.05:
                            val_left = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.3"]["left"]
                        elif not is_v13 and abs(val_left[0]) > 0.05:
                            val_left = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.2"]["left"]
                        set_numeric_field(self.txt_bracket_l_x, val_left[0])
                        set_numeric_field(self.txt_bracket_l_y, val_left[1])
                        set_numeric_field(self.txt_bracket_l_z, val_left[2])
                        set_numeric_field(self.txt_bracket_l_roll, val_left[3])
                        set_numeric_field(self.txt_bracket_l_pitch, val_left[4])
                        set_numeric_field(self.txt_bracket_l_yaw, val_left[5])
                        # Sync back to memory configs
                        self.marker_calibrator.markers_config["Tf_to_marker_left"] = val_left
                        self.joint_calibrator.markers_config["Tf_to_marker_left"] = val_left
                    
                    # Load Right Arm values
                    val_right = marker_data.get("Tf_to_marker_right", None)
                    if val_right and len(val_right) == 6:
                        if is_v13 and abs(val_right[0]) < 0.05:
                            val_right = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.3"]["right"]
                        elif not is_v13 and abs(val_right[0]) > 0.05:
                            val_right = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.2"]["right"]
                        set_numeric_field(self.txt_bracket_r_x, val_right[0])
                        set_numeric_field(self.txt_bracket_r_y, val_right[1])
                        set_numeric_field(self.txt_bracket_r_z, val_right[2])
                        set_numeric_field(self.txt_bracket_r_roll, val_right[3])
                        set_numeric_field(self.txt_bracket_r_pitch, val_right[4])
                        set_numeric_field(self.txt_bracket_r_yaw, val_right[5])
                        # Sync back to memory configs
                        self.marker_calibrator.markers_config["Tf_to_marker_right"] = val_right
                        self.joint_calibrator.markers_config["Tf_to_marker_right"] = val_right
                    
                    self.log_msg(f"[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory")
                    return
            self.log_msg("[WARNING] Could not load bracket design values from setting.yaml.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to load setting.yaml: {e}")

    def _update_marker_key_in_lines(self, lines_list, key_str, new_vals_list):
        from core.config_store import replace_yaml_values
        replace_yaml_values(lines_list, "marker", key_str, new_vals_list)

    def _update_camera_key_in_lines(self, lines_list, key_str, new_vals_list):
        from core.config_store import replace_yaml_values
        replace_yaml_values(lines_list, "camera", key_str, new_vals_list)

    def _joint_offset_patch(self):
        values = {side: {key: float(self.joint_offsets_store[side].get(key, 0.0))
                        for key in ("joint3", "joint5", "joint6")}
                  for side in ("left", "right")}
        if "head" in self.joint_offsets_store:
            values["head"] = {key: float(self.joint_offsets_store["head"].get(key, 0.0))
                              for key in ("pan", "tilt")}
        return {"joint_offset": values}

    def _bracket_patch(self):
        return {"marker": {f"Tf_to_marker_{side}": [
            read_numeric_field(getattr(self, f"txt_bracket_{short}_{axis}"))
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
            for side, short in (("left", "l"), ("right", "r"))}}

    def _publish_joint_offsets(self, values):
        from copy import deepcopy
        applied = deepcopy(self.joint_offsets)
        is_v13 = self.get_robot_version() == "1.3"
        for side in ("left", "right"):
            applied[side].update(
                wrist_pitch=values[side]["joint5"], elbow=values[side]["joint3"],
                wrist_roll=values[side]["joint6"] if is_v13 else 0.0,
                wrist_yaw2=0.0 if is_v13 else values[side]["joint6"])
        self.joint_offsets = applied
        self.joint_calibrator.joint_offsets = applied
        self.marker_calibrator.joint_offsets = applied
        self.update_applied_offset_label()

    def _publish_brackets(self, values):
        for calibrator in (self.marker_calibrator, self.joint_calibrator):
            calibrator.camera_config.update(values)
        if self.marker_st is not None:
            detector = self.marker_st
            if hasattr(detector, "markers_config"):
                detector.markers_config.update(values)
                detector.Tf_to_marker_tf_left = detector.make_transform(values["Tf_to_marker_left"])
                detector.Tf_to_marker_tf_right = detector.make_transform(values["Tf_to_marker_right"])

    def apply_bracket_design_values(self, silent=False):
        from core.config_store import update_yaml
        try:
            updates = self._bracket_patch()
            committed = update_yaml(CONFIG_PATHS["setting_yaml"], updates)
            self._publish_brackets({key: committed["marker"][key] for key in updates["marker"]})
        except Exception as exc:
            self.log_msg(f"[ERROR] Failed to save bracket values: {exc}")
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to save bracket values: {exc}")
            return False
        self.log_msg("[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml")
        if not silent:
            QMessageBox.information(self, "Success", "Bracket design values saved for both arms!")
        return True

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
        self.btn_marker_start.setEnabled(enabled)
        if hasattr(self, 'btn_marker_result'):
            self.btn_marker_result.setEnabled(enabled)
            
        if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
            is_calculating = hasattr(self, 'calc_worker') and self.calc_worker is not None and self.calc_worker.isRunning()
            self.btn_step2_calculate.setEnabled(enabled and not is_calculating)
        
        self.btn_int_capture.setEnabled(enabled)
        self.btn_int_calibrate.setEnabled(enabled)
        self.btn_int_reset.setEnabled(enabled)
        
        if hasattr(self, 'chk_servo_head'):
            self.chk_servo_head.setEnabled(enabled)
        if hasattr(self, 'btn_step1_5_ready'):
            self.btn_step1_5_ready.setEnabled(enabled)
        if hasattr(self, 'btn_step1_5_start'):
            self.btn_step1_5_start.setEnabled(enabled)
            
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

    def on_move_ready_joint_finished(self):
        self.ready_done_joint = True
        self.on_action_finished()

    def on_move_ready_marker_finished(self):
        self.ready_done_marker = True
        self.on_action_finished()

    def get_robot_version(self) -> str:
        return str(getattr(self, "robot_version", "1.2"))

    def move_to_ready_full_auto(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.active_worker = FullAutoReadyWorker(
            self.joint_calibrator,
            self.marker_calibrator,
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_full_auto_ready_finished)
        self.active_worker.start()

    def on_full_auto_ready_finished(self):
        self.set_controls_enabled(True)
        error_msg = getattr(self.active_worker, 'error_msg', None) if self.active_worker else None
        if self.active_worker is not None:
            self.active_worker.wait()
        self.active_worker = None
        
        # Restart poll_timer if appropriate
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        camera_subtab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
        if not camera_subtab_active and not dialog_visible:
            if not self.poll_timer.isActive():
                self.poll_timer.start(200)
                
        if error_msg:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Movement Failed")
            msg_box.setText(f"Ready pose movement failed!\n\nReason:\n{error_msg}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            msg_box.exec()
        else:
            self.ready_done_joint = True
            self.ready_done_marker = True
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Movement Complete")
            msg_box.setText("Robot arms have moved to the initial ready poses successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            msg_box.exec()

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
            return None
        return result_files[0]

    def get_latest_home_reset_path(self, required=True):
        path = Path(CONFIG_PATHS['home_reset_baseline'])
        if path.exists():
            return path
        if required:
            raise RuntimeError(f"No home reset baseline JSON found at {path}")
        return None

    def get_home_reset_path_for_result(self, result_path):
        return self.get_latest_home_reset_path(required=False)

    def apply_home_offset(self):
        try:
            result_path = self.get_latest_result_path()
            baseline_path = self.get_home_reset_path_for_result(result_path)
            
            if hasattr(self, 'apply_offset_dialog') and self.apply_offset_dialog is not None and self.apply_offset_dialog.isVisible():
                self.apply_offset_dialog.raise_()
                self.apply_offset_dialog.activateWindow()
                return

            self.apply_offset_dialog = ApplyHomeOffsetDialog(
                self,
                result_path,
                baseline_path,
                "both",
                include_head=True
            )
            self.apply_offset_dialog.show()
            self.apply_offset_dialog.raise_()
            self.apply_offset_dialog.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Apply Home Offset Error", str(e))
            self.log_msg(f"[ERROR] Apply home offset failed: {e}")

    def home_offset_reset(self, *args, confirm_dialog=True, **kwargs) -> bool:
        if not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return False

        if confirm_dialog:
            msg = (
                "⚠️ [경고] Home Offset Reset 알림\n\n"
                "Home Offset Reset을 진행하면 현재 로봇 관절의 물리적 위치가 새로운 0도(Home Position)로 재설정됩니다.\n\n"
                "진행 순서:\n"
                "1. Direct Teaching 등을 사용하여 양 팔을 홈 자세(Home Pose) 위치에 정확히 맞춥니다.\n"
                "2. 헤드 오프셋도 초기화하려면 헤드를 정면 중앙으로 정렬합니다.\n"
                "3. [확인 (OK)]을 누르면 리셋 프로세스가 시작됩니다.\n\n"
                "※ 리셋 도중 Control Manager가 일시 중지되고 48V 전원이 재인가되며 로봇 연결이 재시작됩니다."
            )
            dialog = QDialog(self)
            dialog.setWindowTitle("Home Offset Reset 확인")
            dialog.setStyleSheet(DARK_STYLESHEET)
            layout = QVBoxLayout(dialog)

            # Image
            img_label = QLabel()
            pixmap = QPixmap(get_asset_path("img/home_offset_position.png"))
            if not pixmap.isNull():
                img_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                img_label.setText("[img/home_offset_position.png not found]")
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

            # Text
            msg_label = QLabel(msg)
            msg_label.setStyleSheet("font-size: 14px; color: white;")
            layout.addWidget(msg_label)

            # Buttons
            btn_layout = QHBoxLayout()
            btn_ok = QPushButton("OK")
            btn_ok.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
            btn_cancel = QPushButton("Cancel")
            btn_cancel.setStyleSheet("background-color: #555; color: white; padding: 5px;")
            
            btn_ok.clicked.connect(dialog.accept)
            btn_cancel.clicked.connect(dialog.reject)
            
            btn_layout.addStretch()
            btn_layout.addWidget(btn_ok)
            btn_layout.addWidget(btn_cancel)
            layout.addLayout(btn_layout)

            if dialog.exec() != QDialog.Accepted:
                return False

        self.set_controls_enabled(False)
        self.btn_home_reset.setEnabled(False)
        
        # Start worker thread
        self.active_worker = HomeOffsetResetWorker(
            self.robot,
            self.robot.model() if (self.robot) else None,
            self.model_input.currentText().strip() if hasattr(self, 'model_input') else "a",
            include_head=True
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_home_offset_reset_finished)
        self.active_worker.start()
        return True

    def on_home_offset_reset_finished(self, result):
        self.set_controls_enabled(True)
        self.btn_home_reset.setEnabled(True)
        if self.active_worker is not None:
            self.active_worker.wait()
        self.active_worker = None
        
        success = False
        error_msg = ""
        if result.get("success", False):
            # Reset software joint offsets to 0.0 for both arms since they are now physically absorbed
            for arm in ["left", "right"]:
                self.joint_offsets_store[arm]["joint3"] = 0.0
                self.joint_offsets_store[arm]["joint5"] = 0.0
                self.joint_offsets_store[arm]["joint6"] = 0.0
                
                self.joint_offsets[arm]["wrist_pitch"] = 0.0
                self.joint_offsets[arm]["wrist_roll"] = 0.0
                self.joint_offsets[arm]["wrist_yaw2"] = 0.0
                self.joint_offsets[arm]["elbow"] = 0.0

            # Save zeroed offsets to setting.yaml and update GUI
            offsets_saved = self.save_offsets_to_yaml()
            self.update_applied_offset_label()



            self.log_msg("Re-connecting and initializing robot...")
            if self.robot:
                self.connect_robot() # Disconnects first
                QApplication.processEvents()
            self.connect_robot() # Connects again
            success = offsets_saved
            if offsets_saved:
                self.log_msg("Home Offset Reset complete!")
                QMessageBox.information(self, "Success", "Home Offset Reset, Power, and Servo Initialization completed successfully! Software joint offsets have been reset to 0.0.")
            else:
                error_msg = "Robot home reset completed, but zeroed software offsets were NOT saved. Do not repeat the physical reset."
                self.log_msg("[ERROR] " + error_msg)
                QMessageBox.warning(self, "Settings Save Failed", error_msg)
        else:
            error_msg = result.get("error", "Some joints failed to reset")
            QMessageBox.warning(self, "Warning", f"Home Offset Reset finished, but some joints failed to reset: {error_msg}")
            success = False

        if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
            self.wizard_widget.set_wizard_busy(False)
            if success:
                self.wizard_widget.mark_step_completed(7, True, "Home Offset Reset complete")
            else:
                self.wizard_widget.mark_step_completed(7, False, error_msg)



    def clear_old_plots(self):
        self.generated_plots = []
        self.current_plot_idx = -1
        self.lbl_plot_title.setText("No Plot Loaded")
        self.plot_label_combined.setPixmap(QPixmap())
        self.btn_plot_prev.setEnabled(False)
        self.btn_plot_next.setEnabled(False)
        plot_dir = CONFIG_PATHS.get("plot_dir")
        if plot_dir and os.path.exists(plot_dir):
            for f_name in os.listdir(plot_dir):
                if f_name.endswith(".png") and "circle_fit_" in f_name:
                    try:
                        os.remove(os.path.join(plot_dir, f_name))
                    except Exception:
                        pass
        if plot_dir:
            txt_dir = os.path.abspath(os.path.join(os.path.dirname(plot_dir), "result_txt"))
            if os.path.exists(txt_dir):
                for f_name in os.listdir(txt_dir):
                    if f_name.endswith(".txt"):
                        try:
                            os.remove(os.path.join(txt_dir, f_name))
                        except Exception:
                            pass
    def apply_full_auto_results(self, silent=False):
        from core.config_store import update_yaml
        if (getattr(self, 'last_full_auto_error', None)
                or not getattr(self, 'last_full_auto_converged', False)):
            self.log_msg("[ERROR] Full Auto results cannot be applied: both arms must finish successfully without cancellation or measurement failure.")
            return False
        if not silent:
            reply = QMessageBox.question(self, "Apply Full Auto Results",
                "Apply all calibrated Joint Offsets and Marker Brackets to setting.yaml?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
        try:
            updates = {**self._joint_offset_patch(), **self._bracket_patch()}
            committed = update_yaml(CONFIG_PATHS["setting_yaml"], updates)
            self._publish_joint_offsets(committed["joint_offset"])
            self._publish_brackets({key: committed["marker"][key] for key in updates["marker"]})
        except Exception as exc:
            self.log_msg(f"[ERROR] Full auto results were not fully applied: {exc}")
            if not silent:
                QMessageBox.critical(self, "Apply Failed", str(exc))
            return False
        self.log_msg("[APPLY] Full auto results (Joints & Brackets) applied successfully.")
        if not silent:
            QMessageBox.information(self, "Apply Complete", "All full auto calibration results have been applied successfully.")
        if getattr(self, "wizard_widget", None) is not None and hasattr(self.wizard_widget, "on_step4_applied"):
            self.wizard_widget.on_step4_applied()
        return True
            

    def start_full_auto(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        self.last_full_auto_error = None
        self.last_full_auto_converged = False
            
        self.clear_old_plots()
            
        self.log_text.clear()
        self.log_msg("[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...")
        
        self.log_msg("[FULL AUTO] Starting a fresh staged calibration; active state is restored if the sequence fails.")
        if self.sim:
            self.log_msg("[MOCK GT] Simulated Ground-Truth Offsets:")
            is_v13 = self.get_robot_version() == "1.3"
            for arm in ["right", "left"]:
                mock_gt = self.marker_st.simulation_model.config['offsets'][arm]
                j6_gt = mock_gt["joint6"]
                j5_gt = mock_gt["joint5_v13"] if is_v13 else mock_gt["joint5_v12"]
                j3_gt = mock_gt["joint3"]
                pos_gt = [x * 1000.0 for x in mock_gt["bracket_pos"]] # convert from m to mm
                rpy_gt = mock_gt["bracket_rpy"]
                self.log_msg(f"  --- {arm.upper()} ARM ---")
                self.log_msg(f"  * Bracket Pos: X: {pos_gt[0]:+.1f}, Y: {pos_gt[1]:+.1f}, Z: {pos_gt[2]:+.1f} mm")
                self.log_msg(f"  * Bracket Rot: R: {rpy_gt[0]:+.2f}, P: {rpy_gt[1]:+.2f}, Y: {rpy_gt[2]:+.2f} deg")
                self.log_msg(f"  * Joint Offsets: Joint 6: {j6_gt:+.2f}°, Joint 5: {j5_gt:+.2f}°, Joint 3: {j3_gt:+.2f}°")
        
        self.set_controls_enabled(False)
        self.btn_full_auto_start.setEnabled(False)
        
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        
        
        self.full_auto_stop_event = threading.Event()
        
        # update mock robot version on calibrators just in case
        v = self.get_robot_version()
        self.marker_calibrator.robot_version = v
        self.joint_calibrator.robot_version = v
        
        self.active_worker = FullAutoWorker(
            self.joint_calibrator,
            self.marker_calibrator,
            stop_event=self.full_auto_stop_event,
            joint_offsets_store=self.joint_offsets_store,
            save_debug=self.chk_save_debug.isChecked(),
            reset_initial_state=True, head_camera_calibrator=self.head_camera_calibrator
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
        if self.robot:
            self.safe_cancel_control()

    def on_full_auto_finished(self):
        self.set_controls_enabled(True)
        if hasattr(self, 'btn_full_auto_start'):
            self.btn_full_auto_start.setEnabled(True)
        
        was_stopped = False
        if hasattr(self, 'full_auto_stop_event') and self.full_auto_stop_event is not None:
            was_stopped = self.full_auto_stop_event.is_set()
            
        error_msg = getattr(self.active_worker, 'error_msg', None) if self.active_worker else None
        self.last_full_auto_error = error_msg
        stability = getattr(self.active_worker, 'arm_convergence', {})
        self.last_full_auto_converged = (not was_stopped and not error_msg
            and all(stability.get(side, False) for side in ('right', 'left')))
        if hasattr(self, 'btn_full_auto_apply'):
            self.btn_full_auto_apply.setEnabled(self.last_full_auto_converged)
        if self.active_worker is not None:
            self.active_worker.wait()
        self.active_worker = None
        self.log_msg("[INFO] Full Auto sequential calibration ended.")
        
        # Restart poll_timer if appropriate (not tab 2 and feed dialog closed)
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        camera_subtab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
        if not camera_subtab_active and not dialog_visible:
            if not self.poll_timer.isActive():
                self.poll_timer.start(200)
                
        if error_msg:
            self.log_msg(f"[ERROR] Full Auto Calibration FAILED: {error_msg}")
        elif not was_stopped:
            if self.last_full_auto_converged:
                self.log_msg("[SUCCESS] Full Auto Sequential Calibration converged. Please review the offsets in the table.")
            else:
                self.log_msg("[WARNING] Full Auto ended without meeting all parameter stability tolerances. Review or repeat before applying.")

    def handle_full_auto_bracket_finished(self, bracket_res):
        if not bracket_res.get('measurement_accepted', False):
            self.log_msg('[ERROR] Rejected bracket measurement was not staged.')
            return
        arm_side = bracket_res['arm_side']
        
        # Update UI text boxes for corresponding arm
        if arm_side == "left":
            set_numeric_field(self.txt_bracket_l_x, bracket_res['x_e']/1000.0)
            set_numeric_field(self.txt_bracket_l_y, bracket_res['y_e']/1000.0)
            set_numeric_field(self.txt_bracket_l_z, bracket_res['z_e']/1000.0)
            set_numeric_field(self.txt_bracket_l_roll, bracket_res['roll_e'])
            set_numeric_field(self.txt_bracket_l_pitch, bracket_res['pitch_e'])
            set_numeric_field(self.txt_bracket_l_yaw, bracket_res['yaw_e'])
        else:
            set_numeric_field(self.txt_bracket_r_x, bracket_res['x_e']/1000.0)
            set_numeric_field(self.txt_bracket_r_y, bracket_res['y_e']/1000.0)
            set_numeric_field(self.txt_bracket_r_z, bracket_res['z_e']/1000.0)
            set_numeric_field(self.txt_bracket_r_roll, bracket_res['roll_e'])
            set_numeric_field(self.txt_bracket_r_pitch, bracket_res['pitch_e'])
            set_numeric_field(self.txt_bracket_r_yaw, bracket_res['yaw_e'])
            
        # Joint offsets are staged only by the joint-result callback.

        if 'plot_path_combined' in bracket_res and bracket_res.get('pass_idx', 1) == 2:
            self.add_and_show_plot(f"[{arm_side.upper()}] FullAuto - Marker Bracket", bracket_res['plot_path_combined'])

        self.log_msg(f"[INFO] Full Auto: Finished bracket calibration for {arm_side.upper()} arm. Values staged in UI (click APPLY BRACKETS to save).")

    def handle_full_auto_joint_finished(self, joint_res):
        arm_side = joint_res['arm_side']
        mode = joint_res.get('mode', 'elbow')
        if not _joint_result_accepted(joint_res):
            self.log_msg(f"[WARNING] {arm_side} {mode}: {joint_res.get('failure_reason')}; "
                         "no new offset staged in UI.")
            return
        
        recommended = joint_res['recommended_joint_offset']
        if mode in ("wrist_roll_v13", "wrist_yaw2"):
            joint_key = "joint6"
        elif mode in ("wrist_pitch_v13", "wrist_pitch"):
            joint_key = "joint5"
        else:
            joint_key = "joint3"
            
        # Update staged offsets store
        self.joint_offsets_store[arm_side][joint_key] = float(recommended)
        
        # Refresh offset monitor table view
        self.update_applied_offset_label()
        
        self.log_msg(f"[INFO] Full Auto: Finished joint calibration for {arm_side.upper()} {mode}. Staged: {recommended:.4f}° (click APPLY OFFSET to save).")
        
        if 'plot_path_combined' in joint_res and joint_res.get('pass_idx', 1) == 2:
            self.add_and_show_plot(f"[{arm_side.upper()}] FullAuto Joint - {mode}", joint_res['plot_path_combined'])

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
        self.ready_worker.finished_signal.connect(self.on_move_ready_joint_finished)
        self.ready_worker.start()

    def start_calibration_joint(self):
        if not self.ready_done_joint:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Prerequisite Check")
            msg_box.setText("Please move the robot to the Ready pose first by clicking 'MOVE TO READY'!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        self.clear_old_plots()

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
        self.original_joint_offset = self.joint_offsets[self.arm_side].get(offset_key, 0.0)
        self._joint_calibration_mode = mode
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Joint Sweep: {mode.upper()}")
        if self.sim:
            mock_gt = self.marker_st.simulation_model.config['offsets'][self.arm_side]
            is_v13 = self.get_robot_version() == "1.3"
            j_gt = {
                "wrist_roll_v13": mock_gt["joint6"],
                "wrist_yaw2": mock_gt["joint6"],
                "wrist_pitch_v13": mock_gt["joint5_v13"],
                "wrist_pitch": mock_gt["joint5_v12"],
                "elbow": mock_gt["joint3"]
            }
            gt_val = j_gt.get(mode, 0.0)
            self.log_msg(f"[MOCK GT] Simulated Target Joint Offset: {gt_val:+.2f}°")
        
        curr_offset = self.joint_offsets[self.arm_side].get(offset_key, 0.0)
        self.active_worker = JointCalibrationWorker(
            self.joint_calibrator, self.arm_side, mode, 
            current_offset_deg=curr_offset,
            save_debug=self.chk_save_debug.isChecked()
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.finished_signal.connect(self.on_calibration_finished_joint)
        self.active_worker.start()

    def _discard_joint_calibration(self, mode, res=None):
        self.joint_sweep_data = None
        self.recommended_joint_offset = None
        if res:
            self._failed_joint_result = dict(res, arm_side=self.arm_side)
        if mode is not None and hasattr(self, 'original_joint_offset'):
            key = self.get_offset_key_for_mode(mode)
            self.joint_offsets[self.arm_side][key] = self.original_joint_offset
            self.joint_calibrator.joint_offsets[self.arm_side][key] = self.original_joint_offset
            self.marker_calibrator.joint_offsets[self.arm_side][key] = self.original_joint_offset
            self.update_applied_offset_label()
        self.on_action_finished()
        self.log_msg('[ERROR] Calibration was not accepted and converged. No new offset to apply; active correction restored.')

    def on_calibration_finished_joint(self, res):
        mode = (res or {}).get('mode', getattr(self, '_joint_calibration_mode', None))
        if not _joint_result_accepted(res):
            UnifiedCalibrationApp._discard_joint_calibration(self, mode, res)
            return
        self._failed_joint_result = None
        self.recommended_joint_offset = res.get('recommended_joint_offset', res.get('optimal_offset'))
        self.finalize_joint_calibration_run(mode, res, converged=True)

    def finalize_joint_calibration_run(self, mode, res, converged=False):
        if not converged or not _joint_result_accepted(res):
            UnifiedCalibrationApp._discard_joint_calibration(self, mode, res)
            return False
        self.on_action_finished()
        self.joint_sweep_data = res

        # Update Plot viewer if plots exist
        if 'plot_path_combined' in res and os.path.exists(res['plot_path_combined']):
            self.add_and_show_plot(f"[{self.arm_side.upper()}] Joint - {mode}", res['plot_path_combined'])

        # Update UI bracket design text fields if new values are calibrated
        if 'x_cal' in res and not np.isnan(res['x_cal']):
            arm = self.arm_side
            x_val, y_val, z_val = res['x_cal'], res['y_cal'], res['z_cal']
            r_val, p_val, yaw_val = res.get('roll_cal', float('nan')), res.get('pitch_cal', float('nan')), res.get('yaw_cal', float('nan'))
            if arm == "left":
                set_numeric_field(self.txt_bracket_l_x, x_val)
                set_numeric_field(self.txt_bracket_l_y, y_val)
                set_numeric_field(self.txt_bracket_l_z, z_val)
                if not np.isnan(r_val):
                    set_numeric_field(self.txt_bracket_l_roll, r_val)
                    set_numeric_field(self.txt_bracket_l_pitch, p_val)
                    set_numeric_field(self.txt_bracket_l_yaw, yaw_val)
            else:
                set_numeric_field(self.txt_bracket_r_x, x_val)
                set_numeric_field(self.txt_bracket_r_y, y_val)
                set_numeric_field(self.txt_bracket_r_z, z_val)
                if not np.isnan(r_val):
                    set_numeric_field(self.txt_bracket_r_roll, r_val)
                    set_numeric_field(self.txt_bracket_r_pitch, p_val)
                    set_numeric_field(self.txt_bracket_r_yaw, yaw_val)
            if not np.isnan(r_val):
                self.log_msg(f"[INFO] Staged calibrated nominal marker values in UI for {arm} arm: X={x_val:.4f}, Y={y_val:.4f}, Z={z_val:.4f}, R={r_val:.2f}, P={p_val:.2f}, Y={yaw_val:.2f}")
            else:
                self.log_msg(f"[INFO] Staged calibrated nominal marker values in UI for {arm} arm: X={x_val:.4f}, Y={y_val:.4f}, Z={z_val:.4f}")

        if mode in ("wrist_roll_v13", "wrist_yaw2"):
            joint_key = "joint6"
            if converged:
                self.wrist_roll_calibrated[self.arm_side] = True
        elif mode in ("wrist_pitch_v13", "wrist_pitch"):
            joint_key = "joint5"
        else:
            joint_key = "joint3"
        self.joint_offsets_store[self.arm_side][joint_key] = float(self.recommended_joint_offset)

        # Revert active offsets to nominal original values in model (until user clicks APPLY)
        offset_key = self.get_offset_key_for_mode(mode)
        self.joint_offsets[self.arm_side][offset_key] = self.original_joint_offset
        self.joint_calibrator.joint_offsets[self.arm_side][offset_key] = self.original_joint_offset
        self.marker_calibrator.joint_offsets[self.arm_side][offset_key] = self.original_joint_offset
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
        elif mode == "wrist_yaw2":
            joint_name = "Joint 6 (Wrist Yaw 2)"
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

        # display bracket design verification results
        if mode in ("wrist_pitch_v13", "wrist_roll_v13", "wrist_yaw2"):
            d = self.joint_sweep_data
            sweep_axis_label = "Joint 6" if mode in ("wrist_roll_v13", "wrist_yaw2") else "Joint 5"
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
            axis_dir = "Z" if mode == "wrist_yaw2" else ("X" if mode == "wrist_roll_v13" else "Y")
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
        self.ready_worker.finished_signal.connect(self.on_move_ready_marker_finished)
        self.ready_worker.start()

    # Move to Center is removed as it is no longer needed

    def start_calibration_marker(self):
        # 1. Prerequisite Check: Joint 6 (Wrist Roll / Wrist Yaw 2) must be calibrated first
        if not self.wrist_roll_calibrated.get(self.arm_side, False):
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Prerequisite Check")
            msg_box.setText(
                "Marker Bracket Calibration requires Joint 6 (Wrist Roll / Wrist Yaw 2) to be calibrated first.\n\n"
                "Joint 6 has not been calibrated yet. Please go to the Joint Calibration tab, select Joint 6, and perform calibration."
            )
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            return

        # 2. Prerequisite Check: Move to Ready Pose first
        if not self.ready_done_marker:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Prerequisite Check")
            msg_box.setText("Please move the robot to the Ready pose first by clicking 'MOVE TO READY'!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        self.clear_old_plots()

        # use_head = self.cb_head_tracking.isChecked()
        use_head = False
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Unified Marker Sweep (Axis 6 & 5) (Head Tracking: {use_head})")
        if self.sim:
            mock_gt = self.marker_st.simulation_model.config['offsets'][self.arm_side]
            pos_gt = [x * 1000.0 for x in mock_gt["bracket_pos"]] # convert from m to mm
            rpy_gt = mock_gt["bracket_rpy"]
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

        if res and res.get('measurement_accepted', False):
            self.marker_data_unified = res
            self.marker_data_5 = res['res_5']
            self.marker_data_6 = res['res_6']
            self.marker_data_4 = res.get('res_4', None)
                
            # Bracket-only results must never overwrite the joint stage.

            # Update UI bracket design text fields
            arm_side = self.arm_side
            x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0
            if arm_side == "left":
                set_numeric_field(self.txt_bracket_l_x, x_m)
                set_numeric_field(self.txt_bracket_l_y, y_m)
                set_numeric_field(self.txt_bracket_l_z, z_m)
                set_numeric_field(self.txt_bracket_l_roll, res['roll_e'])
                set_numeric_field(self.txt_bracket_l_pitch, res['pitch_e'])
                set_numeric_field(self.txt_bracket_l_yaw, res['yaw_e'])
            else:
                set_numeric_field(self.txt_bracket_r_x, x_m)
                set_numeric_field(self.txt_bracket_r_y, y_m)
                set_numeric_field(self.txt_bracket_r_z, z_m)
                set_numeric_field(self.txt_bracket_r_roll, res['roll_e'])
                set_numeric_field(self.txt_bracket_r_pitch, res['pitch_e'])
                set_numeric_field(self.txt_bracket_r_yaw, res['yaw_e'])

            # Sync to memory configs
            new_vals = [x_m, y_m, z_m, res['roll_e'], res['pitch_e'], res['yaw_e']]
            key = f"Tf_to_marker_{arm_side}"
            self.marker_calibrator.camera_config[key] = new_vals
            self.joint_calibrator.camera_config[key] = new_vals
            self.log_msg(f"[INFO] Staged calibrated nominal marker values in UI for {arm_side} arm. (Click APPLY BRACKETS to save)")

            if 'plot_path_combined' in res and os.path.exists(res['plot_path_combined']):
                self.add_and_show_plot(f"[{self.arm_side.upper()}] Marker Bracket", res['plot_path_combined'])
            self.show_unified_result_marker_direct(res)
        else:
            self.log_msg("[ERROR] Marker sweep failed.")

    def show_unified_result_marker_direct(self, res):
        if not res or not res.get('measurement_accepted', False):
            self.log_msg('[ERROR] Bracket measurement was not accepted: ' + str((res or {}).get('failure_reason', 'missing result')))
            return False
        self.log_msg("\nUNIFIED BRACKET CALIBRATION RESULTS")
        self.log_msg(f"Position (EE frame): X={res['x_e']:.3f}, Y={res['y_e']:.3f}, Z={res['z_e']:.3f} mm")
        self.log_msg(f"Rotation (EE frame): Roll={res['roll_e']:.4f}, Pitch={res['pitch_e']:.4f}, Yaw={res['yaw_e']:.4f} deg")
        self.log_msg(f"Observed wrist-axis intersection RMS: {res['axis_intersection_rms_mm']:.4f} mm")
        self.log_msg('J6/bracket twist uses an effective reference.')
        return True

    def show_unified_result_marker(self):
        sweeps = [getattr(self, f'marker_data_{axis}', None) for axis in (4, 5, 6)]
        if any(data is None for data in sweeps):
            self.log_msg('[ERROR] Bracket fitting requires all three axis 4, 5 and 6 sweeps.')
            return False
        try:
            res = self.marker_calibrator.fit_observed_bracket(*sweeps, self.arm_side)
            if not self.show_unified_result_marker_direct(res):
                return False
            res['arm_side'] = self.arm_side
            self.handle_full_auto_bracket_finished(res)
            values = [res['x_e']/1000., res['y_e']/1000., res['z_e']/1000.,
                      res['roll_e'], res['pitch_e'], res['yaw_e']]
            key = f'Tf_to_marker_{self.arm_side}'
            self.marker_calibrator.camera_config[key] = values
            self.joint_calibrator.camera_config[key] = values
            self.marker_data_unified = res
            return True
        except Exception as exc:
            self.log_msg(f'[ERROR] Failed to calculate bracket calibration: {exc}')
            return False

    def open_plot_dialog(self):
        if not hasattr(self, 'plot_dialog') or self.plot_dialog is None:
            self.plot_dialog = PlotViewerDialog(self)
        self.plot_dialog.show()
        self.plot_dialog.raise_()
        self.plot_dialog.activateWindow()
        self.display_current_plot()

    def add_and_show_plot(self, friendly_name, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        
        display_name = friendly_name
        
        # Check if file_path already in list
        existing_idx = -1
        for idx, (_, path) in enumerate(self.generated_plots):
            if path == file_path:
                existing_idx = idx
                break
                
        if existing_idx == -1:
            self.generated_plots.append((display_name, file_path))
            self.current_plot_idx = len(self.generated_plots) - 1
        else:
            self.current_plot_idx = existing_idx
            
        # Do not automatically show the plot. It will be shown only when Full Auto ends/errors, or by manual click.

    def update_navigation_buttons(self):
        self.btn_plot_prev.setEnabled(self.current_plot_idx > 0)
        self.btn_plot_next.setEnabled(self.current_plot_idx < len(self.generated_plots) - 1)
        
    def show_prev_plot(self):
        if self.current_plot_idx > 0:
            self.current_plot_idx -= 1
            self.display_current_plot()
            
    def show_next_plot(self):
        if self.current_plot_idx < len(self.generated_plots) - 1:
            self.current_plot_idx += 1
            self.display_current_plot()
            
    def display_current_plot(self):
        if 0 <= self.current_plot_idx < len(self.generated_plots):
            display_name, file_path = self.generated_plots[self.current_plot_idx]
            self.lbl_plot_title.setText(display_name)
            self.display_plot_image(file_path)
            self.update_navigation_buttons()
        else:
            self.lbl_plot_title.setText("No Plot Loaded")
            self.plot_label_combined.setPixmap(QPixmap())
            self.btn_plot_prev.setEnabled(False)
            self.btn_plot_next.setEnabled(False)

    def display_plot_image(self, file_path):
        if os.path.exists(file_path):
            # Scale to fit the current dialog size dynamically
            if hasattr(self, 'plot_dialog') and self.plot_dialog is not None and self.plot_dialog.isVisible():
                target_w = max(800, self.plot_dialog.width() - 40)
                target_h = max(500, self.plot_dialog.height() - 100)
            else:
                target_w = 900
                target_h = 550
            pix = QPixmap(file_path).scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.plot_label_combined.setPixmap(pix)

    # --- Head Control and Manual Operations ---
    def move_head_manually(self):
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        try:
            yaw = float(self.txt_head_yaw.text())
            pitch = float(self.txt_head_pitch.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Inputs", "Head angles must be valid numbers.")
            return
            
        self.log_msg(f"[MANUAL HEAD] Commands sent - Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°")
            
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
        # Camera 서브탭(Step1 > Camera)이 활성화되어 있거나, Camera Feed 대화상자 또는 마커 미인식 대화상자, 위자드 비디오 슬라이드가 열려있을 때 업데이트
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        prob_dlg_visible = hasattr(self, 'marker_problem_dlg') and self.marker_problem_dlg is not None
        camera_tab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
        wizard_active = hasattr(self, 'wizard_widget') and self.wizard_widget is not None and not self.wizard_widget.isHidden()
        wizard_slide_idx = self.wizard_widget.stacked_widget.currentIndex() if wizard_active else -1
        wizard_slide_mount = wizard_active and (wizard_slide_idx == 0)
        wizard_slide_exp = wizard_active and (wizard_slide_idx == 2)
        wizard_slide_calib = wizard_active and (wizard_slide_idx == 4)
        
        if not camera_tab_active and not dialog_visible and not wizard_slide_mount and not wizard_slide_exp and not wizard_slide_calib and not prob_dlg_visible:
            return

        waiting_for_frame = False
        if not self.sim and self.marker_st is not None:
            worker_owns_camera = self._camera_capture_owned_by_worker()
            if not worker_owns_camera:
                self.marker_st.camera.capture_image()
            img = self.marker_st.camera.get_color_image()
            
            # Sync real-time auto exposure value to UI if auto mode is enabled
            is_auto = hasattr(self, 'chk_auto_exposure') and self.chk_auto_exposure.isChecked()
            if is_auto:
                act_exp = self.marker_st.get_actual_exposure()
                act_exp_int = int(act_exp)
                if hasattr(self, 'spin_exposure'):
                    self.spin_exposure.blockSignals(True)
                    self.spin_exposure.setValue(act_exp_int)
                    self.spin_exposure.blockSignals(False)
                if hasattr(self, 'slider_exposure'):
                    self.slider_exposure.blockSignals(True)
                    self.slider_exposure.setValue(act_exp_int)
                    self.slider_exposure.blockSignals(False)
                if hasattr(self, 'lbl_exposure_ms'):
                    self.lbl_exposure_ms.setText(f"{act_exp / 1000.0:.1f} ms (Auto)")
                
                # Sync Wizard exposure widgets if wizard slide 2 is active
                if wizard_slide_exp and hasattr(self.wizard_widget, 'chk_wiz_auto_exp') and self.wizard_widget.chk_wiz_auto_exp.isChecked():
                    if hasattr(self.wizard_widget, 'spin_wiz_exp'):
                        self.wizard_widget.spin_wiz_exp.blockSignals(True)
                        self.wizard_widget.spin_wiz_exp.setValue(act_exp_int)
                        self.wizard_widget.spin_wiz_exp.blockSignals(False)
                    if hasattr(self.wizard_widget, 'slider_wiz_exp'):
                        self.wizard_widget.slider_wiz_exp.blockSignals(True)
                        self.wizard_widget.slider_wiz_exp.setValue(act_exp_int)
                        self.wizard_widget.slider_wiz_exp.blockSignals(False)
                    if hasattr(self.wizard_widget, 'lbl_wiz_exp_ms'):
                        self.wizard_widget.lbl_wiz_exp_ms.setText(f"{act_exp / 1000.0:.1f} ms (Auto)")
            
            # 백그라운드 마커 검출 및 상태 표시 업데이트
            try:
                if not worker_owns_camera:
                    res_all = self.marker_st.get_marker_transform(sampling_time=0, side="all")
                    detected = bool(res_all and len(res_all) > 0)
                    self.update_marker_indicator(detected)
            except Exception:
                pass
                
            if img is None and worker_owns_camera:
                waiting_for_frame = True
                img = getattr(self, 'current_frame', None)
            if img is None:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                if not waiting_for_frame:
                    cv2.putText(img, "No Camera Detected", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
        else:
            # Mock image
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "UI-ONLY MODE", (440, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 100, 100), 3)

        if img is None:
            return
            
        self.current_frame = img.copy()
        display_img = img.copy()
        if waiting_for_frame:
            cv2.putText(display_img, "Waiting for next camera frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        
        guide_checked = (hasattr(self, 'chk_int_guide') and self.chk_int_guide.isChecked()) or (hasattr(self, 'wizard_widget') and hasattr(self.wizard_widget, 'chk_int_guide') and self.wizard_widget.chk_int_guide.isChecked())
        if guide_checked and (camera_tab_active or wizard_slide_calib):
            num_steps = len(IntrinsicsCalibrator.CALIB_GUIDELINES)
            if self.current_guide_idx < num_steps:
                guideline = IntrinsicsCalibrator.CALIB_GUIDELINES[self.current_guide_idx]
                h, w = display_img.shape[:2]
                pts_pixel = np.array(guideline["pts"] * [w, h], dtype=np.int32)
                
                # Draw filled transparent guide poly
                overlay = display_img.copy()
                cv2.fillPoly(overlay, [pts_pixel], (255, 229, 0)) # Neon Cyan in BGR
                cv2.addWeighted(overlay, 0.15, display_img, 0.85, 0, display_img)
                
                # Draw border poly
                cv2.polylines(display_img, [pts_pixel], isClosed=True, color=(255, 229, 0), thickness=3)
                
                # Draw guide labels
                guide_name = guideline["name"]
                text_title = f"Guide {self.current_guide_idx + 1}/{num_steps}: {guide_name}"
                cv2.putText(display_img, text_title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_img, "Align checkerboard and press CAPTURE (C)", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(display_img, f"All {num_steps} steps captured! Press RUN CALIBRATION", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Convert to QImage and display
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        if camera_tab_active and hasattr(self, 'video_label'):
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if wizard_slide_exp and hasattr(self.wizard_widget, 'wizard_exposure_video_label'):
            self.wizard_widget.wizard_exposure_video_label.setPixmap(pixmap.scaled(self.wizard_widget.wizard_exposure_video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if (wizard_slide_mount or wizard_slide_calib) and hasattr(self.wizard_widget, 'wizard_video_label'):
            self.wizard_widget.wizard_video_label.setPixmap(pixmap.scaled(self.wizard_widget.wizard_video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if dialog_visible:
            w_lbl = max(20, self.feed_dialog.lbl_feed.width())
            h_lbl = max(20, self.feed_dialog.lbl_feed.height())
            self.feed_dialog.lbl_feed.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))
        if prob_dlg_visible and hasattr(self.marker_problem_dlg, 'lbl_live_feed'):
            w_lbl = max(20, self.marker_problem_dlg.lbl_live_feed.width())
            h_lbl = max(20, self.marker_problem_dlg.lbl_live_feed.height())
            self.marker_problem_dlg.lbl_live_feed.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))

    def keyPressEvent(self, event):
        camera_tab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
        wizard_slide4_active = (hasattr(self, 'wizard_widget') and self.wizard_widget.isVisible() and self.wizard_widget.stacked_widget.currentIndex() == 4)
        if event.key() == Qt.Key_C and (camera_tab_active or wizard_slide4_active):
            self.capture_intrinsics_frame()
        super().keyPressEvent(event)

    def capture_intrinsics_frame(self):
        if hasattr(self, 'current_frame'):
            self.captured_images.append(self.current_frame.copy())
            frames = len(self.captured_images)
            self.lbl_captured.setText(f"Captured Frames: {frames}")
            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                if hasattr(self.wizard_widget, 'lbl_captured') and self.wizard_widget.lbl_captured is not None:
                    self.wizard_widget.lbl_captured.setText(f"Captured Frames: {frames} / 16")
            self.log_msg(f"[INTRINSICS] Frame {frames} captured.")
            
            num_steps = len(IntrinsicsCalibrator.CALIB_GUIDELINES)
            if hasattr(self, 'chk_int_guide') and self.chk_int_guide.isChecked() and self.current_guide_idx < num_steps:
                self.current_guide_idx += 1
                if self.current_guide_idx == num_steps:
                    self.log_msg(f"[INTRINSICS] All {num_steps} guided frames captured! You can now run calibration.")

            if frames == 16:
                self.log_msg("[INTRINSICS] 16 frames collected! Automatically running calibration...")
                QApplication.processEvents()
                self.run_intrinsics_calibration()

    def reset_intrinsics_captures(self):
        self.captured_images.clear()
        self.current_guide_idx = 0
        self.lbl_captured.setText(f"Captured Frames: 0")
        self.btn_int_save.setEnabled(False)
        self.log_msg("[INTRINSICS] Capture memory cleared.")

    def on_guide_changed(self, state):
        checked = (state == Qt.Checked or state == 2)
        self.log_msg(f"[INTRINSICS] Guidance overlay {'ENABLED' if checked else 'DISABLED'}.")
        if checked:
            num_steps = len(IntrinsicsCalibrator.CALIB_GUIDELINES)
            self.current_guide_idx = min(num_steps, len(self.captured_images))

    def camera_source_busy(self):
        if len(getattr(self, 'shared_arm_q_list', [])):
            return True
        owners = [self, getattr(self, 'wizard_widget', None)]
        return any(isinstance(value, QThread) and value.isRunning()
                   for owner in owners if owner is not None
                   for value in vars(owner).values())

    def run_intrinsics_calibration(self):
        if len(self.captured_images) < 16:
            self.log_msg("[ERROR] Need all 16 valid frames to run calibration!")
            QMessageBox.warning(self, "Insufficient Data", f"Cannot run calibration: Only {len(self.captured_images)} / 16 frames collected.\nPlease capture all 16 frames first.")
            return
            
        self.log_msg(f"\n[INTRINSICS] Running calibration on {len(self.captured_images)} images. Please wait...")
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        camera = getattr(self.marker_st, 'camera', None)
        self.intrinsics_calibrator.capture_metadata = {
            'calibration_temperature_c': camera.get_camera_temperature() if camera else None,
            'device_name': getattr(self.marker_st, 'camera_model', None)}
        success = self.intrinsics_calibrator.run_calibration_with_images(self.captured_images, None)
        
        QApplication.restoreOverrideCursor()
        
        if success:
            self.log_msg(f"[SUCCESS] Calibration complete! RMS Error: {self.intrinsics_calibrator.rms_error:.4f}")
            self.log_msg("[INTRINSICS] Parameter Standard Error (Uncertainty):")
            self.log_msg(f"  * Focal Length fx: {self.intrinsics_calibrator.std_fx:.4f} pixels")
            self.log_msg(f"  * Focal Length fy: {self.intrinsics_calibrator.std_fy:.4f} pixels")
            self.log_msg(f"  * Principal Point cx: {self.intrinsics_calibrator.std_cx:.4f} pixels")
            self.log_msg(f"  * Principal Point cy: {self.intrinsics_calibrator.std_cy:.4f} pixels")
            
            if self.intrinsics_calibrator.test_rmse is not None:
                self.log_msg(f"[INTRINSICS] Cross-Validation Test RMSE: {self.intrinsics_calibrator.test_rmse:.4f} pixels")
                if self.intrinsics_calibrator.test_rmse < 0.18:
                    self.log_msg("[INTRINSICS] Generalization check: EXCELLENT (low variance, high stability)")
                else:
                    self.log_msg("[INTRINSICS] Generalization check: WARNING (high variance, check board angles)")
            else:
                self.log_msg("[INTRINSICS] Cross-Validation: Not enough frames (min 6 frames needed)")
                
            self.log_msg("[INTRINSICS] Click 'SAVE PARAMETERS' to apply changes.")
            self.btn_int_save.setEnabled(True)
            self.show_intrinsics_verification()
            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                if hasattr(self.wizard_widget, 'lbl_step1_status') and self.wizard_widget.lbl_step1_status is not None:
                    self.wizard_widget.lbl_step1_status.setText(f"Status: Calibration OK (RMS: {self.intrinsics_calibrator.rms_error:.4f})")
                    self.wizard_widget.lbl_step1_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 16px;")
        else:
            self.log_msg("[ERROR] Calibration failed. Check images and board settings.")
            QMessageBox.critical(self, "Calibration Failed", "Calibration failed! Check board visibility and image quality.")
            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                if hasattr(self.wizard_widget, 'lbl_step1_status') and self.wizard_widget.lbl_step1_status is not None:
                    self.wizard_widget.lbl_step1_status.setText("Status: Calibration Failed (Check board settings)")
                    self.wizard_widget.lbl_step1_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def save_intrinsics_calibration(self):
        if self.camera_source_busy():
            self.log_msg('[ERROR] Finish active work and start a new sample session before saving intrinsics.')
            return
        if len(self.captured_images) < 16:
            self.log_msg("[ERROR] Need all 16 frames to save parameters!")
            QMessageBox.warning(self, "Cannot Save", f"Cannot save parameters: Only {len(self.captured_images)} / 16 frames collected.")
            return

        if self.intrinsics_calibrator.cameraMatrix is None or float(self.intrinsics_calibrator.rms_error) <= 0.0:
            self.log_msg("[ERROR] No valid calibration data to save!")
            QMessageBox.warning(self, "Cannot Save", "Calibration has not been successfully executed yet.")
            return

        try:
            camera_model = getattr(self.marker_st, 'camera_model', "")
            data = {
                "device_name": camera_model,
                "camera_matrix": self.intrinsics_calibrator.cameraMatrix.tolist(),
                "dist_coeffs": self.intrinsics_calibrator.distCoeffs.flatten().tolist(),
                "rms_error": float(self.intrinsics_calibrator.rms_error),
                "width": int(self.captured_images[0].shape[1]),
                "height": int(self.captured_images[0].shape[0])
            }
            data.update(getattr(self.intrinsics_calibrator, 'capture_metadata', {}))
            metadata = self.marker_st.save_intrinsics(data)
            self.lbl_intrinsics_source.setText("Actual intrinsics: " + metadata["file"])
            self.log_msg(f"[SUCCESS] Intrinsic parameters saved to: {self.output_yaml}")
            
            self.log_msg('[INTRINSICS] Fixed intrinsics file saved and reloaded for detection.')
            
            # Show save success message box
            self.show_message_box(
                "Save Complete" if Language.instance().current_lang != "ko" else "저장 완료",
                f"Camera intrinsics saved successfully to:\n{self.output_yaml}" if Language.instance().current_lang != "ko" else f"카메라 내부 파라미터가 다음 경로에 성공적으로 저장되었습니다:\n{self.output_yaml}"
            )
        except Exception as e:
            self.log_msg(f"[ERROR] Save failed: {e}")

    def show_intrinsics_verification(self):
        if len(self.captured_images) == 0:
            return
            
        test_img = self.captured_images[-1]
        save_path = os.path.join(CONFIG_PATHS["plot_dir"], "camera_intrinsics_verification.png")
        
        # Delegate image generation to IntrinsicsCalibrator
        self.intrinsics_calibrator.generate_verification_image(test_img, save_path)
        
        # Load inside Plot viewer dialog history
        self.add_and_show_plot("[INTRINSICS] Verification Image", save_path)
        
        # Pop up visual verification dialog directly
        if os.path.exists(save_path):
            dialog = QDialog(self)
            dialog.setWindowTitle("Camera Intrinsics Calibration Verification (Original vs Undistorted)")
            dialog.setStyleSheet(DARK_STYLESHEET)
            
            main_layout = QVBoxLayout(dialog)

            # Left side: Image
            pixmap = QPixmap(save_path)
            scaled_pix = pixmap.scaled(1000, 750, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label = QLabel()
            img_label.setPixmap(scaled_pix)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border: 2px solid #2d2d2d; border-radius: 6px;")
            main_layout.addWidget(img_label)
            
            btn_close = QPushButton("CLOSE")
            btn_close.setMinimumHeight(40)
            btn_close.setStyleSheet("background-color: #37474f; color: white; font-weight: bold; font-size: 13px;")
            btn_close.clicked.connect(dialog.accept)
            main_layout.addWidget(btn_close)
            
            dialog.exec()
            self.log_msg(f"[INTRINSICS] Verification dialog shown. Image saved to: {save_path}")
        else:
            self.log_msg(f"[ERROR] Failed to generate verification image at: {save_path}")

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
        if hasattr(self, 'temp_timer') and self.temp_timer.isActive():
            self.temp_timer.stop()
            
        if not self.sim and self.marker_st is not None:
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

    # Enable High-DPI scaling and crisp pixmaps
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QGuiApplication, QFont
    # Note: HighDPI scaling is natively enabled in Qt6. We guard against DeprecationWarning.
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    
    # Global crisp Korean/English font with subpixel anti-aliasing
    default_font = QFont("Noto Sans CJK KR", 10)
    default_font.setStyleStrategy(QFont.PreferAntialias | QFont.PreferQuality)
    app.setFont(default_font)

    robot = None
    marker_st = None

    marker_st = Marker_Transform(sim=args.ui)
    marker_st.set_marker_type("plate")
    gui = UnifiedCalibrationApp(marker_st, robot, "right", sim=args.ui)
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
