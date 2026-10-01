from ui.core_bridge import idle_core_action
from core.storage import ResultStorage
from core.storage import StoragePaths
import os
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QGroupBox, QCheckBox, QLineEdit, QMessageBox, QDialog,
    QRadioButton, QButtonGroup, QSpinBox, QSlider
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap
from core.language import LanguageManager, tr
from core.calibration.sequences.marker_monitor import MONITOR_WINDOW

def get_asset_path(relative_path):
    return StoragePaths.asset(relative_path)

class HowToMoveArmsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.how_to_move_arms.title"))
        self.resize(750, 520)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #ffffff; font-size: 14px; }
            QGroupBox { border: 1px solid #333333; border-radius: 8px; margin-top: 15px; font-weight: bold; font-size: 15px; color: #e0e0e0; padding: 10px; background-color: #1a1a1a; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 15px; padding: 0 6px; background-color: #1e1e1e; color: #90caf9; }
            QPushButton { background-color: #1e88e5; color: white; font-weight: bold; font-size: 14px; padding: 8px 16px; border-radius: 6px; }
            QPushButton:hover { background-color: #2196f3; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl_title = QLabel(tr("dialogs.how_to_move_arms.lbl_title"))
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffd700;")
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        img_lbl = QLabel()
        pix = QPixmap(get_asset_path("img/teaching_button.png"))
        if not pix.isNull():
            img_lbl.setPixmap(pix.scaled(550, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_lbl.setText("[img/teaching_button.png]")
        img_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_lbl)

        box = QGroupBox(tr("dialogs.how_to_move_arms.box_title"))
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)

        insts = [
            tr("dialogs.how_to_move_arms.step1"),
            tr("dialogs.how_to_move_arms.step2")
        ]
        for txt in insts:
            lbl = QLabel(txt)
            lbl.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
            lbl.setWordWrap(True)
            box_layout.addWidget(lbl)

        warn_lbl = QLabel(tr("dialogs.how_to_move_arms.warn_lbl"))
        warn_lbl.setStyleSheet("font-size: 16px; color: #ff5252; font-weight: bold;")
        warn_lbl.setWordWrap(True)
        warn_lbl.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(warn_lbl)

        layout.addWidget(box)

        btn_close = QPushButton(tr("dialogs.how_to_move_arms.btn_close"))
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignCenter)

class CalibrationWizardWidget(QWidget):
    # Slide order. Everything that depends on a slide position uses these names (main_ui too),
    # so slides can be added or moved by editing this list alone.
    # 2026-09-30: the gripper-removal and marker-bracket slides are one slide, and the separate
    # intrinsics-check slide is gone: the camera slide warns when the connected unit (S/N) has no
    # intrinsics of its own and opens the optional calibration slide from a button.
    # 2026-10-01: the bracket slide comes after the robot connection, because its photo depends on
    # the robot version, which is only known once the robot is connected.
    SLIDE_CAMERA_MOUNT = 0       # 1-1 camera bracket assembly & mounting (+ intrinsics warning/button)
    SLIDE_INTRINSICS_CALIB = 1   # 1-2 intrinsics calibration (optional, entered from 1-1 only)
    SLIDE_ROBOT_CONNECT = 2      # 2 robot connection
    SLIDE_MARKER_BRACKET = 3     # 3 gripper removal & marker bracket attachment (photo by robot version)
    SLIDE_ZERO_POSE = 4          # 3-1 initial zero pose
    SLIDE_HOME_OFFSET = 5        # 3-2 home offset reset
    SLIDE_EXPOSURE = 6           # 3-3 exposure & marker recognition (arms taught into view)
    SLIDE_CALIBRATION = 7        # 4 calibration start
    SLIDE_APPLY = 8              # 5 apply home offset
    SLIDE_COUNT = 9
    # Marker bracket photo per robot version (taken from the connected robot).
    BRACKET_ASSEMBLE_IMAGES = {"1.2": "img/marker_bracket_assemble_1_2.png",
                               "1.3": "img/marker_bracket_assemble_1_3.png"}
    TITLE_KEYS = {
        SLIDE_CAMERA_MOUNT: "wizard.slides.slide_0.title",
        SLIDE_INTRINSICS_CALIB: "wizard.slides.slide_3.title",
        SLIDE_MARKER_BRACKET: "wizard.slides.slide_marker_bracket.title",
        SLIDE_ROBOT_CONNECT: "wizard.slides.slide_4.title",
        SLIDE_ZERO_POSE: "wizard.slides.slide_5.title",
        SLIDE_HOME_OFFSET: "wizard.slides.slide_6.title",
        SLIDE_EXPOSURE: "wizard.slides.slide_exposure.title",
        SLIDE_CALIBRATION: "wizard.slides.slide_7.title",
        SLIDE_APPLY: "wizard.slides.slide_8.title",
    }
    # Slides that show the live camera feed (main_ui.update_video_frame).
    VIDEO_SLIDES = (SLIDE_CAMERA_MOUNT, SLIDE_INTRINSICS_CALIB, SLIDE_EXPOSURE)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(12)

        self.lbl_wizard_title = QLabel()
        self.lbl_wizard_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffd700;")
        self.lbl_wizard_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_wizard_title)

        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget, stretch=1)

        # Navigation Layout
        self.nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton(tr("wizard.btn_prev"))
        self.btn_skip = QPushButton(tr("wizard.btn_skip"))
        self.btn_next = QPushButton(tr("wizard.btn_next"))

        # Make all navigation buttons identical in size and enlarged
        for btn in (self.btn_prev, self.btn_skip, self.btn_next):
            btn.setFixedSize(140, 45)

        self.btn_prev.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_skip.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_next.setStyleSheet("background-color: #1e88e5; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")

        self.btn_prev.clicked.connect(self.go_prev)
        self.btn_skip.clicked.connect(self.go_next)
        self.btn_next.clicked.connect(self.go_next)

        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_skip)
        self.nav_layout.addWidget(self.btn_next)

        self.layout.addLayout(self.nav_layout)

        # State tracking for each step to enable Next
        self.step_completed = [False] * self.SLIDE_COUNT
        self.step_completed[self.SLIDE_CAMERA_MOUNT] = True
        self.step_completed[self.SLIDE_MARKER_BRACKET] = True
        self.step_completed[self.SLIDE_INTRINSICS_CALIB] = False  # Optional (Skip)
        self.step_completed[self.SLIDE_ROBOT_CONNECT] = False
        self.step_completed[self.SLIDE_ZERO_POSE] = False         # Must move to the zero pose
        self.step_completed[self.SLIDE_HOME_OFFSET] = False       # Reset or Skip
        self.step_completed[self.SLIDE_EXPOSURE] = False          # Must confirm the exposure
        self.step_completed[self.SLIDE_CALIBRATION] = False
        self.step_completed[self.SLIDE_APPLY] = True

        self.check_pose_init_done = False
        self.last_marker_monitor = None

        # Unified Timer for Step 1 + Step 2 Calibration
        self.unified_timer = QTimer(self)
        self.unified_timer.timeout.connect(self.update_unified_time)
        self.unified_elapsed = 0

        # Connect language changed signal
        LanguageManager.instance().language_changed.connect(self.on_language_changed)

        self.setup_slides()
        self.stacked_widget.currentChanged.connect(self.update_navigation)
        self.update_navigation(0)

    def on_language_changed(self, lang):
        self.update_navigation(self.stacked_widget.currentIndex())

        # Slide 0
        if hasattr(self, 't0'): self.t0.setText(tr("wizard.slides.slide_0.title"))
        if hasattr(self, 'd0_box'): self.d0_box.setTitle(tr("wizard.slides.slide_0.box_title"))
        if hasattr(self, 'lbl_inst0_1'): self.lbl_inst0_1.setText(tr("wizard.slides.slide_0.inst1"))
        if hasattr(self, 'lbl_inst0_2'): self.lbl_inst0_2.setText(tr("wizard.slides.slide_0.inst2"))
        if hasattr(self, 'cam_info_box'): self.cam_info_box.setTitle(tr("wizard.slides.slide_0.cam_box_title"))
        if hasattr(self, 'lbl_cam_bracket_head'): self.lbl_cam_bracket_head.setText(tr("wizard.slides.slide_0.cap_head"))
        if hasattr(self, 'lbl_cam_bracket_nohead'): self.lbl_cam_bracket_nohead.setText(tr("wizard.slides.slide_0.cap_nohead"))
        if hasattr(self, 'btn_cam_intrinsics'): self.btn_cam_intrinsics.setText(tr("wizard.slides.slide_0.btn_intrinsics"))
        self.refresh_camera_info()

        # Gripper removal & marker bracket attachment
        if hasattr(self, 'd1_2_box'): self.d1_2_box.setTitle(tr("wizard.slides.slide_1.box_title"))
        if hasattr(self, 'lbl_m1'): self.lbl_m1.setText(tr("wizard.slides.slide_1.inst1"))
        if hasattr(self, 'mb_box'): self.mb_box.setTitle(tr("wizard.slides.slide_marker_bracket.box_title"))
        if hasattr(self, 'lbl_mb_cap_assemble'): self.lbl_mb_cap_assemble.setText(tr("wizard.slides.slide_marker_bracket.cap_assemble"))
        if hasattr(self, 'lbl_mb_cap_overview'): self.lbl_mb_cap_overview.setText(tr("wizard.slides.slide_marker_bracket.cap_overview"))
        for i, name in enumerate(("lbl_mb1", "lbl_mb2", "lbl_mb3"), start=1):
            if hasattr(self, name): getattr(self, name).setText(tr(f"wizard.slides.slide_marker_bracket.inst{i}"))

        # Exposure & marker recognition
        if hasattr(self, 't_exp'): self.t_exp.setText(tr("wizard.slides.slide_exposure.title"))
        if hasattr(self, 'd_exp'): self.d_exp.setText(tr("wizard.slides.slide_exposure.inst"))
        if hasattr(self, 'marker_mon_box'): self.marker_mon_box.setTitle(tr("wizard.slides.slide_exposure.monitor_title"))
        if hasattr(self, 'lbl_marker_mon_note'): self.lbl_marker_mon_note.setText(tr("wizard.slides.slide_exposure.monitor_note", window=MONITOR_WINDOW))
        if hasattr(self, 'btn_marker_mon_restart'): self.btn_marker_mon_restart.setText(tr("wizard.slides.slide_exposure.monitor_restart"))
        if hasattr(self, 'marker_mon_labels'): self.update_marker_monitor(self.last_marker_monitor)
        if hasattr(self, 'chk_wiz_auto_exp'): self.chk_wiz_auto_exp.setText(tr("wizard.slides.slide_exposure.auto_exposure"))
        if hasattr(self, 'lbl_wiz_exp_text'): self.lbl_wiz_exp_text.setText(tr("wizard.slides.slide_exposure.exposure_label"))
        if hasattr(self, 'btn_wiz_apply_exp'): self.btn_wiz_apply_exp.setText(tr("wizard.slides.slide_exposure.btn_apply"))
        if hasattr(self, 'btn_wiz_cancel_exp'): self.btn_wiz_cancel_exp.setText(tr("wizard.slides.slide_exposure.btn_cancel"))
        if hasattr(self, 'chk_wiz_exp_confirmed'): self.chk_wiz_exp_confirmed.setText(tr("wizard.slides.slide_exposure.chk_confirmed"))
        if hasattr(self, 'lbl_wiz_exp_status'):
            if self.step_completed[self.SLIDE_EXPOSURE]:
                self.lbl_wiz_exp_status.setText(tr("wizard.slides.slide_exposure.status_confirmed"))
            else:
                self.lbl_wiz_exp_status.setText(tr("wizard.slides.slide_exposure.status_waiting"))

        # Intrinsics calibration (optional)
        if hasattr(self, 't1'): self.t1.setText(tr("wizard.slides.slide_3.title"))
        if hasattr(self, 'lbl_skip_hint1'): self.lbl_skip_hint1.setText(tr("wizard.slides.slide_3.skip_hint"))
        if hasattr(self, 'instr_box'): self.instr_box.setTitle(tr("wizard.slides.slide_3.box_guidelines"))
        if hasattr(self, 'lbl_inst_3_1'): self.lbl_inst_3_1.setText(tr("wizard.slides.slide_3.inst1"))
        if hasattr(self, 'lbl_inst_3_2'): self.lbl_inst_3_2.setText(tr("wizard.slides.slide_3.inst2"))
        if hasattr(self, 'lbl_inst_3_3'): self.lbl_inst_3_3.setText(tr("wizard.slides.slide_3.inst3"))
        if hasattr(self, 'lbl_inst_3_4'): self.lbl_inst_3_4.setText(tr("wizard.slides.slide_3.inst4"))
        if hasattr(self, 'controls_box'): self.controls_box.setTitle(tr("wizard.slides.slide_3.box_controls"))
        if hasattr(self, 'chk_int_guide'): self.chk_int_guide.setText(tr("wizard.slides.slide_3.guide_overlay"))
        if hasattr(self, 'btn_int_capture'): self.btn_int_capture.setText(tr("wizard.slides.slide_3.btn_capture"))
        if hasattr(self, 'btn_int_calibrate'): self.btn_int_calibrate.setText(tr("wizard.slides.slide_3.btn_calibrate"))
        if hasattr(self, 'btn_int_save'): self.btn_int_save.setText(tr("wizard.slides.slide_3.btn_save"))
        if hasattr(self, 'btn_int_reset'): self.btn_int_reset.setText(tr("wizard.slides.slide_3.btn_reset"))
        if hasattr(self, 'stats_box2'): self.stats_box2.setTitle(tr("wizard.slides.slide_3.box_stats"))

        # Slide 5 (Robot Connect)
        if hasattr(self, 't2'): self.t2.setText(tr("wizard.slides.slide_4.title"))
        if hasattr(self, 'd2'): self.d2.setText(tr("wizard.slides.slide_4.inst1"))
        if hasattr(self, 'head_desc'): self.head_desc.setText(tr("wizard.slides.slide_4.head_note"))
        if hasattr(self, 'lbl_bracket_query'): self.lbl_bracket_query.setText(tr("wizard.slides.slide_4.additional_bracket_query"))
        if hasattr(self, 'rdo_bracket_yes'): self.rdo_bracket_yes.setText(tr("wizard.slides.slide_4.yes"))
        if hasattr(self, 'rdo_bracket_no'): self.rdo_bracket_no.setText(tr("wizard.slides.slide_4.no"))
        if hasattr(self, 'conn_box'): self.conn_box.setTitle(tr("wizard.slides.slide_4.box_title"))

        # Slide 6 (Zero Pose)
        if hasattr(self, 't3_1'): self.t3_1.setText(tr("wizard.slides.slide_5.title"))
        if hasattr(self, 'd3_1'): self.d3_1.setText(tr("wizard.slides.slide_5.inst1"))
        if hasattr(self, 'btn_move_zero_init'): self.btn_move_zero_init.setText(tr("wizard.slides.slide_5.btn_move_zero"))

        # Slide 7 (Home Offset Pose)
        if hasattr(self, 't3_2'): self.t3_2.setText(tr("wizard.slides.slide_6.title"))
        if hasattr(self, 'lbl_skip_hint7'): self.lbl_skip_hint7.setText(tr("wizard.slides.slide_6.skip_hint"))
        if hasattr(self, 'btn_how_to_move'): self.btn_how_to_move.setText(tr("wizard.slides.slide_6.btn_how_to_move"))
        if hasattr(self, 'inst3_2_box'): self.inst3_2_box.setTitle(tr("wizard.slides.slide_6.box_title"))
        if hasattr(self, 'lbl_p1'): self.lbl_p1.setText(tr("wizard.slides.slide_6.inst1"))
        if hasattr(self, 'lbl_p1_warn'): self.lbl_p1_warn.setText(tr("wizard.slides.slide_6.shoulder_warn"))
        if hasattr(self, 'lbl_p2'): self.lbl_p2.setText(tr("wizard.slides.slide_6.inst2"))
        if hasattr(self, 'lbl_p3'): self.lbl_p3.setText(tr("wizard.slides.slide_6.inst3"))
        if hasattr(self, 'btn_step3_reset'): self.btn_step3_reset.setText(tr("wizard.slides.slide_6.btn_reset"))

        # Slide 8 (Calibration Pipeline)
        if hasattr(self, 't4'): self.t4.setText(tr("wizard.slides.slide_7.title"))
        if hasattr(self, 'd4_step1'): self.d4_step1.setText(tr("wizard.slides.slide_7.desc"))
        if hasattr(self, 'btn_start_unified'): self.btn_start_unified.setText(tr("wizard.btn_start_calibration"))
        if hasattr(self, 'aux_box4'): self.aux_box4.setTitle(tr("wizard.safety_title"))
        if hasattr(self, 'feed_desc'): self.feed_desc.setText(tr("wizard.slides.slide_7.feed_desc"))
        if hasattr(self, 'btn_feed4'): self.btn_feed4.setText(tr("wizard.btn_open_feed"))
        if hasattr(self, 'stop_desc'): self.stop_desc.setText(tr("wizard.slides.slide_7.stop_desc"))
        if hasattr(self, 'btn_stop4'): self.btn_stop4.setText(tr("wizard.btn_stop_motion"))

        # Slide 9 (Apply Offset)
        if hasattr(self, 't6'): self.t6.setText(tr("wizard.slides.slide_8.title"))
        if hasattr(self, 'd6'): self.d6.setText(tr("wizard.slides.slide_8.desc"))
        if hasattr(self, 'apply_instructions_box'): self.apply_instructions_box.setTitle(tr("wizard.slides.slide_8.box_title"))
        if hasattr(self, 'lbl_apply1'): self.lbl_apply1.setText(tr("wizard.slides.slide_8.inst1"))
        if hasattr(self, 'lbl_apply2'): self.lbl_apply2.setText(tr("wizard.slides.slide_8.inst2"))
        if hasattr(self, 'lbl_apply3'): self.lbl_apply3.setText(tr("wizard.slides.slide_8.inst3"))
        if hasattr(self, 'lbl_apply4'): self.lbl_apply4.setText(tr("wizard.slides.slide_8.inst4"))
        if hasattr(self, 'btn_rollback_zero'):
            self.btn_rollback_zero.setText(tr("wizard.slides.slide_8.btn_rollback_zero"))
        if hasattr(self, 'btn_new_offset_zero'):
            self.btn_new_offset_zero.setText(tr("wizard.slides.slide_8.btn_new_offset_zero"))
        if hasattr(self, 'btn_rollback_preview'):
            self.btn_rollback_preview.setText(tr("wizard.slides.slide_8.btn_rollback_preview"))
        if hasattr(self, 'btn_new_offset_preview'):
            self.btn_new_offset_preview.setText(tr("wizard.slides.slide_8.btn_new_offset_preview"))
        if hasattr(self, 'btn_rollback_joint'):
            self.btn_rollback_joint.setText(tr("wizard.slides.slide_8.btn_rollback_joint"))
        if hasattr(self, 'btn_apply_new_offset'):
            self.btn_apply_new_offset.setText(tr("wizard.slides.slide_8.btn_apply_new_offset"))

    @staticmethod
    def image_label(path, width, height):
        lbl = QLabel()
        pix = QPixmap(get_asset_path(path))
        if not pix.isNull():
            lbl.setPixmap(pix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl.setText(f"[{path} not found]")
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    @staticmethod
    def caption_label(text, style="font-size: 14px; color: #ffd700; font-weight: bold;"):
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    def setup_slides(self):
        # Built in the order below for readability, added to the stack in SLIDE_* order at the end.
        slides = {}

        # -----------------------------------------
        # 1-1. Camera Bracket Assembly & Mounting
        # -----------------------------------------
        slide0 = QWidget()
        l0 = QVBoxLayout(slide0)
        l0.setSpacing(14)
        l0.setAlignment(Qt.AlignCenter)

        self.t0 = QLabel(tr("wizard.slides.slide_0.title"))
        self.t0.setVisible(False)

        # Robot with a head (camera on the head bracket) | without a head (camera on the body).
        cam_bracket_row = QHBoxLayout()
        cam_bracket_row.setSpacing(24)
        cam_bracket_row.setAlignment(Qt.AlignCenter)
        for image, caption_key, attr in (("img/camera_bracket_head.png", "wizard.slides.slide_0.cap_head", "lbl_cam_bracket_head"),
                                         ("img/camera_bracket_nohead.png", "wizard.slides.slide_0.cap_nohead", "lbl_cam_bracket_nohead")):
            col = QVBoxLayout()
            col.setSpacing(6)
            col.addWidget(self.image_label(image, 380, 200))
            caption = self.caption_label(tr(caption_key))
            setattr(self, attr, caption)
            col.addWidget(caption)
            cam_bracket_row.addLayout(col)
        l0.addLayout(cam_bracket_row)

        # What the program actually recognised, so a D435 vs D435I/F or a USB 2 link is visible
        # before calibrating instead of only in the log.
        self.cam_info_box = QGroupBox(tr("wizard.slides.slide_0.cam_box_title"))
        self.cam_info_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.cam_info_box.setFixedWidth(750)
        cam_layout = QVBoxLayout(self.cam_info_box)
        cam_layout.setSpacing(6)
        self.lbl_cam_info = QLabel()
        self.lbl_cam_info.setWordWrap(True)
        self.lbl_cam_info.setTextFormat(Qt.RichText)
        cam_layout.addWidget(self.lbl_cam_info)
        l0.addWidget(self.cam_info_box, alignment=Qt.AlignCenter)
        # A camera without intrinsics of its own (S/N) is warned about in lbl_cam_info; the optional
        # calibration slide is opened from here (the separate 1-4 check slide was removed).
        self.btn_cam_intrinsics = QPushButton(tr("wizard.slides.slide_0.btn_intrinsics"))
        self.btn_cam_intrinsics.setStyleSheet("background-color: #fb8c00; color: #000000; font-weight: bold; font-size: 15px; padding: 10px 20px; border-radius: 6px;")
        self.btn_cam_intrinsics.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(self.SLIDE_INTRINSICS_CALIB))
        cam_layout.addWidget(self.btn_cam_intrinsics, alignment=Qt.AlignCenter)
        self.cam_info_timer = QTimer(self)
        self.cam_info_timer.timeout.connect(self.refresh_camera_info)
        self.cam_info_timer.start(1000)
        self.refresh_camera_info()

        self.d0_box = QGroupBox(tr("wizard.slides.slide_0.box_title"))
        self.d0_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.d0_box.setFixedWidth(750)
        d0_layout = QVBoxLayout(self.d0_box)
        d0_layout.setSpacing(8)

        self.lbl_inst0_1 = QLabel(tr("wizard.slides.slide_0.inst1"))
        self.lbl_inst0_1.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_inst0_1.setWordWrap(True)
        d0_layout.addWidget(self.lbl_inst0_1)

        self.lbl_inst0_2 = QLabel(tr("wizard.slides.slide_0.inst2"))
        self.lbl_inst0_2.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_inst0_2.setWordWrap(True)
        d0_layout.addWidget(self.lbl_inst0_2)

        l0.addWidget(self.d0_box, alignment=Qt.AlignCenter)
        slides[self.SLIDE_CAMERA_MOUNT] = slide0

        # -----------------------------------------
        # 1-2. Gripper Removal & Marker Bracket Attachment
        # -----------------------------------------
        slide_mb = QWidget()
        l_mb = QVBoxLayout(slide_mb)
        l_mb.setSpacing(10)
        l_mb.setAlignment(Qt.AlignCenter)

        self.d1_2_box = QGroupBox(tr("wizard.slides.slide_1.box_title"))
        self.d1_2_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.d1_2_box.setFixedWidth(900)
        d1_2_layout = QVBoxLayout(self.d1_2_box)
        d1_2_layout.setSpacing(8)
        self.lbl_m1 = QLabel(tr("wizard.slides.slide_1.inst1"))
        self.lbl_m1.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_m1.setWordWrap(True)
        self.lbl_m1.setOpenExternalLinks(True)
        d1_2_layout.addWidget(self.lbl_m1)
        l_mb.addWidget(self.d1_2_box, alignment=Qt.AlignCenter)

        # How the bracket bolts to the flange (photo of the connected robot's version: this slide
        # comes after the robot connection) | which marker goes on which arm.
        mb_row = QHBoxLayout()
        mb_row.setSpacing(24)
        mb_row.setAlignment(Qt.AlignCenter)
        col = QVBoxLayout()
        col.setSpacing(6)
        self.img_mb_assemble = self.image_label(self.BRACKET_ASSEMBLE_IMAGES["1.2"], 400, 180)
        col.addWidget(self.img_mb_assemble)
        self.lbl_mb_cap_assemble = self.caption_label(tr("wizard.slides.slide_marker_bracket.cap_assemble"))
        col.addWidget(self.lbl_mb_cap_assemble)
        mb_row.addLayout(col)
        col = QVBoxLayout()
        col.setSpacing(6)
        col.addWidget(self.image_label("img/marker_bracket_overview.png", 400, 180))
        self.lbl_mb_cap_overview = self.caption_label(tr("wizard.slides.slide_marker_bracket.cap_overview"))
        col.addWidget(self.lbl_mb_cap_overview)
        mb_row.addLayout(col)
        l_mb.addLayout(mb_row)
        self.bracket_version = "1.2"

        self.mb_box = QGroupBox(tr("wizard.slides.slide_marker_bracket.box_title"))
        self.mb_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.mb_box.setFixedWidth(900)
        mb_layout = QVBoxLayout(self.mb_box)
        mb_layout.setSpacing(8)
        for i in (1, 2, 3):
            lbl = QLabel(tr(f"wizard.slides.slide_marker_bracket.inst{i}"))
            lbl.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
            lbl.setWordWrap(True)
            setattr(self, f"lbl_mb{i}", lbl)
            mb_layout.addWidget(lbl)
        l_mb.addWidget(self.mb_box, alignment=Qt.AlignCenter)
        slides[self.SLIDE_MARKER_BRACKET] = slide_mb

        # -----------------------------------------
        # 3-3. Camera Exposure & Marker Recognition (after the home offset reset: the robot is
        # connected, so the arms can be taught into view to judge the brightness on the markers)
        # -----------------------------------------
        slide_exp = QWidget()
        slide_exp_layout = QVBoxLayout(slide_exp)
        slide_exp_layout.setSpacing(10)

        self.t_exp = QLabel(tr("wizard.slides.slide_exposure.title"))
        self.t_exp.setVisible(False)

        self.d_exp = QLabel(tr("wizard.slides.slide_exposure.inst"))
        self.d_exp.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.d_exp.setWordWrap(True)
        self.d_exp.setAlignment(Qt.AlignCenter)
        slide_exp_layout.addWidget(self.d_exp)

        content_exp_layout = QHBoxLayout()

        # Left: Live Camera Feed
        exp_left = QVBoxLayout()
        self.wizard_exposure_video_label = QLabel("Camera Feed Loading...")
        self.wizard_exposure_video_label.setAlignment(Qt.AlignCenter)
        self.wizard_exposure_video_label.setMinimumSize(480, 290)
        self.wizard_exposure_video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        exp_left.addWidget(self.wizard_exposure_video_label, 1)

        # Live recognition and jitter of both markers, measured by the core (MarkerMonitor).
        self.marker_mon_box = QGroupBox(tr("wizard.slides.slide_exposure.monitor_title"))
        self.marker_mon_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 15px;}")
        mon_layout = QVBoxLayout(self.marker_mon_box)
        mon_layout.setSpacing(6)
        self.marker_mon_labels = {}
        for side in ("right", "left"):
            lbl = QLabel()
            lbl.setStyleSheet(self.MONITOR_IDLE_STYLE)
            self.marker_mon_labels[side] = lbl
            mon_layout.addWidget(lbl)
        self.lbl_marker_mon_overall = QLabel()
        self.lbl_marker_mon_overall.setStyleSheet(self.MONITOR_IDLE_STYLE)
        mon_layout.addWidget(self.lbl_marker_mon_overall)
        self.lbl_marker_mon_note = QLabel(tr("wizard.slides.slide_exposure.monitor_note", window=MONITOR_WINDOW))
        self.lbl_marker_mon_note.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        self.lbl_marker_mon_note.setWordWrap(True)
        mon_layout.addWidget(self.lbl_marker_mon_note)
        # A calibration sequence or camera reconnect stops the monitor; restart it by hand.
        self.btn_marker_mon_restart = QPushButton(tr("wizard.slides.slide_exposure.monitor_restart"))
        self.btn_marker_mon_restart.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 13px; border-radius: 6px; padding: 4px 12px;")
        self.btn_marker_mon_restart.clicked.connect(self.sync_marker_monitor)
        mon_layout.addWidget(self.btn_marker_mon_restart, alignment=Qt.AlignRight)
        exp_left.addWidget(self.marker_mon_box)
        self.update_marker_monitor(None)
        content_exp_layout.addLayout(exp_left, 3)

        # Right: Exposure Controls
        exp_right = QVBoxLayout()
        exp_ctrl_box = QGroupBox(tr("wizard.slides.slide_exposure.title"))
        exp_ctrl_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 15px;}")
        exp_ctrl_layout = QVBoxLayout()
        exp_ctrl_layout.setSpacing(10)

        self.chk_wiz_auto_exp = QCheckBox(tr("wizard.slides.slide_exposure.auto_exposure"))
        self.chk_wiz_auto_exp.setChecked(True)
        self.chk_wiz_auto_exp.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px;")
        self.chk_wiz_auto_exp.toggled.connect(self.on_wiz_auto_exp_toggled)
        exp_ctrl_layout.addWidget(self.chk_wiz_auto_exp)

        spin_row = QHBoxLayout()
        self.lbl_wiz_exp_text = QLabel(tr("wizard.slides.slide_exposure.exposure_label"))
        self.lbl_wiz_exp_text.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        spin_row.addWidget(self.lbl_wiz_exp_text)

        self.spin_wiz_exp = QSpinBox()
        self.spin_wiz_exp.setRange(1, 200000)  # narrowed to the camera's own range once connected
        self.spin_wiz_exp.setSingleStep(500)
        self.spin_wiz_exp.setValue(6000)
        self.spin_wiz_exp.setEnabled(False)
        self.spin_wiz_exp.setStyleSheet("background-color: #1e1e1e; color: #00e5ff; font-weight: bold; font-size: 14px; padding: 4px;")
        self.spin_wiz_exp.valueChanged.connect(self.on_wiz_exposure_changed)
        spin_row.addWidget(self.spin_wiz_exp)

        self.lbl_wiz_exp_ms = QLabel("6.0 ms")
        self.lbl_wiz_exp_ms.setStyleSheet("color: #ffd700; font-weight: bold; font-size: 14px; min-width: 55px;")
        spin_row.addWidget(self.lbl_wiz_exp_ms)
        exp_ctrl_layout.addLayout(spin_row)

        self.slider_wiz_exp = QSlider(Qt.Horizontal)
        self.slider_wiz_exp.setRange(1, 200000)
        self.slider_wiz_exp.setSingleStep(500)
        self.slider_wiz_exp.setPageStep(5000)
        self.slider_wiz_exp.setValue(6000)
        self.slider_wiz_exp.setEnabled(False)
        self.slider_wiz_exp.valueChanged.connect(self.on_wiz_exposure_changed)
        exp_ctrl_layout.addWidget(self.slider_wiz_exp)

        # Filled in by the main window from the connected camera (its range and unit).
        self.lbl_wiz_exp_range = QLabel("")
        self.lbl_wiz_exp_range.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        exp_ctrl_layout.addWidget(self.lbl_wiz_exp_range)

        # Action Buttons Row (APPLY, CANCEL)
        wiz_btn_row = QHBoxLayout()
        self.btn_wiz_apply_exp = QPushButton(tr("wizard.slides.slide_exposure.btn_apply"))
        self.btn_wiz_apply_exp.setMinimumHeight(40)
        self.btn_wiz_apply_exp.setStyleSheet("background-color: #388e3c; color: white; font-weight: bold; font-size: 14px; border-radius: 6px;")
        self.btn_wiz_apply_exp.clicked.connect(self.on_wiz_apply_exp_clicked)
        wiz_btn_row.addWidget(self.btn_wiz_apply_exp)

        self.btn_wiz_cancel_exp = QPushButton(tr("wizard.slides.slide_exposure.btn_cancel"))
        self.btn_wiz_cancel_exp.setMinimumHeight(40)
        self.btn_wiz_cancel_exp.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 14px; border-radius: 6px;")
        self.btn_wiz_cancel_exp.clicked.connect(self.on_wiz_cancel_exp_clicked)
        wiz_btn_row.addWidget(self.btn_wiz_cancel_exp)
        exp_ctrl_layout.addLayout(wiz_btn_row)

        exp_ctrl_layout.addStretch()

        self.lbl_wiz_exp_status = QLabel(tr("wizard.slides.slide_exposure.status_waiting"))
        self.lbl_wiz_exp_status.setStyleSheet("color: #ff9800; font-size: 13px; font-weight: bold;")
        self.lbl_wiz_exp_status.setWordWrap(True)
        self.lbl_wiz_exp_status.setAlignment(Qt.AlignCenter)
        exp_ctrl_layout.addWidget(self.lbl_wiz_exp_status)

        # Confirmation Checkbox to Enable Next Button
        self.chk_wiz_exp_confirmed = QCheckBox(tr("wizard.slides.slide_exposure.chk_confirmed"))
        self.chk_wiz_exp_confirmed.setChecked(False)
        self.chk_wiz_exp_confirmed.setStyleSheet("""
            QCheckBox {
                color: #80d8ff;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border: 2px solid #0091ea;
                border-radius: 6px;
                background-color: #0d2744;
            }
            QCheckBox:hover {
                border: 2px solid #00e5ff;
            }
            QCheckBox::indicator {
                width: 22px;
                height: 22px;
                border: 2px solid #616161;
                border-radius: 4px;
                background-color: #212121;
            }
            QCheckBox::indicator:checked {
                background-color: #00b0ff;
                border: 2px solid #80d8ff;
            }
        """)
        self.chk_wiz_exp_confirmed.toggled.connect(self.on_wiz_exp_confirmed_toggled)
        exp_ctrl_layout.addWidget(self.chk_wiz_exp_confirmed)

        exp_ctrl_box.setLayout(exp_ctrl_layout)
        exp_right.addWidget(exp_ctrl_box)
        content_exp_layout.addLayout(exp_right, 2)

        slide_exp_layout.addLayout(content_exp_layout)
        slides[self.SLIDE_EXPOSURE] = slide_exp

        # -----------------------------------------
        # Slide 3: Camera Intrinsics Calibration (Optional)
        # -----------------------------------------
        slide1 = QWidget()
        slide1_layout = QVBoxLayout(slide1)

        header1 = QVBoxLayout()
        self.t1 = QLabel(tr("wizard.slides.slide_3.title"))
        self.t1.setVisible(False)

        self.lbl_skip_hint1 = QLabel(tr("wizard.slides.slide_3.skip_hint"))
        self.lbl_skip_hint1.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 20px;")
        self.lbl_skip_hint1.setWordWrap(True)
        self.lbl_skip_hint1.setAlignment(Qt.AlignCenter)
        header1.addWidget(self.lbl_skip_hint1)
        slide1_layout.addLayout(header1)

        content1_layout = QHBoxLayout()

        int_left = QVBoxLayout()
        self.wizard_video_label = QLabel("Camera Feed Loading...")
        self.wizard_video_label.setAlignment(Qt.AlignCenter)
        self.wizard_video_label.setMinimumSize(480, 300)
        self.wizard_video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        int_left.addWidget(self.wizard_video_label, 3)

        self.instr_box = QGroupBox(tr("wizard.slides.slide_3.box_guidelines"))
        self.instr_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        instr_layout = QVBoxLayout()
        self.lbl_inst_3_1 = QLabel(tr("wizard.slides.slide_3.inst1"))
        self.lbl_inst_3_1.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_1.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_1)

        self.lbl_inst_3_2 = QLabel(tr("wizard.slides.slide_3.inst2"))
        self.lbl_inst_3_2.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_2.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_2)

        self.lbl_inst_3_3 = QLabel(tr("wizard.slides.slide_3.inst3"))
        self.lbl_inst_3_3.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_3.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_3)

        self.lbl_inst_3_4 = QLabel(tr("wizard.slides.slide_3.inst4"))
        self.lbl_inst_3_4.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_4.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_4)

        self.instr_box.setLayout(instr_layout)
        int_left.addWidget(self.instr_box, 1)

        self.controls_box = QGroupBox(tr("wizard.slides.slide_3.box_controls"))
        self.controls_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        controls_layout = QVBoxLayout()

        self.chk_int_guide = QCheckBox(tr("wizard.slides.slide_3.guide_overlay"))
        self.chk_int_guide.setChecked(True)
        self.chk_int_guide.setStyleSheet("color: #00e5ff; font-size: 15px; font-weight: bold;")
        self.chk_int_guide.stateChanged.connect(self.parent_app.on_guide_changed)
        controls_layout.addWidget(self.chk_int_guide)

        self.btn_int_capture = QPushButton(tr("wizard.slides.slide_3.btn_capture"))
        self.btn_int_capture.setMinimumHeight(45)
        self.btn_int_capture.setStyleSheet("background-color: #1e88e5; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_capture.clicked.connect(self.step1_capture)
        controls_layout.addWidget(self.btn_int_capture)

        self.btn_int_calibrate = QPushButton(tr("wizard.slides.slide_3.btn_calibrate"))
        self.btn_int_calibrate.setMinimumHeight(45)
        self.btn_int_calibrate.setStyleSheet("background-color: #43a047; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_calibrate.clicked.connect(self.step1_run)
        controls_layout.addWidget(self.btn_int_calibrate)

        self.btn_int_save = QPushButton(tr("wizard.slides.slide_3.btn_save"))
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #fb8c00; color: #000000; font-size: 14px; font-weight: bold;")
        self.btn_int_save.clicked.connect(self.step1_save)
        controls_layout.addWidget(self.btn_int_save)

        self.btn_int_reset = QPushButton(tr("wizard.slides.slide_3.btn_reset"))
        self.btn_int_reset.setMinimumHeight(35)
        self.btn_int_reset.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 13px;")
        self.btn_int_reset.clicked.connect(self.parent_app.reset_intrinsics_captures)
        controls_layout.addWidget(self.btn_int_reset)

        self.controls_box.setLayout(controls_layout)

        int_right = QVBoxLayout()

        self.stats_box2 = QGroupBox(tr("wizard.slides.slide_3.box_stats"))
        self.stats_box2.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        stats_layout2 = QHBoxLayout()
        self.lbl_captured = QLabel(tr("wizard.slides.slide_3.lbl_captured") + "0 / 16")
        self.lbl_captured.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #448aff;")

        self.lbl_temp = QLabel(self.parent_app.get_temp_label_text())
        self.lbl_temp.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #fb8c00;")

        stats_layout2.addWidget(self.lbl_captured)
        stats_layout2.addStretch()
        stats_layout2.addWidget(self.lbl_temp)
        self.stats_box2.setLayout(stats_layout2)

        self.lbl_step1_status = QLabel("Status: Waiting for Capture (Need 16 Frames)")
        self.lbl_step1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step1_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")

        int_right.addWidget(self.stats_box2)
        int_right.addWidget(self.controls_box)
        int_right.addStretch()
        int_right.addWidget(self.lbl_step1_status)
        int_right.addStretch()

        content1_layout.addLayout(int_left, 2)
        content1_layout.addLayout(int_right, 1)
        slide1_layout.addLayout(content1_layout, 1)
        slides[self.SLIDE_INTRINSICS_CALIB] = slide1

        # -----------------------------------------
        # Slide 4: Robot Connection
        # -----------------------------------------
        slide2 = QWidget()
        l2 = QVBoxLayout(slide2)
        l2.setSpacing(12)
        l2.setAlignment(Qt.AlignCenter)

        self.t2 = QLabel(tr("wizard.slides.slide_4.title"))
        self.t2.setVisible(False)

        self.d2 = QLabel(tr("wizard.slides.slide_4.inst1"))
        self.d2.setStyleSheet("font-size: 16px; color: #dddddd;")
        self.d2.setWordWrap(True)
        self.d2.setAlignment(Qt.AlignCenter)
        l2.addWidget(self.d2)

        self.lbl_step2_status = QLabel("Status: Waiting")
        self.lbl_step2_status.setAlignment(Qt.AlignCenter)
        self.lbl_step2_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l2.addWidget(self.lbl_step2_status)

        # Question Label (Centered, Yellow, Bold)
        self.lbl_bracket_query = QLabel(tr("wizard.slides.slide_4.additional_bracket_query"))
        self.lbl_bracket_query.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffd700; margin-top: 10px; margin-bottom: 5px;")
        self.lbl_bracket_query.setAlignment(Qt.AlignCenter)
        l2.addWidget(self.lbl_bracket_query)

        # 2-Column Image + Radio Button Layout
        bracket_layout = QHBoxLayout()
        bracket_layout.setAlignment(Qt.AlignCenter)
        bracket_layout.setSpacing(40)  # Generous spacing between columns

        rdo_style = """
            QRadioButton {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 2px solid #444444;
                border-radius: 6px;
                padding: 8px 30px;
                font-size: 15px;
                font-weight: bold;
            }
            QRadioButton::indicator {
                width: 0px;
                height: 0px;
            }
            QRadioButton:checked {
                background-color: #1e88e5;
                color: #ffffff;
                border: 2px solid #1e88e5;
            }
            QRadioButton:hover {
                border: 2px solid #448aff;
            }
        """

        # Left Column: Yes (Additional Bracket)
        col_yes = QVBoxLayout()
        col_yes.setAlignment(Qt.AlignCenter)
        col_yes.setSpacing(10)

        img_yes = QLabel()
        pix_yes = QPixmap(get_asset_path("img/additional_head_bracket.png"))
        if not pix_yes.isNull():
            img_yes.setPixmap(pix_yes.scaled(350, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_yes.setText("[additional_head_bracket.png not found]")
        img_yes.setAlignment(Qt.AlignCenter)
        img_yes.setStyleSheet("border: 1px solid #444444; border-radius: 4px; background-color: #1e1e1e;")
        col_yes.addWidget(img_yes)

        self.rdo_bracket_yes = QRadioButton(tr("wizard.slides.slide_4.yes"))
        self.rdo_bracket_yes.setStyleSheet(rdo_style)
        col_yes.addWidget(self.rdo_bracket_yes, alignment=Qt.AlignCenter)

        # Right Column: No (Standard/Direct Mount)
        col_no = QVBoxLayout()
        col_no.setAlignment(Qt.AlignCenter)
        col_no.setSpacing(10)

        img_no = QLabel()
        pix_no = QPixmap(get_asset_path("img/standard_head_bracket.png"))
        if not pix_no.isNull():
            img_no.setPixmap(pix_no.scaled(350, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_no.setText("[standard_head_bracket.png not found]")
        img_no.setAlignment(Qt.AlignCenter)
        img_no.setStyleSheet("border: 1px solid #444444; border-radius: 4px; background-color: #1e1e1e;")
        col_no.addWidget(img_no)

        self.rdo_bracket_no = QRadioButton(tr("wizard.slides.slide_4.no"))
        self.rdo_bracket_no.setStyleSheet(rdo_style)
        col_no.addWidget(self.rdo_bracket_no, alignment=Qt.AlignCenter)

        bracket_layout.addLayout(col_yes)
        bracket_layout.addLayout(col_no)
        l2.addLayout(bracket_layout)

        # Group them
        self.bracket_btn_group = QButtonGroup(self)
        self.bracket_btn_group.addButton(self.rdo_bracket_yes)
        self.bracket_btn_group.addButton(self.rdo_bracket_no)

        self.rdo_bracket_no.setChecked(True)
        self.rdo_bracket_yes.toggled.connect(self.on_bracket_radio_changed)
        self.rdo_bracket_no.toggled.connect(self.on_bracket_radio_changed)

        self.conn_box = QGroupBox(tr("wizard.slides.slide_4.box_title"))
        self.conn_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        self.conn_box.setFixedWidth(600)
        conn_layout = QVBoxLayout()
        conn_layout.setSpacing(10)

        ip_row = QHBoxLayout()
        lbl_ip = QLabel("IP/Port:")
        lbl_ip.setStyleSheet("font-size: 15px; font-weight: bold;")
        ip_row.addWidget(lbl_ip)

        is_sim_or_ui = self.parent_app.ui_only or getattr(self.parent_app.core.observer, 'sim', False)
        default_ip = "127.0.0.1:50051" if is_sim_or_ui else "192.168.30.1:50051"
        self.wizard_ip_input = QLineEdit(default_ip)
        self.wizard_ip_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px; font-size: 15px;")
        ip_row.addWidget(self.wizard_ip_input)
        conn_layout.addLayout(ip_row)

        connect_row = QHBoxLayout()
        self.btn_wizard_connect = QPushButton("CONNECT")
        self.btn_wizard_connect.setMinimumWidth(160)
        self.btn_wizard_connect.setStyleSheet("background-color: #1e88e5; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px;")
        self.btn_wizard_connect.clicked.connect(self.step2_connect)
        connect_row.addWidget(self.btn_wizard_connect)

        self.wizard_chk_head = QCheckBox("Head")
        self.wizard_chk_head.setChecked(True)
        self.wizard_chk_head.setVisible(False)
        self.wizard_chk_head.toggled.connect(self.sync_bracket_radio)
        connect_row.addWidget(self.wizard_chk_head)
        conn_layout.addLayout(connect_row)

        self.conn_box.setLayout(conn_layout)
        l2.addWidget(self.conn_box, alignment=Qt.AlignCenter)

        slides[self.SLIDE_ROBOT_CONNECT] = slide2

        # -----------------------------------------
        # Slide 5: 3-1. Initial Zero Position
        # -----------------------------------------
        slide3_1 = QWidget()
        l3_1 = QVBoxLayout(slide3_1)
        l3_1.setSpacing(14)
        l3_1.setAlignment(Qt.AlignCenter)

        self.t3_1 = QLabel(tr("wizard.slides.slide_5.title"))
        self.t3_1.setVisible(False)

        self.d3_1 = QLabel(tr("wizard.slides.slide_5.inst1"))
        self.d3_1.setStyleSheet("font-size: 16px; color: #dddddd; font-weight: bold;")
        self.d3_1.setWordWrap(True)
        self.d3_1.setAlignment(Qt.AlignCenter)
        l3_1.addWidget(self.d3_1)

        self.lbl_step3_1_status = QLabel("Status: Waiting for Zero Position Move")
        self.lbl_step3_1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step3_1_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l3_1.addWidget(self.lbl_step3_1_status)

        self.btn_move_zero_init = QPushButton("Move to Zero Position")
        self.btn_move_zero_init.setMinimumWidth(260)
        self.btn_move_zero_init.setMinimumHeight(45)
        self.btn_move_zero_init.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 15px;")
        self.btn_move_zero_init.clicked.connect(self.step3_1_move_zero)
        l3_1.addWidget(self.btn_move_zero_init, alignment=Qt.AlignCenter)

        slides[self.SLIDE_ZERO_POSE] = slide3_1

        # -----------------------------------------
        # Slide 6: 3-2. Home Offset Position Setup
        # -----------------------------------------
        slide3_2 = QWidget()
        l3_2 = QVBoxLayout(slide3_2)
        l3_2.setSpacing(10)

        self.t3_2 = QLabel(tr("wizard.slides.slide_6.title"))
        self.t3_2.setVisible(False)

        self.lbl_skip_hint7 = QLabel(tr("wizard.slides.slide_6.skip_hint"))
        self.lbl_skip_hint7.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 20px;")
        self.lbl_skip_hint7.setWordWrap(True)
        self.lbl_skip_hint7.setAlignment(Qt.AlignCenter)
        l3_2.addWidget(self.lbl_skip_hint7)

        self.lbl_step7_status = QLabel("Status: Waiting")
        self.lbl_step7_status.setAlignment(Qt.AlignCenter)
        self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l3_2.addWidget(self.lbl_step7_status)

        row3_2 = QHBoxLayout()
        row3_2.setSpacing(15)

        # One column per joint: its photo, then which way to turn it.
        joint_style = "font-size: 15px; color: #ffffff; font-weight: bold;"
        shoulder_col = QVBoxLayout()
        shoulder_col.setSpacing(6)
        shoulder_col.addWidget(self.image_label("img/offset_reset_pose_shoulder_roll.png", 330, 250))
        self.lbl_p1 = self.caption_label(tr("wizard.slides.slide_6.inst1"), joint_style)
        shoulder_col.addWidget(self.lbl_p1)
        self.lbl_p1_warn = self.caption_label(tr("wizard.slides.slide_6.shoulder_warn"),
                                              "font-size: 14px; color: #ff5252; font-weight: bold;")
        shoulder_col.addWidget(self.lbl_p1_warn)
        shoulder_col.addStretch()
        row3_2.addLayout(shoulder_col, 1)

        elbow_col = QVBoxLayout()
        elbow_col.setSpacing(6)
        elbow_col.addWidget(self.image_label("img/offset_reset_pose_elbow.png", 330, 250))
        self.lbl_p2 = self.caption_label(tr("wizard.slides.slide_6.inst2"), joint_style)
        elbow_col.addWidget(self.lbl_p2)
        elbow_col.addStretch()
        row3_2.addLayout(elbow_col, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        self.btn_how_to_move = QPushButton(tr("wizard.slides.slide_6.btn_how_to_move"))
        self.btn_how_to_move.setMinimumHeight(45)
        self.btn_how_to_move.setStyleSheet("background-color: #448aff; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 15px;")
        self.btn_how_to_move.clicked.connect(self.show_how_to_move_arms_dialog)
        right_col.addWidget(self.btn_how_to_move, alignment=Qt.AlignRight)

        self.inst3_2_box = QGroupBox(tr("wizard.slides.slide_6.box_title"))
        self.inst3_2_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        inst3_2_layout = QVBoxLayout(self.inst3_2_box)
        inst3_2_layout.setSpacing(10)

        self.lbl_p3 = QLabel(tr("wizard.slides.slide_6.inst3"))
        self.lbl_p3.setStyleSheet("font-size: 15px; color: #ffd700; font-weight: bold;")
        self.lbl_p3.setWordWrap(True)
        inst3_2_layout.addWidget(self.lbl_p3)

        right_col.addWidget(self.inst3_2_box)
        right_col.addStretch()
        row3_2.addLayout(right_col, 1)
        l3_2.addLayout(row3_2)

        self.btn_step3_reset = QPushButton(tr("wizard.slides.slide_6.btn_reset"))
        self.btn_step3_reset.setMinimumWidth(260)
        self.btn_step3_reset.setMinimumHeight(45)
        self.btn_step3_reset.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 15px;")
        self.btn_step3_reset.clicked.connect(self.step3_reset)
        l3_2.addWidget(self.btn_step3_reset, alignment=Qt.AlignCenter)

        slides[self.SLIDE_HOME_OFFSET] = slide3_2

        # -----------------------------------------
        # Slide 7: 4. Calibration Start (Unified Step 1 + Step 2)
        # -----------------------------------------
        slide4 = QWidget()
        l4 = QVBoxLayout(slide4)
        l4.setAlignment(Qt.AlignCenter)
        l4.setSpacing(16)

        self.t4 = QLabel(tr("wizard.slides.slide_7.title"))
        self.t4.setVisible(False)

        d4_layout = QVBoxLayout()
        self.d4_step1 = QLabel(tr("wizard.slides.slide_7.desc"))
        self.d4_step1.setStyleSheet("font-size: 16px; color: #ffffff; font-weight: bold;")
        self.d4_step1.setWordWrap(True)
        self.d4_step1.setAlignment(Qt.AlignCenter)
        d4_layout.addWidget(self.d4_step1)
        l4.addLayout(d4_layout)

        self.lbl_step4_status = QLabel("Status: Waiting")
        self.lbl_step4_status.setAlignment(Qt.AlignCenter)
        self.lbl_step4_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l4.addWidget(self.lbl_step4_status)

        action_row4 = QHBoxLayout()
        self.btn_start_unified = QPushButton(tr("wizard.btn_start_calibration"))
        self.btn_start_unified.setMinimumHeight(50)
        self.btn_start_unified.setMinimumWidth(260)
        self.btn_start_unified.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 18px; border-radius: 6px; padding: 0 15px;")
        self.btn_start_unified.clicked.connect(self.start_unified_calibration)

        action_row4.addStretch()
        action_row4.addWidget(self.btn_start_unified)
        action_row4.addStretch()
        l4.addLayout(action_row4)

        self.aux_box4 = QGroupBox(tr("wizard.safety_title"))
        self.aux_box4.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 15px;}")
        self.aux_box4.setFixedWidth(640)
        aux_layout4 = QVBoxLayout()
        aux_layout4.setSpacing(10)

        feed_row = QHBoxLayout()
        feed_row.addStretch()
        self.feed_desc = QLabel(tr("wizard.slides.slide_7.feed_desc"))
        self.feed_desc.setStyleSheet("font-size: 14px; color: #ffd700; font-weight: bold;")
        feed_row.addWidget(self.feed_desc)
        self.btn_feed4 = QPushButton(tr("wizard.btn_open_feed"))
        self.btn_feed4.setMinimumWidth(160)
        self.btn_feed4.setStyleSheet("background-color: #fb8c00; color: #000000; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        self.btn_feed4.clicked.connect(self.parent_app.toggle_camera_feed_dialog)
        feed_row.addWidget(self.btn_feed4)
        aux_layout4.addLayout(feed_row)

        stop_row = QHBoxLayout()
        stop_row.addStretch()
        self.stop_desc = QLabel(tr("wizard.slides.slide_7.stop_desc"))
        self.stop_desc.setStyleSheet("font-size: 14px; color: #ff5252; font-weight: bold;")
        stop_row.addWidget(self.stop_desc)
        self.btn_stop4 = QPushButton(tr("wizard.btn_stop_motion"))
        self.btn_stop4.setMinimumWidth(160)
        self.btn_stop4.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        self.btn_stop4.clicked.connect(self.stop_unified_calibration)
        stop_row.addWidget(self.btn_stop4)
        aux_layout4.addLayout(stop_row)

        self.aux_box4.setLayout(aux_layout4)
        l4.addWidget(self.aux_box4, alignment=Qt.AlignCenter)

        slides[self.SLIDE_CALIBRATION] = slide4

        # -----------------------------------------
        # Slide 8: 5. Apply Home Offset
        # -----------------------------------------
        slide6 = QWidget()
        l6 = QVBoxLayout(slide6)
        l6.setAlignment(Qt.AlignCenter)
        l6.setSpacing(12)

        self.t6 = QLabel(tr("wizard.slides.slide_8.title"))
        self.t6.setVisible(False)

        self.d6 = QLabel(tr("wizard.slides.slide_8.desc"))
        self.d6.setStyleSheet("font-size: 16px; color: #dddddd;")
        self.d6.setWordWrap(True)
        self.d6.setAlignment(Qt.AlignCenter)
        l6.addWidget(self.d6)

        self.lbl_step6_status = QLabel("Status: Waiting")
        self.lbl_step6_status.setAlignment(Qt.AlignCenter)
        self.lbl_step6_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l6.addWidget(self.lbl_step6_status)

        apply_row = QHBoxLayout()
        apply_row.setSpacing(15)

        img_apply = QLabel()
        pix_apply = QPixmap(get_asset_path("img/apply_offset.png"))
        if not pix_apply.isNull():
            img_apply.setPixmap(pix_apply.scaled(520, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_apply.setText("[img/apply_offset.png not found]")
        img_apply.setAlignment(Qt.AlignCenter)
        apply_row.addWidget(img_apply)

        self.apply_instructions_box = QGroupBox(tr("wizard.slides.slide_8.box_title"))
        self.apply_instructions_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        apply_instr_layout = QVBoxLayout()
        apply_instr_layout.setSpacing(8)

        self.lbl_apply1 = QLabel(tr("wizard.slides.slide_8.inst1"))
        self.lbl_apply1.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply1.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply1)

        self.lbl_apply2 = QLabel(tr("wizard.slides.slide_8.inst2"))
        self.lbl_apply2.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply2.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply2)

        self.lbl_apply3 = QLabel(tr("wizard.slides.slide_8.inst3"))
        self.lbl_apply3.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply3.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply3)

        self.lbl_apply4 = QLabel(tr("wizard.slides.slide_8.inst4"))
        self.lbl_apply4.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply4.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply4)

        self.apply_instructions_box.setLayout(apply_instr_layout)
        apply_row.addWidget(self.apply_instructions_box)

        l6.addLayout(apply_row)

        # New buttons layout
        btn_container = QVBoxLayout()
        btn_container.setSpacing(10)

        # Row 0: Rollback / New Offset Zero Pose (same moves as the Step 2 apply dialog's "Move to Zero")
        row0_layout = QHBoxLayout()
        row0_layout.setSpacing(15)

        self.btn_rollback_zero = QPushButton(tr("wizard.slides.slide_8.btn_rollback_zero"))
        self.btn_rollback_zero.setMinimumHeight(40)
        self.btn_rollback_zero.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_rollback_zero.clicked.connect(lambda: self.wizard_move_zero("baseline"))
        row0_layout.addWidget(self.btn_rollback_zero)

        self.btn_new_offset_zero = QPushButton(tr("wizard.slides.slide_8.btn_new_offset_zero"))
        self.btn_new_offset_zero.setMinimumHeight(40)
        self.btn_new_offset_zero.setStyleSheet("background-color: #fb8c00; color: #000000; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_new_offset_zero.clicked.connect(lambda: self.wizard_move_zero("optimized"))
        row0_layout.addWidget(self.btn_new_offset_zero)
        btn_container.addLayout(row0_layout)

        # Row 1: Rollback / New Offset Preview Buttons
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(15)

        self.btn_rollback_preview = QPushButton(tr("wizard.slides.slide_8.btn_rollback_preview"))
        self.btn_rollback_preview.setMinimumHeight(40)
        self.btn_rollback_preview.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_rollback_preview.clicked.connect(lambda: self.wizard_move_check("baseline"))
        row1_layout.addWidget(self.btn_rollback_preview)

        self.btn_new_offset_preview = QPushButton(tr("wizard.slides.slide_8.btn_new_offset_preview"))
        self.btn_new_offset_preview.setMinimumHeight(40)
        self.btn_new_offset_preview.setStyleSheet("background-color: #fb8c00; color: #000000; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_new_offset_preview.clicked.connect(lambda: self.wizard_move_check("optimized"))
        row1_layout.addWidget(self.btn_new_offset_preview)
        btn_container.addLayout(row1_layout)

        # Row 2: Rollback Joint / Apply New Offset Buttons
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(15)

        self.btn_rollback_joint = QPushButton(tr("wizard.slides.slide_8.btn_rollback_joint"))
        self.btn_rollback_joint.setMinimumHeight(45)
        self.btn_rollback_joint.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_rollback_joint.clicked.connect(lambda: self.wizard_apply_offset("baseline"))
        row2_layout.addWidget(self.btn_rollback_joint)

        self.btn_apply_new_offset = QPushButton(tr("wizard.slides.slide_8.btn_apply_new_offset"))
        self.btn_apply_new_offset.setMinimumHeight(45)
        self.btn_apply_new_offset.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_apply_new_offset.clicked.connect(lambda: self.wizard_apply_offset("optimized"))
        row2_layout.addWidget(self.btn_apply_new_offset)
        btn_container.addLayout(row2_layout)

        # Why applying the optimized result is disabled (not solved at the current zero in this run,
        # or already applied); previewing it and rollback stay available.
        self.lbl_wiz_opt_locked = QLabel(tr("dialogs.apply_home_offset.opt_locked"))
        self.lbl_wiz_opt_locked.setWordWrap(True)
        self.lbl_wiz_opt_locked.setAlignment(Qt.AlignCenter)
        self.lbl_wiz_opt_locked.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 14px;")
        btn_container.addWidget(self.lbl_wiz_opt_locked)

        l6.addLayout(btn_container)
        self.refresh_apply_gate()

        slides[self.SLIDE_APPLY] = slide6

        if sorted(slides) != list(range(self.SLIDE_COUNT)):
            raise RuntimeError(f"Wizard slides do not match SLIDE_* order: built {sorted(slides)}")
        for index in range(self.SLIDE_COUNT):
            self.stacked_widget.addWidget(slides[index])

    def set_bracket_version(self, version):
        """Show the marker bracket photo of robot version `version` ("1.2" / "1.3")."""
        version = str(version).removeprefix("v")
        if version not in self.BRACKET_ASSEMBLE_IMAGES:
            return
        self.bracket_version = version
        pix = QPixmap(get_asset_path(self.BRACKET_ASSEMBLE_IMAGES[version]))
        if pix.isNull():
            self.img_mb_assemble.setPixmap(QPixmap())
            self.img_mb_assemble.setText(f"[{self.BRACKET_ASSEMBLE_IMAGES[version]} not found]")
        else:
            self.img_mb_assemble.setPixmap(pix.scaled(400, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def sync_bracket_version_from_robot(self):
        """Show the bracket photo of the connected robot's version (called when the slide opens)."""
        if hasattr(self.parent_app, "get_robot_version"):
            self.set_bracket_version(self.parent_app.get_robot_version())

    def camera_info(self):
        core = getattr(self.parent_app, "core", None)
        observer = getattr(core, "observer", None)
        if observer is None or not hasattr(observer, "get_camera_info"):
            return None
        try:
            return observer.get_camera_info()
        except Exception:
            return None

    def refresh_camera_info(self):
        if not hasattr(self, "lbl_cam_info"):
            return
        if self.isVisible() and self.stacked_widget.currentIndex() != self.SLIDE_CAMERA_MOUNT:
            return
        info = self.camera_info()
        ok, warn, bad = "#dddddd", "#ffb74d", "#f44336"
        if not info or not info.get("running"):
            self.lbl_cam_info.setText(f'<span style="font-size:15px; color:{bad}; font-weight:bold;">'
                                      f'{tr("wizard.slides.slide_0.cam_not_connected")}</span>')
            return
        unknown = tr("wizard.slides.slide_0.cam_unknown")
        fps, measured = info.get("fps"), info.get("measured_fps")
        measured_txt = f"{measured:.1f} fps" if measured else tr("wizard.slides.slide_0.cam_measuring")
        resolution_ok = (info.get("width"), info.get("height")) == (1280, 720)
        fps_ok = measured is None or not fps or measured >= 0.8 * fps
        lines = [
            (tr("wizard.slides.slide_0.cam_device", name=info.get("device_name") or unknown,
                serial=info.get("serial_number") or unknown,
                model=(info.get("camera_model") or unknown).upper()), ok),
            (tr("wizard.slides.slide_0.cam_stream", width=info.get("width"), height=info.get("height"),
                fps=fps, measured=measured_txt), ok if resolution_ok and fps_ok else warn),
            (tr("wizard.slides.slide_0.cam_intrinsics", file=info.get("intrinsics_file") or unknown),
             ok if info.get("intrinsics_file") else warn),
        ]
        if not resolution_ok:
            lines.append((tr("wizard.slides.slide_0.cam_warn_resolution"), bad))
        if not fps_ok:
            lines.append((tr("wizard.slides.slide_0.cam_warn_fps"), warn))
        source = info.get("intrinsics_source")
        if source:
            lines.append((tr("wizard.slides.slide_0.cam_source",
                             source=tr(f"wizard.slides.slide_0.cam_source_{source}")),
                          ok if source == "serial" else warn))
        if not info.get("intrinsics_file"):
            lines.append((tr("wizard.slides.slide_0.cam_warn_intrinsics"), warn))
        # A unit without its own calibration (S/N) is the case the operator must notice: the model
        # file came from another unit of the same model, or the factory values are in use.
        if info.get("serial_number") and not info.get("intrinsics_serial_matched"):
            lines.append((tr("wizard.slides.slide_0.cam_warn_serial"), bad))
        self.lbl_cam_info.setText("<br>".join(
            f'<span style="font-size:15px; color:{color}; font-weight:bold;">{text}</span>'
            for text, color in lines))

    def show_how_to_move_arms_dialog(self):
        dlg = HowToMoveArmsDialog(self)
        dlg.exec()

    def on_wiz_auto_exp_toggled(self, checked):
        self.spin_wiz_exp.setEnabled(not checked)
        self.slider_wiz_exp.setEnabled(not checked)
        if hasattr(self.parent_app, 'chk_auto_exposure'):
            self.parent_app.chk_auto_exposure.blockSignals(True)
            self.parent_app.chk_auto_exposure.setChecked(checked)
            self.parent_app.chk_auto_exposure.blockSignals(False)
        if checked:
            self.parent_app.set_camera_auto_mode()
            self.lbl_wiz_exp_status.setText("Status: Switched to AUTO exposure mode.")
            self.lbl_wiz_exp_status.setStyleSheet("color: #00e5ff; font-size: 13px; font-weight: bold;")

    def exposure_ms(self, value):
        helper = getattr(self.parent_app, 'exposure_ms', None)
        return helper(value) if helper else float(value) / 1000.0

    def on_wiz_exposure_changed(self, value):
        self.lbl_wiz_exp_ms.setText(f"{self.exposure_ms(value):.1f} ms")
        if self.slider_wiz_exp.value() != value:
            self.slider_wiz_exp.blockSignals(True)
            self.slider_wiz_exp.setValue(value)
            self.slider_wiz_exp.blockSignals(False)
        if self.spin_wiz_exp.value() != value:
            self.spin_wiz_exp.blockSignals(True)
            self.spin_wiz_exp.setValue(value)
            self.spin_wiz_exp.blockSignals(False)

        if hasattr(self.parent_app, 'spin_exposure'):
            self.parent_app.spin_exposure.blockSignals(True)
            self.parent_app.spin_exposure.setValue(value)
            self.parent_app.spin_exposure.blockSignals(False)
        if hasattr(self.parent_app, 'slider_exposure'):
            self.parent_app.slider_exposure.blockSignals(True)
            self.parent_app.slider_exposure.setValue(value)
            self.parent_app.slider_exposure.blockSignals(False)
        if hasattr(self.parent_app, 'lbl_exposure_ms'):
            self.parent_app.lbl_exposure_ms.setText(f"{self.exposure_ms(value):.1f} ms")

    def on_wiz_apply_exp_clicked(self):
        auto_mode = self.chk_wiz_auto_exp.isChecked()
        exp_val = self.spin_wiz_exp.value()
        if self.parent_app.core.observer is not None and hasattr(self.parent_app.core.observer, 'set_camera_exposure'):
            self.parent_app.core.observer.set_camera_exposure(exp_val, auto_exposure=auto_mode)
        if auto_mode:
            self.lbl_wiz_exp_status.setText("Status: Applied AUTO exposure mode.")
        else:
            self.lbl_wiz_exp_status.setText(f"Status: Applied {exp_val} ({self.exposure_ms(exp_val):.1f} ms) manual exposure.")
        self.lbl_wiz_exp_status.setStyleSheet("color: #4caf50; font-size: 13px; font-weight: bold;")
        self.parent_app.log_msg(f"[Camera] Wizard applied exposure (auto={auto_mode}, exposure={exp_val} = {self.exposure_ms(exp_val):.1f} ms)")

    def on_wiz_cancel_exp_clicked(self):
        self.parent_app.cancel_camera_exposure()
        is_auto = self.parent_app.saved_camera_auto_exposure
        val = self.parent_app.saved_camera_exposure_value

        self.chk_wiz_auto_exp.blockSignals(True)
        self.chk_wiz_auto_exp.setChecked(is_auto)
        self.chk_wiz_auto_exp.blockSignals(False)

        self.spin_wiz_exp.blockSignals(True)
        self.spin_wiz_exp.setValue(val)
        self.spin_wiz_exp.setEnabled(not is_auto)
        self.spin_wiz_exp.blockSignals(False)

        self.slider_wiz_exp.blockSignals(True)
        self.slider_wiz_exp.setValue(val)
        self.slider_wiz_exp.setEnabled(not is_auto)
        self.slider_wiz_exp.blockSignals(False)

        self.lbl_wiz_exp_ms.setText(f"{self.exposure_ms(val):.1f} ms")
        self.lbl_wiz_exp_status.setText("Status: Cancelled. Restored previous exposure setting.")
        self.lbl_wiz_exp_status.setStyleSheet("color: #ff9800; font-size: 13px; font-weight: bold;")

    def on_wiz_exp_confirmed_toggled(self, checked):
        if checked:
            self.parent_app.apply_camera_exposure()
            self.mark_step_completed(self.SLIDE_EXPOSURE,True, tr("wizard.slides.slide_exposure.status_confirmed"))
            self.lbl_wiz_exp_status.setText(tr("wizard.slides.slide_exposure.status_confirmed"))
            self.lbl_wiz_exp_status.setStyleSheet("color: #00e676; font-size: 14px; font-weight: bold;")
        else:
            self.mark_step_completed(self.SLIDE_EXPOSURE,False, tr("wizard.slides.slide_exposure.status_waiting"))
            self.lbl_wiz_exp_status.setText(tr("wizard.slides.slide_exposure.status_waiting"))
            self.lbl_wiz_exp_status.setStyleSheet("color: #ff9800; font-size: 13px; font-weight: bold;")

    def show_how_to_move_arms_dialog(self):
        dlg = HowToMoveArmsDialog(self)
        dlg.exec()

    MONITOR_IDLE_STYLE = "color: #9e9e9e; font-size: 14px; font-weight: bold;"
    MONITOR_OK_STYLE = "color: #4caf50; font-size: 14px; font-weight: bold;"
    MONITOR_WARN_STYLE = "color: #ff9800; font-size: 14px; font-weight: bold;"
    MONITOR_BAD_STYLE = "color: #f44336; font-size: 14px; font-weight: bold;"

    def sync_marker_monitor(self):
        """Run the core marker monitor exactly while the exposure slide is on screen."""
        core = getattr(self.parent_app, "core", None)
        if core is None or not hasattr(core, "start_marker_monitor"):
            return
        wanted = (self.isVisible() and self.stacked_widget.currentIndex() == self.SLIDE_EXPOSURE
                  and core.observer is not None)
        try:
            if wanted and not core.marker_monitor_running and not core.is_busy:
                core.start_marker_monitor()
            elif not wanted and core.marker_monitor_running:
                core.stop_marker_monitor()
        except Exception as error:
            self.parent_app.log_msg(f"[WARN] Marker monitor: {error}")

    def update_marker_monitor(self, info):
        """Show the latest core marker monitor summary (None/running=False = not monitoring)."""
        self.last_marker_monitor = info
        labels = getattr(self, "marker_mon_labels", None)
        if not labels:
            return
        running = bool(info and info.get("running"))
        sides = info.get("sides", {}) if running else {}
        limit = f"{info['max_jitter_mm']:.2f}" if running and info.get("max_jitter_mm") is not None else "-"
        for side, lbl in labels.items():
            name = tr(f"wizard.slides.slide_exposure.monitor_{side}")
            state = sides.get(side)
            if state is None:
                lbl.setText(tr("wizard.slides.slide_exposure.monitor_idle", marker=name))
                lbl.setStyleSheet(self.MONITOR_IDLE_STYLE)
                continue
            jitter = state.get("jitter")
            jitter_text = (tr("wizard.slides.slide_exposure.monitor_jitter",
                              rms=f"{jitter['rms_mm']:.2f}", max=f"{jitter['max_mm']:.2f}")
                           if jitter else "-")
            # Green = recognized and RMS jitter within the limit, orange = recognized but jumping,
            # red = not recognized.
            if not state.get("visible"):
                key, style = "monitor_hidden", self.MONITOR_BAD_STYLE
            elif state.get("stable"):
                key, style = "monitor_stable", self.MONITOR_OK_STYLE
            else:
                key, style = "monitor_unstable", self.MONITOR_WARN_STYLE
            lbl.setText(tr(f"wizard.slides.slide_exposure.{key}", marker=name, limit=limit,
                           rate=f"{state.get('rate', 0.0) * 100:.0f}", jitter=jitter_text))
            lbl.setStyleSheet(style)
        overall = getattr(self, "lbl_marker_mon_overall", None)
        if overall is not None:
            overall.setVisible(running)
            if running and info.get("all_stable"):
                overall.setText(tr("wizard.slides.slide_exposure.monitor_all_ok", limit=limit))
                overall.setStyleSheet(self.MONITOR_OK_STYLE)
            elif running:
                overall.setText(tr("wizard.slides.slide_exposure.monitor_not_ok", limit=limit))
                overall.setStyleSheet(self.MONITOR_WARN_STYLE)
        if hasattr(self, "btn_marker_mon_restart"):
            self.btn_marker_mon_restart.setVisible(not running)

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_marker_monitor()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.sync_marker_monitor()

    def mark_step_completed(self, step_idx, success=True, msg=""):
        if step_idx < len(self.step_completed):
            self.step_completed[step_idx] = success
        self.update_navigation(self.stacked_widget.currentIndex())

        # Map step index to status label
        lbl_name = {
            self.SLIDE_EXPOSURE: "lbl_wiz_exp_status",
            self.SLIDE_INTRINSICS_CALIB: "lbl_step1_status",
            self.SLIDE_ROBOT_CONNECT: "lbl_step2_status",
            self.SLIDE_ZERO_POSE: "lbl_step3_1_status",
            self.SLIDE_HOME_OFFSET: "lbl_step7_status",
            self.SLIDE_CALIBRATION: "lbl_step4_status",
            self.SLIDE_APPLY: "lbl_step6_status",
        }.get(step_idx)

        if lbl_name:
            lbl = getattr(self, lbl_name, None)
            if lbl:
                if success:
                    lbl.setText(f"Status: SUCCESS - {msg}" if msg else "Status: SUCCESS")
                    lbl.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                else:
                    lbl.setText(f"Status: ERROR - {msg}")
                    lbl.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def step3_1_move_zero(self):
        if self.parent_app.move_to_zero_pose():
            self.lbl_step3_1_status.setText("Status: Moving to Zero Position...")
            self.lbl_step3_1_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
            self.set_wizard_busy(True)
        else:
            if not self.parent_app.robot:
                self.mark_step_completed(self.SLIDE_ZERO_POSE,False, "Robot Not Connected")

    def go_prev(self):
        idx = self.stacked_widget.currentIndex()
        # The optional intrinsics calibration (1-2) is entered only from the button on the camera
        # slide (1-1), so going back from the robot connection (2) skips it.
        if idx == self.SLIDE_ROBOT_CONNECT:
            self.stacked_widget.setCurrentIndex(self.SLIDE_CAMERA_MOUNT)
        elif idx > 0:
            self.stacked_widget.setCurrentIndex(idx - 1)
        else:
            if hasattr(self.parent_app, 'overview_container') and self.parent_app.overview_container:
                self.parent_app.overview_container.setVisible(True)
            else:
                if hasattr(self.parent_app, 'overview_title') and self.parent_app.overview_title:
                    self.parent_app.overview_title.setVisible(True)
                if hasattr(self.parent_app, 'overview_link') and self.parent_app.overview_link:
                    self.parent_app.overview_link.setVisible(True)
                if hasattr(self.parent_app, 'overview_duration') and self.parent_app.overview_duration:
                    self.parent_app.overview_duration.setVisible(True)
                self.parent_app.btn_start_wizard.setVisible(True)
                if hasattr(self.parent_app, 'overview_img') and self.parent_app.overview_img:
                    self.parent_app.overview_img.setVisible(True)
            self.setVisible(False)

    def go_next(self):
        idx = self.stacked_widget.currentIndex()
        if idx == self.SLIDE_CAMERA_MOUNT:
            # 1-1 -> 2: the optional calibration (1-2) is opened from its button only.
            self.stacked_widget.setCurrentIndex(self.SLIDE_ROBOT_CONNECT)
        elif idx < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(idx + 1)
            if self.sender() == self.btn_skip:
                self.step_completed[idx] = True
                self.update_navigation(idx + 1)
        else:
            self.parent_app.log_msg("Calibration Wizard Finished.")
            if hasattr(self.parent_app, 'overview_container') and self.parent_app.overview_container:
                self.parent_app.overview_container.setVisible(True)
            else:
                if hasattr(self.parent_app, 'overview_title') and self.parent_app.overview_title:
                    self.parent_app.overview_title.setVisible(True)
                if hasattr(self.parent_app, 'overview_link') and self.parent_app.overview_link:
                    self.parent_app.overview_link.setVisible(True)
                if hasattr(self.parent_app, 'overview_duration') and self.parent_app.overview_duration:
                    self.parent_app.overview_duration.setVisible(True)
                self.parent_app.btn_start_wizard.setVisible(True)
                if hasattr(self.parent_app, 'overview_img') and self.parent_app.overview_img:
                    self.parent_app.overview_img.setVisible(True)
            self.setVisible(False)
            self.stacked_widget.setCurrentIndex(0)

    def update_navigation(self, idx):
        if idx != self.SLIDE_APPLY:
            self.check_pose_init_done = False

        if hasattr(self, "parent_app") and hasattr(self.parent_app, "on_left_tab_changed"):
            self.parent_app.on_left_tab_changed(self.parent_app.left_tabs.currentIndex())
        self.sync_marker_monitor()
        if idx == self.SLIDE_APPLY and hasattr(self, "btn_apply_new_offset"):
            self.refresh_apply_gate()
        if idx == self.SLIDE_MARKER_BRACKET:
            self.sync_bracket_version_from_robot()

        # Update shared top title dynamically to prevent title layout shifts
        if hasattr(self, 'lbl_wizard_title') and idx in self.TITLE_KEYS:
            self.lbl_wizard_title.setText(tr(self.TITLE_KEYS[idx]))

        self.btn_prev.setVisible(True)
        self.btn_prev.setText(tr("wizard.btn_prev"))

        show_skip = idx in (self.SLIDE_INTRINSICS_CALIB, self.SLIDE_HOME_OFFSET)
        self.btn_skip.setVisible(show_skip)
        self.btn_skip.setText(tr("wizard.btn_skip"))

        if hasattr(self, 'lbl_skip_hint1'):
            self.lbl_skip_hint1.setVisible(idx == self.SLIDE_INTRINSICS_CALIB)
        if hasattr(self, 'lbl_skip_hint7'):
            self.lbl_skip_hint7.setVisible(idx == self.SLIDE_HOME_OFFSET)

        enabled = self.step_completed[idx]
        self.btn_next.setEnabled(enabled)

        if enabled:
            self.btn_next.setStyleSheet("background-color: #1e88e5; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setStyleSheet("background-color: #222222; color: #616161; font-weight: bold; font-size: 15px; border-radius: 6px; border: 1px solid #333333;")

        if idx == self.stacked_widget.count() - 1:
            self.btn_next.setText(tr("wizard.btn_finish"))
            self.btn_next.setEnabled(True)
            self.btn_next.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setText(tr("wizard.btn_next"))

    # Step 1: Intrinsics
    def step1_capture(self):
        self.parent_app.capture_intrinsics_frame()
        frames = len(self.parent_app.captured_images)
        self.lbl_captured.setText(f"Captured Frames: {frames} / 16")
        if frames >= 16:
            err = self.parent_app.intrinsics_calibrator.rms_error
            if err is not None and err > 0.0:
                self.lbl_step1_status.setText(f"Status: Calibration OK (RMS: {err:.4f})")
                self.lbl_step1_status.setStyleSheet("color: #d35400; font-weight: bold; font-size: 16px;")
            else:
                self.lbl_step1_status.setText(f"Status: Captured {frames} / 16 frames. Ready to calibrate.")
                self.lbl_step1_status.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_step1_status.setText(f"Status: Captured {frames} / 16 frames (Need 16)")
            self.lbl_step1_status.setStyleSheet("color: #b0bec5; font-weight: bold; font-size: 16px;")

    def step1_run(self):
        if len(self.parent_app.captured_images) < 16:
            self.lbl_step1_status.setText("Status: Need all 16 frames to run calibration!")
            self.lbl_step1_status.setStyleSheet("color: #c0392b; font-weight: bold; font-size: 16px;")
            QMessageBox.warning(self, "Insufficient Data", f"Cannot run calibration: Only {len(self.parent_app.captured_images)} / 16 frames collected.\nPlease capture all 16 frames first.")
            return

        self.parent_app.run_intrinsics_calibration()
        err = self.parent_app.intrinsics_calibrator.rms_error
        if err is not None and err > 0.0:
            self.lbl_step1_status.setText(f"Status: Calibration OK (RMS: {err:.4f})")
            self.lbl_step1_status.setStyleSheet("color: #d35400; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_step1_status.setText("Status: Calibration Failed (Check board settings)")
            self.lbl_step1_status.setStyleSheet("color: #c0392b; font-weight: bold; font-size: 16px;")

    def step1_save(self):
        if len(self.parent_app.captured_images) < 16:
            QMessageBox.warning(self, "Insufficient Data", f"Cannot save parameters: Only {len(self.parent_app.captured_images)} / 16 frames collected.")
            self.mark_step_completed(self.SLIDE_INTRINSICS_CALIB,False, "Need 16 frames to save")
            return

        if self.parent_app.intrinsics_calibrator.cameraMatrix is not None and float(self.parent_app.intrinsics_calibrator.rms_error) > 0.0:
            self.parent_app.save_intrinsics_calibration()
            self.mark_step_completed(self.SLIDE_INTRINSICS_CALIB,True, "Parameters Saved")
        else:
            QMessageBox.warning(self, "Invalid Calibration", "Calibration must be successfully executed before saving parameters.")
            self.mark_step_completed(self.SLIDE_INTRINSICS_CALIB,False, "Calibration not run yet")

    # Step 2: Robot Connection
    def step2_connect(self):
        self.btn_wizard_connect.setText("CONNECTING...")
        self.btn_wizard_connect.setStyleSheet("background-color: #d35400; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px; border-radius: 6px; border: 1px solid #111111;")
        self.btn_wizard_connect.setEnabled(False)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self.parent_app.ip_input.setText(self.wizard_ip_input.text())
        self.parent_app.chk_servo_head.setChecked(self.wizard_chk_head.isChecked())

        self.parent_app.connect_robot()

        self.btn_wizard_connect.setEnabled(True)
        if self.parent_app.robot is not None:
            self.btn_wizard_connect.setText("CONNECTED")
            self.btn_wizard_connect.setStyleSheet("background-color: #34495e; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px; border-radius: 6px; border: 1px solid #111111;")
            self.mark_step_completed(self.SLIDE_ROBOT_CONNECT,True, "Connected to Robot")
        else:
            self.btn_wizard_connect.setText("CONNECT")
            self.btn_wizard_connect.setStyleSheet("background-color: #2b5278; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px; border-radius: 6px; border: 1px solid #111111;")
            self.mark_step_completed(self.SLIDE_ROBOT_CONNECT,False, "Connection Failed")

    def sync_bracket_radio(self):
        is_head = self.wizard_chk_head.isChecked()
        self.rdo_bracket_yes.blockSignals(True)
        self.rdo_bracket_no.blockSignals(True)
        self.rdo_bracket_yes.setChecked(not is_head)
        self.rdo_bracket_no.setChecked(is_head)
        self.rdo_bracket_yes.blockSignals(False)
        self.rdo_bracket_no.blockSignals(False)

    def on_bracket_radio_changed(self):
        if not self.sender().isChecked():
            return
        is_yes = self.rdo_bracket_yes.isChecked()
        self.wizard_chk_head.blockSignals(True)
        self.wizard_chk_head.setChecked(not is_yes)
        self.wizard_chk_head.blockSignals(False)
        if hasattr(self.parent_app, 'on_head_checkbox_changed'):
            self.parent_app.on_head_checkbox_changed(not is_yes)

    # Step 3: Home Offset Reset
    def step3_reset(self):
        reply = QMessageBox.question(
            self,
            "Confirm Home Offset Reset",
            "Are you sure you want to proceed?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            self.lbl_step7_status.setText("Status: Reset cancelled")
            self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")
            return

        if self.parent_app.home_offset_reset(confirm_dialog=False):
            self.lbl_step7_status.setText("Status: Reset in progress...")
            self.lbl_step7_status.setStyleSheet("color: #1e88e5; font-weight: bold; font-size: 16px;")
            self.set_wizard_busy(True)
        else:
            if not self.parent_app.robot:
                self.mark_step_completed(self.SLIDE_HOME_OFFSET,False, "Robot Not Connected")
            else:
                self.lbl_step7_status.setText("Status: Reset cancelled")
                self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")

    def set_wizard_busy(self, busy):
        self.btn_prev.setEnabled(not busy)
        self.btn_skip.setEnabled(not busy)
        if busy:
            self.btn_next.setEnabled(False)
            self.btn_next.setStyleSheet("background-color: #2a2a2a; color: #757575; font-weight: bold; font-size: 15px; border-radius: 6px; border: 1px solid #3d3d3d;")
        else:
            self.update_navigation(self.stacked_widget.currentIndex())
        if hasattr(self, 'btn_step3_reset'):
            self.btn_step3_reset.setEnabled(not busy)

    # -----------------------------------------
    # Unified Step 1, Step 1.5 & Step 2 Calibration Execution
    # -----------------------------------------
    def start_unified_calibration(self):
        self.unified_elapsed = 0
        has_head = getattr(self.parent_app, 'include_head_motion', True)
        if hasattr(self.parent_app, 'robot') and hasattr(self.parent_app.robot, 'model'):
            try:
                m = self.parent_app.robot.model()
                has_head = has_head and (hasattr(m, 'head_idx') and m.head_idx is not None and len(m.head_idx) >= 2)
            except Exception:
                pass

        step_prefix = "[Step 1/3]" if has_head else "[Step 1/2]"
        self.unified_phase_str = f"{step_prefix} Full Auto In Progress"
        self.lbl_step4_status.setText(f"Status: {self.unified_phase_str} (00:00)")
        self.lbl_step4_status.setStyleSheet("color: #1e88e5; font-weight: bold; font-size: 16px;")
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(False)
            self.btn_start_unified.setStyleSheet("background-color: #2a2a2a; color: #757575; font-weight: bold; font-size: 18px; border-radius: 6px; border: 1px solid #3d3d3d;")

        self.unified_timer.start(1000)
        self.parent_app.start_full_auto()
        if hasattr(self.parent_app, 'active_worker') and self.parent_app.active_worker:
            self.parent_app.active_worker.finished_result.connect(self.on_unified_finished)
        else:
            self.stop_unified_calibration_error("Worker not started")

    def on_unified_finished(self, result):
        self.stop_step5(result.success, result.error)

    def update_unified_time(self):
        self.unified_elapsed += 1
        m = self.unified_elapsed // 60
        s = self.unified_elapsed % 60
        curr_text = self.lbl_step4_status.text()
        if hasattr(self, 'unified_phase_str') and self.unified_phase_str:
            self.lbl_step4_status.setText(f"Status: {self.unified_phase_str} ({m:02d}:{s:02d})")
        else:
            prefix = curr_text.split("(")[0].strip() if "(" in curr_text else curr_text
            self.lbl_step4_status.setText(f"{prefix} ({m:02d}:{s:02d})")


    def stop_step5(self, success=True, err_msg=""):
        self.unified_timer.stop()
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(True)
            self.btn_start_unified.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 18px; border-radius: 6px; padding: 0 15px;")

        m = self.unified_elapsed // 60
        s = self.unified_elapsed % 60
        time_str = f"{m:02d}:{s:02d}"
        if success:
            self.mark_step_completed(self.SLIDE_CALIBRATION,True, f"Calibration Pipeline Complete! Total Time: {time_str}")
        else:
            self.mark_step_completed(self.SLIDE_CALIBRATION,False, err_msg)

    def stop_unified_calibration(self):
        self.parent_app.stop_full_auto()
        if hasattr(self.parent_app, 'head_cam_stop_event') and self.parent_app.head_cam_stop_event is not None:
            self.parent_app.head_cam_stop_event.set()
        self.parent_app.request_stop_all_auto_motion()
        self.stop_unified_calibration_error("Cancelled by User")

    def stop_unified_calibration_error(self, err_msg=""):
        self.unified_timer.stop()
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(True)
            self.btn_start_unified.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 18px; border-radius: 6px; padding: 0 15px;")
        self.mark_step_completed(self.SLIDE_CALIBRATION,False, err_msg)

    def get_apply_paths(self):
        result_path = self.parent_app.get_latest_result_path()
        baseline_path = self.parent_app.get_home_reset_path_for_result(result_path)
        return result_path, baseline_path

    @idle_core_action
    def wizard_move_check(self, state):
        self.wizard_preview_move(state, "check")

    @idle_core_action
    def wizard_move_zero(self, state):
        self.wizard_preview_move(state, "zero")

    def wizard_preview_move(self, state, target):
        """Move to the zero pose or the check pose under the baseline or optimized offsets (preview only)."""
        result_path, baseline_path = self.get_apply_paths()
        path = baseline_path if state == "baseline" else result_path

        if not path or not os.path.exists(path):
            QMessageBox.warning(self, tr("common.status_error"),
                                tr("wizard.step6.no_json", state=state))
            return

        self.set_wizard_buttons_enabled(False)
        self.lbl_step6_status.setText(tr("wizard.step6.moving_check" if target == "check" else "wizard.step6.moving_zero",
                                         state=state))
        self.lbl_step6_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")

        from ui.core_bridge import Step2ApplyHomeOffsetWorker
        if target == "check":
            self.wizard_worker = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "move_check",
                json_path=path,
                label=f"{state.capitalize()} Check Position",
                arm="both",
                include_head=self.parent_app.include_head_motion,
                skip_init_pose=self.check_pose_init_done
            )
        else:
            self.wizard_worker = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "move_zero",
                json_path=path,
                label=f"{state.capitalize()} Zero",
                arm="both",
                include_head=self.parent_app.include_head_motion
            )
        self.wizard_worker.log_signal.connect(self.parent_app.log_msg)

        def on_finished(success, error_msg, res):
            self.set_wizard_buttons_enabled(True)
            if success:
                # From the zero pose the next check move must pass the joint ready pose again, as the
                # first one did; only check -> check may skip it.
                self.check_pose_init_done = target == "check"
                self.lbl_step6_status.setText(tr("wizard.step6.arrived_check" if target == "check" else "wizard.step6.arrived_zero",
                                                 state=state))
                self.lbl_step6_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                QMessageBox.information(self, tr("wizard.step6.preview_complete_title"),
                                        tr("wizard.step6.preview_complete_msg" if target == "check" else "wizard.step6.preview_zero_msg",
                                           state=state))
            else:
                self.check_pose_init_done = False   # stopped somewhere on the way
                self.lbl_step6_status.setText(tr("wizard.step6.preview_error_status"))
                self.lbl_step6_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
                QMessageBox.critical(self, tr("wizard.step6.preview_error_title"), error_msg)

        self.wizard_worker.finished_signal.connect(on_finished)
        self.wizard_worker.start()

    @idle_core_action
    def wizard_apply_offset(self, state):
        result_path, baseline_path = self.get_apply_paths()
        path = baseline_path if state == "baseline" else result_path

        if not path or not os.path.exists(path):
            QMessageBox.warning(self, tr("common.status_error"),
                                tr("wizard.step6.no_json", state=state))
            return
        if self.optimized_blocked(state):
            return

        confirm_msg = tr("wizard.step6.confirm_msg", state=state.upper())
        confirm = QMessageBox.question(
            self,
            tr("wizard.step6.confirm_title"),
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        self.set_wizard_buttons_enabled(False)
        self.parent_app.log_msg(f"[INFO] Moving robot to '{state.upper()}' Zero Pose before applying home offset...")
        self.lbl_step6_status.setText(tr("wizard.step6.moving_zero", state=state))
        self.lbl_step6_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")

        from ui.core_bridge import Step2ApplyHomeOffsetWorker
        # Inferred arm
        current_apply_arm = self.parent_app.infer_home_offset_apply_arm("both", result_path)

        self.wizard_worker_move = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_zero",
            json_path=path,
            label=f"{state.capitalize()} Zero",
            arm="both",
            include_head=self.parent_app.include_head_motion
        )
        self.wizard_worker_move.log_signal.connect(self.parent_app.log_msg)

        def on_move_finished(success, error_msg, res):
            # The arm has left the check pose (for the zero pose, or stopped part way).
            self.check_pose_init_done = False
            if not success:
                self.set_wizard_buttons_enabled(True)
                self.lbl_step6_status.setText(tr("wizard.step6.move_zero_error_status"))
                self.lbl_step6_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
                QMessageBox.critical(self, tr("wizard.step6.move_zero_error_title"),
                                     tr("wizard.step6.move_zero_error_msg", error_msg=error_msg))
                return

            arm_to_apply = res.get("arm", current_apply_arm)
            self.parent_app.log_msg(f"[INFO] Arrived at '{state.upper()}' Zero Pose. Now resetting and applying home offset...")
            self.lbl_step6_status.setText(tr("wizard.step6.applying_offset", state=state))

            self.wizard_worker_apply = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "apply",
                arm=arm_to_apply,
                include_head=self.parent_app.include_head_motion,
                json_path=result_path if state == "optimized" else None
            )
            self.wizard_worker_apply.log_signal.connect(self.parent_app.log_msg)

            def on_apply_finished(app_success, app_error_msg, app_res):
                self.set_wizard_buttons_enabled(True)
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
                            if arm_to_apply == "both" or arm_to_apply == arm:
                                self.parent_app.joint_offsets_store[arm]["joint3"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint5"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint6"] = 0.0

                                self.parent_app.joint_offsets[arm]["wrist_pitch"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_roll"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_yaw2"] = 0.0
                                self.parent_app.joint_offsets[arm]["elbow"] = 0.0

                        # Save zeroed offsets to setting.yaml and update GUI
                        self.parent_app.save_offsets_to_yaml()
                        self.parent_app.update_applied_offset_label()

                        # Zero out baseline json if it exists to prevent accidental unsafe rollback later
                        if baseline_path and os.path.exists(baseline_path):
                            try:
                                import json
                                data = ResultStorage.load(baseline_path)

                                if "right_arm_joint_offset_deg" in data and (arm_to_apply == "both" or arm_to_apply == "right"):
                                    data["right_arm_joint_offset_deg"] = [0.0] * len(data["right_arm_joint_offset_deg"])
                                if "left_arm_joint_offset_deg" in data and (arm_to_apply == "both" or arm_to_apply == "left"):
                                    data["left_arm_joint_offset_deg"] = [0.0] * len(data["left_arm_joint_offset_deg"])
                                if "head_joint_offset_deg" in data and data["head_joint_offset_deg"] is not None and self.parent_app.include_head_motion:
                                    data["head_joint_offset_deg"] = [0.0] * len(data["head_joint_offset_deg"])

                                if "right_arm_joint_offset_deg" in data and "left_arm_joint_offset_deg" in data:
                                    data["joint_offset_deg"] = data["right_arm_joint_offset_deg"] + data["left_arm_joint_offset_deg"]
                                elif "joint_offset_deg" in data:
                                    data["joint_offset_deg"] = [0.0] * len(data["joint_offset_deg"])

                                ResultStorage.save(baseline_path, data)
                                self.parent_app.log_msg(f"[INFO] Zeroed out applied arm offsets in baseline json: {baseline_path}")
                            except Exception as e:
                                self.parent_app.log_msg(f"[WARN] Failed to zero out baseline json: {e}")

                        self.lbl_step6_status.setText(tr("wizard.step6.success_status", state=state.upper()))
                        self.lbl_step6_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                        self.mark_step_completed(self.SLIDE_APPLY,True, f"'{state.upper()}' home offset applied.")

                        QMessageBox.information(self, tr("wizard.step6.success_title"),
                                                tr("wizard.step6.success_msg", state=state.upper()))
                    else:
                        self.lbl_step6_status.setText(tr("wizard.step6.partial_fail_status"))
                        self.lbl_step6_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 16px;")
                        QMessageBox.warning(self, tr("wizard.step6.partial_fail_title"),
                                            tr("wizard.step6.partial_fail_msg"))
                else:
                    self.lbl_step6_status.setText(tr("wizard.step6.apply_error_status"))
                    self.lbl_step6_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
                    QMessageBox.critical(self, tr("wizard.step6.apply_error_title"), app_error_msg)

            self.wizard_worker_apply.finished_signal.connect(on_apply_finished)
            self.wizard_worker_apply.start()

        self.wizard_worker_move.finished_signal.connect(on_move_finished)
        self.wizard_worker_move.start()

    def optimized_apply_allowed(self):
        """The optimized result may be applied only while the core allows it (solved from samples at
        the current robot zero in this run, not applied yet). Previewing it is always possible."""
        core = getattr(self.parent_app, "core", None)
        if core is None or not hasattr(core, "result_apply_allowed") or core.applicable_result_path is None:
            return False   # fails closed; the lock label says why
        result_path, _ = self.get_apply_paths()
        return bool(result_path and os.path.exists(result_path) and core.result_apply_allowed(result_path))

    def refresh_apply_gate(self, enabled=True):
        allowed = self.optimized_apply_allowed()
        result_path, _ = self.get_apply_paths()
        exists = bool(result_path and os.path.exists(result_path))
        for btn in (self.btn_new_offset_zero, self.btn_new_offset_preview):
            btn.setEnabled(enabled and exists)
        self.btn_apply_new_offset.setEnabled(enabled and allowed)
        if hasattr(self, "lbl_wiz_opt_locked"):
            self.lbl_wiz_opt_locked.setVisible(not allowed)

    def optimized_blocked(self, state):
        if state != "optimized" or self.optimized_apply_allowed():
            return False
        QMessageBox.warning(self, tr("dialogs.apply_home_offset.opt_locked_title"),
                            tr("dialogs.apply_home_offset.opt_locked"))
        self.refresh_apply_gate()
        return True

    def set_wizard_buttons_enabled(self, enabled):
        self.btn_rollback_zero.setEnabled(enabled)
        self.btn_rollback_preview.setEnabled(enabled)
        self.btn_rollback_joint.setEnabled(enabled)
        self.refresh_apply_gate(enabled)
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
