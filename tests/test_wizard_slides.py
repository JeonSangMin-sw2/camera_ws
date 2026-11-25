"""Wizard slide order, images, texts, and the exposure-slide marker monitor. Offline, offscreen."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import re
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtWidgets import QApplication

from core.calibration import CalibrationCore
import tempfile
from core.calibration.data import (DEFAULT_MAX_MARKER_JITTER_MM, load_max_marker_jitter_mm,
                                   marker_monitor_summary, marker_position_jitter_mm)
from core.calibration.sequences.marker_monitor import MarkerMonitor
from core.calibration.sequences.result import SequenceCancelled
from core.storage import ConfigStorage
from core.language import tr
from core.robot.robot_core import RobotOperations
from ui.marker_monitor_panel import MarkerMonitorPanel

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeObserver:
    """Right marker jitters `right_noise_m` per axis; left marker is missing on every other call
    unless `left_always`."""
    sim = True
    last_frame = {}

    def __init__(self, right_noise_m=0.0005, left_always=False):
        self.calls = {"right": 0, "left": 0}
        self.rng = np.random.default_rng(1)
        self.kwargs = []
        self.right_noise_m, self.left_always = right_noise_m, left_always

    def get_marker_transform(self, sampling_time=0, side="left", **kwargs):
        self.kwargs.append(kwargs)
        self.calls[side] += 1
        if side == "left" and not self.left_always and self.calls[side] % 2 == 0:
            return None
        T = np.eye(4)
        T[:3, 3] = [0.06 if side == "right" else -0.06, 0.0, 0.35]
        if side == "right":
            T[:3, 3] += self.rng.normal(0, self.right_noise_m, 3)
        return [T.ravel()]


class TestJitterMaths(unittest.TestCase):
    def test_jitter_is_distance_from_the_mean(self):
        pts = np.array([[0.001, 0, 0], [-0.001, 0, 0], [0, 0.001, 0], [0, -0.001, 0]])
        jitter = marker_position_jitter_mm(pts)
        self.assertAlmostEqual(jitter["rms_mm"], 1.0)
        self.assertAlmostEqual(jitter["max_mm"], 1.0)

    def test_still_marker_has_no_jitter_and_one_point_has_none(self):
        self.assertEqual(marker_position_jitter_mm([[0.1, 0.2, 0.3]] * 5), {"rms_mm": 0.0, "max_mm": 0.0})
        self.assertIsNone(marker_position_jitter_mm([[0.1, 0.2, 0.3]]))

    def test_summary_counts_detections(self):
        p = np.array([0.0, 0.0, 0.3])
        summary = marker_monitor_summary({"right": [p, p, None, p], "left": [None, None], "none": []})
        self.assertTrue(summary["right"]["visible"])
        self.assertAlmostEqual(summary["right"]["rate"], 0.75)
        self.assertEqual(summary["right"]["jitter"], {"rms_mm": 0.0, "max_mm": 0.0})
        self.assertFalse(summary["left"]["visible"])
        self.assertEqual(summary["left"]["rate"], 0.0)
        self.assertIsNone(summary["left"]["jitter"])
        self.assertEqual(summary["none"]["samples"], 0)
        self.assertIsNone(summary["right"]["stable"])

    def test_stable_means_visible_and_rms_within_the_limit(self):
        still = [np.array([0.0, 0.0, 0.3])] * 3
        jumpy = [np.array([0.0, 0.0, 0.3]), np.array([0.0006, 0.0, 0.3])]   # RMS 0.3 mm
        summary = marker_monitor_summary({"still": still, "jumpy": jumpy, "lost": still + [None]}, 0.2)
        self.assertTrue(summary["still"]["stable"])
        self.assertAlmostEqual(summary["jumpy"]["jitter"]["rms_mm"], 0.3)
        self.assertFalse(summary["jumpy"]["stable"])
        self.assertFalse(summary["lost"]["stable"])
        near = [np.array([0.0, 0.0, 0.3]), np.array([0.00039, 0.0, 0.3])]  # RMS 0.195 mm
        self.assertTrue(marker_monitor_summary({"near": near}, 0.2)["near"]["stable"])

    def test_limit_comes_from_setting_yaml(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "setting.yaml"
        ConfigStorage.save(path, {"exposure_check": {"max_marker_jitter_rms_mm": 0.35}})
        self.assertEqual(load_max_marker_jitter_mm(path), (0.35, True))
        ConfigStorage.save(path, {})
        self.assertEqual(load_max_marker_jitter_mm(path), (DEFAULT_MAX_MARKER_JITTER_MM, False))
        ConfigStorage.save(path, {"exposure_check": {"max_marker_jitter_rms_mm": -1}})
        with self.assertRaises(ValueError):
            load_max_marker_jitter_mm(path)
        shipped = ConfigStorage.load(ROOT / "config" / "setting.yaml")["exposure_check"]["max_marker_jitter_rms_mm"]
        self.assertEqual(shipped, 0.2)
        self.assertEqual(DEFAULT_MAX_MARKER_JITTER_MM, 0.2)


class TestMarkerMonitor(unittest.TestCase):
    def setUp(self):
        self.core = CalibrationCore()
        self.core.observer = FakeObserver()
        self.events = []
        self.core.on_event = lambda kind, value: self.events.append((kind, value))
        self.addCleanup(self.core.stop_marker_monitor)

    def monitor_events(self):
        return [v for k, v in self.events if k == "marker_monitor"]

    def test_reports_recognition_and_jitter_from_raw_detections(self):
        self.core.start_marker_monitor()
        self.assertTrue(wait_for(lambda: len(self.monitor_events()) >= 5))
        self.core.stop_marker_monitor()
        self.assertFalse(self.core.marker_monitor_running)
        running = [e for e in self.monitor_events() if e.get("running")]
        last = running[-1]["sides"]
        self.assertGreater(last["right"]["jitter"]["rms_mm"], 0.2)
        self.assertFalse(last["right"]["stable"])          # ~0.9 mm RMS > 0.2 mm
        self.assertAlmostEqual(last["left"]["rate"], 0.5, delta=0.2)
        self.assertEqual(running[-1]["max_jitter_mm"], 0.2)
        self.assertFalse(running[-1]["all_stable"])
        self.assertEqual(self.monitor_events()[-1], {"running": False})
        self.assertTrue(all(kw.get("use_filter") is False for kw in self.core.observer.kwargs))

    def test_still_markers_are_all_stable(self):
        self.core.observer = FakeObserver(right_noise_m=0.00002, left_always=True)
        self.core.start_marker_monitor()
        self.assertTrue(wait_for(lambda: len([e for e in self.monitor_events() if e.get("running")]) >= 3))
        self.core.stop_marker_monitor()
        last = [e for e in self.monitor_events() if e.get("running")][-1]
        self.assertTrue(last["all_stable"], last)

    def test_exposure_change_drops_the_samples_taken_before_it(self):
        self.core.start_marker_monitor()
        self.assertTrue(wait_for(lambda: any(e.get("running") and e["sides"]["right"]["samples"] >= 5
                                             for e in self.monitor_events())))
        window = self.core.restart_marker_monitor_window()
        self.assertTrue(wait_for(lambda: any(e.get("window") == window for e in self.monitor_events())))
        self.core.stop_marker_monitor()
        running = [e for e in self.monitor_events() if e.get("running")]
        first_new = next(e for e in running if e["window"] == window)
        self.assertEqual(first_new["sides"]["right"]["samples"], 1)

    def test_sequence_start_stops_it(self):
        self.core.start_marker_monitor()
        self.assertTrue(wait_for(lambda: self.monitor_events()))
        self.core.prepare_run()
        try:
            self.assertFalse(self.core.marker_monitor_running)
        finally:
            self.core.cancel()
            self.core.run("step1", prepared=True)

    def test_not_started_during_a_sequence_or_without_a_camera(self):
        self.core.prepare_run()
        try:
            with self.assertRaises(RuntimeError):
                self.core.start_marker_monitor()
        finally:
            self.core.cancel()
            self.core.run("step1", prepared=True)
        self.core.observer = None
        with self.assertRaises(RuntimeError):
            self.core.start_marker_monitor()

    def test_earlier_stop_does_not_block_it(self):
        self.core.stop_event.set()
        self.core.start_marker_monitor()
        self.assertTrue(wait_for(lambda: any(e.get("running") for e in self.monitor_events())))

    def test_cancel_and_errors_end_it_quietly(self):
        core = MagicMock()
        core.observer.get_marker_transform.side_effect = SequenceCancelled()
        monitor = MarkerMonitor(core)
        monitor.start()
        self.assertTrue(wait_for(lambda: not monitor.running))
        core.emit.assert_called_with("marker_monitor", {"running": False})
        core.observer.get_marker_transform.side_effect = RuntimeError("camera gone")
        monitor.start()
        self.assertTrue(wait_for(lambda: not monitor.running))
        self.assertIn("camera gone", core.log_msg.call_args[0][0])


class TestWizardSlides(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        from main_ui import UnifiedCalibrationApp
        cls.window = UnifiedCalibrationApp(ui_only=True)
        cls.window.show()  # the wizard only counts as visible inside a shown window
        cls.wiz = cls.window.wizard_widget
        cls.W = type(cls.wiz)

    def setUp(self):
        self.wiz.setVisible(True)
        self.wiz.step_completed = [True] * self.W.SLIDE_COUNT
        self.wiz.stacked_widget.setCurrentIndex(0)
        self.window.left_tabs.setCurrentIndex(self.window.TAB_WIZARD)
        self.window.reset_marker_monitor_confirmation()
        self.addCleanup(self.window.core.stop_marker_monitor)
        # Navigation must not depend on the repository's own reset record or the connected robot.
        pending = patch.object(self.W, "home_reset_pending", return_value=(False, None))
        pending.start()
        self.addCleanup(pending.stop)
        version = self.window.robot_version
        self.window.robot_version = "1.2"
        self.addCleanup(setattr, self.window, "robot_version", version)

    def walk(self):
        """Indices visited by Next from the first slide to the last, then by Previous back to the first."""
        forward = [self.wiz.stacked_widget.currentIndex()]
        while self.wiz.stacked_widget.currentIndex() != self.W.SLIDE_APPLY:
            self.wiz.go_next()
            forward.append(self.wiz.stacked_widget.currentIndex())
        backward = [forward[-1]]
        while self.wiz.stacked_widget.currentIndex() != 0:
            self.wiz.go_prev()
            backward.append(self.wiz.stacked_widget.currentIndex())
        return forward, backward

    def test_order_skips_the_optional_calibration_both_ways(self):
        W = self.W
        self.assertEqual(self.wiz.stacked_widget.count(), W.SLIDE_COUNT)
        self.assertEqual(sorted(W.TITLE_KEYS), list(range(W.SLIDE_COUNT)))
        # A camera without intrinsics of its own: every slide, the intrinsics ones first.
        forward, backward = self.walk()
        self.assertEqual(forward, list(range(W.SLIDE_COUNT)))
        self.assertEqual(backward, forward[::-1])
        # A camera registered by serial: the wizard opens on the camera connection (2026-11-24), Next goes
        # straight to the robot connection, and Previous from the camera slide goes back to the start page.
        with patch.object(self.wiz, "intrinsics_registered", return_value=True):
            self.wiz.stacked_widget.setCurrentIndex(0)
            self.window.show_wizard_ui()
            self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_CAMERA_CONNECT)
            self.wiz.go_next()
            self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_ROBOT_CONNECT)
            self.wiz.go_prev()
            self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_CAMERA_CONNECT)
            self.wiz.go_prev()
            self.assertFalse(self.wiz.isVisible())

    def test_v13_passes_over_the_gripper_removal(self):
        # 2026-11-12: a v1.3 marker bracket goes on the gripper, so there is nothing to remove.
        self.window.robot_version = "1.3"
        forward, backward = self.walk()
        self.assertNotIn(self.W.SLIDE_GRIPPER_REMOVAL, forward + backward)
        self.assertIn(self.W.SLIDE_MARKER_BRACKET, forward)

    def test_unapplied_reset_on_record_passes_over_the_reset_slide(self):
        # 2026-11-12: a reset whose calibration was never applied (the app was closed) is not redone.
        W = self.W
        with patch.object(W, "home_reset_pending", return_value=(True, "2026-11-12T10:00:00")), \
                patch.object(self.wiz, "confirm_home_reset_passed_over", return_value=True) as ask:
            forward, backward = self.walk()
        self.assertNotIn(W.SLIDE_HOME_OFFSET, forward + backward)
        ask.assert_called_once()
        # "Go to the Reset Step" on the notice opens it anyway (e.g. the record is another robot's).
        self.wiz.stacked_widget.setCurrentIndex(W.SLIDE_EXPOSURE)
        with patch.object(W, "home_reset_pending", return_value=(True, "2026-11-12T10:00:00")), \
                patch.object(self.wiz, "confirm_home_reset_passed_over", return_value=False):
            self.wiz.go_next()
        self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_HOME_OFFSET)

    def test_new_slides_sit_where_requested(self):
        W = self.W
        # 2026-11-12: robot connection first, gripper-change pose, camera, gripper removal, marker bracket,
        # then exposure (ends at the zero pose) -> home offset reset -> calibration -> apply. The separate
        # zero-pose slide was dropped the same day: the exposure slide's zero button does that move. Later the
        # same day the intrinsics (board printing, then the calibration) moved before the robot, numbered 0.
        # 2026-11-24: the camera connection slide went first (the app no longer opens the camera at start).
        order = [W.SLIDE_CAMERA_CONNECT, W.SLIDE_INTRINSICS_PREP, W.SLIDE_INTRINSICS_CALIB, W.SLIDE_ROBOT_CONNECT,
                 W.SLIDE_GRIPPER_POSE, W.SLIDE_CAMERA_MOUNT, W.SLIDE_GRIPPER_REMOVAL, W.SLIDE_MARKER_BRACKET,
                 W.SLIDE_EXPOSURE, W.SLIDE_HOME_OFFSET, W.SLIDE_CALIBRATION, W.SLIDE_CHECK_POSE, W.SLIDE_APPLY]
        self.assertEqual(order, list(range(W.SLIDE_COUNT)))
        self.assertFalse(hasattr(W, "SLIDE_ZERO_POSE"))
        numbers = [tr(W.TITLE_KEYS[i]).split(".")[0] for i in range(W.SLIDE_COUNT)]
        self.assertEqual(numbers, ["0", "0-1", "0-2", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        self.assertTrue((ROOT / W.CHARUCO_BOARD_PDF).is_file())
        self.assertIn(W.CHARUCO_BOARD_PDF, self.wiz.lbl_prep_steps.text())
        for index, word in ((W.SLIDE_EXPOSURE, "Exposure"),
                            (W.SLIDE_HOME_OFFSET, "Home Offset"), (W.SLIDE_GRIPPER_POSE, "Gripper")):
            self.assertIn(word, tr(W.TITLE_KEYS[index], lang="en"))
        self.assertIn(W.SLIDE_EXPOSURE, W.VIDEO_SLIDES)
        # The marker bracket text no longer assumes the gripper came off (v1.3 brackets go on it).
        for key in ("cap_assemble", "inst1"):
            for lang in ("en", "ko"):
                self.assertNotIn("ripper" if lang == "en" else "그리퍼",
                                 tr(f"wizard.slides.slide_marker_bracket.{key}", lang=lang))

    def test_check_pose_gather_and_spread_run_only_while_held(self):
        # 2026-11-24 (user): after the check pose is reached, the hands come together / apart while held, on a
        # slide of its own (check pose buttons, then the arrows), before the zero pose and the apply.
        W, wiz = self.W, self.wiz
        self.assertIs(wiz.btn_check_gather.window(), wiz.btn_new_offset_preview.window())
        check_slide = wiz.stacked_widget.widget(W.SLIDE_CHECK_POSE)
        for btn in (wiz.btn_rollback_preview, wiz.btn_new_offset_preview, wiz.btn_check_gather, wiz.btn_check_spread):
            self.assertTrue(check_slide.isAncestorOf(btn))
        for btn in (wiz.btn_rollback_zero, wiz.btn_new_offset_zero, wiz.btn_rollback_joint, wiz.btn_apply_new_offset):
            self.assertTrue(wiz.stacked_widget.widget(W.SLIDE_APPLY).isAncestorOf(btn))
        self.assertEqual((wiz.btn_check_gather.text(), wiz.btn_check_spread.text()), ("→   ←", "←   →"))
        wiz.stacked_widget.setCurrentIndex(W.SLIDE_CHECK_POSE)
        wiz.update_navigation(W.SLIDE_CHECK_POSE)
        self.assertFalse(wiz.btn_check_gather.isEnabled() or wiz.btn_check_spread.isEnabled())

        class FakeSignal:
            def __init__(self):
                self.slots = []

            def connect(self, slot):
                self.slots.append(slot)

        created = []

        class FakeWorker:
            def __init__(self, app, task_type, **options):
                self.task_type, self.options = task_type, options
                self.log_signal, self.finished_signal = FakeSignal(), FakeSignal()
                created.append(self)

            def start(self):
                pass

        wiz.check_pose_state = "optimized"
        wiz.refresh_check_gap_buttons()
        self.assertTrue(wiz.btn_check_gather.isEnabled() and wiz.btn_check_spread.isEnabled())
        with patch("ui.core_bridge.Step2ApplyHomeOffsetWorker", FakeWorker), \
                patch.object(wiz, "get_apply_paths", return_value=("result.json", "baseline.json")), \
                patch.object(self.window, "safe_cancel_control") as cancel:
            wiz.btn_check_gather.pressed.emit()
            self.assertEqual((created[0].task_type, created[0].options["direction"], created[0].options["json_path"]),
                             ("check_gap", "gather", "result.json"))
            self.assertTrue(wiz.btn_check_gather.isEnabled())       # held button stays enabled for its release
            self.assertFalse(wiz.btn_check_spread.isEnabled())
            wiz.btn_check_gather.released.emit()
            cancel.assert_called_once()
            created[0].finished_signal.slots[0](False, "Stopped by user", {})
        self.assertEqual(wiz.lbl_check_status.text(), tr("wizard.step6.gap_stopped"))
        self.assertTrue(wiz.btn_check_spread.isEnabled())
        # No hands-together pose for v1.3 yet: both stay off.
        self.window.robot_version = "1.3"
        wiz.refresh_check_gap_buttons()
        self.assertFalse(wiz.btn_check_gather.isEnabled())
        wiz.check_pose_state = None

    def test_camera_connection_slide_gates_next_on_an_open_camera(self):
        # 2026-11-24 (user): the camera is connected here, after the how-to, instead of at app start.
        W = self.W
        self.wiz.stacked_widget.setCurrentIndex(W.SLIDE_CAMERA_CONNECT)
        self.wiz.update_navigation(W.SLIDE_CAMERA_CONNECT)
        self.assertTrue(self.wiz.btn_next.isEnabled())          # UI-only window: no camera needed
        self.assertFalse(self.wiz.btn_cc_connect.isEnabled())
        with patch.object(self.window, "ui_only", False):
            with patch.object(self.wiz, "camera_info", return_value=None):
                self.wiz.update_navigation(W.SLIDE_CAMERA_CONNECT)
                self.assertFalse(self.wiz.btn_next.isEnabled())
                self.assertTrue(self.wiz.btn_cc_connect.isEnabled())
            info = {"running": True, "device_name": "Intel RealSense D405", "serial_number": "230422270977",
                    "intrinsics_serial_matched": False}
            with patch.object(self.wiz, "camera_info", return_value=info):
                self.wiz.update_navigation(W.SLIDE_CAMERA_CONNECT)
                self.assertTrue(self.wiz.btn_next.isEnabled())
                self.assertIn("230422270977", self.wiz.lbl_cc_status.text())
                self.assertFalse(self.wiz.btn_cc_connect.isEnabled())   # one camera open: no second connect
            seen = {}

            def connect_now(**_):
                seen["text"], seen["enabled"] = self.wiz.lbl_cc_status.text(), self.wiz.btn_cc_connect.isEnabled()
            with patch.object(self.wiz, "camera_info", return_value=None), \
                    patch.object(self.window, "reconnect_camera", side_effect=connect_now) as connect, \
                    patch("ui.wizard_widget.QMessageBox.warning") as warning:
                self.wiz.update_navigation(W.SLIDE_CAMERA_CONNECT)
                self.wiz.btn_cc_connect.click()
                connect.assert_called_once_with(show_dialog=False)
                self.assertEqual(seen, {"text": tr("wizard.slides.slide_camera_connect.st_connecting"), "enabled": False})
                warning.assert_called_once()
                self.assertTrue(self.wiz.btn_cc_connect.isEnabled())      # failed: try again

    def test_intrinsics_slides_follow_the_camera_serial(self):
        W = self.W
        self.assertFalse(hasattr(self.wiz, "btn_cam_intrinsics"))   # no longer opened from the camera slide
        registered = {"running": True, "serial_number": "123456789", "intrinsics_serial_matched": True}
        with patch.object(self.wiz, "camera_info", return_value=registered):
            self.assertTrue(self.wiz.slide_passed_over(W.SLIDE_INTRINSICS_PREP)
                            and self.wiz.slide_passed_over(W.SLIDE_INTRINSICS_CALIB))
        for info in (None, {**registered, "intrinsics_serial_matched": False}):
            with patch.object(self.wiz, "camera_info", return_value=info):
                self.assertFalse(self.wiz.slide_passed_over(W.SLIDE_INTRINSICS_PREP))
        # The board slide moves on with Next; skipping the calibration slide goes to the robot connection.
        self.wiz.step_completed = [False] * W.SLIDE_COUNT
        self.wiz.step_completed[W.SLIDE_INTRINSICS_PREP] = True
        self.wiz.stacked_widget.setCurrentIndex(W.SLIDE_INTRINSICS_PREP)
        self.assertTrue(self.wiz.btn_next.isEnabled() and not self.wiz.btn_skip.isVisible())
        self.wiz.go_next()
        self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_INTRINSICS_CALIB)
        self.assertTrue(self.wiz.btn_skip.isVisible() and not self.wiz.btn_next.isEnabled())
        self.wiz.btn_skip.click()
        self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_ROBOT_CONNECT)

    def test_camera_mount_slide_no_longer_lists_the_camera(self):
        # 2026-11-12 (user): the camera is checked on the distortion calibration slides (shown first for a
        # camera without intrinsics of its own), so the mount slide drops its recognised-camera box.
        self.assertFalse(hasattr(self.wiz, "cam_info_box") or hasattr(self.wiz, "lbl_cam_info"))
        self.assertEqual(tr("wizard.slides.slide_0.cam_unknown", lang="en"), "unknown")  # still used by the Camera tab

    def test_bracket_photo_follows_the_connected_robot_version(self):
        # The slide comes after the robot connection, so the version is known when it opens.
        version = self.window.robot_version
        try:
            for robot_version in ("1.3", "1.2"):
                self.window.robot_version = robot_version
                self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_ROBOT_CONNECT)
                self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_MARKER_BRACKET)
                self.assertEqual(self.wiz.bracket_version, robot_version)
                self.assertFalse(self.wiz.img_mb_assemble.pixmap().isNull())
        finally:
            self.window.robot_version = version
        self.assertFalse(hasattr(self.wiz, "rdo_mb_version"))
        for path in self.W.BRACKET_ASSEMBLE_IMAGES.values():
            self.assertTrue((ROOT / path).is_file(), path)

    def test_marker_monitor_runs_only_on_the_exposure_slide(self):
        core = self.window.core
        core.observer = FakeObserver()
        panel = self.wiz.marker_monitor_panel
        try:
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_CALIBRATION)
            self.assertFalse(core.marker_monitor_running)
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_EXPOSURE)
            self.assertTrue(core.marker_monitor_running)
            self.assertTrue(wait_for(lambda: "%" in panel.labels["right"].text()))
            # Right marker jumps ~0.9 mm RMS: recognized but not stable -> orange, overall not OK.
            self.assertEqual(panel.labels["right"].styleSheet(), panel.WARN_STYLE)
            self.assertEqual(panel.lbl_overall.styleSheet(), panel.WARN_STYLE)
            core.stop_marker_monitor()
            core.observer = FakeObserver(right_noise_m=0.00002, left_always=True)
            self.wiz.sync_marker_monitor()
            self.assertTrue(wait_for(lambda: panel.lbl_overall.styleSheet() == panel.OK_STYLE))
            # The wizard checks the right marker only (2026-11-12): its test pose shows one arm.
            self.assertEqual(list(panel.labels), ["right"])
            self.assertEqual(panel.labels["right"].styleSheet(), panel.OK_STYLE)
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_CALIBRATION)
            self.assertFalse(core.marker_monitor_running)
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_EXPOSURE)
            self.wiz.setVisible(False)
            self.assertFalse(core.marker_monitor_running)
        finally:
            core.stop_marker_monitor()
            core.observer = None

    def test_camera_brightness_tab_has_the_same_monitor(self):
        window, core = self.window, self.window.core
        self.assertIsInstance(window.marker_monitor_panel, MarkerMonitorPanel)
        self.assertIsNot(window.marker_monitor_panel, self.wiz.marker_monitor_panel)
        core.observer = FakeObserver(right_noise_m=0.00002, left_always=True)
        try:
            window.left_tabs.setCurrentIndex(window.TAB_CAMERA)
            window.camera_tabs.setCurrentIndex(window.CAMERA_SUB_INTRINSICS)
            self.assertFalse(core.marker_monitor_running)
            window.camera_tabs.setCurrentIndex(window.CAMERA_SUB_EXPOSURE)
            self.assertTrue(core.marker_monitor_running)
            panel = window.marker_monitor_panel
            self.assertTrue(wait_for(lambda: panel.lbl_overall.styleSheet() == panel.OK_STYLE))
            window.left_tabs.setCurrentIndex(window.TAB_ADVANCED)
            self.assertFalse(core.marker_monitor_running)
        finally:
            core.stop_marker_monitor()
            core.observer = None
            window.left_tabs.setCurrentIndex(window.TAB_WIZARD)

    def test_exposure_is_flagged_as_not_saved_in_both_places(self):
        warning = tr("wizard.slides.slide_exposure.not_saved_warn")
        self.assertIn("not saved", tr("wizard.slides.slide_exposure.not_saved_warn", lang="en"))
        self.assertEqual(self.wiz.lbl_wiz_exp_not_saved.text(), warning)
        self.assertEqual(self.window.lbl_exposure_not_saved.text(), warning)

    def test_exposure_slide_auto_button_then_zero(self):
        window, wiz, W = self.window, self.wiz, self.W
        wiz.stacked_widget.setCurrentIndex(W.SLIDE_EXPOSURE)
        wiz.step_completed[W.SLIDE_EXPOSURE] = False
        wiz.chk_wiz_exp_confirmed.setChecked(False)
        calls, finish = [], {}

        def fake_auto(on_finished):
            calls.append(("auto",))
            finish["cb"] = on_finished
            return True

        def fake_zero(on_finished):
            calls.append(("zero",))
            finish["cb"] = on_finished
            return True

        with patch.object(window, "run_auto_exposure", side_effect=fake_auto), \
                patch.object(window, "move_to_zero_for_exposure", side_effect=fake_zero), \
                patch.object(window, "apply_camera_exposure"):
            wiz.btn_exp_auto.click()
            self.assertEqual(wiz.exposure_motion_status[:2], ("moving", "auto"))
            self.assertFalse(wiz.btn_exp_zero.isEnabled() or wiz.btn_prev.isEnabled())
            self.assertTrue(wiz.btn_exp_stop.isEnabled())  # Stop stays usable while moving
            finish["cb"](True, "", {"applied": "manual", "exposure": 4000, "camera_model": "D405", "trials": []})
            self.assertEqual(wiz.exposure_motion_status[:2], ("done", "auto"))
            self.assertTrue(wiz.btn_exp_zero.isEnabled() and wiz.btn_prev.isEnabled())
            # The applied value shows in both exposure controls and box 2 asks whether to keep it; the hand
            # controls and the 3) check stay hidden (2026-11-12, user).
            self.assertFalse(wiz.chk_wiz_auto_exp.isChecked())
            self.assertEqual((wiz.spin_wiz_exp.value(), window.spin_exposure.value()), (4000, 4000))
            self.assertIn("4000", wiz.lbl_exp_choice.text())
            self.assertFalse(wiz.chk_wiz_exp_confirmed.isChecked())
            self.assertTrue(wiz.btn_exp_accept.isEnabled())
            self.assertFalse(any(w.isVisibleTo(wiz) for w in (wiz.exp_manual_panel, wiz.chk_wiz_exp_confirmed, wiz.btn_exp_zero)))
            self.assertFalse(wiz.step_completed[W.SLIDE_EXPOSURE] or wiz.btn_next.isEnabled())
            # Yes ticks 3) and goes straight on to 4) the zero pose; Next waits for the pose.
            wiz.btn_exp_accept.click()
            self.assertTrue(wiz.chk_wiz_exp_confirmed.isChecked())
            self.assertEqual(wiz.exposure_motion_status[:2], ("moving", "zero"))
            self.assertFalse(wiz.btn_next.isEnabled())
            finish["cb"](False, "cancelled")
            self.assertEqual(wiz.exposure_motion_status, ("failed", "zero", "cancelled"))
            self.assertFalse(wiz.btn_next.isEnabled())
            # 4) is in the hand panel now (2026-11-17), so on this path Yes is how the zero pose is retried.
            self.assertTrue(wiz.btn_exp_accept.isEnabled())
            wiz.btn_exp_accept.click()
            finish["cb"](True, "")
            self.assertTrue(wiz.step_completed[W.SLIDE_EXPOSURE] and wiz.btn_next.isEnabled())
            # Moving the arm out again (auto exposure) leaves the zero pose: Next waits once more.
            wiz.btn_exp_auto.click()
            self.assertFalse(wiz.step_completed[W.SLIDE_EXPOSURE] or wiz.chk_wiz_exp_confirmed.isChecked())
            self.assertFalse(wiz.btn_exp_accept.isEnabled())
            finish["cb"](True, "", {"applied": "manual", "exposure": 4000, "camera_model": "D405", "trials": []})
        self.assertEqual(calls, [("auto",), ("zero",), ("zero",), ("auto",)])
        # 'Adjust by hand' opens the controls, the 3) check and 4) the zero pose.
        wiz.chk_exp_manual.setChecked(True)
        self.assertTrue(all(w.isVisibleTo(wiz) for w in (wiz.exp_manual_panel, wiz.chk_wiz_exp_confirmed, wiz.btn_exp_zero)))
        wiz.chk_exp_manual.setChecked(False)

    def test_auto_exposure_without_a_good_value_leaves_it_to_the_operator(self):
        window, wiz = self.window, self.wiz
        wiz.chk_wiz_exp_confirmed.setChecked(False)
        finish = {}
        with patch.object(window, "run_auto_exposure", side_effect=lambda cb: finish.setdefault("cb", cb) and True), \
                patch.object(window, "set_camera_auto_mode"):
            wiz.btn_exp_auto.click()
            finish["cb"](True, "", {"applied": None, "trials": []})
        self.assertTrue(wiz.btn_exp_auto.isEnabled())
        self.assertFalse(wiz.chk_wiz_exp_confirmed.isChecked())
        self.assertTrue(wiz.chk_wiz_auto_exp.isChecked())
        self.assertEqual(wiz.lbl_exp_choice.text(), tr("wizard.slides.slide_exposure.auto_none"))
        # Nothing to keep: no Yes, and the hand controls open by themselves.
        self.assertFalse(wiz.btn_exp_accept.isEnabled())
        self.assertTrue(wiz.chk_exp_manual.isChecked() and not wiz.exp_manual_panel.isHidden())
        wiz.chk_exp_manual.setChecked(False)

    def test_gripper_pose_button_moves_then_reports_the_power(self):
        wiz, W = self.wiz, self.W
        wiz.stacked_widget.setCurrentIndex(W.SLIDE_GRIPPER_POSE)
        wiz.step_completed[W.SLIDE_GRIPPER_POSE] = False
        finish = {}

        def fake_move(on_finished):
            finish["cb"] = on_finished
            return True

        # 2026-11-24 (user): the move runs only while the button is held.
        with patch.object(self.window, "move_to_gripper_pose", side_effect=fake_move), \
                patch.object(self.window, "safe_cancel_control") as cancel:
            wiz.btn_gripper_pose.pressed.emit()
            self.assertEqual(wiz.gripper_pose_status[0], "moving")
            self.assertTrue(wiz.btn_gripper_pose.isEnabled())       # must stay enabled to see the release
            self.assertFalse(wiz.btn_next.isEnabled())
            self.assertFalse(hasattr(wiz, "btn_gp_stop"))              # letting go is the stop
            finish["cb"](True, "", {"right": 0.0, "left": 0.0})
            wiz.btn_gripper_pose.released.emit()                    # after arriving: nothing to stop
            cancel.assert_not_called()
        self.assertEqual(wiz.gripper_pose_status[0], "done")
        self.assertIn("right 0 V", wiz.lbl_gripper_pose_status.text())
        self.assertTrue(wiz.step_completed[W.SLIDE_GRIPPER_POSE] and wiz.btn_next.isEnabled())
        # Let go before the pose: the motion is cancelled and the slide says how to carry on.
        with patch.object(self.window, "move_to_gripper_pose", side_effect=fake_move), \
                patch.object(self.window, "safe_cancel_control") as cancel:
            wiz.btn_gripper_pose.pressed.emit()
            wiz.btn_gripper_pose.released.emit()
            cancel.assert_called_once()
            finish["cb"](False, "Stopped by user", {})
        self.assertEqual(wiz.gripper_pose_status[0], "stopped")
        self.assertFalse(wiz.step_completed[W.SLIDE_GRIPPER_POSE])
        with patch.object(self.window, "move_to_gripper_pose", side_effect=fake_move):
            wiz.btn_gripper_pose.pressed.emit()
            finish["cb"](False, "still on", {})
        self.assertEqual(wiz.gripper_pose_status, ("failed", "still on"))
        self.assertFalse(wiz.step_completed[W.SLIDE_GRIPPER_POSE])
        self.assertTrue(wiz.btn_skip.isVisible())

    def test_v13_gripper_pose_text_says_the_power_stays_on(self):
        wiz, W = self.wiz, self.W
        version = self.window.robot_version
        try:
            self.window.robot_version = "1.3"
            wiz.stacked_widget.setCurrentIndex(W.SLIDE_ROBOT_CONNECT)
            wiz.stacked_widget.setCurrentIndex(W.SLIDE_GRIPPER_POSE)
            self.assertEqual(wiz.lbl_gp_desc.text(), tr("wizard.slides.slide_gripper_pose.desc_v13"))
            finish = {}
            with patch.object(self.window, "move_to_gripper_pose", side_effect=lambda cb: finish.setdefault("cb", cb) and True):
                wiz.btn_gripper_pose.pressed.emit()
                finish["cb"](True, "", {})
            self.assertEqual(wiz.lbl_gripper_pose_status.text(), tr("wizard.slides.slide_gripper_pose.status_done_v13"))
            self.window.robot_version = "1.2"
            wiz.stacked_widget.setCurrentIndex(W.SLIDE_ROBOT_CONNECT)
            wiz.stacked_widget.setCurrentIndex(W.SLIDE_GRIPPER_POSE)
            self.assertEqual(wiz.lbl_gp_desc.text(), tr("wizard.slides.slide_gripper_pose.desc"))
        finally:
            self.window.robot_version = version
            wiz.set_gripper_pose_status("idle")

    def test_no_readjustment_window_after_stop(self):
        # 2026-11-17 (user): the error after Stop is expected, the posture readjustment window was not.
        window = self.window
        window.core.stop_event.set()
        requests = []
        spy = lambda *args: requests.append(args)
        window.marker_problem_signal.connect(spy)
        try:
            self.assertFalse(window.prompt_marker_problem_teaching("right", mode="marker"))
            self.assertEqual(requests, [])
            # A request already queued before Stop: the window side gives up without a dialog.
            evt, res = threading.Event(), {"resolved": True}
            with patch.object(window, "show_message_box") as box:
                window._on_marker_problem_requested("right", evt, res)
            box.assert_not_called()
            self.assertTrue(evt.is_set())
            self.assertFalse(res["resolved"])
        finally:
            window.marker_problem_signal.disconnect(spy)
            window.core.stop_event.clear()

    def test_robot_address_by_where_the_program_runs(self):
        from ui import wizard_widget as wizard_mod
        from PySide6.QtWidgets import QDialog
        wiz = self.wiz
        self.assertEqual(wiz.wizard_ip_input.text(), "127.0.0.1:50051")      # no camera: simulator address
        wiz.btn_ip_upc.click()
        self.assertEqual(wiz.wizard_ip_input.text(), "192.168.30.1:50051")

        def typed(text, result):
            def fake_exec(dialog):
                dialog.address_input.setText(text)
                return result
            return fake_exec

        with patch.object(wizard_mod.RobotIpDialog, "exec", typed("192.168.8.130/23", QDialog.Accepted)):
            wiz.btn_ip_external.click()
        self.assertEqual(wiz.wizard_ip_input.text(), "192.168.8.130:50051")
        with patch.object(wizard_mod.RobotIpDialog, "exec", typed("10.0.0.5", QDialog.Rejected)):
            wiz.btn_ip_external.click()
        self.assertEqual(wiz.wizard_ip_input.text(), "192.168.8.130:50051")
        self.assertEqual(wizard_mod.normalize_robot_address(" 192.168.12.1 "), "192.168.12.1:50051")
        self.assertEqual(wizard_mod.normalize_robot_address("192.168.12.1:50052"), "192.168.12.1:50052")
        self.assertEqual(wizard_mod.normalize_robot_address(""), "")
        dlg = wizard_mod.RobotIpDialog("", wiz)
        self.assertIn("192.168.12.1:50051", dlg.lbl_hotspot.text())
        self.assertTrue((ROOT / "img" / "robot_oled_ip.jpg").is_file())
        wiz.wizard_ip_input.setText("127.0.0.1:50051")

    def test_camera_reconnect_ignores_clicks_while_it_runs(self):
        # 2026-11-12: every extra click on RECONNECT ran another reconnect once the first finished.
        window = self.window
        calls = []

        def slow_connect():
            calls.append(1)
            self.assertFalse(window.btn_reconnect_cam.isEnabled())
            self.assertEqual(window.btn_reconnect_cam.text(), tr("common.reconnecting_camera"))
            window.reconnect_camera(show_dialog=False)       # a click arriving meanwhile
            window.btn_reconnect_cam.click()                  # disabled: ignored

        with patch.object(window.core, "connect_camera", side_effect=slow_connect), \
                patch.object(window, "init_camera_exposure_state"), patch.object(window, "refresh_camera_tab_info"):
            window.reconnect_camera(show_dialog=False)
        self.assertEqual(len(calls), 1)
        self.assertTrue(window.btn_reconnect_cam.isEnabled())
        self.assertEqual(window.btn_reconnect_cam.text(), tr("common.reconnect_camera"))

    def test_every_referenced_image_exists(self):
        # A line listing several paths is a fallback list (first existing one is used): one must exist.
        for source in (ROOT / "ui" / "wizard_widget.py", ROOT / "main_ui.py"):
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
                paths = re.findall(r'"(img/[^"]+\.png)"', line)
                if paths:
                    self.assertTrue(any((ROOT / p).is_file() for p in paths), f"{source.name}:{number}: {paths}")

    def test_stage_line_on_top_follows_the_slides(self):
        # 2026-11-12 (user): prepare - mount - calibrate - check result, over every slide.
        W, wiz = self.W, self.wiz
        self.assertEqual(sorted(W.SLIDE_STAGE), list(range(W.SLIDE_COUNT)))
        expected = {W.SLIDE_ROBOT_CONNECT: ["now", "todo", "todo", "todo"],
                    W.SLIDE_MARKER_BRACKET: ["done", "now", "todo", "todo"],
                    W.SLIDE_HOME_OFFSET: ["done", "done", "now", "todo"],
                    W.SLIDE_CHECK_POSE: ["done", "done", "done", "now"],
                    W.SLIDE_APPLY: ["done", "done", "done", "now"]}
        for idx, states in expected.items():
            wiz.stacked_widget.setCurrentIndex(idx)
            self.assertEqual(wiz.stage_progress.states, states, idx)
        self.assertEqual(wiz.stage_progress.labels[1].text(), tr("wizard.progress.mount"))
        self.assertTrue(wiz.stage_progress.isVisibleTo(wiz))

    def test_run_line_follows_the_core_progress(self):
        from core.calibration.sequences.result import SequenceResult
        wiz = self.wiz
        wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_CALIBRATION)
        self.assertEqual(wiz.run_progress.states, ["todo"] * 4)
        wiz.run_phase, wiz.run_detail, wiz.run_skipped, wiz.run_outcome = 0, None, set(), None
        wiz.unified_timer.start(1000)
        try:
            wiz.on_core_progress(SequenceResult("step1"))
            self.assertEqual(wiz.run_progress.states, ["now", "todo", "todo", "todo"])
            wiz.on_core_progress(SequenceResult("step1_5"))
            head = SequenceResult("step1_5", completed={"head_camera": {"skipped": True}})
            wiz.on_core_progress(SequenceResult("full", completed={"step1": SequenceResult("step1"), "step1_5": head}))
            wiz.on_core_progress(SequenceResult("step2"))
            wiz.on_core_progress({"stage": "collect", "pose_index": 23, "pose_total": 60, "samples": 23})
            self.assertEqual(wiz.run_progress.states, ["done", "skipped", "now", "todo"])
            self.assertEqual(wiz.run_progress.labels[2].text(), tr("wizard.progress.run_collect_n", n=23, total=60))
            self.assertEqual(wiz.run_progress.labels[1].text(),
                             tr("wizard.progress.skipped", name=tr("wizard.progress.run_head")))
            wiz.on_core_progress(SequenceResult("step2", partial={"dataset_path": "x.npz"}))
            self.assertEqual(wiz.run_progress.states[3], "now")
            # An event from an older nested result never moves the line back.
            wiz.on_core_progress(SequenceResult("step1"))
            self.assertEqual(wiz.run_progress.states[3], "now")
        finally:
            wiz.stop_step5(True)
        self.assertEqual(wiz.run_progress.states, ["done", "skipped", "done", "done"])
        wiz.on_core_progress(SequenceResult("step1"))   # after the run: ignored
        self.assertEqual(wiz.run_progress.states, ["done", "skipped", "done", "done"])

    def test_run_cautions_carry_their_log_lines(self):
        # 2026-11-24 (user): a caution says what went wrong; clicking it shows the log lines behind it.
        from core.calibration.sequences.result import SequenceResult
        wiz = self.wiz
        wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_CALIBRATION)
        line = "[WARN] wrist_yaw2: the two J6 estimates differ by 0.90 deg (> 0.5 deg)."
        step1 = SequenceResult("step1", completed={"warnings": [{"key": "j6_methods", "params": {"arm": "left", "diff": 0.9}}],
                                                   "warning_log": [line]})
        wiz.run_warnings, wiz.run_warning_logs = {}, {}

        def reset():
            wiz.run_warnings, wiz.run_warning_logs = {}, {}
            wiz.run_phase, wiz.run_detail, wiz.run_skipped, wiz.run_outcome = -1, None, set(), None
            wiz.refresh_run_progress()
        self.addCleanup(reset)
        wiz.collect_run_warnings(SequenceResult("full", completed={"step1": step1}))
        wiz.run_phase, wiz.run_outcome = 3, "completed"
        wiz.refresh_run_progress()
        self.assertIn(tr("wizard.run_warn.click_for_log"), wiz.lbl_run_warnings.text())
        text = wiz.run_warning_log_text()
        self.assertIn(line, text)
        self.assertIn(tr("wizard.progress.run_wrist"), text)
        # Step 2 settled them (J6 solved again): the stage goes back to green.
        wiz.collect_run_warnings(SequenceResult("full", completed={"step1": SequenceResult("step1")}))
        wiz.refresh_run_progress()
        self.assertEqual(wiz.run_warnings, {})
        self.assertFalse(wiz.lbl_run_warnings.isVisible())

    def test_steps_read_top_to_bottom_with_their_buttons(self):
        # 2026-11-12 (user): the teaching-button guide (text, then its button on the right) sits above the
        # steps; steps 1-3 each have a big number and their text on one row, photos / the reset button below.
        # The "skip if already done" note is gone (a reset on record is passed over by itself). Exposure:
        # 3) the check, then 4) the zero pose last.
        wiz, W = self.wiz, self.W
        wiz.stacked_widget.setCurrentIndex(W.SLIDE_HOME_OFFSET)
        QApplication.processEvents()
        self.assertIs(wiz.btn_how_to_move.parentWidget(), wiz.teach_box)
        tops = [lbl.mapToGlobal(lbl.rect().topLeft()).y() for lbl in (wiz.lbl_p1, wiz.lbl_p2, wiz.lbl_p3)]
        self.assertLess(wiz.teach_box.mapToGlobal(wiz.teach_box.rect().bottomLeft()).y(), tops[0])
        self.assertEqual(len(set(tops)), 1, tops)
        self.assertGreater(wiz.btn_step3_reset.mapToGlobal(wiz.btn_step3_reset.rect().topLeft()).y(), tops[0])
        self.assertFalse(hasattr(wiz, "lbl_skip_hint7") or hasattr(wiz, "inst3_2_box"))
        box = wiz.exp_ctrl_box.parentWidget().layout()
        order = [box.indexOf(w) for w in (wiz.exp_motion_box, wiz.exp_ctrl_box)]
        self.assertEqual(order, sorted(order))
        # 2026-11-17 (user): 3) and 4) sit on one row inside the hand panel, 3) on the left.
        for w in (wiz.chk_wiz_exp_confirmed, wiz.btn_exp_zero):
            self.assertTrue(wiz.exp_manual_panel.isAncestorOf(w))
        wiz.stacked_widget.setCurrentIndex(W.SLIDE_EXPOSURE)
        wiz.chk_exp_manual.setChecked(True)
        QApplication.processEvents()
        chk, zero = (w.mapToGlobal(w.rect().center()) for w in (wiz.chk_wiz_exp_confirmed, wiz.btn_exp_zero))
        self.assertLess(chk.x(), zero.x())
        self.assertLess(abs(chk.y() - zero.y()), 10)
        wiz.chk_exp_manual.setChecked(False)
        self.assertTrue(tr("wizard.slides.slide_exposure.chk_confirmed").startswith("3)"))
        self.assertTrue(tr("wizard.slides.slide_exposure.btn_zero").startswith("4)"))

    def test_new_texts_exist_in_both_languages(self):
        slides = ConfigStorage.load(ROOT / "config" / "ui_config" / "i18n.yaml")["wizard"]["slides"]
        needed = {
            "slide_intrinsics_prep": ("title", "notice", "steps", "btn_open"),
            "slide_0": ("title", "cap_head", "cap_nohead", "inst1", "inst2", "cam_unknown"),
            "slide_gripper_pose": ("title", "desc", "desc_v13", "btn_move", "status_idle", "status_moving",
                                   "status_done", "status_done_v13", "status_failed"),
            "slide_1": ("title", "box_title", "inst1"),
            "slide_marker_bracket": ("title", "box_title", "cap_assemble", "cap_overview", "inst1", "inst2", "inst3"),
            "slide_6": ("inst1", "inst2", "inst3", "shoulder_warn", "pending_title", "pending_msg", "btn_go_reset",
                        "teach_title", "teach_desc", "btn_how_to_move", "btn_reset"),
            "slide_exposure": ("title", "inst", "monitor_title", "monitor_right", "monitor_left", "monitor_idle",
                               "monitor_stable", "monitor_unstable", "monitor_hidden", "monitor_all_ok",
                               "monitor_side_ok", "monitor_not_ok", "monitor_jitter", "monitor_note",
                               "monitor_restart", "monitor_confirmed", "not_saved_warn", "motion_title",
                               "motion_note", "manual_title", "btn_auto", "btn_zero", "target_auto", "target_zero",
                               "auto_applied", "auto_camera_auto", "auto_none", "motion_status_idle",
                               "motion_status_moving", "motion_status_done", "motion_status_failed"),
        }
        for slide, keys in needed.items():
            for key in keys:
                for lang in ("en", "ko"):
                    self.assertTrue(slides[slide][key][lang].strip(), (slide, key, lang))


def monitor_info(right=None, left=None, limit=0.2):
    """A core marker_monitor event; a side is (visible, stable) or None for 'not sampled'."""
    def side(state):
        visible, stable = state
        return {"visible": visible, "stable": stable, "rate": 1.0 if visible else 0.0, "samples": 30,
                "jitter": {"rms_mm": 0.05 if stable else 0.9, "max_mm": 1.0} if visible else None}
    sides = {name: side(state) for name, state in (("right", right), ("left", left)) if state is not None}
    return {"running": True, "sides": sides, "max_jitter_mm": limit,
            "all_stable": all(s["stable"] for s in sides.values())}


class TestMarkerMonitorPanel(unittest.TestCase):
    """One arm at a time is shown to the camera, so a marker stays confirmed once it was stable at the
    current exposure, and an exposure change clears that."""
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = MarkerMonitorPanel()

    def test_markers_confirmed_one_at_a_time(self):
        panel = self.panel
        panel.update_monitor(monitor_info(right=(True, True), left=(False, False)))
        self.assertEqual(panel.confirmed, {"right": True, "left": False})
        self.assertEqual(panel.labels["left"].styleSheet(), panel.BAD_STYLE)
        self.assertEqual(panel.lbl_overall.styleSheet(), panel.WARN_STYLE)
        # The right arm went back to zero and the left arm came into view.
        panel.update_monitor(monitor_info(right=(False, False), left=(True, True)))
        self.assertTrue(panel.all_confirmed)
        self.assertEqual(panel.labels["right"].text(),
                         tr("wizard.slides.slide_exposure.monitor_confirmed",
                            marker=tr("wizard.slides.slide_exposure.monitor_right")))
        self.assertEqual(panel.labels["right"].styleSheet(), panel.OK_STYLE)
        self.assertEqual(panel.lbl_overall.styleSheet(), panel.OK_STYLE)

    def test_jumping_marker_is_not_confirmed(self):
        self.panel.update_monitor(monitor_info(right=(True, False), left=(True, False)))
        self.assertEqual(self.panel.confirmed, {"right": False, "left": False})
        self.assertEqual(self.panel.labels["right"].styleSheet(), self.panel.WARN_STYLE)

    def test_confirmation_survives_a_monitor_stop_but_not_an_exposure_change(self):
        panel = self.panel
        panel.update_monitor(monitor_info(right=(True, True), left=(True, True)))
        panel.update_monitor({"running": False})  # a robot move stops the monitor
        self.assertTrue(panel.all_confirmed)
        self.assertEqual(panel.lbl_overall.styleSheet(), panel.OK_STYLE)
        panel.reset_confirmation()
        self.assertEqual(panel.confirmed, {"right": False, "left": False})
        self.assertEqual(panel.labels["right"].styleSheet(), panel.IDLE_STYLE)

    def test_summary_from_before_the_exposure_change_does_not_confirm(self):
        panel = self.panel
        panel.reset_confirmation(3)
        stable = monitor_info(right=(True, True), left=(True, True))
        panel.update_monitor(dict(stable, window=2))  # still on its way when the exposure changed
        self.assertEqual(panel.confirmed, {"right": False, "left": False})
        self.assertEqual(panel.lbl_overall.styleSheet(), panel.WARN_STYLE)
        panel.update_monitor(dict(stable, window=3))
        self.assertTrue(panel.all_confirmed)

    def test_exposure_change_restarts_the_monitor_window(self):
        from main_ui import UnifiedCalibrationApp
        window = UnifiedCalibrationApp(ui_only=True)
        self.addCleanup(window.close)
        before = window.core.restart_marker_monitor_window()
        window.apply_camera_exposure()
        stable = monitor_info(right=(True, True), left=(True, True))
        window._on_marker_monitor(dict(stable, window=before))
        self.assertFalse(any(any(panel.confirmed.values()) for panel in window.marker_monitor_panels()))
        window._on_marker_monitor(dict(stable, window=before + 1))
        self.assertTrue(all(panel.all_confirmed for panel in window.marker_monitor_panels()))

    def test_every_exposure_change_clears_the_confirmation(self):
        from main_ui import UnifiedCalibrationApp
        window = UnifiedCalibrationApp(ui_only=True)
        self.addCleanup(window.close)
        panels = window.marker_monitor_panels()
        self.assertEqual(len(panels), 2)
        for change in (window.apply_camera_exposure, window.cancel_camera_exposure,
                       window.set_camera_auto_mode, window.init_camera_exposure_state,
                       window.wizard_widget.on_wiz_apply_exp_clicked):
            window._on_marker_monitor(monitor_info(right=(True, True), left=(True, True)))
            self.assertTrue(all(panel.all_confirmed for panel in panels), change.__name__)
            change()
            self.assertFalse(any(any(panel.confirmed.values()) for panel in panels), change.__name__)


class FakeFlangeRobot:
    """Tool flange outputs start at 12 V; `lag` state reads pass before a 0 V command shows up."""
    def __init__(self, accept=True, lag=0, stuck=False):
        self.voltage = {"right": 12, "left": 12}
        self.commands, self.accept, self.lag, self.stuck = [], accept, lag, stuck

    def set_tool_flange_output_voltage(self, name, voltage):
        self.commands.append((name, voltage))
        return self.accept

    def get_state(self):
        if self.lag > 0:
            self.lag -= 1
        elif not self.stuck:
            self.voltage.update({name: voltage for name, voltage in self.commands})
        return MagicMock(tool_flange_right=MagicMock(output_voltage=self.voltage["right"]),
                         tool_flange_left=MagicMock(output_voltage=self.voltage["left"]))


class TestToolFlangePowerOff(unittest.TestCase):
    def test_both_sides_are_set_to_zero_and_read_back(self):
        robot = FakeFlangeRobot(lag=2)
        with patch("core.robot.robot_core.TOOL_FLANGE_POLL_S", 0.0):
            voltages = RobotOperations.power_off_tool_flanges(robot)
        self.assertEqual(robot.commands, [("right", 0), ("left", 0)])
        self.assertEqual(voltages, {"right": 0.0, "left": 0.0})

    def test_a_refused_command_or_an_output_that_stays_on_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            RobotOperations.power_off_tool_flanges(FakeFlangeRobot(accept=False))
        with patch("core.robot.robot_core.TOOL_FLANGE_OFF_TIMEOUT_S", 0.05), \
                patch("core.robot.robot_core.TOOL_FLANGE_POLL_S", 0.01):
            with self.assertRaisesRegex(RuntimeError, "still on"):
                RobotOperations.power_off_tool_flanges(FakeFlangeRobot(stuck=True))

    def test_core_refuses_without_a_robot_or_during_a_sequence(self):
        core = CalibrationCore()
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            core.power_off_tool_flanges()
        core.robot = FakeFlangeRobot()
        core.prepare_run()
        try:
            with self.assertRaisesRegex(RuntimeError, "during a calibration sequence"):
                core.power_off_tool_flanges()
        finally:
            core._prepared = False
            core._run_lock.release()
        self.assertEqual(core.power_off_tool_flanges(), {"right": 0.0, "left": 0.0})


class TestReadyPoseHeadZero(unittest.TestCase):
    """The exposure slide comes before the zero-pose slide, so its arm ready pose also zeroes the head."""
    def setUp(self):
        self.core = CalibrationCore()
        self.core.robot = object()
        self.order = []
        calibrator = self.core.joint_calibrator
        calibrator.movej = MagicMock(side_effect=lambda *a, **k: self.order.append("head") or True)
        calibrator.perform_move_to_ready_pose = MagicMock(
            side_effect=lambda *a, **k: self.order.append("ready") or True)

    def test_head_goes_to_zero_before_the_arm_pose(self):
        self.core.include_head_motion = True
        result = self.core.run("ready", kind="joint", arm_side="right", mode="wrist_pitch", head_zero=True)
        self.assertTrue(result.success, result.error)
        self.assertEqual(self.order, ["head", "ready"])
        kwargs = self.core.joint_calibrator.movej.call_args.kwargs
        self.assertEqual(kwargs["head"], [0.0, 0.0])
        self.assertFalse(kwargs["apply_offsets"])
        self.assertEqual(self.core.joint_calibrator.perform_move_to_ready_pose.call_args.kwargs["mode"], "wrist_pitch")

    def test_head_is_left_alone_by_default_and_without_head_motion(self):
        self.core.include_head_motion = True
        self.assertTrue(self.core.run("ready", kind="joint", arm_side="right", mode="wrist_pitch").success)
        self.core.include_head_motion = False
        self.assertTrue(self.core.run("ready", kind="joint", arm_side="left", mode="wrist_pitch", head_zero=True).success)
        self.assertEqual(self.order, ["ready", "ready"])

    def test_a_failed_head_move_stops_before_the_arm_moves(self):
        self.core.include_head_motion = True
        self.core.joint_calibrator.movej = MagicMock(return_value=False)
        result = self.core.run("ready", kind="joint", arm_side="right", mode="wrist_pitch", head_zero=True)
        self.assertFalse(result.success)
        self.assertEqual(self.order, [])


if __name__ == "__main__":
    unittest.main()
