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

    def test_order_skips_the_optional_calibration_both_ways(self):
        W = self.W
        self.assertEqual(self.wiz.stacked_widget.count(), W.SLIDE_COUNT)
        self.assertEqual(sorted(W.TITLE_KEYS), list(range(W.SLIDE_COUNT)))
        forward = [0]
        while self.wiz.stacked_widget.currentIndex() != W.SLIDE_APPLY:
            self.wiz.go_next()
            forward.append(self.wiz.stacked_widget.currentIndex())
        self.assertEqual(forward, [i for i in range(W.SLIDE_COUNT) if i != W.SLIDE_INTRINSICS_CALIB])
        backward = [forward[-1]]
        while self.wiz.stacked_widget.currentIndex() != 0:
            self.wiz.go_prev()
            backward.append(self.wiz.stacked_widget.currentIndex())
        self.assertEqual(backward, forward[::-1])

    def test_new_slides_sit_where_requested(self):
        W = self.W
        # 2026-09-30: gripper removal + marker bracket are one slide. 2026-10-01: it comes right after
        # the robot connection, because its photo depends on the robot version.
        self.assertEqual(W.SLIDE_MARKER_BRACKET, W.SLIDE_ROBOT_CONNECT + 1)
        # 2026-10-01: exposure (3-1) -> zero pose (3-2) -> home offset reset (3-3).
        self.assertEqual(W.SLIDE_EXPOSURE, W.SLIDE_MARKER_BRACKET + 1)
        self.assertEqual(W.SLIDE_ZERO_POSE, W.SLIDE_EXPOSURE + 1)
        self.assertEqual(W.SLIDE_HOME_OFFSET, W.SLIDE_ZERO_POSE + 1)
        self.assertEqual(W.SLIDE_CALIBRATION, W.SLIDE_HOME_OFFSET + 1)
        numbers = [tr(W.TITLE_KEYS[i]).split(".")[0] for i in range(W.SLIDE_COUNT)]
        self.assertEqual(numbers, ["1-1", "1-2", "2", "3", "3-1", "3-2", "3-3", "4", "5"])
        for index, word in ((W.SLIDE_EXPOSURE, "Exposure"), (W.SLIDE_ZERO_POSE, "Zero"),
                            (W.SLIDE_HOME_OFFSET, "Home Offset")):
            self.assertIn(word, tr(W.TITLE_KEYS[index], lang="en"))
        self.assertIn(W.SLIDE_EXPOSURE, W.VIDEO_SLIDES)
        self.assertFalse(hasattr(W, "SLIDE_GRIPPER") or hasattr(W, "SLIDE_INTRINSICS_CHECK"))

    def test_intrinsics_calibration_is_opened_from_the_camera_slide(self):
        W = self.W
        self.wiz.btn_cam_intrinsics.click()
        self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_INTRINSICS_CALIB)
        self.wiz.go_next()
        self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_ROBOT_CONNECT)
        self.wiz.stacked_widget.setCurrentIndex(W.SLIDE_INTRINSICS_CALIB)
        self.wiz.go_prev()
        self.assertEqual(self.wiz.stacked_widget.currentIndex(), W.SLIDE_CAMERA_MOUNT)

    def test_camera_without_its_own_intrinsics_is_warned(self):
        base = {"running": True, "device_name": "Intel RealSense D405", "serial_number": "123456789",
                "camera_model": "D405", "width": 1280, "height": 720, "fps": 30, "measured_fps": 30.0,
                "intrinsics_file": "camera_intrinsics.yaml"}
        warning = tr("wizard.slides.slide_0.cam_warn_serial")
        with patch.object(self.wiz, "camera_info", return_value={**base, "intrinsics_serial_matched": False,
                                                                  "intrinsics_source": "model"}):
            self.wiz.refresh_camera_info()
            self.assertIn(warning, self.wiz.lbl_cam_info.text())
        with patch.object(self.wiz, "camera_info", return_value={**base, "intrinsics_serial_matched": True,
                                                                  "intrinsics_source": "serial"}):
            self.wiz.refresh_camera_info()
            self.assertNotIn(warning, self.wiz.lbl_cam_info.text())

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
            self.assertEqual(panel.labels["left"].styleSheet(), panel.OK_STYLE)
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

    def test_exposure_slide_buttons_move_one_arm_then_zero(self):
        window, wiz, W = self.window, self.wiz, self.W
        wiz.stacked_widget.setCurrentIndex(W.SLIDE_EXPOSURE)
        wiz.step_completed[W.SLIDE_ZERO_POSE] = False
        calls, finish = [], {}

        def fake_ready(arm_side, on_finished):
            calls.append(("ready", arm_side))
            finish["cb"] = on_finished
            return True

        def fake_zero(on_finished):
            calls.append(("zero",))
            finish["cb"] = on_finished
            return True

        with patch.object(window, "move_arm_to_exposure_pose", side_effect=fake_ready), \
                patch.object(window, "move_to_zero_for_exposure", side_effect=fake_zero):
            for side in ("right", "left"):
                wiz.btn_exp_ready[side].click()
                self.assertEqual(wiz.exposure_motion_status[:2], ("moving", f"ready_{side}"))
                self.assertFalse(wiz.btn_exp_zero.isEnabled() or wiz.btn_prev.isEnabled())
                self.assertTrue(wiz.btn_exp_stop.isEnabled())  # Stop stays usable while moving
                finish["cb"](True, "")
                self.assertEqual(wiz.exposure_motion_status[:2], ("done", f"ready_{side}"))
                self.assertTrue(wiz.btn_exp_zero.isEnabled() and wiz.btn_prev.isEnabled())
            wiz.btn_exp_zero.click()
            finish["cb"](False, "cancelled")
            self.assertEqual(wiz.exposure_motion_status, ("failed", "zero", "cancelled"))
        self.assertEqual(calls, [("ready", "right"), ("ready", "left"), ("zero",)])
        # The zero button here does not complete the zero-pose slide (3-2).
        self.assertFalse(wiz.step_completed[W.SLIDE_ZERO_POSE])

    def test_exposure_ready_pose_is_the_j5_pose_of_the_robot_version(self):
        window = self.window
        version, robot = window.robot_version, window.robot
        try:
            window.robot = object()
            for robot_version, mode in (("1.2", "wrist_pitch"), ("1.3", "wrist_pitch_v13")):
                window.robot_version = robot_version
                with patch("main_ui.MoveToReadyWorker") as worker, \
                        patch.object(window, "_sync_core_state", return_value=window.core):
                    self.assertTrue(window.move_arm_to_exposure_pose("left", lambda *_: None))
                worker.assert_called_once_with(window.core, "left", mode, head_zero=True)
        finally:
            window.robot_version, window.robot = version, robot

    def test_gripper_power_button_reports_the_result(self):
        wiz = self.wiz
        with patch.object(self.window, "power_off_tool_flanges", return_value={"right": 0.0, "left": 0.0}):
            wiz.btn_tool_power_off.click()
        self.assertEqual(wiz.tool_power_status[0], "off")
        self.assertIn("right 0 V", wiz.lbl_tool_power_status.text())
        with patch.object(self.window, "power_off_tool_flanges", side_effect=RuntimeError("still on")):
            wiz.btn_tool_power_off.click()
        self.assertEqual(wiz.tool_power_status, ("failed", "still on"))

    def test_every_referenced_image_exists(self):
        # A line listing several paths is a fallback list (first existing one is used): one must exist.
        for source in (ROOT / "ui" / "wizard_widget.py", ROOT / "main_ui.py"):
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
                paths = re.findall(r'"(img/[^"]+\.png)"', line)
                if paths:
                    self.assertTrue(any((ROOT / p).is_file() for p in paths), f"{source.name}:{number}: {paths}")

    def test_new_texts_exist_in_both_languages(self):
        slides = ConfigStorage.load(ROOT / "config" / "ui_config" / "i18n.yaml")["wizard"]["slides"]
        needed = {
            "slide_0": ("title", "cap_head", "cap_nohead", "inst1", "btn_intrinsics", "cam_warn_serial", "cam_source",
                        "cam_source_serial", "cam_source_model", "cam_source_factory", "cam_source_file"),
            "slide_1": ("box_title", "inst1"),
            "slide_marker_bracket": ("title", "box_title", "cap_assemble", "cap_overview", "inst1", "inst2", "inst3",
                                     "tool_power_note", "btn_tool_power_off", "tool_power_idle", "tool_power_off",
                                     "tool_power_failed"),
            "slide_6": ("inst1", "inst2", "shoulder_warn"),
            "slide_exposure": ("title", "inst", "monitor_title", "monitor_right", "monitor_left", "monitor_idle",
                               "monitor_stable", "monitor_unstable", "monitor_hidden", "monitor_all_ok",
                               "monitor_not_ok", "monitor_jitter", "monitor_note", "monitor_restart",
                               "monitor_confirmed", "not_saved_warn", "motion_title", "motion_note",
                               "btn_ready_right", "btn_ready_left", "btn_zero", "target_ready_right",
                               "target_ready_left", "target_zero", "motion_status_idle", "motion_status_moving",
                               "motion_status_done", "motion_status_failed"),
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
