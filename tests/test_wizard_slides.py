"""Wizard slide order, images, texts, and the exposure-slide marker monitor. Offline, offscreen."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import re
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PySide6.QtWidgets import QApplication

from core.calibration import CalibrationCore
import tempfile
from core.calibration.data import (DEFAULT_MAX_MARKER_JITTER_MM, load_max_marker_jitter_mm,
                                   marker_monitor_summary, marker_position_jitter_mm)
from core.calibration.sequences.marker_monitor import MarkerMonitor
from core.calibration.sequences.result import SequenceCancelled
from core.storage import ConfigStorage

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
        self.assertEqual(W.SLIDE_MARKER_BRACKET, W.SLIDE_GRIPPER + 1)
        self.assertEqual(W.SLIDE_EXPOSURE, W.SLIDE_HOME_OFFSET + 1)
        self.assertEqual(W.SLIDE_CALIBRATION, W.SLIDE_EXPOSURE + 1)
        self.assertIn(W.SLIDE_EXPOSURE, W.VIDEO_SLIDES)

    def test_marker_monitor_runs_only_on_the_exposure_slide(self):
        core = self.window.core
        core.observer = FakeObserver()
        try:
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_CALIBRATION)
            self.assertFalse(core.marker_monitor_running)
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_EXPOSURE)
            self.assertTrue(core.marker_monitor_running)
            self.assertTrue(wait_for(lambda: "%" in self.wiz.marker_mon_labels["right"].text()))
            # Right marker jumps ~0.9 mm RMS: recognized but not stable -> orange, overall not OK.
            self.assertEqual(self.wiz.marker_mon_labels["right"].styleSheet(), self.W.MONITOR_WARN_STYLE)
            self.assertEqual(self.wiz.lbl_marker_mon_overall.styleSheet(), self.W.MONITOR_WARN_STYLE)
            core.stop_marker_monitor()
            core.observer = FakeObserver(right_noise_m=0.00002, left_always=True)
            self.wiz.sync_marker_monitor()
            self.assertTrue(wait_for(lambda: self.wiz.lbl_marker_mon_overall.styleSheet() == self.W.MONITOR_OK_STYLE))
            self.assertEqual(self.wiz.marker_mon_labels["left"].styleSheet(), self.W.MONITOR_OK_STYLE)
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_CALIBRATION)
            self.assertFalse(core.marker_monitor_running)
            self.wiz.stacked_widget.setCurrentIndex(self.W.SLIDE_EXPOSURE)
            self.wiz.setVisible(False)
            self.assertFalse(core.marker_monitor_running)
        finally:
            core.stop_marker_monitor()
            core.observer = None

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
            "slide_0": ("title", "cap_head", "cap_nohead", "inst1"),
            "slide_1": ("title", "box_title", "inst1"),
            "slide_marker_bracket": ("title", "box_title", "cap_assemble", "cap_overview", "inst1", "inst2", "inst3"),
            "slide_6": ("inst1", "inst2", "shoulder_warn"),
            "slide_exposure": ("title", "inst", "monitor_title", "monitor_right", "monitor_left", "monitor_idle",
                               "monitor_stable", "monitor_unstable", "monitor_hidden", "monitor_all_ok",
                               "monitor_not_ok", "monitor_jitter", "monitor_note", "monitor_restart"),
        }
        for slide, keys in needed.items():
            for key in keys:
                for lang in ("en", "ko"):
                    self.assertTrue(slides[slide][key][lang].strip(), (slide, key, lang))


if __name__ == "__main__":
    unittest.main()
