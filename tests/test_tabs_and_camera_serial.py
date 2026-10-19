"""2026-09-30 UI rework: top tabs Wizard / Camera / Advanced, and per-camera (serial) intrinsics.

Tabs: the old top tabs (Wizard / Step 1 / Step 2) were small and faint and the camera tools sat inside
Step 1. Now Camera (intrinsics calibration, brightness) is a top tab and Step 1 / Step 2 live under
Advanced; code checks tabs by name (TAB_*), not by bare index.

Serial intrinsics: a new camera unit should get its own calibration. config/camera_intrinsics/<serial>.yaml
is used before the per-model store, and Save always (re)writes it -- the serial file wins at the next
start, so leaving an old one would undo a recalibration. Offline, no camera or robot.
"""
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

import core.storage as storage
from core.storage import ConfigStorage, camera_intrinsics_serial_path, save_camera_intrinsics
from core.marker_detection import Marker_Transform
from test_camera_model_config import CameraModelConfigCase

SERIAL = "123456789012"
ROOT = Path(__file__).resolve().parents[1]


class TestSerialIntrinsics(CameraModelConfigCase):
    def resolve_unit(self, device_name, serial):
        observer = Marker_Transform.__new__(Marker_Transform)
        observer.camera = SimpleNamespace(device_name=device_name, serial_number=serial)
        observer.marker_detection = SimpleNamespace(markers_config=None)
        observer._load_all_configs()
        return observer

    def unit_file(self, serial=SERIAL, fx=555.0):
        data = dict(ConfigStorage.load(self.config_path("camera_intrinsics_d405.yaml")))
        data["camera_matrix"] = [[fx, 0.0, 640.0], [0.0, fx, 360.0], [0.0, 0.0, 1.0]]
        path = camera_intrinsics_serial_path(serial)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ConfigStorage.save(path, data)
        return path

    def test_serial_file_lives_in_the_config_camera_intrinsics_folder(self):
        path = camera_intrinsics_serial_path(SERIAL)
        self.assertEqual(os.path.dirname(path), os.path.join(self.folder, "config", "camera_intrinsics"))
        self.assertTrue(path.endswith(f"{SERIAL}.yaml"))
        self.assertIsNone(camera_intrinsics_serial_path(None))
        self.assertNotIn("..", camera_intrinsics_serial_path("../x"))

    def test_without_a_serial_file_the_model_store_is_used_and_flagged(self):
        observer = self.resolve_unit("Intel RealSense D405", SERIAL)
        self.assertFalse(observer.intrinsics_serial_matched)
        self.assertEqual(observer.intrinsics_source, "model")
        self.assertEqual(observer.camera_serial, SERIAL)

    def test_the_units_own_file_wins_over_the_model_store(self):
        self.unit_file(fx=555.0)
        observer = self.resolve_unit("Intel RealSense D405", SERIAL)
        self.assertTrue(observer.intrinsics_serial_matched)
        self.assertEqual(observer.intrinsics_source, "serial")
        working = ConfigStorage.load(observer.active_intrinsics_path)
        self.assertAlmostEqual(working["camera_matrix"][0][0], 555.0)
        self.assertEqual(str(working["serial_number"]), SERIAL)
        self.assertEqual(working["device_name"], "D405")
        info = {"intrinsics_serial_matched": observer.intrinsics_serial_matched}
        self.assertTrue(info["intrinsics_serial_matched"])

    def test_another_units_working_file_is_kept_in_its_own_serial_file(self):
        other = dict(ConfigStorage.load(self.config_path("camera_intrinsics_d405.yaml")))
        other["serial_number"] = "OTHER999"
        other["camera_matrix"] = [[777.0, 0.0, 640.0], [0.0, 777.0, 360.0], [0.0, 0.0, 1.0]]
        ConfigStorage.save(storage.CONFIG_PATHS["camera_intrinsics"], other)
        self.unit_file(fx=555.0)

        self.resolve_unit("Intel RealSense D405", SERIAL)

        archived = ConfigStorage.load(camera_intrinsics_serial_path("OTHER999"))
        self.assertAlmostEqual(archived["camera_matrix"][0][0], 777.0)

    def test_simulation_has_no_serial_and_keeps_the_file(self):
        observer = self.resolve_unit(None, None)
        self.assertIsNone(observer.camera_serial)
        self.assertFalse(observer.intrinsics_serial_matched)

    def test_save_writes_working_model_and_serial_files_and_overwrites_the_serial_file(self):
        self.unit_file(fx=555.0)
        data = {"camera_matrix": [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]],
                "dist_coeffs": [0.0] * 5, "rms_error": 0.2, "width": 1280, "height": 720}

        paths = save_camera_intrinsics(data, family="D405", serial=SERIAL)

        self.assertEqual(paths, [storage.CONFIG_PATHS["camera_intrinsics"], self.config_path("camera_intrinsics_d405.yaml"),
                                 camera_intrinsics_serial_path(SERIAL)])
        for path in paths:
            saved = ConfigStorage.load(path)
            self.assertAlmostEqual(saved["camera_matrix"][0][0], 600.0, msg=path)
            self.assertEqual(saved["device_name"], "D405")
        self.assertEqual(str(ConfigStorage.load(paths[-1])["serial_number"]), SERIAL)
        self.assertEqual(len(save_camera_intrinsics(data, family="D405", serial=None)), 2)

    def test_a_save_while_connected_is_recognized_without_a_reconnect(self):
        # 2026-10-08: after Save the camera still read "not registered" until it was reconnected.
        # The UI wrote the status onto core.observer, a wrapper; the flags live on its engine.
        from core.calibration import CalibrationCore
        from core.calibration.observation import ObservationSource
        from core.marker_detection import Marker_Detection
        engine = self.resolve_unit("Intel RealSense D405", SERIAL)
        self.assertFalse(engine.intrinsics_serial_matched)
        engine.camera = None
        engine.marker_detection = Marker_Detection()
        engine.width, engine.height = 1280, 720
        engine._factory_intrinsics, engine._factory_dist_coeffs = [640.0, 360.0, 900.0, 900.0], [0.0] * 5
        core = CalibrationCore()
        self.addCleanup(core.close)
        core.observer = ObservationSource(engine, core.stop_event)

        save_camera_intrinsics({"camera_matrix": [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]],
                                "dist_coeffs": [0.0] * 5, "rms_error": 0.2, "width": 1280, "height": 720},
                               family="D405", serial=SERIAL)
        with patch("core.marker_detection.calib_intrinsics_mode", "full"):
            self.assertTrue(core.reload_camera_intrinsics())

        self.assertTrue(engine.intrinsics_serial_matched)
        self.assertEqual(engine.intrinsics_source, "serial")
        self.assertAlmostEqual(engine.marker_detection.fx, 600.0)
        # Nothing may sit on the wrapper, where it would hide the engine's value.
        self.assertNotIn("intrinsics_serial_matched", vars(core.observer))


class TestTopTabs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        from main_ui import UnifiedCalibrationApp
        cls.window = UnifiedCalibrationApp(ui_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def test_wizard_camera_advanced_with_their_sub_tabs(self):
        w = self.window
        self.assertEqual(w.left_tabs.count(), 3)
        self.assertEqual(w.left_tabs.objectName(), "mainTabs")
        self.assertEqual(w.camera_tabs.count(), 2)
        self.assertEqual(w.advanced_tabs.count(), 2)
        self.assertIs(w.camera_tabs.widget(w.CAMERA_SUB_INTRINSICS).isAncestorOf(w.video_label), True)
        self.assertIs(w.camera_tabs.widget(w.CAMERA_SUB_EXPOSURE).isAncestorOf(w.spin_exposure), True)
        self.assertIs(w.camera_tabs.widget(w.CAMERA_SUB_EXPOSURE).isAncestorOf(w.video_label_exposure), True)
        self.assertIs(w.advanced_tabs.widget(w.ADV_SUB_STEP1).isAncestorOf(w.workflow_tabs), True)

    def test_tab_checks_use_the_names(self):
        w = self.window
        w.left_tabs.setCurrentIndex(w.TAB_CAMERA)
        w.camera_tabs.setCurrentIndex(w.CAMERA_SUB_EXPOSURE)
        self.assertTrue(w.is_camera_tab_active())
        self.assertFalse(w.is_intrinsics_subtab_active())
        w.camera_tabs.setCurrentIndex(w.CAMERA_SUB_INTRINSICS)
        self.assertTrue(w.is_intrinsics_subtab_active())
        w.left_tabs.setCurrentIndex(w.TAB_ADVANCED)
        w.advanced_tabs.setCurrentIndex(w.ADV_SUB_STEP2)
        self.assertTrue(w.is_step2_view_active())
        self.assertFalse(w.is_camera_tab_active())

    def test_shared_boxes_follow_step1_and_step2_under_advanced(self):
        w = self.window
        w.left_tabs.setCurrentIndex(w.TAB_ADVANCED)
        w.advanced_tabs.setCurrentIndex(w.ADV_SUB_STEP2)
        self.assertTrue(w.advanced_tabs.widget(w.ADV_SUB_STEP2).isAncestorOf(w.conn_head_box))
        w.advanced_tabs.setCurrentIndex(w.ADV_SUB_STEP1)
        self.assertTrue(w.advanced_tabs.widget(w.ADV_SUB_STEP1).isAncestorOf(w.conn_head_box))

    def test_camera_header_without_camera(self):
        from core.language import tr
        self.window.refresh_camera_tab_info()
        self.assertEqual(self.window.lbl_camera_tab_info.text(), tr("main_tabs.camera_info_none"))
        self.assertEqual(self.window.lbl_camera_serial_status.text(), "")

    def test_camera_header_says_whether_the_serial_is_registered(self):
        # 2026-10-08: the header named the intrinsics yaml; the operator wants to see at a glance
        # whether this unit already has its own calibration (green) or runs on the defaults (red).
        from core.language import tr
        w = self.window
        base = {"camera_model": "d405", "serial_number": SERIAL, "intrinsics_file": "camera_intrinsics.yaml"}
        cases = (({**base, "intrinsics_serial_matched": True, "intrinsics_source": "serial"},
                  "main_tabs.camera_serial_registered", "#00e676"),
                 ({**base, "intrinsics_serial_matched": False, "intrinsics_source": "model"},
                  "main_tabs.camera_serial_unregistered", "#ff1744"),
                 ({**base, "serial_number": None, "intrinsics_serial_matched": False, "intrinsics_source": "file"},
                  "main_tabs.camera_serial_unknown", "#ff1744"))
        self.addCleanup(w.refresh_camera_tab_info)
        for info, key, color in cases:
            with patch.object(w.core, "observer", SimpleNamespace(get_camera_info=lambda info=info: info)):
                w.refresh_camera_tab_info()
                w.wizard_widget.refresh_intrinsics_serial_status()
            for label in (w.lbl_camera_serial_status, w.wizard_widget.lbl_int_serial_status):
                self.assertEqual(label.text(), tr(key), key)
                self.assertIn(color, label.styleSheet(), key)
            self.assertIn("D405", w.lbl_camera_tab_info.text())
            self.assertNotIn(".yaml", w.lbl_camera_tab_info.text() + w.lbl_camera_serial_status.text())
        for key in ("camera_serial_registered", "camera_serial_unregistered", "camera_serial_unknown"):
            self.assertNotEqual(tr(f"main_tabs.{key}", lang="en"), tr(f"main_tabs.{key}", lang="ko"))


class TestIntrinsicsBusy(unittest.TestCase):
    """2026-10-08: nothing showed that the intrinsics solve was running (RUN CALIBRATION, or the
    automatic run after the 16th capture). Both intrinsics screens now show a busy indicator and
    lock their buttons until the worker's result arrives."""
    BUTTONS = ("btn_int_capture", "btn_int_calibrate", "btn_int_save", "btn_int_reset")

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        from main_ui import UnifiedCalibrationApp
        cls.window = UnifiedCalibrationApp(ui_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def setUp(self):
        self.window.captured_images.clear()
        self.addCleanup(self.window.captured_images.clear)
        self.addCleanup(self.window.set_intrinsics_busy, False)

    def panels(self):
        return (self.window, self.window.wizard_widget)

    def test_busy_shows_the_indicator_and_locks_the_buttons(self):
        w = self.window
        before = {(panel, name): getattr(panel, name).isEnabled() for panel in self.panels() for name in self.BUTTONS}
        self.assertFalse(before[w, "btn_int_save"])
        self.assertTrue(all(panel.int_busy.isHidden() for panel in self.panels()))

        w.set_intrinsics_busy(True)

        for panel in self.panels():
            self.assertFalse(panel.int_busy.isHidden())
            self.assertEqual((panel.int_busy.bar.minimum(), panel.int_busy.bar.maximum()), (0, 0))
            for name in self.BUTTONS:
                self.assertFalse(getattr(panel, name).isEnabled(), name)
        self.assertTrue(w.wizard_widget.lbl_step1_status.isHidden())

        w.set_intrinsics_busy(False)

        after = {(panel, name): getattr(panel, name).isEnabled() for panel in self.panels() for name in self.BUTTONS}
        self.assertEqual(after, before)
        self.assertTrue(all(panel.int_busy.isHidden() for panel in self.panels()))
        self.assertFalse(w.wizard_widget.lbl_step1_status.isHidden())

    def test_capture_is_ignored_while_busy(self):
        w = self.window
        with patch.object(w, "current_frame", np.zeros((4, 4, 3), np.uint8), create=True):
            w.set_intrinsics_busy(True)
            w.capture_intrinsics_frame()
            self.assertEqual(len(w.captured_images), 0)
            w.set_intrinsics_busy(False)
            w.capture_intrinsics_frame()
            self.assertEqual(len(w.captured_images), 1)

    def test_the_run_is_busy_from_start_until_its_result(self):
        import main_ui
        w = self.window
        started = []

        class FakeWorker:
            def __init__(self, core, name, options=None):
                self.finished_result = SimpleNamespace(connect=lambda slot: None)

            def start(self):
                started.append(w.intrinsics_busy)

        w.captured_images.extend(np.zeros((4, 4, 3), np.uint8) for _ in range(16))
        with patch.object(main_ui, "SequenceWorker", FakeWorker):
            w.run_intrinsics_calibration()
        self.assertEqual(started, [False])
        self.assertTrue(w.intrinsics_busy)
        self.assertFalse(w.int_busy.isHidden())

        with patch.object(main_ui.QMessageBox, "critical"):
            w._on_intrinsics_finished(SimpleNamespace(success=False))
        self.assertFalse(w.intrinsics_busy)
        self.assertTrue(w.int_busy.isHidden())
        self.assertTrue(w.btn_int_calibrate.isEnabled())
        self.assertFalse(w.btn_int_save.isEnabled())

    def test_too_few_frames_never_turn_busy(self):
        import main_ui
        with patch.object(main_ui.QMessageBox, "warning"):
            self.window.run_intrinsics_calibration()
        self.assertFalse(self.window.intrinsics_busy)
        self.assertTrue(self.window.int_busy.isHidden())

    def test_the_wizard_capture_that_starts_the_run_leaves_the_status_to_the_result(self):
        w, wiz = self.window, self.window.wizard_widget
        wiz.lbl_step1_status.setText("before")

        def capture():
            w.intrinsics_busy = True

        with patch.object(w, "capture_intrinsics_frame", capture):
            wiz.step1_capture()
        self.assertEqual(wiz.lbl_step1_status.text(), "before")


class TestLanguagesAreNotMixed(unittest.TestCase):
    """2026-10-08: some texts showed both languages whatever was selected ("확인 (Close)",
    "... (적용 완료했습니다)"). Each text now holds its own language only."""
    HANGUL = re.compile(r"[가-힣]")
    # English left inside a Korean text on purpose: a file name, or a state whose button reads the same.
    KEPT = re.compile(r"\.yaml|Baseline|Optimized")

    def leaves(self, node, path=()):
        if isinstance(node, dict) and {"en", "ko"} <= set(node):
            yield ".".join(path), node["en"], node["ko"]
        elif isinstance(node, dict):
            for key, value in node.items():
                yield from self.leaves(value, path + (str(key),))

    def setUp(self):
        self.texts = list(self.leaves(ConfigStorage.load(ROOT / "config" / "ui_config" / "i18n.yaml")))
        self.assertGreater(len(self.texts), 200)

    def test_english_texts_hold_no_korean(self):
        self.assertEqual([key for key, en, _ko in self.texts if self.HANGUL.search(en)], [])

    def test_korean_texts_do_not_repeat_themselves_in_english(self):
        mixed = []
        for key, _en, ko in self.texts:
            for inner in re.findall(r"\(([^()]*)\)", re.sub(r"\{[^}]*\}", "", ko)):
                if re.search(r"[A-Za-z]{3,}", inner) and not self.HANGUL.search(inner) and not self.KEPT.search(inner):
                    mixed.append((key, inner))
        self.assertEqual(mixed, [])

    def test_ui_source_texts_hold_no_korean(self):
        import ast
        found = []
        for source in (ROOT / "main_ui.py", *sorted((ROOT / "ui").glob("*.py"))):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                          if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body
                          and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
                        and self.HANGUL.search(node.value)):
                    found.append((source.name, node.lineno, node.value))
        # The language picker shows each language in its own name.
        self.assertEqual([text for _name, _line, text in found], ["한국어"], found)


if __name__ == "__main__":
    unittest.main()
