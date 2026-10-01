"""2026-09-30 UI rework: top tabs Wizard / Camera / Advanced, and per-camera (serial) intrinsics.

Tabs: the old top tabs (Wizard / Step 1 / Step 2) were small and faint and the camera tools sat inside
Step 1. Now Camera (intrinsics calibration, brightness) is a top tab and Step 1 / Step 2 live under
Advanced; code checks tabs by name (TAB_*), not by bare index.

Serial intrinsics: a new camera unit should get its own calibration. config/camera_intrinsics/<serial>.yaml
is used before the per-model store, and Save always (re)writes it -- the serial file wins at the next
start, so leaving an old one would undo a recalibration. Offline, no camera or robot.
"""
import os
import unittest
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

import core.storage as storage
from core.storage import ConfigStorage, camera_intrinsics_serial_path, save_camera_intrinsics
from core.marker_detection import Marker_Transform
from test_camera_model_config import CameraModelConfigCase

SERIAL = "123456789012"


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


if __name__ == "__main__":
    unittest.main()
