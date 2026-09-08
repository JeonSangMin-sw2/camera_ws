"""Real persistence paths with temporary configs; no robot motion or user writes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace, MethodType
import tempfile
import unittest
from unittest.mock import patch, Mock

import yaml
from PySide6.QtWidgets import QApplication, QLineEdit
from main_ui import UnifiedCalibrationApp
from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
from core.paths import CONFIG_PATHS


OLD = [.001216910260022361, .05452662320206416, -.04985474186323428,
       90.0604531953877, -.011647405483456354, -.03223181369789695]
NEW = [.0010028638699574037, .05455610378110098, -.0499611612241565,
       90.15735293769633, .09764915699520482, -.02854435260503006]
BROKEN = 'camera: {}\nmarker:\n  pose: [1, 2, 3, 4, 5, 6]\n    90.1, 0.2, 0.3]\n'


def app_double():
    app = SimpleNamespace(logs=[], ui_only=True, marker_st=None,
        last_full_auto_error=None, last_full_auto_converged=True,
        get_robot_version=lambda: '1.2', update_applied_offset_label=lambda: None,
        joint_offsets_store={s: {'joint3': .1, 'joint5': .2, 'joint6': .3} for s in ('left', 'right')},
        joint_offsets={s: {'elbow': 0., 'wrist_pitch': 0., 'wrist_roll': 0., 'wrist_yaw2': 0.} for s in ('left', 'right')},
        joint_calibrator=SimpleNamespace(camera_config={}), marker_calibrator=SimpleNamespace(camera_config={}))
    app.log_msg = app.logs.append
    for name in ('save_offsets_to_yaml', 'apply_joint_offset', 'apply_bracket_design_values',
                 '_update_marker_key_in_lines', '_update_camera_key_in_lines', 'apply_full_auto_results',
                 '_joint_offset_patch', '_bracket_patch', '_publish_joint_offsets', '_publish_brackets'):
        setattr(app, name, MethodType(getattr(UnifiedCalibrationApp, name), app))
    for side in ('l', 'r'):
        for axis, value in zip(('x', 'y', 'z', 'roll', 'pitch', 'yaw'), NEW):
            setattr(app, f'txt_bracket_{side}_{axis}', QLineEdit(str(value)))
    return app


class PersistenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_intrinsics_save_failure_keeps_detector_unchanged(self):
        from core.marker_detection import Marker_Transform
        obj = SimpleNamespace(camera=Mock(), width=640, height=480,
            marker_detection=Mock(), intrinsics_metadata={'source': 'factory'},
            camera_config={'intrinsics_source': 'factory'})
        persist = Mock(side_effect=OSError('disk full'))
        with patch('core.camera_intrinsics.select_intrinsics', return_value=([1]*4, [0]*5, {'source': 'calibrated'})):
            with self.assertRaises(OSError):
                Marker_Transform.apply_intrinsics_source(obj, 'calibrated', persist=persist)
        obj.marker_detection.set_intrinsics_param.assert_not_called()
        self.assertEqual(obj.intrinsics_metadata, {'source': 'factory'})
        self.assertEqual(obj.camera_config['intrinsics_source'], 'factory')

    def test_clear_failure_preserves_staged_and_applied_offsets(self):
        from PySide6.QtWidgets import QMessageBox
        app = app_double()
        before = deepcopy(app.joint_offsets_store), deepcopy(app.joint_offsets)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            path.write_text(BROKEN)
            with patch.dict(CONFIG_PATHS, setting_yaml=str(path)), patch('main_ui.QMessageBox.question', return_value=QMessageBox.Yes):
                UnifiedCalibrationApp.clear_joint_offset(app)
            self.assertEqual(before, (app.joint_offsets_store, app.joint_offsets))
            self.assertFalse(any('[CLEAR]' in msg for msg in app.logs))

    def test_full_auto_and_head_round_trip_and_replace_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            path.write_text(yaml.dump({'camera': {'mount_to_cam': OLD, 'mount_to_cam_nominal': OLD},
                'marker': {'Tf_to_marker_left': OLD}, 'joint_offset': {'custom': 99}}, default_flow_style=None))
            app = app_double()
            obj = SimpleNamespace(camera_config={'mount_to_cam_nominal': OLD}, app=app)
            result = {'success': True, 'calibrated_mount_to_cam': NEW, 'head_offsets_deg': {'pan': .4, 'tilt': .5}}
            with patch.dict(CONFIG_PATHS, setting_yaml=str(path)):
                for _ in range(3):
                    self.assertTrue(app.apply_full_auto_results(silent=True))
                    self.assertTrue(HeadCameraCalibrator.apply_calibration_results(obj, result))
                saved = yaml.safe_load(path.read_text())
                self.assertEqual(saved['camera']['mount_to_cam_nominal'], OLD)
                self.assertEqual(saved['camera']['mount_to_cam'], NEW)
                self.assertEqual(saved['marker']['Tf_to_marker_left'], NEW)
                self.assertEqual(saved['joint_offset']['custom'], 99)
                before = path.read_bytes(), deepcopy(app.joint_offsets), deepcopy(obj.camera_config)
                app.joint_offsets_store['left']['joint5'] = 42
                with patch('core.config_store.os.replace', side_effect=OSError('disk full')):
                    self.assertFalse(app.apply_full_auto_results(silent=True))
                    self.assertFalse(app.apply_joint_offset())
                    self.assertFalse(HeadCameraCalibrator.apply_calibration_results(obj, {**result, 'calibrated_mount_to_cam': OLD}))
                self.assertEqual(before, (path.read_bytes(), app.joint_offsets, obj.camera_config))

    def test_wrapped_flow_updates_reparse_and_preserve_precision(self):
        for section in ('marker', 'camera'):
            with self.subTest(section=section):
                key = 'Tf_to_marker_left' if section == 'marker' else 'mount_to_cam'
                original = yaml.dump({section: {key: OLD, 'keep': {'custom': 42}}}, default_flow_style=None)
                lines = original.splitlines(keepends=True)
                getattr(UnifiedCalibrationApp, f'_update_{section}_key_in_lines')(None, lines, key, NEW)
                try:
                    result = yaml.safe_load(''.join(lines))
                except yaml.YAMLError as exc:
                    self.fail(f'Production updater left invalid continuation: {exc}')
                self.assertEqual(result[section][key], NEW)
                self.assertEqual(result[section]['keep'], {'custom': 42})

    def test_head_parse_failure_preserves_all_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            path.write_text(BROKEN)
            app = app_double()
            obj = SimpleNamespace(camera_config={'mount_to_cam': OLD.copy()}, app=app)
            before = deepcopy(obj.camera_config)
            with patch.dict(CONFIG_PATHS, setting_yaml=str(path)):
                ok = HeadCameraCalibrator.apply_calibration_results(obj,
                    {'success': True, 'calibrated_mount_to_cam': NEW, 'head_offsets_deg': {'pan': .4, 'tilt': .5}})
            self.assertFalse(ok)
            self.assertEqual(obj.camera_config, before)
            self.assertEqual(app.joint_calibrator.camera_config, {})
            self.assertNotIn('head', app.joint_offsets_store)
            self.assertEqual(path.read_text(), BROKEN)

    def test_joint_save_rejects_corrupt_document(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            path.write_text(BROKEN)
            app = app_double()
            with patch.dict(CONFIG_PATHS, setting_yaml=str(path)):
                ok = app.save_offsets_to_yaml()
            self.assertIs(ok, False)
            self.assertEqual(path.read_text(), BROKEN)
            self.assertFalse(any('[SUCCESS]' in msg for msg in app.logs))

    def test_full_auto_invalid_bracket_does_not_partially_save_joints(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            original = 'camera: {}\nmarker: {}\njoint_offset: {custom: 99}\n'
            path.write_text(original)
            app = app_double()
            before = deepcopy(app.joint_offsets)
            app.txt_bracket_l_x.setText('not-a-number')
            with patch.dict(CONFIG_PATHS, setting_yaml=str(path)):
                app.apply_full_auto_results(silent=True)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(app.joint_offsets, before)
            self.assertFalse(any('applied successfully' in msg for msg in app.logs))


class AtomicStoreTests(unittest.TestCase):
    def store(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('core.config_store'), 'Shared transactional config storage is missing')
        from core.config_store import update_yaml
        return update_yaml

    def test_replace_failure_preserves_original_and_removes_temporary_file(self):
        update = self.store()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            original = 'camera: {intrinsics_source: factory}\n'
            path.write_text(original)
            with patch('core.config_store.os.replace', side_effect=OSError('injected replace failure')):
                with self.assertRaises(OSError):
                    update(path, {'camera': {'intrinsics_source': 'calibrated'}})
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_multiple_updates_preserve_unknown_keys_precision_and_numpy_values(self):
        import numpy as np
        update = self.store()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            path.write_text(yaml.dump({'camera': {'mount_to_cam': OLD, 'user': '메모'},
                                      'joint_offset': {'left': {'custom': 7}}}, default_flow_style=None))
            for _ in range(3):
                update(path, {'camera': {'mount_to_cam': np.array(NEW)},
                              'joint_offset': {'left': {'joint3': np.float64(.123456789)}}})
            doc = yaml.safe_load(path.read_text())
            self.assertEqual(doc['camera']['mount_to_cam'], NEW)
            self.assertEqual(doc['camera']['user'], '메모')
            self.assertEqual(doc['joint_offset']['left'], {'custom': 7, 'joint3': .123456789})

    def test_invalid_existing_yaml_never_overwritten(self):
        update = self.store()
        for original in (BROKEN, '[]\n', 'camera: {}\ncamera: {}\n', 'camera: 12\n', ''):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'setting.yaml'
                path.write_text(original)
                with self.assertRaises((ValueError, yaml.YAMLError)):
                    update(path, {'camera': {'intrinsics_source': 'calibrated'}})
                self.assertEqual(path.read_text(), original)

    def test_nonfinite_updates_never_written(self):
        update = self.store()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'setting.yaml'
            path.write_text('camera: {}\n')
            for value in (float('nan'), float('inf'), float('-inf')):
                with self.assertRaises(ValueError):
                    update(path, {'camera': {'mount_to_cam': [value]*6}})
                self.assertEqual(path.read_text(), 'camera: {}\n')


if __name__ == '__main__':
    unittest.main()
