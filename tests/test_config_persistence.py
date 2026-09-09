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
import numpy as np
from PySide6.QtWidgets import QApplication, QLineEdit
from main_ui import UnifiedCalibrationApp
from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
from core.config_store import CONFIG_PATHS


OLD = [.001216910260022361, .05452662320206416, -.04985474186323428,
       90.0604531953877, -.011647405483456354, -.03223181369789695]
NEW = [.0010028638699574037, .05455610378110098, -.0499611612241565,
       90.15735293769633, .09764915699520482, -.02854435260503006]
BROKEN = 'camera: {}\nmarker:\n  pose: [1, 2, 3, 4, 5, 6]\n    90.1, 0.2, 0.3]\n'


def app_double():
    app = SimpleNamespace(logs=[], sim=True, marker_st=None,
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
        obj = Marker_Transform(sim=True)
        before = deepcopy(obj.intrinsics_metadata)
        data = yaml.safe_load(Path(CONFIG_PATHS['camera_intrinsics']).read_text())
        with patch('core.config_store.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                obj.save_intrinsics(data)
        self.assertEqual(obj.intrinsics_metadata, before)

    def test_missing_camera_selects_sim_but_runtime_failure_propagates(self):
        from core.marker_detection import Marker_Transform, CameraUnavailableError
        with patch('core.marker_detection.RealSenseCamera', side_effect=CameraUnavailableError('missing')):
            self.assertTrue(Marker_Transform().sim)
        with patch('core.marker_detection.RealSenseCamera', side_effect=RuntimeError('stream failed')):
            with self.assertRaisesRegex(RuntimeError, 'stream failed'):
                Marker_Transform()

    def test_application_import_and_sim_start_without_realsense_driver(self):
        import subprocess
        import sys
        command = "import sys; sys.modules['pyrealsense2'] = None; import main_ui; from core.marker_detection import Marker_Transform; assert Marker_Transform(sim=True).sim"
        result = subprocess.run([sys.executable, '-c', command], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sim_intrinsics_are_fixed_yaml_and_invalid_reload_preserves_state(self):
        from core.marker_detection import Marker_Transform
        obj = Marker_Transform(sim=True)
        self.assertAlmostEqual(obj.marker_detection.fx, 660.3380121560385)
        before = deepcopy(obj.intrinsics_metadata)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'camera_intrinsics.yaml'
            data = yaml.safe_load(Path(CONFIG_PATHS['camera_intrinsics']).read_text())
            data['camera_matrix'][0][0] = float('nan')
            path.write_text(yaml.safe_dump(data))
            with patch.dict(CONFIG_PATHS, camera_intrinsics=str(path)):
                with self.assertRaises(ValueError):
                    obj.reload_intrinsics()
        self.assertEqual(obj.intrinsics_metadata, before)

    def test_intrinsics_reject_nonfinite_or_nonpositive_dimensions(self):
        from core.marker_detection import Marker_Transform
        obj = Marker_Transform(sim=True)
        data = yaml.safe_load(Path(CONFIG_PATHS['camera_intrinsics']).read_text())
        for width in (float('inf'), 0, -1, 640.5):
            obj.width = data['width'] = width
            with self.assertRaises(ValueError):
                obj._validate_intrinsics(data)

    def test_sim_requires_encoder_source_and_returns_flat_metres(self):
        from core.marker_detection import Marker_Transform
        obj = Marker_Transform(sim=True)
        obj.set_marker_type('plate')
        with self.assertRaisesRegex(RuntimeError, 'robot'):
            obj.get_marker_transform()
        robot = SimpleNamespace(get_state=lambda: SimpleNamespace(position=np.zeros(24)))
        obj.bind_robot(robot, '1.2')
        right, left = np.eye(4), np.eye(4)
        right[0, 3], left[0, 3] = 0.4, -0.3
        with patch.object(type(obj.simulation_model), 'marker_pose', side_effect=lambda robot, q, side, rng: right if side == 'right' else left):
            observed = obj.get_marker_transform(side='all')
        self.assertEqual(np.asarray(observed).shape, (2, 16))
        self.assertEqual(observed[0][3], 0.4)
        self.assertEqual(observed[1][3], -0.3)

    def test_numeric_field_keeps_unchanged_precision(self):
        from main_ui import set_numeric_field, read_numeric_field
        field = QLineEdit()
        set_numeric_field(field, OLD[0])
        self.assertEqual(read_numeric_field(field), OLD[0])
        field.setText('0.25')
        self.assertEqual(read_numeric_field(field), 0.25)

    def test_disconnected_home_workers_report_failure(self):
        from main_ui import HomeOffsetResetWorker, MoveHomeOffsetWorker, ApplyCurrentPoseWorker
        workers = [HomeOffsetResetWorker(None, None, 'm', False),
                   MoveHomeOffsetWorker(None, None, 'both', '/unused', False, 'preview'),
                   ApplyCurrentPoseWorker(None, None, 'both', False)]
        for worker in workers:
            outcomes = []
            worker.finished_signal.connect(outcomes.append)
            worker.run()
            self.assertEqual(len(outcomes), 1)
            result = outcomes[0]
            self.assertFalse(result.get('success') if isinstance(result, dict) else result)

    def test_real_missing_frames_timeout_and_real_translations_use_metres(self):
        import time
        from core.marker_detection import Marker_Transform
        obj = Marker_Transform(sim=True)
        obj.set_marker_type('plate')
        obj.sim = False
        obj.camera = SimpleNamespace(camera_monitoring=True, get_color_image=lambda: None,
                                     get_depth_image=lambda: None)
        started = time.monotonic()
        self.assertIsNone(obj.get_marker_transform(sampling_time=.03))
        self.assertLess(time.monotonic() - started, .25)
        obj.camera.get_color_image = lambda: np.zeros((2, 2, 3), dtype=np.uint8)
        transform = np.eye(4)
        transform[0, 3] = 400.0
        with patch.object(obj.marker_detection, 'detect', return_value=[('plate_right', transform.ravel())]):
            observed = obj.get_marker_transform(side='right')
        self.assertEqual(np.asarray(observed).shape, (1, 16))
        self.assertEqual(observed[0][3], .4)

    def test_camera_rejects_profile_failure_without_resolution_fallback(self):
        import threading
        from core.camera_processing import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.lock = threading.Lock()
        camera.config, camera.serial_number = Mock(), 'offline'
        camera.pipeline = SimpleNamespace(start=Mock(side_effect=[RuntimeError('profile rejected'), RuntimeError('fallback attempted'), RuntimeError('fallback attempted')]))
        rs = SimpleNamespace(stream=SimpleNamespace(color=1), format=SimpleNamespace(bgr8=2), config=Mock)
        with patch('core.camera_processing.rs', rs, create=True):
            with self.assertRaisesRegex(RuntimeError, 'profile rejected'):
                RealSenseCamera.initialize_camera(camera, 1280, 720, 30)

    def test_capture_failure_never_reuses_cached_marker_frame(self):
        import threading
        import time
        from core.marker_detection import Marker_Transform
        from core.camera_processing import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.camera_running = True
        camera.camera_monitoring = False
        camera.color_image = np.zeros((2, 2, 3), dtype=np.uint8)
        camera.color_frame_received_at = time.monotonic()
        camera.depth_image = None
        camera.fps = 30
        camera.lock = threading.Lock()
        camera.pipeline = SimpleNamespace(wait_for_frames=Mock(side_effect=RuntimeError('camera disconnected')))
        provider = Marker_Transform(sim=True)
        provider.sim = False
        provider.camera = camera
        provider.set_marker_type('plate')
        pose = np.eye(4)
        pose[0, 3] = 400.
        with patch.object(provider.marker_detection, 'detect', return_value=[('plate_right', pose.ravel())]):
            self.assertIsNone(provider.get_marker_transform(.03, side='right'))
        self.assertIsNone(camera.get_color_image())
        self.assertFalse(provider.sim)

    def test_monitored_camera_expires_old_frame_and_rejects_stopped_camera(self):
        import threading
        import time
        from core.camera_processing import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.camera_running = True
        camera.camera_monitoring = True
        camera.color_image = np.zeros((2, 2, 3), dtype=np.uint8)
        camera.color_frame_received_at = time.monotonic() - 10
        camera.fps = 30
        camera.lock = threading.Lock()
        self.assertIsNone(camera.get_color_image())
        camera.color_frame_received_at = time.monotonic()
        self.assertIsNotNone(camera.get_color_image())
        camera.camera_running = False
        self.assertIsNone(camera.get_color_image())

    def test_reconnect_during_sample_session_preserves_all_consumers(self):
        app = SimpleNamespace(logs=[], shared_arm_q_list=[np.zeros(14)], marker_st=object())
        app.log_msg = app.logs.append
        app.camera_source_busy = MethodType(UnifiedCalibrationApp.camera_source_busy, app)
        previous = app.marker_st
        self.assertFalse(UnifiedCalibrationApp.reconnect_camera(app, show_dialog=False))
        self.assertIs(app.marker_st, previous)

    def test_intrinsics_calibrator_save_reloads_provider(self):
        from core.marker_detection import Marker_Transform
        from core.calibration.IntrinsicsCalibrator import IntrinsicsCalibrator
        provider = Marker_Transform(sim=True)
        calibrator = IntrinsicsCalibrator()
        calibrator.marker_st = provider
        calibrator.cameraMatrix = np.array([[700., 0, 630], [0, 701, 350], [0, 0, 1]])
        calibrator.distCoeffs = np.zeros(5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'camera_intrinsics.yaml'
            path.write_bytes(Path(CONFIG_PATHS['camera_intrinsics']).read_bytes())
            with patch.dict(CONFIG_PATHS, camera_intrinsics=str(path)):
                calibrator._save_results(str(path), 1280, 720)
                self.assertEqual(provider.marker_detection.fx, 700.)
                self.assertEqual(provider.intrinsics_metadata['file'], str(path))

    def test_app_inherits_provider_version_and_rebinds_every_consumer(self):
        from core.marker_detection import Marker_Transform
        provider = Marker_Transform(sim=True, robot_version='1.3')
        app = UnifiedCalibrationApp(provider, None)
        try:
            self.assertEqual(app.get_robot_version(), '1.3')
            for calibrator in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
                self.assertIs(calibrator.marker_st, provider)
                self.assertEqual(calibrator.robot_version, '1.3')
            replacement = Marker_Transform(sim=True, robot_version='1.3')
            with patch('main_ui.Marker_Transform', return_value=replacement):
                self.assertTrue(app.reconnect_camera(show_dialog=False))
            self.assertIs(app.marker_detector, replacement.marker_detection)
            self.assertIs(app.intrinsics_calibrator.marker_st, replacement)
            for calibrator in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
                self.assertIs(calibrator.marker_st, replacement)
        finally:
            app.close()

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
