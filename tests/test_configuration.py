"""Configuration ownership and isolation; no camera or robot connections."""
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import yaml

from core import config_store


class ConfigurationTests(unittest.TestCase):
    def test_base_initialization_does_not_report_success_after_servo_failure(self):
        from unittest.mock import MagicMock, patch
        import rby1_sdk as rby
        from core.calibration.CalibratorBase import BaseCalibrator
        robot = MagicMock()
        robot.get_robot_info.return_value.robot_model_name = 'm'
        robot.get_control_manager_state.return_value.state = rby.ControlManagerState.State.Enabled
        robot.is_servo_on.return_value = False
        robot.servo_on.return_value = False
        with patch('core.calibration.CalibratorBase.rby.create_robot', return_value=robot), patch('time.sleep'):
            self.assertIsNone(BaseCalibrator.initialize_robot('127.0.0.1:50051', 'm', include_head=False))

    def test_separate_camera_device_never_returns_a_stale_frame(self):
        import importlib.util
        import threading
        import time
        import numpy as np
        self.assertIsNotNone(importlib.util.find_spec('core.camera_processing'))
        from core.camera_processing import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.camera_running = True
        camera.color_frame_received_at = time.monotonic() - 10
        camera.fps = 30
        camera.color_image = np.ones((2, 2, 3), dtype=np.uint8)
        camera.lock = threading.Lock()
        self.assertIsNone(camera.get_color_image())
        camera.color_frame_received_at = time.monotonic()
        self.assertIsNotNone(camera.get_color_image())
        camera.camera_running = False
        self.assertIsNone(camera.get_color_image())

    def test_camera_protocol_declares_monitoring_state_used_by_marker_consumers(self):
        from core.camera_processing import CameraDevice
        self.assertIn('camera_monitoring', CameraDevice.__annotations__)

    def test_camera_initialization_discards_previous_session_frames(self):
        from core.camera_processing import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.lock = threading.Lock()
        camera.camera_running = False
        camera.camera_monitoring = False
        camera.fps = 30
        camera.serial_number = 'offline'
        camera.color_image = np.ones((1, 1, 3), dtype=np.uint8)
        camera.depth_image = np.ones((1, 1), dtype=np.uint16)
        camera.left_ir_image = np.ones((1, 1), dtype=np.uint8)
        camera.right_ir_image = np.ones((1, 1), dtype=np.uint8)
        camera.color_frame_received_at = time.monotonic()
        camera.depth_frame_received_at = time.monotonic()
        camera.infrared_frame_received_at = time.monotonic()
        camera.actual_exposure = 6000.0
        camera.baseline = .065
        camera.config = Mock()
        intrinsics = SimpleNamespace(fx=1., fy=1., ppx=.5, ppy=.5, coeffs=[0.] * 5)
        video_profile = SimpleNamespace(get_intrinsics=lambda: intrinsics)
        stream_profile = SimpleNamespace(as_video_stream_profile=lambda: video_profile)
        profile = SimpleNamespace(
            get_device=lambda: SimpleNamespace(query_sensors=lambda: []),
            get_stream=lambda stream: stream_profile,
        )
        camera.pipeline = SimpleNamespace(
            start=lambda config: profile,
            wait_for_frames=lambda timeout_ms: None,
        )
        fake_rs = SimpleNamespace(
            stream=SimpleNamespace(color=1),
            format=SimpleNamespace(bgr8=2),
            option=SimpleNamespace(enable_auto_exposure=3),
        )
        with patch('core.camera_processing.rs', fake_rs, create=True):
            camera.initialize_camera(1280, 720, 30)
        self.assertTrue(camera.camera_running)
        self.assertIsNone(camera.color_image)
        self.assertIsNone(camera.depth_image)
        self.assertIsNone(camera.left_ir_image)
        self.assertIsNone(camera.right_ir_image)

    def test_camera_stop_invalidates_all_cached_frames(self):
        from core.camera_processing import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.lock = threading.Lock()
        camera.thread = None
        camera.pipeline = SimpleNamespace(stop=Mock())
        camera.camera_running = True
        camera.camera_monitoring = True
        camera.fps = 30
        camera.color_image = np.ones((1, 1, 3), dtype=np.uint8)
        camera.depth_image = np.ones((1, 1), dtype=np.uint16)
        camera.left_ir_image = np.ones((1, 1), dtype=np.uint8)
        camera.right_ir_image = np.ones((1, 1), dtype=np.uint8)
        now = time.monotonic()
        camera.color_frame_received_at = now
        camera.depth_frame_received_at = now
        camera.infrared_frame_received_at = now
        camera.stream_off()
        self.assertFalse(camera.camera_running)
        self.assertFalse(camera.camera_monitoring)
        self.assertIsNone(camera.get_color_image())
        self.assertIsNone(camera.get_depth_image())
        self.assertEqual(camera.get_infrared_images(), (None, None))

    def test_camera_depth_and_infrared_getters_require_fresh_running_frames(self):
        from core.camera_processing import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.lock = threading.Lock()
        camera.camera_running = True
        camera.fps = 30
        camera.depth_image = np.ones((1, 1), dtype=np.uint16)
        camera.left_ir_image = np.ones((1, 1), dtype=np.uint8)
        camera.right_ir_image = np.ones((1, 1), dtype=np.uint8)
        old = time.monotonic() - 10
        camera.depth_frame_received_at = old
        camera.infrared_frame_received_at = old
        self.assertIsNone(camera.get_depth_image())
        self.assertEqual(camera.get_infrared_images(), (None, None))
        now = time.monotonic()
        camera.depth_frame_received_at = now
        camera.infrared_frame_received_at = now
        self.assertIsNotNone(camera.get_depth_image())
        left, right = camera.get_infrared_images()
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)

    def test_paths_are_independent_of_cwd_and_do_not_overwrite_external_config(self):
        self.assertTrue(hasattr(config_store, 'Paths'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / 'bundle'
            (bundle / 'config').mkdir(parents=True)
            (bundle / 'config' / 'setting.yaml').write_text('camera: {}\n')
            paths = config_store.Paths(root / 'installed', bundle=bundle)
            paths.prepare_templates()
            setting = Path(paths.config['setting_yaml'])
            setting.write_text('camera: {custom: true}\n')
            paths.prepare_templates()
            self.assertEqual(setting.read_text(), 'camera: {custom: true}\n')
            self.assertTrue(Path(paths.config['ready_poses_yaml']).is_absolute())

    def test_relative_bundle_override_is_resolved_when_paths_are_created(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / 'bundle'
            elsewhere = root / 'elsewhere'
            (bundle / 'config').mkdir(parents=True)
            elsewhere.mkdir()
            (bundle / 'config' / 'setting.yaml').write_text('camera: {}\n')
            try:
                os.chdir(root)
                paths = config_store.Paths(root / 'installed', bundle=Path('bundle'))
                os.chdir(elsewhere)
                paths.prepare_templates()
            finally:
                os.chdir(original_cwd)
            self.assertEqual(Path(paths.config['setting_yaml']).read_text(), 'camera: {}\n')

    def test_asset_paths_use_bundle_without_relocating_editable_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            installed = base / 'installed'
            bundle = base / 'bundle'
            paths = config_store.Paths(installed, bundle=bundle)
            self.assertEqual(paths.asset('img/help.png'), str((bundle / 'img/help.png').resolve()))
            self.assertEqual(paths.config['setting_yaml'], str((installed / 'config/setting.yaml').resolve()))
        source_root = Path(config_store.__file__).resolve().parents[1]
        self.assertEqual(config_store.get_asset_path('img/help.png'),
                         str((source_root / 'img/help.png').resolve()))

    def test_robot_settings_control_runtime_and_remain_instance_local(self):
        self.assertTrue(hasattr(config_store, 'RobotConfig'))
        from core.calibration.CalibratorBase import BaseCalibrator
        with tempfile.TemporaryDirectory() as directory:
            original = config_store.RobotConfig.load()
            document = deepcopy(original.document)
            document['calibration']['joint_sweep_seconds']['wrist_yaw2'] = 23.5
            path = Path(directory) / 'ready.yaml'
            path.write_text(yaml.safe_dump(document))
            loaded = config_store.RobotConfig.load(path)
            self.assertEqual(loaded.joint_sweep_seconds['wrist_yaw2'], 23.5)
            loaded.nominal_brackets['1.2']['left'][0] = .9
            self.assertEqual(config_store.RobotConfig.load(path).nominal_brackets['1.2']['left'][0], 0.)
            from unittest.mock import patch
            with patch.dict(config_store.CONFIG_PATHS, ready_poses_yaml=str(path)):
                runtime = BaseCalibrator()
                self.assertEqual(runtime.JOINT_SWEEP_SECONDS['wrist_yaw2'], 23.5)
                self.assertEqual(runtime.NOMINAL_BRACKET_TEMPLATES['1.2']['left'][0], 0.)

    def test_invalid_motor_config_is_rejected_before_motion(self):
        self.assertTrue(hasattr(config_store, 'RobotConfig'))
        for field, value in [('cand_joint', 7), ('cand_joint', -1), ('sweep_range_A', float('nan'))]:
            with self.subTest(field=field):
                document = deepcopy(config_store.RobotConfig.load().document)
                document['calibration']['joint_configs']['wrist_yaw2'][field] = value
                with self.assertRaises(ValueError):
                    config_store.RobotConfig(document)
        document = deepcopy(config_store.RobotConfig.load().document)
        document['calibration']['joint_sweep_seconds']['elbow'] = 0
        with self.assertRaises(ValueError):
            config_store.RobotConfig(document)

    def test_robot_config_rejects_wrong_offset_key_for_mode(self):
        document = deepcopy(config_store.RobotConfig.load().document)
        document['calibration']['joint_configs']['wrist_yaw2']['offset_key'] = 'elbow'
        with self.assertRaises(ValueError):
            config_store.RobotConfig(document)

    def test_robot_config_requires_ready_pose_modes_for_each_version(self):
        for version, mode in [('v1.2', 'wrist_yaw2'), ('v1.3', 'wrist_roll')]:
            with self.subTest(version=version, mode=mode):
                document = deepcopy(config_store.RobotConfig.load().document)
                del document[version]['joint'][mode]
                with self.assertRaises(ValueError):
                    config_store.RobotConfig(document)

    def test_robot_config_requires_finite_unit_nominal_axes(self):
        invalid_vectors = ([0., 0., 0.], [0., 0., 2.], [0., float('nan'), 1.], [0., 1.])
        for vector in invalid_vectors:
            with self.subTest(vector=vector):
                document = deepcopy(config_store.RobotConfig.load().document)
                document['calibration']['marker_configs']['axis_4']['n_nom_v12'] = vector
                with self.assertRaises(ValueError):
                    config_store.RobotConfig(document)

    def test_joint_calibration_forwards_instance_sweep_duration(self):
        from core.calibration.JointCalibrator import JointCalibrator
        calibrator = JointCalibrator.__new__(JointCalibrator)
        calibrator.JOINT_SWEEP_SECONDS = {'wrist_yaw2': 23.5}
        calibrator.JOINT_CONFIGS = {'wrist_yaw2': {
            'offset_key': 'wrist_yaw2', 'offset_range': [-30., 30.],
            'sweep_joint_A': 6, 'sweep_joint_B': 5,
        }}
        calibrator.joint_offsets = {'right': {'wrist_yaw2': 0.}}
        calibrator.stop_requested = False
        calibrator.robot = None
        seen = []

        def reject_measurement(*args, **kwargs):
            seen.append(kwargs['sweep_duration'])
            return {'measurement_accepted': False, 'failure_reason': 'test stop'}

        calibrator.perform_calibration_sweep_continuous = reject_measurement
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                config_store.CONFIG_PATHS, txt_dir=directory):
            calibrator.perform_joint_calibration('right', 'wrist_yaw2', pass_idx=2)
        self.assertEqual(seen, [23.5])

    def test_j6_reference_uses_robot_config_nominal_not_saved_estimate(self):
        from core.calibration.JointCalibrator import JointCalibrator
        calibrator = JointCalibrator.__new__(JointCalibrator)
        calibrator.robot_version = '1.2'
        calibrator.robot_parameters = SimpleNamespace(nominal_brackets={
            '1.2': {'right': [0., 0., 0., 0., 0., 0.]},
        })
        calibrator.camera_config = {
            'Tf_to_marker_right_v12': [0., 0., 0., 90., 0., 0.],
        }
        circles = iter([
            {'axis': np.array([1., 0., 0.]), 'residual_rms_m': 0., 'center_m': np.zeros(3),
             'radius': 1., 'c_opt': np.zeros(3), 'rmse': 0.},
            {'axis': np.array([0., 1., 0.]), 'residual_rms_m': 0., 'center_m': np.zeros(3),
             'radius': 1., 'c_opt': np.zeros(3), 'rmse': 0.},
        ])
        calibrator.fit_observed_circle = lambda data: next(circles)
        captured = []

        def estimate(poses_a, axis_a, poses_b, axis_b, reference):
            captured.append(reference)
            return {'measurement_accepted': True, 'optimal_offset': 0.}

        with patch('core.calibration.JointCalibrator.estimate_j6_reference', side_effect=estimate):
            result = calibrator.compute_calibration_results(
                'right', 'wrist_yaw2', np.zeros((1, 4, 4)), np.zeros((1, 4, 4)))
        self.assertTrue(result['measurement_accepted'])
        np.testing.assert_allclose(captured[0], np.eye(3), atol=1e-12)

    @staticmethod
    def _marker_with_observed_axes(tool_lengths=None):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        calibrator = MarkerCalibrator.__new__(MarkerCalibrator)
        calibrator.robot_version = '1.2'
        if tool_lengths is not None:
            calibrator.robot_parameters = SimpleNamespace(tool_lengths=tool_lengths,
                nominal_brackets=calibrator.NOMINAL_BRACKET_TEMPLATES)
        calibrator.fit_observed_circle = lambda poses, direction: {}
        calibrator.refine_adjacent_circles = lambda datasets, circles, directions: circles
        calibrator.fit_common_wrist_pivot = lambda datasets: (np.zeros(3), 0.)
        directions = {
            'axis_4': np.array([0., 0., 1.]),
            'axis_5': np.array([0., 1., 0.]),
            'axis_6': np.array([0., 0., 1.]),
        }
        calibrator.observed_axis_line = lambda data: (
            directions[data['name']], np.zeros(3),
            np.eye(3) - np.outer(directions[data['name']], directions[data['name']]),
        )
        return calibrator

    def test_marker_fit_uses_instance_robot_config_tool_length(self):
        calibrator = self._marker_with_observed_axes({'1.2': .2, '1.3': .3})
        datasets = [
            {'name': 'axis_4', 'captured_poses': np.zeros((1, 4, 4)), 'sweep_direction': 1},
            {'name': 'axis_5', 'captured_poses': np.zeros((1, 4, 4)), 'sweep_direction': 1,
             'commanded_reference_j6_deg': 0.},
            {'name': 'axis_6', 'captured_poses': np.zeros((1, 4, 4)), 'sweep_direction': 1},
        ]
        result = calibrator.fit_observed_bracket(*datasets, 'right')
        self.assertTrue(result['measurement_accepted'], result.get('failure_reason'))
        self.assertAlmostEqual(result['z_e'], 200.)

    def test_marker_fit_has_default_tool_length_for_constructor_free_geometry_tests(self):
        calibrator = self._marker_with_observed_axes()
        datasets = [
            {'name': 'axis_4', 'captured_poses': np.zeros((1, 4, 4)), 'sweep_direction': 1},
            {'name': 'axis_5', 'captured_poses': np.zeros((1, 4, 4)), 'sweep_direction': 1,
             'commanded_reference_j6_deg': 0.},
            {'name': 'axis_6', 'captured_poses': np.zeros((1, 4, 4)), 'sweep_direction': 1},
        ]
        result = calibrator.fit_observed_bracket(*datasets, 'right')
        self.assertTrue(result['measurement_accepted'], result.get('failure_reason'))
        self.assertAlmostEqual(result['z_e'], 126.1)

    def test_nominal_design_does_not_mutate_saved_estimates_or_other_calibrators(self):
        self.assertTrue(hasattr(config_store, 'RobotConfig'))
        from core.calibration.CalibratorBase import BaseCalibrator
        setting = Path(config_store.CONFIG_PATHS['setting_yaml'])
        before = setting.read_bytes()
        first, second = BaseCalibrator(), BaseCalibrator()
        first.NOMINAL_BRACKET_TEMPLATES['1.2']['left'][0] = 10.
        self.assertEqual(second.NOMINAL_BRACKET_TEMPLATES['1.2']['left'][0], 0.)
        self.assertEqual(setting.read_bytes(), before)

    def test_language_notifications_and_fallback_without_qt(self):
        self.assertTrue(hasattr(config_store, 'Language'))
        language = config_store.Language()
        language.translations = {'title': {'ko': '보정', 'en': 'Calibration'}}
        events = []
        language.subscribe(events.append)
        language.set_language('korean')
        language.set_language('ko')
        self.assertEqual(events, ['ko'])
        self.assertEqual(language.get('title'), '보정')
        self.assertEqual(language.get('missing', default='fallback'), 'fallback')
        language.unsubscribe(events.append)
        language.set_language('en')
        self.assertEqual(events, ['ko'])
        command = "import sys; sys.modules['PySide6'] = None; from core.config_store import Language, RobotConfig; assert RobotConfig.load().joint_sweep_seconds['elbow'] > 0; Language()"
        result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_language_does_not_retain_bound_listener_owner(self):
        language = config_store.Language()
        events = []

        class Listener:
            def changed(self, code):
                events.append(code)

        listener = Listener()
        listener_ref = weakref.ref(listener)
        language.subscribe(listener.changed)
        del listener
        gc.collect()
        self.assertIsNone(listener_ref())
        language.set_language('ko')
        self.assertEqual(events, [])
        self.assertEqual(language._listeners, [])


if __name__ == '__main__':
    unittest.main()
