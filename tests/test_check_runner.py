"""Supported verification CLI safety and current raw sweep replay."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class ReplayConfigurationTests(unittest.TestCase):
    def test_checkpoint_restores_accepted_brackets_through_actual_ui_save(self):
        from copy import deepcopy
        from types import MethodType, SimpleNamespace
        from unittest.mock import patch
        from PySide6.QtWidgets import QApplication, QLineEdit
        import yaml
        import run_connected_calibration as connected
        from main_ui import UnifiedCalibrationApp
        from core.config_store import CONFIG_PATHS
        qt = QApplication.instance() or QApplication([])
        logs = []
        app = SimpleNamespace(log_msg=logs.append, marker_st=None,
            last_full_auto_error=None, last_full_auto_converged=True,
            get_robot_version=lambda: '1.2', update_applied_offset_label=lambda: None,
            joint_offsets={side: {} for side in ('right', 'left')})
        for name in ('joint_calibrator', 'marker_calibrator', 'head_camera_calibrator'):
            setattr(app, name, SimpleNamespace(camera_config={}))
        for short in ('r', 'l'):
            for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
                setattr(app, f'txt_bracket_{short}_{axis}', QLineEdit('0'))
        for name in ('handle_full_auto_bracket_finished', '_joint_offset_patch',
                     '_bracket_patch', '_publish_joint_offsets', '_publish_brackets',
                     'apply_full_auto_results'):
            setattr(app, name, MethodType(getattr(UnifiedCalibrationApp, name), app))
        checkpoint = {'step1': 'passed',
            'joint_offsets': {side: dict(joint3=-.123456, joint5=.654321, joint6=-2.123456)
                              for side in ('right', 'left')},
            'brackets': {'right': [.00054321, -.05412345, -.04612345, 90.123456, .123456, -179.987654],
                         'left': [.00098765, .05456789, -.04967891, 90.654321, -.234567, .345678]}}
        original = deepcopy(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'setting.yaml'
            path.write_text('camera: {intrinsics_source: calibrated}\n')
            with patch.dict(CONFIG_PATHS, setting_yaml=str(path)):
                connected.restore_step1_checkpoint(app, checkpoint)
                self.assertTrue(app.apply_full_auto_results(silent=True), logs)
            saved = yaml.safe_load(path.read_text())
        for side, vector in checkpoint['brackets'].items():
            np.testing.assert_allclose(saved['marker']['Tf_to_marker_' + side], vector, atol=1e-15)
        self.assertEqual(saved['joint_offset'], checkpoint['joint_offsets'])
        self.assertEqual(checkpoint, original)
        self.assertFalse(any('Rejected bracket' in log for log in logs), logs)
        with self.assertRaises(ValueError):
            connected.restore_step1_checkpoint(app, {**checkpoint, 'step1': 'failed'})

    def test_connected_head_worker_uses_ui_values_not_constructor_defaults(self):
        from types import SimpleNamespace
        import run_connected_calibration as connected
        field = lambda value: SimpleNamespace(text=lambda: str(value))
        app = SimpleNamespace(head_camera_calibrator=object(), step1_5_pan_range=field(10.),
                              step1_5_tilt_range=field(8.), step1_5_num_steps=field(11))
        worker = connected.make_head_sweep_worker(app, None)
        self.assertEqual((worker.pan_range, worker.tilt_range, worker.num_steps), (10., 8., 11))

    def test_checkpoint_requires_matching_version_mode_and_production_source(self):
        import json
        import run_connected_calibration as connected
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step1.json'
            saved = {'robot_version': '1.2', 'include_head': True, 'step1': 'passed',
                     'sweep_seconds': 'production_defaults'}
            path.write_text(json.dumps(saved))
            (path.parent / 'source_manifest.json').write_text(json.dumps({'main_ui.py': 'abc', 'core/model.py': '123'}))
            inputs = {'simulation_yaml': 'truth-hash', 'ready_poses_yaml': 'geometry-hash',
                      'camera_intrinsics': 'intrinsic-hash', 'cold_start_settings': {}}
            (path.parent / 'checkpoint_inputs.json').write_text(json.dumps(inputs))
            source = {'main_ui.py': 'abc', 'core/model.py': '123', 'tests/run_connected_calibration.py': 'new-driver'}
            self.assertEqual(connected.load_step1_checkpoint(path, '1.2', True, source, inputs, None), saved)
            for version, head, hashes in [('1.3', True, source), ('1.2', False, source),
                                         ('1.2', True, {**source, 'core/model.py': 'changed'})]:
                with self.assertRaises(ValueError):
                    connected.load_step1_checkpoint(path, version, head, hashes, inputs, None)
            for key in inputs:
                with self.assertRaises(ValueError):
                    connected.load_step1_checkpoint(path, '1.2', True, source, {**inputs, key: 'changed'}, None)
            with self.assertRaises(ValueError):
                connected.load_step1_checkpoint(path, '1.2', True, source, inputs, 2.)

    def test_legacy_checkpoint_context_ignores_only_replaced_estimates(self):
        from copy import deepcopy
        import yaml
        import run_connected_calibration as connected
        with tempfile.TemporaryDirectory() as directory:
            paths = {key: str(Path(directory) / filename) for key, filename in (
                ('simulation_yaml', 'simulation.yaml'), ('ready_poses_yaml', 'ready_poses.yaml'),
                ('camera_intrinsics', 'camera_intrinsics.yaml'), ('setting_yaml', 'setting.yaml'))}
            for path in paths.values():
                Path(path).write_text('{}')
            settings = {'camera': {'mount_to_cam': [0]*6, 'intrinsics_source': 'calibrated'},
                        'marker': {'Tf_to_marker_right': [0]*6, 'Tf_to_marker_right_v12': [1]*6},
                        'joint_offset': {'right': {'joint6': 1}}}
            Path(paths['setting_yaml']).write_text(yaml.safe_dump(settings))
            before = connected.checkpoint_input_context(paths)
            head_disabled_before = connected.checkpoint_input_context(paths, include_head=False)
            fitted = deepcopy(settings)
            fitted['camera']['mount_to_cam'] = [2]*6
            fitted['marker']['Tf_to_marker_right'] = [3]*6
            fitted['joint_offset']['right']['joint6'] = 4
            Path(paths['setting_yaml']).write_text(yaml.safe_dump(fitted))
            self.assertEqual(connected.checkpoint_input_context(paths), before)
            self.assertNotEqual(connected.checkpoint_input_context(paths, include_head=False),
                                head_disabled_before)
            fitted['marker']['Tf_to_marker_right_v12'] = [5]*6
            Path(paths['setting_yaml']).write_text(yaml.safe_dump(fitted))
            self.assertNotEqual(connected.checkpoint_input_context(paths), before)

    def test_explicit_camera_context_needs_no_current_setting_file(self):
        from unittest.mock import patch
        from core.config_store import CONFIG_PATHS, RobotConfig
        from core.calibration_core import get_both_arm_config
        from calibration_support import OfflineRobot
        camera = {'mount_to_cam': [.01, .02, .03, 1., 2., 3.],
                  'head_base_to_cam': [.04, .05, .06, 4., 5., 6.]}
        with patch.dict(CONFIG_PATHS, setting_yaml='/tmp/camera-check-no-such-settings.yaml'):
            result = get_both_arm_config(OfflineRobot().model(), '1.2', camera_config=camera)
        self.assertEqual(result['mount_to_cam_nom'], camera['mount_to_cam'])
        self.assertEqual(result['head_base_to_cam_nom'], camera['head_base_to_cam'])
        self.assertEqual(result['ee_to_marker_nom'], RobotConfig.load().nominal_brackets['1.2'])


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tests' / 'run_calibration_checks.py'


class CheckRunnerTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run([sys.executable, str(RUNNER), *arguments],
                              cwd=ROOT, capture_output=True, text=True, timeout=20)

    def test_help_is_available_without_robot_or_camera(self):
        self.assertTrue(RUNNER.exists(), 'Supported verification entry point is missing')
        result = self.invoke('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ('unit', 'replay', 'connected'):
            self.assertIn(command, result.stdout)

    def test_motion_requires_explicit_simulator_confirmation_and_new_output(self):
        self.assertTrue(RUNNER.exists(), 'Supported verification entry point is missing')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'evidence'
            result = self.invoke('connected', '--output', str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('keep')
            result = self.invoke('connected', '--output', str(output), '--confirm-local-simulator')
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_current_pose_blocks_preserve_iteration_boundaries(self):
        from replay_j6_sweeps import read_blocks
        poses = np.tile(np.eye(4), (12, 1, 1))
        poses[:, 0, 3] = np.linspace(.01, .02, 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sweep.txt'
            with path.open('w') as stream:
                np.savetxt(stream, poses.reshape(-1, 16), header='ordered T_camera_marker; translation metres; no encoder/FK')
                np.savetxt(stream, poses.reshape(-1, 16), header='ordered T_camera_marker; translation metres; no encoder/FK')
            before = path.read_bytes()
            blocks = read_blocks(path)
            self.assertEqual(len(blocks), 2)
            np.testing.assert_array_equal(blocks[0].reshape(-1, 4, 4), poses)
            self.assertEqual(path.read_bytes(), before)

    def test_npz_replay_uses_recorded_estimates_without_touching_user_settings(self):
        import json
        from copy import deepcopy
        from calibration_support import OfflineRobot
        from calibration_support import data
        from core.calibration_core import save_npz_dataset
        from core.config_store import CONFIG_PATHS
        from core.marker_detection import SimulationModel
        robot = OfflineRobot('1.2')
        sim = SimulationModel.create('1.2')
        qa, qh, observations = data(robot, sim, count=24)
        camera = deepcopy(sim.config)
        for side in ('right', 'left'):
            camera['Tf_to_marker_' + side] = camera['brackets']['1.2'][side]
        settings_path = Path(CONFIG_PATHS['setting_yaml'])
        before_settings = settings_path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observations.npz'
            save_npz_dataset(path, qa, observations, qh, metadata={
                'robot_version': '1.2', 'source': 'test_pose_sensor',
                'estimation_camera_snapshot': camera, 'head_motion_enabled': True})
            before = path.read_bytes()
            output = Path(directory) / 'replay'
            result = self.invoke('replay', str(path), '--output', str(output))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads((output / 'optimizer.json').read_text())
            self.assertTrue(saved['diagnostics']['converged'])
            self.assertIn('camera_forward_zero', saved)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(settings_path.read_bytes(), before_settings)


if __name__ == '__main__':
    unittest.main()
