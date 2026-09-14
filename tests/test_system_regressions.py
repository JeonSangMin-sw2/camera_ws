"""Failure-injection coverage for system boundaries; no hardware is connected."""
import importlib
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtWidgets import QApplication
from core.calibration import CalibrationCore
from core.calibration.sequences.step1 import execute_step1_sequence
from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
from core.calibration.CalibratorBase import BaseCalibrator
from core.storage import ConfigStorage, DatasetStorage
from ui.core_bridge import SequenceWorker, MoveToReadyWorker


class TestFailureBoundaries(unittest.TestCase):
    def setUp(self):
        self.core = CalibrationCore()

    def test_joint_nonconvergence_is_failure_with_partial_result(self):
        self.core.robot = MagicMock()
        self.core.observer = SimpleNamespace(last_frame={}, sim=True)
        value = {"converged": False, "recommended_joint_offset": 1.0}
        with patch.object(self.core.joint_calibrator, "perform_joint_calibration", return_value=value):
            result = self.core.run("joint", arm_side="right", mode="wrist_pitch")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.partial["joint_result"], value)
        self.assertNotIn("joint", result.completed)

    def test_step1_nonconvergence_stops_before_marker_sweep(self):
        joint, marker = self.core.joint_calibrator, self.core.marker_calibrator
        joint.robot = MagicMock()
        joint.robot_version = marker.robot_version = "1.2"
        with patch.object(joint, "perform_move_to_ready_pose", return_value=True), patch.object(joint, "perform_joint_calibration", return_value={"converged": False, "recommended_joint_offset": 2.0}), patch.object(marker, "perform_calibration_sweep") as sweep:
            result = execute_step1_sequence(joint, marker, self.core.joint_offsets_store)
        self.assertEqual(result.status, "failed", result.error)
        sweep.assert_not_called()
        self.assertFalse(result.partial["right_wrist_pitch"]["converged"])

    def test_display_callback_failure_cannot_strand_core(self):
        self.core.on_event = MagicMock(side_effect=RuntimeError("display destroyed"))
        with self.assertLogs('core.calibration.calibration_core', level='ERROR'):
            result = self.core.run("unknown")
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.core._contexts, [])
        self.assertFalse(self.core.is_busy)
        self.core.on_event = None
        self.assertEqual(self.core.run("unknown").status, "failed")

    def test_close_releases_unstarted_worker_reservation(self):
        worker = SequenceWorker(self.core, "step1")
        self.core.close()
        self.assertFalse(self.core.is_busy)
        worker.run()
        self.assertEqual(worker.result.status, "failed")
        self.assertFalse(self.core.is_busy)

    def test_close_disconnects_robot_and_rejects_new_commands(self):
        robot = MagicMock()
        self.core.robot = robot
        self.core.close()
        self.core.close()
        robot.disconnect.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.core.run('step1')
        self.assertFalse(self.core.is_busy)

    def test_failed_camera_reconnect_clears_calibrator_references(self):
        previous = MagicMock()
        self.core.observer = previous
        with patch("core.marker_detection.Marker_Transform", side_effect=RuntimeError("camera init failed")):
            with self.assertRaisesRegex(RuntimeError, "camera init failed"):
                self.core.connect_camera()
        previous.close.assert_called_once()
        self.assertIsNone(self.core.observer)
        self.assertTrue(all(c.marker_st is None for c in self.core.calibrators))
        self.assertFalse(self.core.get_monitor_snapshot()["connected"])

    def test_step2_without_head_samples_accepts_headless_robot(self):
        # The UI may retain include_head_motion=True on a headless robot. No
        # object-dtype array of None values should be forwarded to the solver.
        self.core.model = SimpleNamespace(head_idx=[])
        samples = [{"q_arm": np.zeros(14), "q_head": None, "marker": np.tile(np.eye(4), (2, 1, 1))}]
        with patch("core.calibration.sequences.step2.optimize_step2", return_value={}) as solve:
            result = self.core.run("step2", samples=samples)
        self.assertTrue(result.success, result.error)
        self.assertIsNone(solve.call_args.kwargs["q_head_list"])

    def test_home_offset_sdk_failure_retains_changed_and_uncertain_joints(self):
        from core.robot import home_offset
        robot = MagicMock()
        robot.home_offset_reset.side_effect = [True, RuntimeError('connection lost')]
        model = SimpleNamespace(right_arm_idx=[0, 1, 2], left_arm_idx=[], head_idx=[])
        with patch.object(home_offset, 'validate_home_offset_joint_limits', return_value=(True, '')), patch.object(home_offset, 'wait_motion'):
            with self.assertRaises(home_offset.HomeOffsetError) as caught:
                home_offset.reset_current_pose_home_offsets(robot, model, arm='right', include_head=False)
        progress = caught.exception.partial['home']
        self.assertEqual(progress['reset_joints'], ['right_arm_0'])
        self.assertEqual(progress['uncertain_joints'], ['right_arm_1'])
        self.assertEqual(robot.home_offset_reset.call_count, 2)
        robot.power_off.assert_not_called()


class TestHeadQualityGate(unittest.TestCase):
    def test_failed_solver_cannot_be_accepted_or_applied(self):
        solver = HeadCameraCalibrator()
        solver.robot = SimpleNamespace(model=lambda: SimpleNamespace(head_idx=[0, 1]),
                                       get_state=lambda: SimpleNamespace(position=np.zeros(2)),
                                       get_dynamics=lambda: None)
        angles = np.linspace(-10, 10, 5)
        points = np.array([[.2, .01 * x, .3 + .001 * x * x] for x in range(5)])
        solution = SimpleNamespace(x=np.r_[np.zeros(5), [.2, 0, .3]], success=False)
        module = importlib.import_module("core.calibration.HeadCameraCalibrator")
        with patch.object(module, "least_squares", return_value=solution), patch.object(BaseCalibrator, "compute_fk", return_value=np.eye(4)):
            result = solver._compute_head_camera_solution(points, points, angles, angles, [0.] * 6, np.eye(3))
        self.assertFalse(result["success"])
        self.assertIsNone(solver.calibrated_results)
        self.assertFalse(solver.apply_calibration_results(result))


class TestStorageFailures(unittest.TestCase):
    def test_invalid_measurements_are_rejected_before_solver_construction(self):
        core = CalibrationCore()
        core.model = SimpleNamespace(right_arm_idx=list(range(7)), left_arm_idx=list(range(7, 14)), head_idx=[])
        core.include_head_motion = False
        sample = {"q_arm": np.full(14, np.nan), "q_head": None, "marker": np.tile(np.eye(4), (2, 1, 1))}
        with patch('core.calibration.sequences.step2.QPCalibrationOptimizer') as solver:
            result = core.run('step2', samples=[sample])
        self.assertEqual(result.status, 'failed')
        self.assertIn('finite', result.error)
        solver.assert_not_called()

    def test_npz_encoder_failure_keeps_prior_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.npz'
            DatasetStorage.save(path, q=np.arange(5))
            with patch('core.storage.np.savez_compressed', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    DatasetStorage.save(path, q=np.arange(9))
            np.testing.assert_array_equal(DatasetStorage.load(path)['q'], np.arange(5))
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_yaml_updates_do_not_cross_commented_section_boundary(self):
        for updater, section in ((ConfigStorage.update_camera_key_in_lines, 'camera'), (ConfigStorage.update_marker_key_in_lines, 'marker')):
            lines = [f'{section}:\n', '  unrelated: 1\n', 'other: # section comment\n', '  target: [9, 9, 9, 9, 9, 9]\n']
            updater(lines, 'target', [1.] * 6)
            self.assertEqual(lines[-1], '  target: [9, 9, 9, 9, 9, 9]\n')
            self.assertIn('target:', lines[1])


class TestUiFailurePropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_ready_motion_failure_does_not_unlock_sweep(self):
        from main_ui import UnifiedCalibrationApp
        window = UnifiedCalibrationApp(ui_only=True)
        for kind in ('joint', 'marker'):
            window.on_move_ready_joint_finished(False, 'cancelled') if kind == 'joint' else window.on_move_ready_marker_finished(False, 'cancelled')
            self.assertFalse(getattr(window, 'ready_done_' + kind))
        worker = MoveToReadyWorker(window.core, "right")
        received = []
        worker.finished_signal.connect(lambda success, error: received.append(success))
        worker.run()
        self.assertEqual(received, [False])
        window.close()

    def test_core_baseline_path_is_not_lost_at_next_ui_sync(self):
        from main_ui import UnifiedCalibrationApp
        from core.calibration import SequenceResult
        window = UnifiedCalibrationApp(ui_only=True)
        window.core.last_home_reset_path = 'baseline-test.json'
        window._on_core_result(SequenceResult('home').finish('completed'))
        window._sync_core_state()
        self.assertEqual(window.core.last_home_reset_path, 'baseline-test.json')
        window.close()

    def test_apply_is_rejected_while_core_is_running(self):
        from main_ui import UnifiedCalibrationApp
        window = UnifiedCalibrationApp(ui_only=True)
        window.core.prepare_run()
        with patch.object(window.head_camera_calibrator, 'apply_calibration_results') as apply:
            window.apply_results_step1_5()
        apply.assert_not_called()
        window.core.cancel()
        window.core.run('step1', prepared=True)
        window.close()

    def test_worker_lifetime_is_retained_until_qt_thread_finishes(self):
        core = CalibrationCore()
        worker = SequenceWorker(core, 'unknown')
        worker.start()
        self.assertIn(worker, SequenceWorker.active_for_core(core))
        self.assertTrue(worker.wait(2000))
        self.app.processEvents()
        self.assertNotIn(worker, SequenceWorker.active_for_core(core))
        core.close()

    def test_main_entrypoint_starts_and_exits_without_hardware_from_other_cwd(self):
        root = str(Path(__file__).resolve().parents[1])
        source = '''
import sys
sys.path.insert(0, sys.argv[1])
sys.argv = ['main_ui.py', '--ui']
from unittest.mock import patch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import main_ui
original_exec = QApplication.exec
def timed_exec():
    QTimer.singleShot(150, QApplication.instance().quit)
    return original_exec()
with patch.object(QApplication, 'exec', side_effect=timed_exec), patch('rby1_sdk.create_robot', side_effect=AssertionError('hardware forbidden')):
    try:
        main_ui.main()
    except SystemExit as exit_status:
        assert exit_status.code == 0, exit_status.code
print('ENTRYPOINT_OK')
'''
        with tempfile.TemporaryDirectory() as folder:
            process = subprocess.run([sys.executable, '-c', source, root], cwd=folder,
                                     env=dict(os.environ, QT_QPA_PLATFORM='offscreen'),
                                     capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('ENTRYPOINT_OK', process.stdout)


class TestCameraIdentity(unittest.TestCase):
    def test_unknown_serial_does_not_reset_another_camera(self):
        from core.camera_processing import RealSenseCamera, CameraUnavailableError
        device = MagicMock()
        device.get_info.return_value = 'attached-device'
        with patch('core.camera_processing.rs.context') as context:
            context.return_value.query_devices.return_value = [device]
            with self.assertRaises(CameraUnavailableError):
                RealSenseCamera(serial_number='missing-device')
        device.hardware_reset.assert_not_called()

    def test_reset_reenumeration_keeps_selected_camera(self):
        import core.camera_processing as camera_module
        def device(serial):
            value = MagicMock()
            value.get_info.side_effect = lambda key: serial if key == camera_module.rs.camera_info.serial_number else 'test-camera'
            value.first_depth_sensor.return_value.supports.return_value = False
            value.first_depth_sensor.return_value.get_depth_scale.return_value = .001
            return value
        first, second = device('first'), device('second')
        with patch.object(camera_module.rs, 'context') as context, patch.object(camera_module.time, 'sleep'), patch.object(camera_module.rs, 'pipeline'), patch.object(camera_module.rs, 'config'), patch.object(camera_module.rs, 'spatial_filter'), patch.object(camera_module.rs, 'temporal_filter'), patch.object(camera_module.rs, 'hole_filling_filter'):
            context.return_value.query_devices.side_effect = [[first, second], [second, first]]
            camera = camera_module.RealSenseCamera(serial_number='first')
        self.assertEqual(camera.serial_number, 'first')
        self.assertEqual(camera.device_number, 1)
        first.hardware_reset.assert_called_once()
        second.hardware_reset.assert_not_called()
