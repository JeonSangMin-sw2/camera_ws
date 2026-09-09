"""Domain services with offline providers; no hardware power, servo or motion."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from core import robot_motion
from core import calibration_core
import re
from unittest.mock import MagicMock
import rby1_sdk as rby
import tempfile
import json
from pathlib import Path


class SweepLogRoutingTests(unittest.TestCase):
    def test_detail_is_saved_not_displayed_while_progress_remains_visible(self):
        from PySide6.QtWidgets import QApplication, QTextEdit
        from main_ui import UnifiedCalibrationApp
        from core.config_store import CONFIG_PATHS
        qt = QApplication.instance() or QApplication([])
        view = QTextEdit()
        app = SimpleNamespace(log_text=view)
        detail = '[SWEEP COMMAND] right J6: -20 -> 20\ncenter_deg=[0 1 2 3 4 5 6]'
        with tempfile.TemporaryDirectory() as directory, patch.dict(CONFIG_PATHS, txt_dir=directory):
            UnifiedCalibrationApp._log_msg_slot(app, detail)
            UnifiedCalibrationApp._log_msg_slot(app, '[INFO] Sweeping J6...')
            UnifiedCalibrationApp._log_msg_slot(app, '[SWEEP QUALITY] accepted')
            self.assertNotIn('[SWEEP COMMAND]', view.toPlainText())
            self.assertIn('[INFO] Sweeping J6...', view.toPlainText())
            self.assertIn('[SWEEP QUALITY] accepted', view.toPlainText())
            self.assertEqual((Path(directory)/'sweep_commands.log').read_text(), detail+'\n')

    def test_detail_write_failure_is_visible_without_leaking_command(self):
        from PySide6.QtWidgets import QApplication, QTextEdit
        from main_ui import UnifiedCalibrationApp
        from core.config_store import CONFIG_PATHS
        qt = QApplication.instance() or QApplication([])
        view = QTextEdit()
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory)/'not_a_directory'
            blocker.write_text('keep')
            with patch.dict(CONFIG_PATHS, txt_dir=str(blocker)):
                UnifiedCalibrationApp._log_msg_slot(SimpleNamespace(log_text=view), '[SWEEP COMMAND] private detail')
            self.assertIn('[WARNING]', view.toPlainText())
            self.assertNotIn('private detail', view.toPlainText())
            self.assertEqual(blocker.read_text(), 'keep')


class ControlManagerRobotDouble:
    """Replicate SDK enable's early return while already enabled."""
    def __init__(self, unlimited, failure=None, expose_mode=True):
        self.enabled = True
        self.unlimited = unlimited
        self.failure = failure
        self.expose_mode = expose_mode
        self.operations = []
        self.servo_enabled = True

    def connect(self): return True
    def disconnect(self): self.operations.append('disconnect')
    def get_robot_info(self): return SimpleNamespace(robot_model_name='m')
    def is_power_on(self, pattern): return True
    def is_servo_on(self, pattern): return self.servo_enabled

    def servo_on(self, pattern):
        self.operations.append('servo_on')
        if self.enabled:
            raise RuntimeError('Control manager must be disabled before servo on')
        self.servo_enabled = True
        return True

    def get_control_manager_state(self):
        state = SimpleNamespace(state=(rby.ControlManagerState.State.Enabled if self.enabled
                                       else rby.ControlManagerState.State.Idle))
        if self.expose_mode:
            state.unlimited_mode_enabled = self.unlimited
        return state

    def disable_control_manager(self):
        self.operations.append('disable')
        if self.failure == 'disable':
            return False
        self.enabled = False
        return True

    def enable_control_manager(self, *, unlimited_mode_enabled):
        self.operations.append(('enable', unlimited_mode_enabled))
        if self.enabled:
            return True
        if self.failure == 'enable':
            return False
        self.enabled = True
        if self.failure != 'mode_not_applied':
            self.unlimited = unlimited_mode_enabled
        return True


class MotionServiceTests(unittest.TestCase):
    def test_servo_off_initialization_uses_available_sdk_state_api(self):
        for enabled in (True, False):
            with self.subTest(manager_enabled=enabled):
                robot = ControlManagerRobotDouble(False)
                robot.enabled = enabled
                robot.servo_enabled = False
                with patch('core.robot_motion.rby.create_robot', return_value=robot), patch('core.robot_motion.time.sleep'):
                    result = robot_motion.initialize_robot_connection('offline', 'm')
                self.assertIs(result, robot)
                self.assertTrue(robot.servo_enabled)
                self.assertTrue(robot.enabled)
                expected = (['disable'] if enabled else []) + ['servo_on', ('enable', False)]
                self.assertEqual(robot.operations, expected)

    def test_servo_off_disable_failure_disconnects_before_enabling_servo(self):
        robot = ControlManagerRobotDouble(False, failure='disable')
        robot.servo_enabled = False
        with patch('core.robot_motion.rby.create_robot', return_value=robot), patch('core.robot_motion.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Control manager disable failed'):
                robot_motion.initialize_robot_connection('offline', 'm')
        self.assertFalse(robot.servo_enabled)
        self.assertEqual(robot.operations, ['disable', 'disconnect'])

    def test_enabled_manager_switches_unlimited_mode_in_both_directions(self):
        for before, requested in ((False, True), (True, False)):
            with self.subTest(before=before, requested=requested):
                robot = ControlManagerRobotDouble(before)
                with patch('core.robot_motion.rby.create_robot', return_value=robot), patch('core.robot_motion.time.sleep'):
                    result = robot_motion.initialize_robot_connection('offline', 'm', unlimited_mode_enabled=requested)
                self.assertIs(result, robot)
                self.assertEqual(robot.unlimited, requested)
                self.assertEqual(robot.operations, ['disable', ('enable', requested)])

    def test_unavailable_mode_uses_checked_disable_and_reenable(self):
        robot = ControlManagerRobotDouble(True, expose_mode=False)
        with patch('core.robot_motion.rby.create_robot', return_value=robot), patch('core.robot_motion.time.sleep'):
            robot_motion.initialize_robot_connection('offline', 'm', unlimited_mode_enabled=False)
        self.assertFalse(robot.unlimited)
        self.assertEqual(robot.operations, ['disable', ('enable', False)])

    def test_failed_mode_switch_cannot_return_success(self):
        for failure, message in (('disable', 'Control manager disable failed'),
                                 ('enable', 'Control manager enable failed'),
                                 ('mode_not_applied', 'Control manager mode mismatch')):
            with self.subTest(failure=failure):
                robot = ControlManagerRobotDouble(True, failure=failure)
                with patch('core.robot_motion.rby.create_robot', return_value=robot), patch('core.robot_motion.time.sleep'):
                    with self.assertRaisesRegex(RuntimeError, message):
                        robot_motion.initialize_robot_connection('offline', 'm', unlimited_mode_enabled=False)
                self.assertEqual(robot.operations[-1], 'disconnect')
                if failure == 'disable':
                    self.assertEqual(robot.operations, ['disable', 'disconnect'])

    def test_dataset_selects_matching_marker_for_single_arm_recording(self):
        right, left = np.eye(4), np.eye(4)
        right[0, 3], left[0, 3] = .4, -.4
        qa, qh, markers, arms = calibration_core.select_calibration_dataset(
            np.zeros((2, 7)), None, np.array([[right, left], [right, left]]), ['right', 'left'])
        self.assertEqual(arms, ['right'])
        self.assertEqual(markers.shape, (2, 4, 4))
        np.testing.assert_allclose(markers[:, 0, 3], [.4, .4])

    def test_offline_sample_estimate_keeps_existing_33_pose_rule(self):
        count, start, approximate = robot_motion.estimate_collection_samples(
            None, None, robot_motion.AutoCollectionConfig(max_x=.36, step_x_m=.02), False)
        self.assertEqual(count, 132)
        self.assertEqual(start, .3)
        self.assertTrue(approximate)

    def test_home_comparison_uses_baseline_minus_optimized_without_rounding(self):
        from core import homeoffset_core
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / 'result.json'
            baseline = Path(folder) / 'baseline.json'
            result.write_text(json.dumps({'right_arm_joint_offset_deg': [1.123456789]}))
            baseline.write_text(json.dumps({'right_arm_joint_offset_deg': [2.]}))
            compared = homeoffset_core.compare_home_offset_files(result, baseline)
        self.assertAlmostEqual(compared['right']['difference'][0], .876543211, places=12)
        self.assertIsNone(compared['head']['difference'])

    def test_capture_metadata_is_an_independent_snapshot(self):
        camera = {'mount_to_cam': [1., 2., 3., 0., 0., 0.]}
        metadata = calibration_core.build_capture_metadata(camera, '1.3', False,
            source_metadata={'source': 'simulation', 'truth': {'unchanged': True}})
        camera['mount_to_cam'][0] = 9.
        self.assertEqual(metadata['estimation_camera_snapshot']['mount_to_cam'][0], 1.)
        self.assertEqual(metadata['truth'], {'unchanged': True})
        self.assertFalse(metadata['head_motion_enabled'])

    def test_marker_service_stops_when_ready_motion_rejected(self):
        calibrator = SimpleNamespace(get_robot_version=lambda: '1.2', is_v13=lambda: False,
            perform_move_to_ready_pose=lambda *args, **kwargs: False)
        self.assertIsNone(calibration_core.calibrate_marker_bracket(calibrator, 'right'))

    def test_empty_collection_plan_rejected_without_capture(self):
        state = calibration_core.CollectionState(motion_plan=[])
        service = calibration_core.AutoCollectionService(
            robot=object(), model=object(), dyn_model=None,
            marker_transform=None, config=robot_motion.AutoCollectionConfig(), state=state)
        with self.assertRaisesRegex(RuntimeError, 'empty'):
            service.run()

    def test_three_failed_observations_abort_and_do_not_append_dataset(self):
        state = calibration_core.CollectionState(
            motion_plan=[{'desc': 'offline pose'}] * 5, ready=True)
        service = calibration_core.AutoCollectionService(
            robot=object(), model=object(), dyn_model=None, marker_transform=None,
            config=robot_motion.AutoCollectionConfig(), state=state, include_head_motion=False)
        with patch('core.robot_motion.build_incremental_motion_plan', return_value=state.motion_plan), \
             patch('core.robot_motion.execute_auto_motion_step'), \
             patch('core.calibration_core.capture_calibration_sample', return_value=(None, None, None)), \
             patch('core.calibration_core.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, '3 consecutive'):
                service.run()
        self.assertEqual(state.pose_index, 3)
        self.assertEqual(state.arm_samples, [])
        self.assertEqual(state.marker_samples, [])

    def test_initialization_filters_head_and_rejects_servo_failure(self):
        robot = MagicMock()
        robot.get_robot_info.return_value.robot_model_name = 'm'
        robot.get_control_manager_state.return_value.state = rby.ControlManagerState.State.Enabled
        robot.is_servo_on.return_value = False
        robot.servo_on.return_value = False
        with patch('core.robot_motion.rby.create_robot', return_value=robot), patch('core.robot_motion.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Servo on failed'):
                robot_motion.initialize_robot_connection('offline', 'm', servo='right_arm.*|head.*', include_head=False)
        pattern = robot.servo_on.call_args.args[0]
        self.assertIsNone(re.fullmatch(pattern, 'head_0'))
        self.assertIsNotNone(re.fullmatch(pattern, 'right_arm_0'))
        robot.enable_control_manager.assert_not_called()

    def test_disabled_head_only_move_sends_no_command(self):
        robot = SimpleNamespace(send_command=lambda *args: self.fail('Head motion was forbidden'))
        self.assertFalse(robot_motion.move_joints_checked(robot, head=[.1, .2], include_head=False))

    def test_square_targets_keep_arm_separation_and_four_point_order(self):
        targets = robot_motion.calibration_square_targets(.12)
        self.assertEqual(len(targets), 4)
        np.testing.assert_allclose([r[:3, 3] for r, _ in targets],
            [[.35, -.05, 0], [.35, -.12, .07], [.35, -.19, 0], [.35, -.12, -.07]])
        np.testing.assert_allclose([l[:3, 3] for _, l in targets],
            [[.35, .19, 0], [.35, .12, .07], [.35, .05, 0], [.35, .12, -.07]])
        np.testing.assert_allclose(targets[0][0][:3, :3],
            [[0, -1, 0], [0, 0, -1], [1, 0, 0]], atol=1e-15)
        np.testing.assert_allclose(targets[0][1][:3, :3],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-15)

    def test_no_head_ready_pose_requests_teaching_without_head_state_access(self):
        queries = []
        taught = []
        def observe(**kwargs):
            queries.append(kwargs['side'])
            return None if len(queries) == 1 else np.eye(4)
        marker = SimpleNamespace(get_marker_transform=observe)
        with patch('core.robot_motion.move_to_auto_ready_pose'), patch('core.robot_motion.time.sleep'):
            result = robot_motion.prepare_capture_pose(
                object(), ['right', 'left'], 10, include_head_motion=False,
                marker_transform=marker, head_idx=[0, 1],
                teaching_callback=lambda side: taught.append(side) or True)
        self.assertIsNone(result)
        self.assertEqual(taught, ['right'])
        self.assertEqual(queries, ['right', 'left', 'right', 'left'])

    def test_cancelled_teaching_stops_before_visibility_success(self):
        marker = SimpleNamespace(get_marker_transform=lambda **kwargs: None)
        with patch('core.robot_motion.move_to_auto_ready_pose'), patch('core.robot_motion.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'teaching canceled'):
                robot_motion.prepare_capture_pose(
                    object(), ['right', 'left'], 10, include_head_motion=False,
                    marker_transform=marker, teaching_callback=lambda side: False)

    def test_single_arm_ready_checks_only_requested_marker(self):
        sides = []
        def observe(**kwargs):
            sides.append(kwargs['side'])
            if kwargs['side'] == 'left':
                self.fail('Inactive arm must not require teaching')
            return np.eye(4)
        with patch('core.robot_motion.move_to_auto_ready_pose'), patch('core.robot_motion.time.sleep'):
            robot_motion.prepare_capture_pose(object(), ['right'], 10,
                include_head_motion=False, marker_transform=SimpleNamespace(get_marker_transform=observe))
        self.assertEqual(sides, ['right', 'right'])

    def test_head_centering_command_failure_cannot_publish_success(self):
        pose = np.eye(4)
        pose[:3, 3] = [.1, .1, 1.]
        robot = SimpleNamespace(get_state=lambda: SimpleNamespace(position=np.zeros(2)),
            send_command=lambda *args: SimpleNamespace(get=lambda: SimpleNamespace(finish_code='failed')))
        marker = SimpleNamespace(get_marker_transform=lambda **kwargs: pose)
        with patch('core.robot_motion.move_to_auto_ready_pose'), patch('core.robot_motion.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Head centering move failed'):
                robot_motion.prepare_capture_pose(robot, ['right'], 10, include_head_motion=True,
                    marker_transform=marker, head_idx=[0, 1])
