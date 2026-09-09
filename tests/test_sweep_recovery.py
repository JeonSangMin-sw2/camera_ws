"""Observation loss must reach teaching without mixing camera postures."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from core.calibration.CalibratorBase import BaseCalibrator, SweepObservationError
from core.calibration.JointCalibrator import JointCalibrator


class SweepRecoveryTests(unittest.TestCase):
    def test_partial_visibility_loss_requests_whole_measurement_recovery(self):
        # More than ten poses survive, so the old low-count check missed this.
        class FakeThread:
            def __init__(self):
                self.remaining = 40
                self.finished = False
            def start(self): pass
            def is_alive(self):
                if self.remaining:
                    self.remaining -= 1
                    return True
                if not self.finished:
                    self.finished = True
                    self.run()
                return False
            def join(self): pass
        observations = []
        for i in range(40):
            pose = np.eye(4)
            pose[0, 3] = i * .001
            observations.append([pose] if i < 25 else None)
        cal = BaseCalibrator.__new__(BaseCalibrator)
        cal.stop_requested = False
        cal.robot = SimpleNamespace(model=lambda: SimpleNamespace(right_arm_idx=list(range(7))),
                                    get_dynamics=lambda: None, cancel_control=Mock(),
                                    get_state=Mock(return_value=SimpleNamespace(position=np.arange(7.))))
        cal.marker_st = SimpleNamespace(get_marker_transform=Mock(side_effect=observations))
        cal.movej = Mock(return_value=True)
        encoders = []
        with patch('core.calibration.CalibratorBase.threading.Thread', FakeThread), \
             patch('core.calibration.CalibratorBase.time.sleep'):
            with self.assertRaisesRegex(ValueError, 'visibility'):
                cal.perform_single_joint_sweep('right', 5, np.zeros(7), -10., 10., 20.,
                                              defer_recovery=True, encoder_samples=encoders)
        self.assertEqual(len(encoders), 25)
        np.testing.assert_array_equal(encoders[0], np.arange(7.))
        self.assertIsNot(encoders[0], encoders[1])

    def test_fit_failure_prompts_once_and_restarts_whole_measurement(self):
        cal = JointCalibrator.__new__(JointCalibrator)
        cal.stop_requested = False
        rejected = dict(measurement_accepted=False, retryable_observation=True,
                        failure_reason='Poor circle observation')
        accepted = dict(measurement_accepted=True, optimal_offset=.02)
        cal._perform_calibration_sweep_once = Mock(side_effect=[rejected, accepted])
        cal.perform_move_to_ready_pose = Mock(return_value=True)
        cal.marker_problem_callback = Mock(return_value=True)
        result = cal.perform_calibration_sweep_continuous('right', 'wrist_yaw2',
                    current_offset_deg=-.3, first_starting_pose=np.zeros(7))
        self.assertIs(result, accepted)
        cal.marker_problem_callback.assert_called_once_with('right')
        self.assertEqual(cal._perform_calibration_sweep_once.call_count, 2)
        self.assertIsNone(cal._perform_calibration_sweep_once.call_args.kwargs['first_starting_pose'])
        self.assertEqual(cal._perform_calibration_sweep_once.call_args.kwargs['current_offset_deg'], -.3)

    def test_cancellation_and_geometry_rejection_do_not_retry(self):
        for retryable, resolved in ((True, False), (False, True)):
            with self.subTest(retryable=retryable, resolved=resolved):
                cal = JointCalibrator.__new__(JointCalibrator)
                cal.stop_requested = False
                rejected = dict(measurement_accepted=False, retryable_observation=retryable)
                cal._perform_calibration_sweep_once = Mock(return_value=rejected)
                cal.perform_move_to_ready_pose = Mock(return_value=True)
                cal.marker_problem_callback = Mock(return_value=resolved)
                result = cal.perform_calibration_sweep_continuous('right', 'wrist_yaw2')
                self.assertFalse(result and result.get('measurement_accepted'))
                self.assertEqual(cal._perform_calibration_sweep_once.call_count, 1)
                self.assertEqual(cal.marker_problem_callback.call_count, int(retryable))

    def test_repeated_bad_observations_have_bounded_retry(self):
        cal = JointCalibrator.__new__(JointCalibrator)
        cal.stop_requested = False
        cal._perform_calibration_sweep_once = Mock(return_value=dict(
            measurement_accepted=False, retryable_observation=True))
        cal.perform_move_to_ready_pose = Mock(return_value=True)
        cal.marker_problem_callback = Mock(return_value=True)
        result = cal.perform_calibration_sweep_continuous('right', 'wrist_yaw2')
        self.assertFalse(result['measurement_accepted'])
        self.assertEqual(cal._perform_calibration_sweep_once.call_count, 2)
        cal.marker_problem_callback.assert_called_once()

    def test_circle_fit_failure_is_classified_for_teaching(self):
        cal = JointCalibrator.__new__(JointCalibrator)
        result = cal.compute_calibration_results('right', 'wrist_yaw2',
            np.tile(np.eye(4), (20, 1, 1)), np.tile(np.eye(4), (20, 1, 1)))
        self.assertFalse(result['measurement_accepted'])
        self.assertTrue(result.get('retryable_observation'))

    def test_real_sequence_discards_first_a_after_b_loses_visibility(self):
        cal = JointCalibrator.__new__(JointCalibrator)
        cal.stop_requested = False
        cal.robot = SimpleNamespace(model=lambda: SimpleNamespace(right_arm_idx=list(range(7))),
                                    get_state=lambda: SimpleNamespace(position=np.zeros(7)))
        cal.marker_st = SimpleNamespace(get_marker_transform=Mock(return_value=[np.eye(4)]))
        cal.get_robot_version = lambda: '1.2'
        cal.get_ready_pose = lambda *args: np.zeros(7)
        cal.perform_move_to_ready_pose = Mock(return_value=True)
        old_a, new_a, new_b = [np.tile(np.eye(4), (20, 1, 1)) for _ in range(3)]
        cal.perform_single_joint_sweep = Mock(side_effect=[old_a,
            SweepObservationError('B visibility insufficient'), new_a, new_b])
        def teach(side):
            taught = np.zeros(7)
            taught[0] = .35
            cal.user_taught_ready_poses = {side: {'wrist_yaw2': taught}}
            return True
        cal.marker_problem_callback = teach
        cal.save_observed_points = Mock()
        cal.compute_legacy_j6_results = Mock(return_value=dict(measurement_accepted=True))
        result = cal.perform_calibration_sweep_continuous('right', 'wrist_yaw2',
                    current_offset_deg=-.3, save_debug=True)
        self.assertTrue(result['measurement_accepted'])
        calls = cal.perform_single_joint_sweep.call_args_list
        self.assertEqual([call.args[1] for call in calls], [6, 5, 6, 5])
        self.assertEqual(calls[0].args[2][0], 0.)
        self.assertEqual(calls[2].args[2][0], .35)
        self.assertAlmostEqual(calls[2].args[2][6], np.deg2rad(-.3))
        self.assertEqual([call.args[3:6] for call in calls],
                         [(-15., 15., 20.), (-10., 10., 20.)] * 2)
        self.assertEqual(cal.compute_legacy_j6_results.call_count, 1)
        self.assertIs(cal.compute_legacy_j6_results.call_args.args[1], new_a)
        self.assertIs(cal.compute_legacy_j6_results.call_args.args[3], new_b)
        self.assertEqual(cal.save_observed_points.call_count, 2)
        self.assertIs(cal.save_observed_points.call_args_list[0].args[2], new_a)
        self.assertIs(cal.save_observed_points.call_args_list[1].args[2], new_b)

    def test_stop_during_teaching_prevents_restart(self):
        cal = JointCalibrator.__new__(JointCalibrator)
        cal.stop_requested = False
        cal._perform_calibration_sweep_once = Mock(return_value=dict(
            measurement_accepted=False, retryable_observation=True))
        cal.perform_move_to_ready_pose = Mock(return_value=True)
        def teach(side):
            cal.stop_requested = True
            return True
        cal.marker_problem_callback = teach
        self.assertIsNone(cal.perform_calibration_sweep_continuous('right', 'wrist_yaw2'))
        cal._perform_calibration_sweep_once.assert_called_once()


if __name__ == '__main__':
    unittest.main()
