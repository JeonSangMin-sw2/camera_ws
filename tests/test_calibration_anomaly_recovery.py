"""Regression and contract tests for runtime anomaly detection and recovery.

Tests the interaction between JointCalibrator/MarkerCalibrator and the user teaching callback
without requiring any physical robot hardware, GUI, or camera.
"""
import unittest
import numpy as np

from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator


class DummyRobotModel:
    left_arm_idx = list(range(7))
    right_arm_idx = list(range(7, 14))
    head_idx = [14, 15]
    robot_joint_names = [f"joint_{i}" for i in range(16)]


class DummyRobotState:
    position = [0.0] * 16


class DummyRobot:
    def model(self):
        return DummyRobotModel()

    def get_state(self):
        return DummyRobotState()

    def get_dynamics(self):
        return None


class DummyMarkerST:
    sim = True

    def get_marker_transform(self, sampling_time=2.0, side="right"):
        return np.eye(4)


class TestCalibrationAnomalyRecovery(unittest.TestCase):
    def setUp(self):
        self.jc = JointCalibrator(marker_st=DummyMarkerST(), robot=DummyRobot())
        self.jc.joint_offsets = {"right": {"elbow": 0.0, "wrist_pitch": 0.0, "wrist_yaw2": 0.0, "wrist_roll": 0.0}}

        self.mc = MarkerCalibrator(marker_st=DummyMarkerST(), robot=DummyRobot())
        self.mc.joint_offsets = {"right": {"elbow": 0.0, "wrist_pitch": 0.0, "wrist_yaw2": 0.0, "wrist_roll": 0.0}}

    def test_joint_anomaly_triggers_callback_and_restarts_iteration_1(self):
        """Verify that an anomaly (e.g. large center_dist or runaway optimal_offset)
        triggers the posture readjustment callback, resets staged_offset to initial,
        and restarts from Iteration 1."""
        callback_call_count = [0]
        callback_modes = []

        def mock_marker_problem_callback(arm_side, **kwargs):
            callback_call_count[0] += 1
            callback_modes.append(kwargs.get('mode'))
            return True  # User adjusted posture successfully

        self.jc.marker_problem_callback = mock_marker_problem_callback
        self.jc.perform_move_to_ready_pose = lambda arm_side, mode="marker", log_callback=None: True

        sweep_invocation_count = [0]
        staged_offsets_received = []

        def mock_continuous_sweep(arm_side, mode, **kwargs):
            sweep_invocation_count[0] += 1
            cur_off = kwargs.get('current_offset_deg', 0.0)
            staged_offsets_received.append(cur_off)

            if sweep_invocation_count[0] == 1:
                # Anomalous sweep (center_dist > 40mm for wrist_pitch)
                return {
                    'angle_between_normals': 83.16,
                    'center_dist': 56.6,
                    'r_A': 75.0,
                    'r_B': 75.2,
                    'optimal_offset': -3.22,
                    'sign': 1.0,
                }
            else:
                # Clean sweep after posture readjustment
                return {
                    'angle_between_normals': 0.02,
                    'center_dist': 3.5,
                    'r_A': 75.0,
                    'r_B': 75.1,
                    'optimal_offset': 0.01,
                    'sign': 1.0,
                }

        self.jc.perform_calibration_sweep_continuous = mock_continuous_sweep

        logs = []
        res = self.jc.perform_joint_calibration(
            arm_side="right",
            mode="wrist_pitch",
            log_callback=logs.append,
            current_offset_deg=0.0,
            sweep_duration=1.0
        )

        self.assertIsNotNone(res, "Calibration should succeed after readjustment")
        self.assertEqual(callback_call_count[0], 1, "Teaching callback should be called exactly once")
        self.assertTrue(res['converged'], "Should converge after restarted clean sweep")
        self.assertEqual(staged_offsets_received[1], 0.0, "Restarted Iteration 1 must use initial offset (0.0)")
        self.assertTrue(any("Runtime measurement anomaly detected in Iteration 1" in l for l in logs))
        self.assertTrue(any("restarting calibration from Iteration 1" in l for l in logs))

    def test_joint_consecutive_anomaly_aborts_after_2_retries(self):
        """Verify that consecutive anomalies abort after 2 retries without infinite looping."""
        callback_call_count = [0]

        def mock_marker_problem_callback(arm_side, **kwargs):
            callback_call_count[0] += 1
            return True

        self.jc.marker_problem_callback = mock_marker_problem_callback
        self.jc.perform_move_to_ready_pose = lambda arm_side, mode="marker", log_callback=None: True

        def mock_continuous_sweep_always_bad(*args, **kwargs):
            return {
                'angle_between_normals': 85.0,
                'center_dist': 60.0,
                'r_A': 75.0,
                'r_B': 75.0,
                'optimal_offset': -4.0,
                'sign': 1.0,
            }

        self.jc.perform_calibration_sweep_continuous = mock_continuous_sweep_always_bad

        logs = []
        res = self.jc.perform_joint_calibration(
            arm_side="right",
            mode="wrist_pitch",
            log_callback=logs.append,
            current_offset_deg=0.0
        )

        self.assertIsNone(res, "Calibration should abort and return None after consecutive failures")
        self.assertEqual(callback_call_count[0], 2, "Callback should be called max 2 times (retry limit)")
        self.assertTrue(any("retry limit reached" in l for l in logs))

    def test_step_clamping_limits_large_steps(self):
        """Verify that step_correction is clamped to [-1.5, 1.5] degrees per iteration."""
        self.jc.marker_problem_callback = None
        self.jc.perform_move_to_ready_pose = lambda arm_side, mode="marker", log_callback=None: True

        staged_offsets_history = []

        def mock_continuous_sweep(arm_side, mode, **kwargs):
            cur_off = kwargs.get('current_offset_deg', 0.0)
            staged_offsets_history.append(cur_off)
            if len(staged_offsets_history) == 1:
                return {
                    'angle_between_normals': 3.0,
                    'center_dist': 5.0,
                    'r_A': 75.0,
                    'r_B': 75.0,
                    'optimal_offset': 3.0,
                    'sign': 1.0,
                }
            else:
                return {
                    'angle_between_normals': 0.02,
                    'center_dist': 5.0,
                    'r_A': 75.0,
                    'r_B': 75.0,
                    'optimal_offset': 0.01,
                    'sign': 1.0,
                }

        self.jc.perform_calibration_sweep_continuous = mock_continuous_sweep

        logs = []
        res = self.jc.perform_joint_calibration(
            arm_side="right",
            mode="wrist_pitch",
            log_callback=logs.append,
            current_offset_deg=0.0
        )

        # Iteration 1 start: 0.0
        # Correction was 3.0, but clamped to 1.5
        # Therefore Iteration 2 start should be 1.5
        self.assertAlmostEqual(staged_offsets_history[1], 1.5, places=3,
                               msg=f"Expected clamped staged offset 1.5, got {staged_offsets_history[1]}")

    def test_marker_anomaly_triggers_callback_and_retries(self):
        """Verify that in MarkerCalibrator, fitted circle axis deviation > 35 deg triggers callback
        and retries the sweep cleanly."""
        callback_call_count = [0]
        callback_modes = []

        def mock_marker_problem_callback(arm_side, **kwargs):
            callback_call_count[0] += 1
            callback_modes.append(kwargs.get('mode'))
            return True

        self.mc.marker_problem_callback = mock_marker_problem_callback
        self.mc.perform_move_to_ready_pose = lambda arm_side, mode="marker", log_callback=None: True

        sweep_count = [0]

        def mock_single_joint_sweep(arm_side, joint_i, cur_initial_pos, start_deg, end_deg, sweep_duration, **kwargs):
            sweep_count[0] += 1
            poses = [np.eye(4) for _ in range(30)]
            q_fulls = [[0.0] * 16 for _ in range(30)]
            return list(zip(q_fulls, poses))

        self.mc.perform_single_joint_sweep = mock_single_joint_sweep

        def mock_circle_fit(captured_poses, captured_angles, axis_prior=None, robust=True):
            return {
                'axis_opt': np.array([0.0, 0.0, 1.0]),
                'rmse': 1.2,
                'c_opt': np.array([0, 0, 0]),
                'radius_opt': 100.0,
            }

        self.mc.fit_circle_3d_and_6dof_misalignment = mock_circle_fit

        def mock_extract_axis(captured_poses, target_ideal):
            if sweep_count[0] == 1:
                # Return orthogonal vector (90 deg deviation in marker frame)
                v = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(v, target_ideal)) > 0.9:
                    v = np.array([0.0, 0.0, 1.0])
                return v
            else:
                # Normal clean sweep in marker frame
                return np.array(target_ideal)

        self.mc.extract_axis_from_rotations = mock_extract_axis

        logs = []
        res = self.mc.perform_calibration_sweep(
            arm_side="right",
            axis_mode=4,
            log_callback=logs.append
        )

        self.assertIsNotNone(res, "Marker sweep should succeed after readjustment")
        self.assertEqual(callback_call_count[0], 1, "Teaching callback should be called once")
        self.assertEqual(callback_modes[0], "marker", "Callback mode must be 'marker'")
        self.assertEqual(sweep_count[0], 2, "Sweep should have run twice (1 bad + 1 good retry)")
        self.assertTrue(any("Runtime measurement anomaly detected for Marker Axis 4" in l for l in logs))
        self.assertTrue(any("Restarting Marker Axis 4 sweep" in l for l in logs))

    def test_wrist_yaw2_orthogonal_normal_convergence(self):
        """Verify that in wrist_yaw2 (which has perpendicular axes at ~90 deg and center_dist ~56 mm),
        an angle_between_normals of 89.09 deg and center_dist of 56.69 mm is recognized as
        completely normal (angle_dev = 0.91 deg < 35 deg) and does NOT trigger an anomaly alert."""
        callback_call_count = [0]

        def mock_marker_problem_callback(arm_side, **kwargs):
            callback_call_count[0] += 1
            return True

        self.jc.marker_problem_callback = mock_marker_problem_callback
        self.jc.perform_move_to_ready_pose = lambda arm_side, mode="marker", log_callback=None: True

        def mock_continuous_sweep(arm_side, mode, **kwargs):
            return {
                'angle_between_normals': 89.0859,
                'center_dist': 56.6907,
                'r_A': 50.0,
                'r_B': 173.0,
                'optimal_offset': -0.04,
                'sign': 1.0,
            }

        self.jc.perform_calibration_sweep_continuous = mock_continuous_sweep

        logs = []
        res = self.jc.perform_joint_calibration(
            arm_side="right",
            mode="wrist_yaw2",
            log_callback=logs.append,
            current_offset_deg=0.0
        )

        self.assertIsNotNone(res)
        self.assertEqual(callback_call_count[0], 0, "Should NOT trigger anomaly callback for normal 89 deg orthogonal axes")
        self.assertTrue(res['converged'], "Should converge cleanly")
        self.assertFalse(any("Runtime measurement anomaly detected" in l for l in logs))


if __name__ == "__main__":
    unittest.main()
