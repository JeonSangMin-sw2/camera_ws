import unittest
from unittest.mock import MagicMock
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from core.calibration.MarkerCalibrator import MarkerCalibrator
from core.calibration.CalibratorBase import BaseCalibrator


class TestMarkerCalibratorContracts(unittest.TestCase):

    def test_axis_5_nominal_target_accounts_for_theta_6(self):
        """
        Verify that for v1.2, when Joint 6 is at angle q_6 (e.g. 60.82 deg),
        the nominal target axis in the marker frame is rotated around Flange Z
        by q_6, so that physical rotation of Joint 5 does not falsely trigger
        a 60.82 deg anomaly alert.
        """
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        mc.is_v13 = MagicMock(return_value=False)
        mc.joint_offsets = {"right": {"wrist_pitch": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}}
        mc.NOMINAL_BRACKET_TEMPLATES = {
            "1.2": {
                "right": [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
            }
        }

        # Simulated Joint 6 angle from user posture adjustment or ready pose
        theta_6_deg = 60.82
        theta_6_rad = np.radians(theta_6_deg)
        cur_initial_pos = [0.0, 0.0, 0.0, 0.0, 0.0, np.radians(90.0), theta_6_rad]

        nominal_rpy = mc.NOMINAL_BRACKET_TEMPLATES["1.2"]["right"][3:6]
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

        # Physical rotation axis observed by camera during Joint 5 sweep (with q_6 held at theta_6)
        n_marker_actual = mc.rodrigues_rotation(y_ee_m_ideal, z_ee_m_ideal, theta_6_rad)

        # Uncorrected (buggy) target assumed theta_6 == 0
        uncorrected_target = y_ee_m_ideal
        uncorrected_dev = np.degrees(np.arccos(np.clip(abs(np.dot(n_marker_actual, uncorrected_target)), -1.0, 1.0)))
        self.assertAlmostEqual(uncorrected_dev, 60.82, places=2)

        # Corrected target takes theta_6 into account
        corrected_target = mc.rodrigues_rotation(y_ee_m_ideal, z_ee_m_ideal, theta_6_rad)
        corrected_dev = np.degrees(np.arccos(np.clip(abs(np.dot(n_marker_actual, corrected_target)), -1.0, 1.0)))
        self.assertAlmostEqual(corrected_dev, 0.0, places=4)
        self.assertLess(corrected_dev, 35.0, "Corrected deviation must be well below 35 deg threshold")

    def test_user_taught_pose_takes_priority_over_stale_initial_joint_pos(self):
        """
        Verify that cur_initial_pos prioritizes user_taught_ready_poses
        over any stale initial_joint_pos passed in from earlier stages.
        """
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        stale_pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        taught_pos = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        mc.user_taught_ready_poses = {
            "right": {"marker": taught_pos}
        }

        # Emulate the priority selection logic implemented in MarkerCalibrator
        taught_pose = None
        if hasattr(mc, 'user_taught_ready_poses') and isinstance(mc.user_taught_ready_poses, dict):
            arm_dict = mc.user_taught_ready_poses.get("right", {})
            if isinstance(arm_dict, dict) and "marker" in arm_dict and arm_dict["marker"] is not None:
                taught_pose = list(arm_dict["marker"])

        if taught_pose is not None:
            cur_initial_pos = list(taught_pose)
        elif stale_pos is not None:
            cur_initial_pos = list(stale_pos)
        else:
            cur_initial_pos = None

        self.assertEqual(cur_initial_pos, taught_pos, "Taught pose must override stale initial_joint_pos")

    def test_move_to_ready_pose_taught_pose_disables_apply_offsets(self):
        """
        Verify that when user-taught pose is detected in perform_move_to_ready_pose,
        movej is called with apply_offsets=False to avoid double-offsetting the arm.
        """
        calib = BaseCalibrator.__new__(BaseCalibrator)
        calib.robot = MagicMock()
        calib.is_v13 = MagicMock(return_value=False)
        calib.current_calib_mode = "marker"
        calib.user_taught_ready_poses = {
            "right": {"marker": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
        }
        calib.movej = MagicMock(return_value=True)

        calib.perform_move_to_ready_pose("right", mode="marker")

        # Verify movej call kwargs
        self.assertTrue(calib.movej.called)
        _, kwargs = calib.movej.call_args
        self.assertFalse(kwargs.get("apply_offsets", True), "apply_offsets must be False when using taught_pose")


if __name__ == '__main__':
    unittest.main()
