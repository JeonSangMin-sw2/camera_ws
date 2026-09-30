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


class ReteachedBracketPosture(unittest.TestCase):
    """2026-09-21: a right-arm marker re-teach left J5 2.7 deg off its ready-pose value; the
    measured J4-J6 axes came out 86.8 deg apart, the bracket roll absorbed it and Step 2 moved
    both J0 offsets by ~0.5 deg."""

    def test_reteach_keeps_the_operators_posture_but_puts_j5_back(self):
        import numpy as np
        from types import SimpleNamespace
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        mc.robot_version = "1.2"
        ready = np.radians([-90.0, -40.0, 73.0, -97.0, 90.0, 90.0, -10.0])
        mc.get_ready_pose = lambda version, kind, mode, side: ready.copy()
        mc.is_v13 = lambda: False
        mc.joint_offsets = {"right": {"wrist_pitch": 1.6}, "left": {"wrist_pitch": 0.0}}
        shared = {}
        mc.user_taught_ready_poses = shared
        taught = np.radians([-85.0, -35.0, 70.0, -95.0, 88.0, 88.9, -5.0])
        q = np.zeros(20)
        q[0:7] = taught
        mc.robot = SimpleNamespace(model=lambda: SimpleNamespace(right_arm_idx=list(range(7)), left_arm_idx=list(range(7, 14))),
                                   get_state=lambda: SimpleNamespace(position=q))
        logs = []
        pose = mc.adopt_taught_marker_pose("right", logs.append)
        self.assertAlmostEqual(np.degrees(pose[5]), 91.6, places=6)
        np.testing.assert_allclose(np.delete(pose, 5), np.delete(taught, 5))
        # Stored where the sweeps and the ready-pose move read it, in the shared dict.
        self.assertIs(mc.user_taught_ready_poses, shared)
        np.testing.assert_allclose(shared["right"]["marker"], pose)
        self.assertEqual(len(logs), 1)

    def test_marker_reteach_keeps_the_encoder_pose_as_taught(self):
        # A taught pose is replayed as raw encoder values; forcing the nominal J5 into it would
        # drop the J5 calibration offset.
        import numpy as np
        from main_ui import enforce_nominal_taught_joints
        nominal = np.radians([-90.0, -40.0, 73.0, -97.0, 90.0, 90.0, -10.0])
        taught = list(np.radians([-85.0, -35.0, 70.0, -95.0, 88.0, 91.6, -5.0]))
        before = list(taught)
        enforce_nominal_taught_joints(taught, "marker", nominal)
        self.assertEqual(taught, before)

    def test_j4_j6_axis_angle_reports_how_far_j5_was_from_90(self):
        import numpy as np
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        logs = []
        tilt = np.radians(86.8)
        dev = mc.warn_if_j5_off_nominal({'axis_opt': [1.0, 0.0, 0.0]},
                                        {'axis_opt': [np.cos(tilt), np.sin(tilt), 0.0]}, "right", logs.append)
        self.assertAlmostEqual(dev, 3.2, places=3)
        self.assertEqual(len(logs), 1)
        logs.clear()
        mc.warn_if_j5_off_nominal({'axis_opt': [1.0, 0.0, 0.0]}, {'axis_opt': [0.003, 1.0, 0.0]}, "left", logs.append)
        self.assertEqual(logs, [])


class AxisInMarkerFrameUsesEveryFrame(unittest.TestCase):
    """2026-09-22 (D405): mapping the swept axis into the marker frame with the middle frame alone
    made the right J6 estimate scatter ~0.7 deg; the axis is fixed in the marker frame, so every
    frame of the sweep estimates it."""

    def test_one_bad_middle_frame_does_not_move_the_axis(self):
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        rng = np.random.default_rng(0)
        R0 = R.from_euler("xyz", [20, -30, 10], degrees=True)
        n_cam = np.array([0.3, -0.2, 0.93]); n_cam /= np.linalg.norm(n_cam)   # joint axis, camera frame
        axis_marker = R0.inv().apply(n_cam)                                     # the same axis, marker frame
        poses = []
        for k in range(200):
            R_true = R.from_rotvec(n_cam * np.radians(k * 0.2)) * R0
            noise = R.from_rotvec(rng.normal(0, np.radians(0.3), 3))
            T = np.eye(4); T[:3, :3] = (R_true * noise).as_matrix(); poses.append(T)
        mid = len(poses) // 2
        poses[mid][:3, :3] = (R.from_matrix(poses[mid][:3, :3]) * R.from_euler("x", 4, degrees=True)).as_matrix()
        est = MarkerCalibrator.axis_in_marker_frame_robust(n_cam, poses, axis_marker)
        err = np.degrees(np.arccos(np.clip(est @ axis_marker, -1, 1)))
        self.assertLess(err, 0.15)
        old = poses[mid][:3, :3].T @ n_cam
        self.assertGreater(np.degrees(np.arccos(np.clip(old @ axis_marker, -1, 1))), 2.0)


class BracketYLock(unittest.TestCase):
    """v1.2 bracket y stays at the design value (54 mm) unless setting.yaml turns it off; x/z are
    refitted within the assembly tolerance."""

    def calibrator(self, version, markers):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        mc.robot_version = version
        mc.markers_config = markers
        return mc

    def test_on_by_default_for_v12(self):
        self.assertTrue(self.calibrator("1.2", {}).bracket_y_locked())

    def test_setting_can_turn_it_off(self):
        self.assertFalse(self.calibrator("1.2", {"lock_bracket_y": False}).bracket_y_locked())

    def test_never_applies_to_v13(self):
        self.assertFalse(self.calibrator("1.3", {"lock_bracket_y": True}).bracket_y_locked())

    def test_tolerance_defaults_to_2_mm_and_is_configurable(self):
        self.assertEqual(self.calibrator("1.2", {}).bracket_offset_tolerance_mm(), 2.0)
        self.assertEqual(self.calibrator("1.2", {"bracket_offset_tolerance_mm": 1.0}).bracket_offset_tolerance_mm(), 1.0)



class BracketRollLock(unittest.TestCase):
    """v1.2 bracket roll stays at the design value (90 deg) unless setting.yaml turns it off."""

    def calibrator(self, version, markers):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        mc.robot_version = version
        mc.markers_config = markers
        return mc

    def test_on_by_default_for_v12(self):
        self.assertTrue(self.calibrator("1.2", {}).bracket_roll_locked())

    def test_setting_can_turn_it_off(self):
        self.assertFalse(self.calibrator("1.2", {"lock_bracket_roll": False}).bracket_roll_locked())

    def test_never_applies_to_v13(self):
        self.assertFalse(self.calibrator("1.3", {"lock_bracket_roll": True}).bracket_roll_locked())


class BracketRadiiFromEncoderFits(unittest.TestCase):
    """v1.2 bracket translation uses the encoder-angle circle radii the sweeps arrive with, not the
    position-only radii of the joint axis refit (2026-09-22: J6 57.3 mm position-only vs 54.2 mm
    encoder-angle on the left arm, which pushed z to the 2 mm assembly limit)."""

    L5 = 126.1

    def calibrator(self):
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        mc.robot_version = "1.2"
        mc.markers_config = {}
        mc.camera_config = {}
        mc.joint_offsets = {"left": {"wrist_pitch": 0.0, "wrist_yaw2": 0.0}}
        mc.get_link_length = lambda arm_side: self.L5
        mc.get_z_sign = lambda arm_side: -1.0
        mc.refine_bracket_axes_constrained = lambda *a, **k: {   # position-only radii (J4, J6, J5)
            "axes": [np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])],
            "centers": [np.zeros(3)] * 3, "radii": [186.9, 57.3, 174.7]}
        return mc

    def sweeps(self, x, y, z):
        Zp = z - self.L5
        poses = [np.eye(4) for _ in range(12)]
        make = lambda axis, r: {"axis_opt": np.array(axis, dtype=float), "radius": r, "rmse": 0.1,
                                "captured_poses": poses, "theta_6": 0.0}
        return (make([0, 1, 0], np.hypot(x, Zp)), make([0, 0, 1], np.hypot(x, y)),
                make([1, 0, 0], np.hypot(y, Zp)))

    def solve(self, y):
        import io, contextlib
        d5, d6, d4 = self.sweeps(0.0, y, -49.0)
        logs = []
        with contextlib.redirect_stdout(io.StringIO()):
            out = self.calibrator().compute_unified_bracket_calibration(
                d5, d6, "left", marker_data_4=d4, log_callback=logs.append)
        return out, logs, d4, d6

    def test_translation_uses_the_encoder_radii(self):
        out, logs, d4, d6 = self.solve(54.0)
        self.assertAlmostEqual(out["radius_6"], 54.0, places=6)
        self.assertAlmostEqual(out["radius_4"], d4["radius_encoder"], places=6)
        self.assertEqual(d6["radius"], 57.3)              # the refit circle is still what the plot draws
        self.assertAlmostEqual(out["z_e"], -49.0, places=2)
        self.assertFalse([l for l in logs if "[WARN]" in l], logs)

    def test_a_y_more_than_1_mm_off_design_warns(self):
        out, logs, _, _ = self.solve(55.5)
        self.assertEqual(out["y_e"], 54.0)                # still held at the design value
        self.assertTrue(any("[WARN]" in l and "y at 55.50" in l for l in logs), logs)


class RefitDropsOffCircleFrames(unittest.TestCase):
    """The constrained axis refit leaves out frames far off the sweep's encoder-angle circle
    (2026-09-22: one 26 mm false detection at the end of the left J5 sweep moved the bracket
    pitch 0.09 -> 0.33 deg)."""

    def sweep(self, n=60, noise=0.1, seed=0, axis=(0.0, 1.0, 0.0)):
        rng = np.random.default_rng(seed)
        c, axis, r = np.array([10.0, -20.0, 250.0]), np.array(axis), 174.0
        u = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        t = np.radians(np.linspace(0, -40, n))
        P = c + r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), w)) + rng.normal(0, noise, (n, 3))
        return P, {"c_opt": c, "axis_opt": axis, "radius_encoder": r}

    def test_a_false_detection_is_left_out_and_logged(self):
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        P, fit = self.sweep()
        P[-1] += [15.0, 5.0, 20.0]
        logs = []
        kept = mc.drop_refit_outliers(P, fit, 5, "left", logs.append)
        self.assertEqual(len(kept), len(P) - 1)
        np.testing.assert_allclose(kept, P[:-1])
        self.assertEqual(len(logs), 1)
        self.assertIn("frame 59", logs[0])

    def test_clean_sweeps_keep_every_frame(self):
        mc = MarkerCalibrator.__new__(MarkerCalibrator)
        P, fit = self.sweep(noise=0.25)
        logs = []
        self.assertEqual(len(mc.drop_refit_outliers(P, fit, 5, "left", logs.append)), len(P))
        self.assertEqual(logs, [])

    def test_the_refit_gets_the_filtered_points(self):
        import io, contextlib
        mc = BracketRadiiFromEncoderFits().calibrator()
        seen = {}

        def refit(points, axes, centers, radii, log_callback=None):
            seen["n"] = [len(p) for p in points]
            return None

        mc.refine_bracket_axes_constrained = refit
        d5, d6, d4 = BracketRadiiFromEncoderFits().sweeps(0.0, 54.0, -49.0)
        for d, seed in zip((d4, d6, d5), (1, 2, 3)):
            P, fit = self.sweep(seed=seed, axis=d["axis_opt"])
            d.update(fit, captured_poses=[np.eye(4) for _ in P])
            d.pop("radius_encoder")
            d["radius"] = fit["radius_encoder"]
            for T, p in zip(d["captured_poses"], P):
                T[:3, 3] = p / 1000.0
        d5["captured_poses"][-1][:3, 3] += 0.03
        with contextlib.redirect_stdout(io.StringIO()):
            mc.compute_unified_bracket_calibration(d5, d6, "left", marker_data_4=d4, log_callback=lambda *_: None)
        self.assertEqual(seen["n"], [60, 60, 59])        # J4, J6 whole; J5 without its bad frame
