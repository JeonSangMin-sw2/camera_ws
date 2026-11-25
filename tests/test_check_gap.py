"""Wizard check pose: bring the hands together / apart while a button is held (2026-11-24, user).

home_offset.check_gap_motion picks the target and the time for one press; ready_poses.yaml holds the
hands-together pose. Offline, no robot.
"""
import unittest
from pathlib import Path

import numpy as np

from core.robot.home_offset import CHECK_GAP_TRAVEL_S, check_gap_motion, has_check_gather_pose
from core.storage import ConfigStorage

ROOT = Path(__file__).resolve().parents[1]
CHECK = np.zeros(14)
GATHER = np.radians(np.r_[np.full(7, 5.0), np.full(7, -5.0)])


class CheckGapMotionTest(unittest.TestCase):
    def test_gather_from_the_check_pose_takes_the_full_travel(self):
        target, seconds = check_gap_motion(CHECK, CHECK, GATHER, "gather")
        np.testing.assert_allclose(target, GATHER)
        self.assertAlmostEqual(seconds, CHECK_GAP_TRAVEL_S)

    def test_a_press_after_a_release_carries_on_at_the_same_speed(self):
        halfway = CHECK + 0.5 * (GATHER - CHECK)
        self.assertAlmostEqual(check_gap_motion(halfway, CHECK, GATHER, "gather")[1], CHECK_GAP_TRAVEL_S / 2)
        self.assertAlmostEqual(check_gap_motion(halfway, CHECK, GATHER, "spread")[1], CHECK_GAP_TRAVEL_S / 2)

    def test_spread_goes_back_to_the_check_pose(self):
        target, seconds = check_gap_motion(GATHER, CHECK, GATHER, "spread")
        np.testing.assert_allclose(target, CHECK)
        self.assertAlmostEqual(seconds, CHECK_GAP_TRAVEL_S)

    def test_already_there(self):
        self.assertEqual(check_gap_motion(GATHER, CHECK, GATHER, "gather"), (None, 0.0))
        self.assertEqual(check_gap_motion(CHECK, CHECK, GATHER, "spread"), (None, 0.0))

    def test_arms_away_from_the_check_pose_are_refused(self):
        elsewhere = CHECK.copy()
        elsewhere[3] = np.radians(10.0)
        with self.assertRaises(RuntimeError):
            check_gap_motion(elsewhere, CHECK, GATHER, "gather")

    def test_bad_direction(self):
        with self.assertRaises(ValueError):
            check_gap_motion(CHECK, CHECK, GATHER, "up")


class GatherPoseConfigTest(unittest.TestCase):
    def setUp(self):
        self.poses = ConfigStorage.load(ROOT / "config" / "ready_poses.yaml")

    def test_v12_has_it_and_v13_not_yet(self):
        # v1.3's check pose already has the hands at t5 y +-70 mm, inside the +-100 mm of the v1.2 pose.
        self.assertTrue(has_check_gather_pose("1.2"))
        self.assertFalse(has_check_gather_pose("1.3"))

    def test_v12_pose_is_mirror_symmetric_and_near_the_check_pose(self):
        gather = self.poses["v1.2"]["check_calib_gather"]
        right, left = np.array(gather["right_arm"]), np.array(gather["left_arm"])
        mirror = np.array([1, -1, -1, 1, -1, 1, -1])     # left = mirror * right for these arms
        np.testing.assert_allclose(left, mirror * right, atol=0.02)
        check = np.array(self.poses["v1.2"]["check_calib"]["right_arm"])
        self.assertLess(np.max(np.abs(right - check)), 10.0)


if __name__ == "__main__":
    unittest.main()
