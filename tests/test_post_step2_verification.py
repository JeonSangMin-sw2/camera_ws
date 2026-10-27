"""Post-Step-2 verification without a second move to the ready pose (user, 2026-10-27).

Live Step 2 collection ends on a restore_baseline step, i.e. at the ready pose, so the verification takes that
sample as the ready pose: the right arm's joints mirrored onto the left, minus the offsets, is commanded straight
from where the robot stands. Offline, no robot.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.calibration.sequences import verify

RIGHT = np.radians([-70.0, -40.0, 20.0, -110.0, 30.0, -40.0, 35.0])
OFFSETS = {"right_arm_joint_offset_deg": [0.5, -0.4, 0.3, 3.4, 0.6, 3.3, -1.1],
           "left_arm_joint_offset_deg": [0.1, -2.5, 0.4, 2.9, -1.8, 0.2, 1.7],
           "head_joint_offset_deg": [-0.6, -0.4], "mount_to_cam_new": [0.044, 0.009, 0.0635, -90.0, -0.7, -90.0]}


def make_core():
    state = np.zeros(20)
    robot = SimpleNamespace(get_state=lambda: SimpleNamespace(position=state))
    core = SimpleNamespace(robot=robot, model=MagicMock(), log_msg=lambda *_: None, include_head_motion=True,
                           get_robot_version=lambda: "1.2", marker_calibrator=MagicMock(), observer=SimpleNamespace(sim=True))
    core.marker_calibrator.movej.return_value = True
    return core


class TestNoSecondReadyPoseMove(unittest.TestCase):
    def run_verify(self, reference):
        core = make_core()
        context = MagicMock()
        cfg = {"arm_idx": list(range(2, 16)), "ee_links": {"right": "ee_right", "left": "ee_left"}}
        sym = {"dx_mm": 0, "dy_mm": 0, "dz_mm": 0, "dist_mm": 0, "drot_deg": 0, "separation_mm": 260}
        err = {"pos_err_mm": 0.1, "rot_err_deg": 0.1}
        evaluation = {"per_arm": {"right": err, "left": err}, "symmetry": sym, "measured_symmetry": sym}
        sample = (np.r_[RIGHT, verify.MIRROR_JOINT_SIGN * RIGHT], np.zeros(2), [np.eye(4), np.eye(4)])
        with patch.object(verify, "move_to_auto_ready_pose") as ready, \
                patch.object(verify, "get_both_arm_config", return_value=cfg), \
                patch.object(verify, "get_head_config", return_value={"head_idx": [0, 1]}), \
                patch.object(verify, "_evaluate", return_value=evaluation), \
                patch.object(verify, "_link_separations", return_value=np.full(8, 200.0)), \
                patch.object(verify, "capture_one_sample", return_value=sample) as capture:
            verify.run_post_step2_verification(core, context, optimization=OFFSETS, reference=reference)
        return ready, capture, core.marker_calibrator.movej

    def test_with_the_last_step2_sample_the_robot_goes_straight_to_the_corrected_pose(self):
        q_last = np.r_[RIGHT, verify.MIRROR_JOINT_SIGN * RIGHT + np.radians(0.3)]   # left a little off symmetric
        ready, capture, movej = self.run_verify((q_last, np.radians([1.0, -2.0])))
        ready.assert_not_called()
        self.assertEqual(capture.call_count, 1)                                 # only at the corrected pose
        kwargs = movej.call_args.kwargs
        np.testing.assert_allclose(kwargs["right_arm"], RIGHT - np.radians(OFFSETS["right_arm_joint_offset_deg"]))
        np.testing.assert_allclose(kwargs["left_arm"], verify.MIRROR_JOINT_SIGN * RIGHT
                                   - np.radians(OFFSETS["left_arm_joint_offset_deg"]))
        np.testing.assert_allclose(kwargs["head"], np.radians([1.0, -2.0]) - np.radians(OFFSETS["head_joint_offset_deg"]))

    def test_without_a_reference_it_still_moves_to_the_ready_pose(self):
        ready, capture, movej = self.run_verify(None)
        ready.assert_called_once()
        self.assertEqual(capture.call_count, 2)                                 # ready pose and corrected pose


class TestCollectionMarksBaselineSamples(unittest.TestCase):
    def test_restore_baseline_samples_are_flagged(self):
        from core.calibration import CalibrationCore
        from core.calibration.sequences import collection
        core = CalibrationCore()
        core.robot = MagicMock()
        core.observer = SimpleNamespace(sim=True, last_frame={"frame_id": 1})
        core.log_msg = lambda *_: None
        good = (np.zeros(14), np.zeros(2), np.stack([np.eye(4), np.eye(4)]))
        plan = [{"type": "joint", "desc": "a"}, {"type": "restore_baseline", "desc": "b"}]
        with patch.object(collection, "get_both_arm_config", return_value={"arm_idx": list(range(14))}), \
                patch.object(collection, "get_head_config", return_value={"head_idx": [14, 15]}), \
                patch.object(collection, "execute_auto_motion_step", side_effect=plan), \
                patch.object(collection, "capture_one_sample", return_value=good):
            result = core.run("collect", plan=plan)
        self.assertEqual([s["baseline"] for s in result.completed["samples"]], [False, True])


if __name__ == "__main__":
    unittest.main()
