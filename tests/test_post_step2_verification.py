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
    core.marker_calibrator.ready_poses = {}          # no check_calib_view: the check-pose step is skipped
    core.get_marker_bracket = lambda side: [0.0, -0.054 if side == "right" else 0.054, -0.048, 90.0, 0.0,
                                            180.0 if side == "right" else 0.0]
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


class TestCheckPoseView(unittest.TestCase):
    """The check pose seen by the head camera (user, 2026-11-24): offset-corrected, head down, J6 turned mirrored."""
    CHECK_R = np.radians([-24.67, -39.97, 11.25, -107.95, -123.90, 62.21, 58.40])
    VIEW = {"head_deg": [0.0, 20.0], "j6_turn_deg": -42.0}

    def run_view(self, measured_dist, markers=True, dx=0.0, dy=0.2):
        core = make_core()
        core.marker_calibrator.ready_poses = {"v1.2": {"check_calib_view": self.VIEW}}
        core.marker_calibrator.get_ready_pose.side_effect = (
            lambda _v, _t, _m, side: self.CHECK_R if side == "right" else verify.MIRROR_JOINT_SIGN * self.CHECK_R)
        cfg = {"arm_idx": list(range(2, 16)), "ee_links": {"right": "ee_right", "left": "ee_left"}}
        sym = {"dx_mm": dx, "dy_mm": dy, "dz_mm": -measured_dist, "dist_mm": float(np.linalg.norm([dx, dy, measured_dist])),
               "drot_deg": 0.1, "separation_mm": 300}
        err = {"pos_err_mm": 0.3, "rot_err_deg": 0.2}
        evaluation = {"per_arm": {"right": err, "left": err}, "symmetry": sym, "measured_symmetry": sym}
        sample = (np.zeros(14), np.zeros(2), [np.eye(4), np.eye(4)] if markers else None)
        with patch.object(verify, "_evaluate", return_value=evaluation),                 patch.object(verify, "_link_separations", return_value=np.full(8, 200.0)),                 patch.object(verify, "capture_one_sample", return_value=sample):
            value = verify._check_pose_view(core, MagicMock(), cfg, [0, 1], ("link_head_2", [0] * 6), OFFSETS,
                                            lambda *_: None)
        return value, core.marker_calibrator.movej.call_args_list

    def test_commands_the_corrected_check_pose_and_stays_there(self):
        value, calls = self.run_view(0.8)
        self.assertEqual(len(calls), 1)
        there = calls[0].kwargs
        turn = np.radians(-42.0)
        np.testing.assert_allclose(there["right_arm"], self.CHECK_R + np.r_[np.zeros(6), turn]
                                   - np.radians(OFFSETS["right_arm_joint_offset_deg"]))
        np.testing.assert_allclose(there["left_arm"], verify.MIRROR_JOINT_SIGN * self.CHECK_R + np.r_[np.zeros(6), -turn]
                                   - np.radians(OFFSETS["left_arm_joint_offset_deg"]))
        np.testing.assert_allclose(there["head"], np.radians([0.0, 20.0]) - np.radians(OFFSETS["head_joint_offset_deg"]))
        self.assertFalse(there["apply_offsets"])
        self.assertTrue(value["seen"])
        self.assertEqual((value["gap_mm"], value["limit_mm"]), (0.8, 2.0))
        # The raw measurement for an offline re-solve (user, 2026-11-24), JSON-ready.
        import json
        saved = json.loads(json.dumps(value))
        np.testing.assert_allclose(saved["q_cmd_arm"], np.r_[there["right_arm"], there["left_arm"]])
        np.testing.assert_allclose(saved["q_cmd_head"], there["head"])
        self.assertEqual(np.asarray(saved["markers"]["left"]).shape, (4, 4))
        self.assertEqual(saved["view"], {"head_deg": [0.0, 20.0], "j6_turn_deg": -42.0})
        self.assertEqual(saved["camera"]["link"], "link_head_2")
        self.assertEqual(saved["brackets"]["left"][1], 0.054)

    def test_both_hands_moved_sideways_together_is_not_judged(self):
        # 2026-11-25 16:55 (second robot): camera x -0.50, y -3.57, z -0.50 mm, the plates matched within 1 mm by eye.
        value, _ = self.run_view(0.50, dx=-0.50, dy=-3.57)
        self.assertAlmostEqual(value["gap_mm"], float(np.hypot(0.5, 0.5)))
        self.assertLess(value["gap_mm"], value["limit_mm"])
        self.assertAlmostEqual(value["measured_symmetry"]["dy_mm"], -3.57)     # still recorded

    def test_front_back_and_height_are_judged(self):
        # 2026-11-24 19:38 (first robot): x -1.88, z -4.21 mm, left hand about 4 mm low by eye.
        value, _ = self.run_view(4.21, dx=-1.88, dy=1.77)
        self.assertAlmostEqual(value["gap_mm"], float(np.hypot(1.88, 4.21)))
        self.assertGreater(value["gap_mm"], value["limit_mm"])

    def test_markers_not_seen_is_reported_not_passed(self):
        value, calls = self.run_view(0.0, markers=False)
        self.assertEqual((value["seen"], value["reason"]), (False, "markers"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(value["q_arm"]), 14)                               # still recorded, with what was sent
        self.assertNotIn("markers", value)

    def test_without_a_view_entry_it_is_skipped(self):
        core = make_core()
        self.assertIsNone(verify._check_pose_view(core, MagicMock(), {}, [0, 1], None, OFFSETS, lambda *_: None))
        core.marker_calibrator.movej.assert_not_called()


class TestCheckPoseViewSaved(unittest.TestCase):
    def test_goes_into_the_result_json(self):
        import json
        import tempfile
        from pathlib import Path
        from core.calibration import CalibrationCore
        core = CalibrationCore()
        self.addCleanup(core.close)
        logs = []
        core.log_msg = logs.append
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result_x.json"
            path.write_text(json.dumps({"right_arm_joint_offset_deg": [0.0] * 7}), encoding="utf-8")
            core._save_check_pose_view(str(path), {"check_pose_view": {"seen": True, "gap_mm": 4.94}})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["check_pose_view"], {"seen": True, "gap_mm": 4.94})
            self.assertEqual(data["right_arm_joint_offset_deg"], [0.0] * 7)
            core._save_check_pose_view(str(Path(tmp) / "missing.json"), {"check_pose_view": {"seen": False}})
        self.assertTrue(any("[WARN]" in line and "not saved" in line for line in logs))   # not silent


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
