"""Step 2 wrist-diversity poses (head robots, setting.yaml step2.wrist_diversity_poses).

2026-09-30: the base Step 2 plan moves each joint only a few degrees, so J2/J4/J6 offset
combinations were nearly invisible to Step 2 yet moved the hands 1-2 mm per degree at the check pose.
A fixed list of extra wrist poses (ready_poses.yaml <version>.step2_wrist_diversity) is appended to the
plan. They are optional: a pose near a joint limit or with a marker out of view is skipped and never
aborts Step 2. Offline, no robot.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.calibration import CalibrationCore
from core.calibration.data import load_step2_wrist_diversity
from core.robot import motion
from core.storage import ConfigStorage

ROOT = Path(__file__).resolve().parents[1]
NQ = 24
RIGHT, LEFT, HEAD = list(range(2, 9)), list(range(9, 16)), [0, 1]
POSE = {"right": [10.0, -5.0, 20.0], "left": [-8.0, 4.0, -15.0], "head": [2.5, -5.0]}


class TestLoader(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.setting = Path(folder.name) / "setting.yaml"
        self.ready = Path(folder.name) / "ready_poses.yaml"
        ConfigStorage.save(self.ready, {"v1.2": {"step2_wrist_diversity": [POSE, POSE]}})

    def test_switch_off_gives_nothing(self):
        ConfigStorage.save(self.setting, {"step2": {"wrist_diversity_poses": False}})
        self.assertEqual(load_step2_wrist_diversity("1.2", self.setting, self.ready), ([], False))
        ConfigStorage.save(self.setting, {"step2": {}})
        self.assertEqual(load_step2_wrist_diversity("1.2", self.setting, self.ready), ([], False))

    def test_switch_on_reads_the_version_list(self):
        ConfigStorage.save(self.setting, {"step2": {"wrist_diversity_poses": True}})
        steps, enabled = load_step2_wrist_diversity("v1.2", self.setting, self.ready)
        self.assertTrue(enabled)
        self.assertEqual(steps, [POSE, POSE])
        self.assertEqual(load_step2_wrist_diversity("1.3", self.setting, self.ready), ([], True))

    def test_malformed_entry_is_an_error(self):
        ConfigStorage.save(self.setting, {"step2": {"wrist_diversity_poses": True}})
        ConfigStorage.save(self.ready, {"v1.2": {"step2_wrist_diversity": [{"right": [1, 2], "left": [1, 2, 3], "head": [0, 0]}]}})
        with self.assertRaises(ValueError):
            load_step2_wrist_diversity("1.2", self.setting, self.ready)

    def test_shipped_list(self):
        ConfigStorage.save(self.setting, {"step2": {"wrist_diversity_poses": True}})
        steps, _ = load_step2_wrist_diversity("1.2", self.setting, ROOT / "config" / "ready_poses.yaml")
        self.assertEqual(len(steps), 24)
        wrist = np.array([s["right"] + s["left"] for s in steps])
        head = np.array([s["head"] for s in steps])
        self.assertLessEqual(np.abs(wrist).max(), 45.0)        # J4 +-40, J5 +-30, J6 +-45 deg
        self.assertLessEqual(np.abs(head).max(), 12.5)
        shipped = ConfigStorage.load(ROOT / "config" / "setting.yaml")["step2"]
        self.assertIn("wrist_diversity_poses", shipped)          # explicit on/off, never implicit


def fake_robot(upper=np.pi):
    names = [f"j{i}" for i in range(NQ)]
    model = SimpleNamespace(right_arm_idx=RIGHT, left_arm_idx=LEFT, head_idx=HEAD, robot_joint_names=names)
    dyn = MagicMock()
    dyn.get_limit_q_lower.return_value = np.full(NQ, -np.pi)
    dyn.get_limit_q_upper.return_value = np.full(NQ, upper) if np.isscalar(upper) else upper
    q = np.zeros(NQ)
    q[RIGHT] = np.radians([-72, -73, 27, -110, 25, -41, 37])
    q[LEFT] = np.radians([-72, 73, -27, -110, -25, -41, -37])
    return SimpleNamespace(model=lambda: model, get_dynamics=lambda: dyn, get_state=lambda: SimpleNamespace(position=q.copy()))


def build_plan(include_head, poses):
    with patch.object(motion, "compute_fk", return_value=(None, np.eye(4))):
        return motion.build_incremental_motion_plan(fake_robot(), None, motion.AutoCollectionConfig(),
                                                    include_head_motion=include_head, wrist_diversity=poses)


class TestPlan(unittest.TestCase):
    def test_appended_once_as_optional_joint_steps_then_baseline(self):
        base = build_plan(True, [])
        plan = build_plan(True, [POSE, POSE])
        self.assertEqual(len(plan), len(base) + 3)
        extra = plan[len(base):]
        for step in extra[:2]:
            self.assertEqual(step["type"], "joint")
            self.assertTrue(step["optional"])
            self.assertEqual(step["offsets_by_arm"], {"right": {4: 10.0, 5: -5.0, 6: 20.0}, "left": {4: -8.0, 5: 4.0, 6: -15.0}})
            self.assertEqual((step["head_pan_offset_deg"], step["head_tilt_offset_deg"]), (2.5, -5.0))
        self.assertEqual(extra[2]["type"], "restore_baseline")
        self.assertFalse(any(s.get("optional") for s in base))

    def test_not_added_without_head(self):
        self.assertEqual(len(build_plan(False, [POSE])), len(build_plan(False, [])))


class TestExecute(unittest.TestCase):
    def run_step(self, robot):
        step = build_plan(True, [POSE])[-2]
        motion.reset_motion_state()
        sent = []
        with patch.object(motion, "send_auto_motion_cmd", side_effect=lambda **kw: sent.append(kw)), \
                patch.object(motion, "wait_motion"), patch.object(motion, "compute_fk", return_value=(None, np.eye(4))):
            out = motion.execute_auto_motion_step(robot, motion.AutoCollectionConfig(), step, ["right", "left"])
        return out, sent, step

    def test_per_arm_offsets_without_mirroring_and_head_offsets(self):
        robot = fake_robot()
        base = robot.get_state().position
        out, sent, step = self.run_step(robot)
        self.assertIs(out, step)
        np.testing.assert_allclose(sent[0]["q_right"][4:], base[RIGHT][4:] + np.radians([10, -5, 20]))
        np.testing.assert_allclose(sent[0]["q_left"][4:], base[LEFT][4:] + np.radians([-8, 4, -15]))
        np.testing.assert_allclose(sent[0]["q_right"][:4], base[RIGHT][:4])
        np.testing.assert_allclose(sent[0]["head_position"], np.radians([2.5, -5.0]))

    def test_near_a_joint_limit_is_skipped_without_moving(self):
        upper = np.full(NQ, np.pi)
        upper[RIGHT[6]] = np.radians(37 + 20 + 2)            # target 57 deg, limit 59 < 5 deg margin
        out, sent, _ = self.run_step(fake_robot(upper))
        self.assertIsNone(out)
        self.assertEqual(sent, [])


class TestCollectionSkipsOptionalFailures(unittest.TestCase):
    def setUp(self):
        self.core = CalibrationCore()
        self.core.robot = MagicMock()
        self.core.observer = SimpleNamespace(sim=True, last_frame={"frame_id": 1})
        self.logs = []
        self.core.log_msg = self.logs.append

    def collect(self, plan, moves, captures):
        from core.calibration.sequences import collection
        with patch.object(collection, "get_both_arm_config", return_value={"arm_idx": list(range(14))}), \
                patch.object(collection, "get_head_config", return_value={"head_idx": [14, 15]}), \
                patch.object(collection, "execute_auto_motion_step", side_effect=moves), \
                patch.object(collection, "capture_one_sample", side_effect=captures) as capture:
            return self.core.run("collect", plan=plan), capture

    def test_optional_misses_do_not_abort(self):
        good = (np.zeros(14), np.zeros(2), np.stack([np.eye(4), np.eye(4)]))
        missed = (None, None, None)
        plan = [{"optional": True, "desc": "wrist 1"}, {"optional": True, "desc": "wrist 2"},
                {"optional": True, "desc": "wrist 3"}, {"optional": True, "desc": "wrist 4"}, {"desc": "base"}]
        moves = [plan[0], None, plan[2], plan[3], plan[4]]      # wrist 2 is out of limits
        result, capture = self.collect(plan, moves, [missed, missed, missed, good])
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(len(result.completed["samples"]), 1)
        self.assertEqual(capture.call_count, 4)                  # no capture at the skipped pose
        self.assertTrue(any("skipped (joint limit)" in line for line in self.logs))
        self.assertEqual(sum("skipped (marker not in view)" in line for line in self.logs), 3)

    def test_regular_misses_still_abort(self):
        missed = (None, None, None)
        plan = [{}, {}, {}]
        result, _ = self.collect(plan, plan, [missed] * 3)
        self.assertEqual(result.status, "failed")
        self.assertIn("three consecutive", result.error)


if __name__ == "__main__":
    unittest.main()
