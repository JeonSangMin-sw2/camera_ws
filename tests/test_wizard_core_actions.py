"""Core side of the 2026-11-12 wizard changes: reset-on-record check, gripper-change pose, auto exposure.
Offline: fake robot, camera and calibrator."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.calibration import CalibrationCore
from core.calibration.data import AUTO_EXPOSURE_CANDIDATES_US, choose_auto_exposure, load_gripper_change_pose
from core.calibration.sequences import auto_exposure
from core.calibration.sequences.result import SequenceContext
from core.robot.home_offset import home_reset_pending
from core.storage import ConfigStorage, ResultStorage

ROOT = Path(__file__).resolve().parents[1]


class TestHomeResetPending(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "home_reset_baseline.json"

    def save(self, right, left):
        ResultStorage.save(self.path, {"right_arm_joint_offset_deg": right, "left_arm_joint_offset_deg": left,
                                       "metadata": {"type": "home_reset_baseline", "created_at": "2026-11-12T10:00:00"}})

    def test_no_file_and_applied_record_are_not_pending(self):
        self.assertEqual(home_reset_pending(self.path), (False, None))
        self.save([0.0] * 7, [0.0] * 7)          # what Home Offset Apply leaves behind
        self.assertEqual(home_reset_pending(self.path), (False, "2026-11-12T10:00:00"))

    def test_offsets_left_mean_the_reset_was_not_applied(self):
        self.save([0.0, 1.2, 0, 0, 0, 0, 0], [0.0] * 7)
        self.assertEqual(home_reset_pending(self.path), (True, "2026-11-12T10:00:00"))


class TestGripperChangePose(unittest.TestCase):
    def test_shipped_poses(self):
        v12, v13 = load_gripper_change_pose("1.2"), load_gripper_change_pose("v1.3")
        # v1.2: J3 + J5 = -180 turns the flange up; mirrored J1 between the arms.
        self.assertEqual(v12["right"][3] + v12["right"][5], -180.0)
        self.assertEqual(v12["right"][1], -v12["left"][1])
        self.assertGreater(v12["right"][5], -90.0)            # clear of the J5 limit
        # v1.3: only the elbow at 90 deg (user, 2026-11-12).
        self.assertEqual(v13["right"], [0.0, 0.0, 0.0, -90.0, 0.0, 0.0, 0.0])
        self.assertEqual(v13["left"], v13["right"])

    def test_missing_or_short_entry_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ready_poses.yaml"
            ConfigStorage.save(path, {"v1.2": {}})
            with self.assertRaises(ValueError):
                load_gripper_change_pose("1.2", path)
            ConfigStorage.save(path, {"v1.2": {"gripper_change": {"right_arm": [0] * 6, "left_arm": [0] * 7}}})
            with self.assertRaises(ValueError):
                load_gripper_change_pose("1.2", path)


class TestChooseAutoExposure(unittest.TestCase):
    @staticmethod
    def trial(exposure, rate, jitter, turned=((1.0, 0.05), (1.0, 0.05))):
        return {"exposure": exposure, "rate": rate, "jitter_mm": jitter,
                "turned": [{"rate": r, "jitter_mm": j} for r, j in turned]}

    def test_steadiest_qualified_value_wins_darker_on_a_tie(self):
        trials = [self.trial(1500, 0.6, 0.30),                                # often lost
                  self.trial(2500, 1.0, 0.08), self.trial(4000, 1.0, 0.08),
                  self.trial(6000, 1.0, 0.05, turned=((0.5, 0.05), (1.0, 0.05)))]   # lost when turned
        self.assertEqual(choose_auto_exposure(trials, 0.2)["exposure"], 2500)
        trials.append(self.trial(8500, 1.0, 0.04))
        self.assertEqual(choose_auto_exposure(trials, 0.2)["exposure"], 8500)

    def test_near_tie_takes_the_darker_exposure(self):
        # 2026-11-24 16:30 run: 16000 us won on 0.033 mm against 6000 us at 0.044 mm.
        trials = [self.trial(6000, 1.0, 0.044, turned=((1.0, 0.037), (1.0, 0.036))),
                  self.trial(8500, 1.0, 0.036, turned=((1.0, 0.026), (1.0, 0.097))),
                  self.trial(16000, 1.0, 0.033, turned=((1.0, 0.023), (1.0, 0.024)))]
        self.assertEqual(choose_auto_exposure(trials, 0.2)["exposure"], 6000)
        trials[0] = self.trial(6000, 1.0, 0.060, turned=((1.0, 0.037), (1.0, 0.036)))   # 0.027 worse: not a tie
        self.assertEqual(choose_auto_exposure(trials, 0.2)["exposure"], 16000)

    def test_none_when_nothing_holds_the_limit_or_was_not_turned(self):
        self.assertIsNone(choose_auto_exposure([self.trial(4000, 1.0, 0.25)], 0.2))
        self.assertIsNone(choose_auto_exposure([self.trial(4000, 1.0, 0.05, turned=())], 0.2))


class FakeCamera:
    """Right marker: lost below 2500 us, jitter falls with exposure up to 8500 us and rises above."""
    def __init__(self, model="D405"):
        self.model, self.exposure, self.auto = model, None, None
        self.rng = np.random.default_rng(0)

    def get_camera_info(self):
        return {"camera_model": self.model}

    def set_camera_exposure(self, value, auto_exposure=False):
        self.exposure, self.auto = value, auto_exposure
        return True

    def get_marker_transform(self, sampling_time=0, side="right", use_filter=None):
        if self.exposure < 2500:
            return None
        noise_m = {2500: 0.00015, 4000: 0.0001, 6000: 0.00006, 8500: 0.00004}.get(self.exposure, 0.0003)
        T = np.eye(4)
        T[:3, 3] = [0.06, 0.0, 0.35] + self.rng.normal(0, noise_m, 3)
        return [T.ravel()]


def fake_core(camera, version="1.2"):
    core = CalibrationCore()
    core.observer = camera
    q = np.zeros(24)
    model = SimpleNamespace(right_arm_idx=list(range(2, 9)), left_arm_idx=list(range(9, 16)), head_idx=[0, 1])
    core.robot = SimpleNamespace(model=lambda: model, get_state=lambda: SimpleNamespace(position=q))
    core.robot_version = version
    core.include_head_motion = True
    core.joint_calibrator = MagicMock()
    core.joint_calibrator.movej.return_value = True
    core.joint_calibrator.perform_move_to_ready_pose.return_value = True
    core.log_msg = MagicMock()
    return core


class TestAutoExposureSequence(unittest.TestCase):
    def run_sequence(self, core):
        with patch.object(auto_exposure, "SAMPLE_PERIOD_S", 0.001):
            return auto_exposure.run_auto_exposure(core, SequenceContext("auto_exposure"),
                                                   settle_s=0, still_s=0.05, turned_s=0.03)

    def test_d405_tries_every_value_and_applies_the_steadiest(self):
        camera = FakeCamera()
        core = fake_core(camera, version="1.3")
        result = self.run_sequence(core)
        self.assertEqual([t["exposure"] for t in result["trials"]], list(AUTO_EXPOSURE_CANDIDATES_US))
        self.assertEqual((result["applied"], result["exposure"]), ("manual", 8500))
        self.assertEqual((camera.exposure, camera.auto), (8500, False))
        core.joint_calibrator.perform_move_to_ready_pose.assert_called_once_with(
            arm_side="right", mode="wrist_pitch_v13", log_callback=core.log_msg)
        # Pitched both ways (J5 +-15 deg) and back to the ready pose.
        pitches = [np.degrees(c.kwargs["right_arm"][5]) for c in core.joint_calibrator.movej.call_args_list
                   if "right_arm" in c.kwargs]
        self.assertEqual([round(p) for p in pitches], [-15, 15, 0])

    def test_other_cameras_use_their_own_auto_exposure(self):
        camera = FakeCamera(model="D435")
        result = self.run_sequence(fake_core(camera))
        self.assertEqual((result["applied"], result["trials"]), ("auto", []))
        self.assertTrue(camera.auto)

    def test_no_good_value_turns_auto_exposure_back_on(self):
        camera = FakeCamera()
        camera.get_marker_transform = lambda **kwargs: None
        result = self.run_sequence(fake_core(camera))
        self.assertIsNone(result["applied"])
        self.assertTrue(camera.auto)


class TestGripperPoseSequence(unittest.TestCase):
    def test_moves_both_arms_then_switches_the_tool_power_off(self):
        core = fake_core(FakeCamera())
        order = []
        core.joint_calibrator.movej.side_effect = lambda *a, **k: order.append(("move", k)) or True
        with patch("core.calibration.calibration_core.RobotOperations.power_off_tool_flanges",
                   side_effect=lambda robot: order.append(("power",)) or {"right": 0.0, "left": 0.0}):
            result = core.run("gripper_pose")
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual([o[0] for o in order], ["move", "power"])
        move = order[0][1]
        np.testing.assert_allclose(np.degrees(move["right_arm"]), load_gripper_change_pose("1.2")["right"])
        np.testing.assert_allclose(np.degrees(move["left_arm"]), load_gripper_change_pose("1.2")["left"])
        self.assertEqual((move["head"], move["apply_offsets"]), ([0.0, 0.0], False))
        self.assertEqual(result.completed["gripper_pose"], {"right": 0.0, "left": 0.0})

    def test_move_time_follows_the_remaining_travel(self):
        # 2026-11-24 (user): resumed after the held button was let go, the move keeps its pace.
        core = fake_core(FakeCamera())
        core.run("gripper_pose")
        self.assertAlmostEqual(core.joint_calibrator.movej.call_args.kwargs["minimum_time"], 110.0 / 18.0)
        q = core.robot.get_state().position
        q[[5, 12]] = np.radians(-55.0)                  # J3 (-110) and J5 (-70) of both arms half way there
        q[[7, 14]] = np.radians(-35.0)
        core.run("gripper_pose")
        self.assertAlmostEqual(core.joint_calibrator.movej.call_args.kwargs["minimum_time"], 55.0 / 18.0)

    def test_v13_moves_but_keeps_the_gripper_powered(self):
        # 2026-11-12 (user): a v1.3 marker bracket goes on the gripper, so its 12 V stays on.
        core = fake_core(FakeCamera(), version="1.3")
        with patch("core.calibration.calibration_core.RobotOperations.power_off_tool_flanges") as power:
            result = core.run("gripper_pose")
        self.assertEqual(result.status, "completed", result.error)
        move = core.joint_calibrator.movej.call_args.kwargs
        np.testing.assert_allclose(np.degrees(move["right_arm"]), load_gripper_change_pose("1.3")["right"])
        power.assert_not_called()
        self.assertEqual(result.completed["gripper_pose"], {})

    def test_failed_move_leaves_the_power_alone(self):
        core = fake_core(FakeCamera())
        core.joint_calibrator.movej.return_value = False
        with patch("core.calibration.calibration_core.RobotOperations.power_off_tool_flanges") as power:
            result = core.run("gripper_pose")
        self.assertEqual(result.status, "failed")
        power.assert_not_called()


if __name__ == "__main__":
    unittest.main()
