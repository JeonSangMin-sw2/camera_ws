"""Step 2 orientation-noise floor (step2.min_rot_noise_deg) and the v1.2 J6 two-method report.

2026-09-29 (D405): Step 2 weighted residuals by 1/sigma with sigma from its own residual RMS and a
0.01 deg orientation floor. J4 bent to absorb a marker-orientation bias, the orientation RMS fell to
0.114 deg, the orientation weight rose further, and J4 moved by up to 1.6 deg and both J0 by
1.5-1.8 deg. A 0.3 deg floor removed it offline while leaving two D435 runs within 0.09 deg.
Offline, no robot.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.calibration import CalibrationCore
from core.calibration.calibration_optimizer import QPCalibrationOptimizer, ResidualNoiseEstimator
from core.calibration.data import DEFAULT_STEP2_MIN_ROT_NOISE_DEG, load_step2_min_rot_noise_deg
from core.calibration.JointCalibrator import choose_wrist_yaw2_offset
from core.calibration.sequences import step2 as step2_mod
from core.storage import ConfigStorage

ROOT = Path(__file__).resolve().parents[1]
NQ = 24


def fake_robot():
    dyn = MagicMock()
    dyn.get_limit_q_lower.return_value = np.full(NQ, -3.0)
    dyn.get_limit_q_upper.return_value = np.full(NQ, 3.0)
    model = SimpleNamespace(robot_joint_names=[f"j{i}" for i in range(NQ)])
    return SimpleNamespace(get_dynamics=lambda: dyn, model=lambda: model,
                           get_state=lambda: SimpleNamespace(position=np.zeros(NQ)))


def make_optimizer(**kwargs):
    return QPCalibrationOptimizer(robot=fake_robot(), arm_idx=list(range(2, 16)),
                                  ee_links={"right": "ee_right", "left": "ee_left"},
                                  mount_to_cam_nom=[0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
                                  ee_to_marker_nom={"right": [0, -0.054, -0.049, 90, 0, 180],
                                                    "left": [0, 0.054, -0.049, 90, 0, 0]},
                                  head_idx=[0, 1], estimate_measurement_noise=True, **kwargs)


class TestNoiseFloor(unittest.TestCase):
    def test_orientation_noise_never_drops_below_the_floor(self):
        est = ResidualNoiseEstimator(enabled=True, update_rate=1.0, min_rot_std_rad=np.radians(0.3))
        tiny_rot = np.radians(0.05)
        est.update([[tiny_rot, 0, 0, 1e-4, 0, 0]] * 10)
        self.assertAlmostEqual(np.degrees(est.rot_std_rad), 0.3)
        self.assertAlmostEqual(est.weights()[0], 1.0 / np.radians(0.3))
        self.assertAlmostEqual(est.as_dict()["measurement_noise_min_rot_std_deg"], 0.3)

    def test_optimizer_applies_the_floor_it_is_given(self):
        self.assertAlmostEqual(np.degrees(make_optimizer().noise_estimator.min_rot_std_rad), 0.01)
        floored = make_optimizer(min_rot_noise_std_rad=np.radians(0.3))
        self.assertAlmostEqual(np.degrees(floored.noise_estimator.min_rot_std_rad), 0.3)
        self.assertGreaterEqual(np.degrees(floored.noise_estimator.rot_std_rad), 0.3)


class TestFloorConfig(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "setting.yaml"

    def test_value_missing_and_invalid(self):
        ConfigStorage.save(self.path, {"step2": {"min_rot_noise_deg": 0.25}})
        self.assertEqual(load_step2_min_rot_noise_deg(self.path), (0.25, True))
        ConfigStorage.save(self.path, {"step2": {}})
        self.assertEqual(load_step2_min_rot_noise_deg(self.path), (DEFAULT_STEP2_MIN_ROT_NOISE_DEG, False))
        ConfigStorage.save(self.path, {"step2": {"min_rot_noise_deg": 0}})
        with self.assertRaises(ValueError):
            load_step2_min_rot_noise_deg(self.path)

    def test_shipped_value_is_0_3(self):
        shipped = ConfigStorage.load(ROOT / "config" / "setting.yaml")["step2"]["min_rot_noise_deg"]
        self.assertEqual(shipped, 0.3)
        self.assertEqual(DEFAULT_STEP2_MIN_ROT_NOISE_DEG, 0.3)


class Stop(Exception):
    pass


class TestOptimizeStep2PassesTheFloor(unittest.TestCase):
    def test_first_optimizer_gets_the_configured_floor(self):
        created = []

        def fake_optimizer(**kwargs):
            created.append(kwargs)
            raise Stop()

        core = CalibrationCore()
        core.on_event = lambda kind, value: None
        core.robot = MagicMock()
        core.model = MagicMock()
        arm_cfg = {"arm_idx": list(range(2, 16)), "ee_links": {"right": "ee_right", "left": "ee_left"},
                   "ee_to_marker_nom": {"right": [0] * 6, "left": [0] * 6},
                   "mount_to_cam_nom": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]}
        n = 5
        q_arm, q_head, T = np.zeros((n, 14)), np.zeros((n, 2)), np.tile(np.eye(4), (n, 2, 1, 1))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(step2_mod, "QPCalibrationOptimizer", fake_optimizer), \
                patch.object(step2_mod, "get_both_arm_config", return_value=arm_cfg), \
                patch.object(step2_mod, "get_head_config", return_value={"head_idx": [0, 1]}), \
                patch.object(step2_mod, "validate_dataset"), \
                patch.object(step2_mod, "load_step2_min_rot_noise_deg", return_value=(0.3, True)):
            with self.assertRaises(Stop):
                step2_mod.optimize_step2(core, ["right", "left"], True, True, q_arm, q_head, T,
                                         str(Path(folder) / "result_test.json"))
        self.assertEqual(len(created), 1)
        self.assertAlmostEqual(created[0]["min_rot_noise_std_rad"], np.radians(0.3))

    def test_every_step2_optimizer_is_built_with_the_floor(self):
        source = (ROOT / "core" / "calibration" / "sequences" / "step2.py").read_text(encoding="utf-8")
        constructions = source.count("QPCalibrationOptimizer(")
        self.assertEqual(constructions, 3)                      # pass 1, pass 2, single arm
        self.assertEqual(source.count("min_rot_noise_std_rad=min_rot_noise_rad"), constructions)


class TestWristYaw2Choice(unittest.TestCase):
    def test_orientation_is_used_by_default_and_both_are_reported(self):
        self.assertEqual(choose_wrist_yaw2_offset(2.60, 2.39, "orientation"), (2.60, 2.39 - 2.60))

    def test_position_is_used_when_selected(self):
        chosen, diff = choose_wrist_yaw2_offset(1.60, 2.53, "position")
        self.assertEqual(chosen, 2.53)
        self.assertAlmostEqual(diff, 0.93)

    def test_missing_position_falls_back_to_orientation(self):
        self.assertEqual(choose_wrist_yaw2_offset(1.60, None, "position"), (1.60, None))


if __name__ == "__main__":
    unittest.main()
