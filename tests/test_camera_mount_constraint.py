"""Design-accurate camera mount (camera_info.yaml mount_to_cam_position_range_m).

2026-10-08: with the camera position free by +-25 mm, a Step 1 J6 off by about 1 deg was absorbed by
J4 and the camera height, and both J0 followed the camera's fore-aft position. With the key, the
camera stays inside a range around its design pose with its angles fixed, and Step 2 solves J6.
2026-10-16: the angles held are Step 1.5's, which estimates the optical-axis rotation in the range mode too
(a camera turned -0.65 deg about it had put -1.1 deg into both J1).
Offline, no robot; every file is in a temp folder.
"""
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from core.calibration import CalibrationCore
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.calibration_optimizer import QPCalibrationOptimizer, make_transform, se3_exp
from core.calibration.data import load_camera_mount_constraint, mount_range_ends
from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
from core.calibration.sequences import step2 as step2_mod
from core.storage import CONFIG_PATHS, ConfigStorage

D2R = np.pi / 180.0
NQ = 24
MOUNT = [0.0433, 0.009, 0.067, -90.0, 0.0, -90.0]
RANGE = [[-0.003, 0.001], [0.0, 0.0], [-0.001, 0.001]]
STEP1 = {"right": {"joint3": -1.9, "joint5": -0.3, "joint6": 0.8},
         "left": {"joint3": -1.8, "joint5": -0.1, "joint6": -1.2}, "head": {"pan": -0.3, "tilt": -1.2}}


class TestLoader(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.setting = Path(folder.name) / "setting.yaml"
        self.info = Path(folder.name) / "camera_info.yaml"
        ConfigStorage.save(self.setting, {"camera": {"device_name": "D405"}})

    def load(self, **entry):
        ConfigStorage.save(self.info, {"D405": {"mount_to_cam": MOUNT, **entry}, "D435": {"mount_to_cam": MOUNT}})
        return load_camera_mount_constraint(self.setting, self.info)

    def test_a_model_without_the_key_has_no_constraint(self):
        self.assertIsNone(self.load())
        ConfigStorage.save(self.setting, {"camera": {"device_name": "D435"}})
        self.assertIsNone(self.load(mount_to_cam_position_range_m=RANGE))

    def test_the_connected_models_design_pose_and_range_are_returned(self):
        constraint = self.load(mount_to_cam_position_range_m=RANGE)
        self.assertEqual(constraint["model"], "D405")
        self.assertEqual(constraint["mount_to_cam"], MOUNT)
        np.testing.assert_allclose(constraint["position_range_m"], RANGE)

    def test_a_malformed_range_is_an_error_not_a_free_camera(self):
        for bad in ([[-0.003, 0.001], [0.0, 0.0]], [0.001, 0.001, 0.001],
                    [[0.001, 0.003], [0.0, 0.0], [-0.001, 0.001]], [[-0.003, "wide"], [0.0, 0.0], [-0.001, 0.001]]):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.load(mount_to_cam_position_range_m=bad)

    def test_shipped_d405_entry(self):
        root = Path(__file__).resolve().parents[1] / "config"
        entry = ConfigStorage.load(root / "camera_info.yaml")["D405"]
        # 2026-10-08: the bracket height was measured (67 mm), so z went from +-1 mm to +-0.5 mm.
        np.testing.assert_allclose(entry["mount_to_cam_position_range_m"],
                                   [[-0.003, 0.001], [0.0, 0.0], [-0.0005, 0.0005]])
        self.assertNotIn("mount_to_cam_position_range_m", ConfigStorage.load(root / "camera_info.yaml")["D435"])


class TestRangeEnds(unittest.TestCase):
    def test_only_axes_with_room_that_sit_at_an_end_are_named(self):
        self.assertEqual(mount_range_ends([0.001, 0.0, -0.0004], RANGE), ["x at the upper end (+1.0 mm)"])
        self.assertEqual(mount_range_ends([-0.003, 0.0, -0.001], RANGE),
                         ["x at the lower end (-3.0 mm)", "z at the lower end (-1.0 mm)"])
        self.assertEqual(mount_range_ends([-0.0015, 0.0, 0.0], RANGE), [])


def fake_robot():
    dynamics = MagicMock()
    dynamics.get_limit_q_lower.return_value = np.full(NQ, -3.0)
    dynamics.get_limit_q_upper.return_value = np.full(NQ, 3.0)
    model = SimpleNamespace(robot_joint_names=[f"j{i}" for i in range(NQ)])
    return SimpleNamespace(get_dynamics=lambda: dynamics, model=lambda: model,
                           get_state=lambda: SimpleNamespace(position=np.zeros(NQ)))


def make_optimizer(mount_to_cam_nom=MOUNT, **options):
    return QPCalibrationOptimizer(robot=fake_robot(), arm_idx=list(range(2, 16)),
                                  ee_links={"right": "ee_right", "left": "ee_left"}, mount_to_cam_nom=mount_to_cam_nom,
                                  ee_to_marker_nom={"right": [0, -0.054, -0.049, 90, 0, 180],
                                                    "left": [0, 0.054, -0.049, 90, 0, 0]},
                                  head_idx=[0, 1], optimize_head=True, apply_joint_offset_limits=True,
                                  joint_offsets_to_apply=STEP1, camera_rot_bound_rad=3.0 * D2R,
                                  camera_pos_bound_m=0.025, **options)


class TestOptimizerBounds(unittest.TestCase):
    def bounds(self, optimizer, xi):
        dim = optimizer.total_dim()
        return optimizer._build_qp_bounds(dim, np.zeros(14), np.zeros(2), np.asarray(xi, dtype=float))

    def test_without_a_range_the_camera_and_j6_are_as_before(self):
        optimizer = make_optimizer()
        low, high = self.bounds(optimizer, np.zeros(6))
        self.assertAlmostEqual(high[-4], 3.0 * D2R)             # optical-axis rotation free within its bound
        np.testing.assert_allclose(high[-3:], 0.025)
        np.testing.assert_allclose(optimizer.joint_offset_upper[[6, 13]] - optimizer.joint_offset_lower[[6, 13]], 0.1 * D2R)

    def test_range_holds_the_angles_and_boxes_the_position_in_mount_link_axes(self):
        optimizer = make_optimizer(camera_pos_range_m=RANGE)
        xi = np.array([0.0, 0.0, 0.0, 0.0002, -0.0003, 0.0004])
        low, high = self.bounds(optimizer, xi)
        np.testing.assert_allclose(low[-6:-3], 0.0, atol=1e-15)
        np.testing.assert_allclose(high[-6:-3], 0.0, atol=1e-15)
        nominal = make_transform(MOUNT)
        reached = []
        for corner in itertools.product(*zip(low[-3:] + xi[3:], high[-3:] + xi[3:])):
            moved = nominal @ se3_exp(np.concatenate([np.zeros(3), corner]))
            reached.append(moved[:3, 3] - nominal[:3, 3])
        reached = np.array(reached)
        np.testing.assert_allclose(reached.min(axis=0), np.array(RANGE)[:, 0], atol=1e-12)
        np.testing.assert_allclose(reached.max(axis=0), np.array(RANGE)[:, 1], atol=1e-12)

    def test_a_nominal_turned_about_the_optical_axis_keeps_the_box_in_mount_link_axes(self):
        turned = make_transform(MOUNT)
        turned[:3, :3] = turned[:3, :3] @ Rotation.from_euler('z', -0.65, degrees=True).as_matrix()
        yaw, pitch, roll = Rotation.from_matrix(turned[:3, :3]).as_euler('ZYX', degrees=True)
        optimizer = make_optimizer(camera_pos_range_m=RANGE, mount_to_cam_nom=MOUNT[:3] + [roll, pitch, yaw])
        low, high = self.bounds(optimizer, np.zeros(6))
        reached = np.array([(turned @ se3_exp(np.concatenate([np.zeros(3), corner])))[:3, 3] - turned[:3, 3]
                            for corner in itertools.product(*zip(low[-3:], high[-3:]))])
        np.testing.assert_allclose(reached.min(axis=0), np.array(RANGE)[:, 0], atol=5e-5)
        np.testing.assert_allclose(reached.max(axis=0), np.array(RANGE)[:, 1], atol=5e-5)

    def test_a_tilted_nominal_camera_is_refused(self):
        with self.assertRaises(ValueError):
            QPCalibrationOptimizer(robot=fake_robot(), arm_idx=list(range(2, 16)),
                                   ee_links={"right": "ee_right", "left": "ee_left"},
                                   mount_to_cam_nom=MOUNT[:3] + [-90.0, 10.0, -90.0],
                                   ee_to_marker_nom={"right": [0] * 6, "left": [0] * 6}, head_idx=[0, 1],
                                   camera_pos_range_m=RANGE)

    def test_unanchored_j6_gets_the_other_joints_range_and_j3_j5_stay_held(self):
        optimizer = make_optimizer(anchor_joint6=False)
        np.testing.assert_allclose(optimizer.joint_offset_lower[[6, 13]], -15.0 * D2R)
        np.testing.assert_allclose(optimizer.joint_offset_upper[[6, 13]], 15.0 * D2R)
        held = [3, 5, 10, 12]
        np.testing.assert_allclose(optimizer.joint_offset_upper[held] - optimizer.joint_offset_lower[held], 0.1 * D2R)

    def test_unanchored_j5_gets_the_other_joints_range_and_j3_j6_stay_held(self):
        optimizer = make_optimizer(anchor_joint5=False)
        np.testing.assert_allclose(optimizer.joint_offset_lower[[5, 12]], -15.0 * D2R)
        np.testing.assert_allclose(optimizer.joint_offset_upper[[5, 12]], 15.0 * D2R)
        held = [3, 6, 10, 13]
        np.testing.assert_allclose(optimizer.joint_offset_upper[held] - optimizer.joint_offset_lower[held], 0.1 * D2R)


class Stop(Exception):
    pass


class TestStep2UsesTheConstraint(unittest.TestCase):
    def created_with(self, constraint, step15_camera=None, version="1.2"):
        created = []

        def fake_optimizer(**kwargs):
            created.append(kwargs)
            raise Stop()

        core = CalibrationCore()
        core.on_event = lambda kind, value: None
        core.robot, core.model = MagicMock(), MagicMock()
        core.robot_version = version
        arm = {"arm_idx": list(range(2, 16)), "ee_links": {"right": "ee_right", "left": "ee_left"},
               "ee_to_marker_nom": {"right": [0] * 6, "left": [0] * 6},
               "mount_to_cam_nom": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]}
        # The calibrators load the real setting.yaml; the camera Step 2 starts from must not depend on it.
        for key in ("mount_to_cam", "mount_to_cam_nominal"):
            core.marker_calibrator.camera_config[key] = list(arm["mount_to_cam_nom"])
        core.head_camera_calibrator.calibrated_results = (
            None if step15_camera is None else {"calibrated_mount_to_cam": list(step15_camera)})
        n = 5
        q_arm, q_head, T = np.zeros((n, 14)), np.zeros((n, 2)), np.tile(np.eye(4), (n, 2, 1, 1))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(step2_mod, "QPCalibrationOptimizer", fake_optimizer), \
                patch.object(step2_mod, "get_both_arm_config", return_value=arm), \
                patch.object(step2_mod, "get_head_config", return_value={"head_idx": [0, 1]}), \
                patch.object(step2_mod, "validate_dataset"), \
                patch.object(step2_mod, "load_camera_mount_constraint", return_value=constraint):
            with self.assertRaises(Stop):
                step2_mod.optimize_step2(core, ["right", "left"], True, True, q_arm, q_head, T,
                                         str(Path(folder) / "result_test.json"))
        return created[0]

    def test_without_a_constraint_step2_is_built_as_before(self):
        kwargs = self.created_with(None)
        self.assertIsNone(kwargs["camera_pos_range_m"])
        self.assertTrue(kwargs["anchor_joint6"])
        self.assertEqual(kwargs["mount_to_cam_nom"], [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])

    def test_with_a_constraint_the_camera_starts_at_design_and_j6_is_solved(self):
        kwargs = self.created_with({"model": "D405", "mount_to_cam": MOUNT, "position_range_m": np.array(RANGE)})
        np.testing.assert_allclose(kwargs["camera_pos_range_m"], RANGE)
        self.assertFalse(kwargs["anchor_joint6"])
        self.assertEqual(kwargs["mount_to_cam_nom"], MOUNT)

    def test_with_a_constraint_the_angles_are_step15s_and_the_position_the_design(self):
        step15 = [0.0443, 0.009, 0.0625, -90.0, -0.65, -90.0]
        kwargs = self.created_with({"model": "D405", "mount_to_cam": MOUNT, "position_range_m": np.array(RANGE)},
                                   step15_camera=step15)
        self.assertEqual(kwargs["mount_to_cam_nom"], MOUNT[:3] + step15[3:])

    def test_both_passes_get_the_constraint(self):
        source = (Path(step2_mod.__file__)).read_text(encoding="utf-8")
        self.assertEqual(source.count("camera_pos_range_m=camera_pos_range_m"), 2)
        self.assertEqual(source.count("anchor_joint6=not solve_j6"), 2)
        self.assertEqual(source.count("anchor_joint5=not solve_j5"), 2)

    def test_v13_holds_j6_at_step1_and_solves_j5_with_a_constraint(self):
        kwargs = self.created_with({"model": "D405", "mount_to_cam": MOUNT, "position_range_m": np.array(RANGE)},
                                   version="1.3")
        np.testing.assert_allclose(kwargs["camera_pos_range_m"], RANGE)
        self.assertTrue(kwargs["anchor_joint6"])
        self.assertFalse(kwargs["anchor_joint5"])

    def test_v12_and_v13_without_a_constraint_hold_j5(self):
        self.assertTrue(self.created_with({"model": "D405", "mount_to_cam": MOUNT,
                                           "position_range_m": np.array(RANGE)})["anchor_joint5"])
        self.assertTrue(self.created_with(None, version="1.3")["anchor_joint5"])


class TestStep15StaysInTheRange(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.dict(CONFIG_PATHS, txt_dir=folder.name, result_dir=folder.name, plot_dir=folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def solve(self, true_height_offset_m, position_range_m, optical_deg=0.0):
        mount_r = Rotation.from_euler('xyz', MOUNT[3:], degrees=True).as_matrix()
        true_r = mount_r @ Rotation.from_euler('z', optical_deg, degrees=True).as_matrix()
        true_t = np.array(MOUNT[:3]) + [0.0, 0.0, true_height_offset_m]
        marker = np.array([.30, -.04, .02])

        def head_r(p, t):
            return (Rotation.from_euler('z', p, degrees=True).as_matrix()
                    @ Rotation.from_euler('y', t, degrees=True).as_matrix())

        def phase(joint, name):
            samples = []
            for a in np.linspace(-10., 10., 21):
                pan, tilt = (0., a) if joint == 1 else (a, 0.)
                samples.append({"t": None, "side": "right", "deg": float(a),
                                "p": true_r.T @ (head_r(pan, tilt).T @ marker - true_t)})
            return {"joint": joint, "name": name, "enc_t": None, "enc_deg": None, "missing": 0, "samples": samples}

        solver = HeadCameraCalibrator.__new__(HeadCameraCalibrator)
        solver.robot = SimpleNamespace(model=lambda: SimpleNamespace(head_idx=[0, 1]), get_dynamics=lambda: None,
                                       get_state=lambda: SimpleNamespace(position=np.zeros(2)))

        def fk(robot, dynamics, q, ee_link, base_link):
            transform = np.eye(4)
            transform[:3, :3] = head_r(*np.rad2deg(q))
            return transform

        with patch.object(BaseCalibrator, 'compute_fk', side_effect=fk):
            return solver._solve_head_camera([phase(1, "Tilt"), phase(0, "Pan")], MOUNT, mount_r,
                                             marker_priors={"right": marker}, position_range_m=position_range_m)

    def test_a_camera_higher_than_the_range_allows_ends_at_the_range_end_and_says_so(self):
        result = self.solve(0.004, RANGE)
        camera = result["calibrated_mount_to_cam"]
        self.assertAlmostEqual(camera[2], MOUNT[2] + 0.001, places=6)
        self.assertAlmostEqual(camera[1], MOUNT[1], places=6)
        self.assertLessEqual(camera[0], MOUNT[0] + 0.001 + 1e-6)
        self.assertGreaterEqual(camera[0], MOUNT[0] - 0.003 - 1e-6)
        # Only the optical-axis rotation (Euler pitch for this mount) may move; the head-axis ones stay.
        self.assertEqual((result["cam_rot_diff_deg"]["roll"], result["cam_rot_diff_deg"]["yaw"]), (0.0, 0.0))
        self.assertIn("z at the upper end (+1.0 mm)", result["camera_at_range_end"])

    def test_a_camera_turned_about_its_optical_axis_is_found_inside_the_range(self):
        result = self.solve(0.0, RANGE, optical_deg=-0.65)
        mount_r = Rotation.from_euler('xyz', MOUNT[3:], degrees=True).as_matrix()
        found = Rotation.from_euler('ZYX', result["calibrated_mount_to_cam"][3:][::-1], degrees=True).as_matrix()
        np.testing.assert_allclose(np.degrees(Rotation.from_matrix(mount_r.T @ found).as_rotvec()), [0.0, 0.0, -0.65],
                                   atol=1e-3)
        np.testing.assert_allclose(result["calibrated_mount_to_cam"][:3], MOUNT[:3], atol=2e-5)

    def test_a_camera_at_its_design_pose_is_inside_the_range(self):
        result = self.solve(0.0, RANGE)
        np.testing.assert_allclose(result["calibrated_mount_to_cam"][:3], MOUNT[:3], atol=2e-5)
        self.assertEqual(result["camera_at_range_end"], [])

    def test_without_a_range_the_result_has_no_range_report(self):
        result = self.solve(0.004, None)
        self.assertNotIn("camera_at_range_end", result)
        self.assertAlmostEqual(result["calibrated_mount_to_cam"][2], MOUNT[2] + 0.004, places=4)


if __name__ == "__main__":
    unittest.main()
