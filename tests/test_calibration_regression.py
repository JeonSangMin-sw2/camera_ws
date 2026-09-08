"""Read-only regression suite. Optional CALIBRATION_ROBOT_ADDRESS connects SDK
kinematics only; no power, servo, motion, or home-offset writes are issued.
"""
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rby1_sdk.dynamics as rd
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from core.calibration_optimizer import QPCalibrationOptimizer, make_transform, se3_exp, se3_log
from core.calibration_core import load_npz_dataset, save_npz_dataset, generate_sim_measurements
from core.simulation_model import SimulationModel
from core.camera_intrinsics import select_intrinsics
from core.homeoffset_core import load_offset_from_json


class OfflineRobot:
    def __init__(self, version='1.2'):
        self.urdf = Path(os.environ.get('RBY1_MODEL_DIR', '/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf')) / f'model_v{version}.urdf'
        self.dynamics = rd.Robot(rd.load_robot_from_urdf(str(self.urdf), 'base'))
        names = self.dynamics.get_joint_names()
        self.meta = SimpleNamespace(robot_joint_names=names,
            right_arm_idx=[names.index(f'right_arm_{i}') for i in range(7)],
            left_arm_idx=[names.index(f'left_arm_{i}') for i in range(7)],
            head_idx=[names.index(f'head_{i}') for i in range(2)])
        self.state = SimpleNamespace(position=np.zeros(len(names)))
    def model(self): return self.meta
    def get_state(self): return self.state
    def get_dynamics(self): return self.dynamics


def transform_vector(T):
    return [*T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)]


def data(robot, simulation, count=32, noise=False):
    rng = np.random.default_rng(991)
    model = robot.model()
    arms = list(model.right_arm_idx) + list(model.left_arm_idx)
    qa = rng.uniform(-1.4, 1.4, (count, 14))
    qh = rng.uniform(-0.3, 0.3, (count, 2))
    observations = []
    for a, h in zip(qa, qh):
        q = np.array(robot.get_state().position, copy=True)
        q[arms], q[model.head_idx] = a, h
        observations.append([simulation.marker_pose(robot, q, side, rng, noisy=noise) for side in ('right', 'left')])
    return qa, qh, np.asarray(observations)


def optimizer(robot, simulation, head=True, free_camera=True, reference=None):
    m, cfg = robot.model(), simulation.config
    return QPCalibrationOptimizer(robot=robot,
        arm_idx=list(m.right_arm_idx) + list(m.left_arm_idx),
        head_idx=m.head_idx, use_head_kinematics=head, optimize_head=head,
        optimize_camera=free_camera, head_tilt_reference_rad=reference,
        head_zero_convention='effective_zero',
        ee_links={s: f'ee_{s}' for s in ('right', 'left')},
        ee_to_marker_nom={s: transform_vector(simulation.bracket_transform(s)) for s in ('right', 'left')},
        mount_to_cam_nom=cfg['mount_to_cam'], head_base_to_cam_nom=cfg['head_base_to_cam'],
        lambda_cam_pos=0, lambda_cam_rot=0, estimate_measurement_noise=False,
        camera_pos_bound_m=.01, camera_rot_bound_rad=np.deg2rad(3), max_iter=80,
        qp_kwargs={'eps_abs': 1e-9, 'eps_rel': 1e-9, 'max_iter': 20000})


class MathTests(unittest.TestCase):
    def test_se3_against_matrix_exponential(self):
        for angle in (0, 1e-10, 1e-6, 1e-4, .1, np.pi - 1e-7):
            xi = np.r_[np.array([1., 2., 3.]) / np.sqrt(14) * angle, [.001, .002, -.003]]
            w = xi[:3]
            algebra = np.zeros((4, 4))
            algebra[:3, :3] = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]
            algebra[:3, 3] = xi[3:]
            np.testing.assert_allclose(se3_exp(xi), expm(algebra), atol=2e-12)
            np.testing.assert_allclose(se3_log(se3_exp(xi)), xi, atol=2e-10)

    def test_dataset_metadata_and_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.npz'
            save_npz_dataset(path, np.zeros((1, 14)), np.eye(4)[None], metadata={'source': 'test'})
            self.assertEqual(load_npz_dataset(path, True)[3]['source'], 'test')
            np.savez(path, q=np.zeros((1, 14)), marker=np.eye(4)[None])
            self.assertEqual(load_npz_dataset(path, True)[3]['schema_version'], 0)

    def test_intrinsics_selection(self):
        path = Path(__file__).resolve().parents[1] / 'config/camera_intrinsics.yaml'
        _, _, metadata = select_intrinsics('calibrated', [0, 0, 1, 1], np.zeros(5), 1280, 720, path)
        self.assertEqual(metadata['calibration_temperature_c'], 38)
        self.assertNotIn('connected_serial', metadata)
        for alias in ('per_device', 'transferred'):
            _, _, legacy = select_intrinsics(alias, [], [], 1280, 720, path)
            self.assertEqual(legacy['source'], 'calibrated')
        with self.assertRaises(ValueError):
            select_intrinsics('calibrated', [], [], 640, 480, path)

    def test_gauge_head_home_export_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            with path.open('w') as stream:
                json.dump({'joint_offset_deg': [1]*14, 'head_joint_offset_deg': [1, 0], 'head_tilt_independent': False}, stream)
            arm, head = load_offset_from_json(path)
            self.assertIsNone(head)
            np.testing.assert_allclose(arm, np.deg2rad(np.ones(14)))


class KinematicsTests(unittest.TestCase):
    def test_shared_truth_and_optimizer_flag_invariance(self):
        robot, sim = OfflineRobot(), SimulationModel.create()
        before = sim.metadata()['truth_sha256']
        detached = sim.config
        detached['mount_to_cam'][0] += .02
        self.assertEqual(before, sim.metadata()['truth_sha256'])
        qa, qh, _ = data(robot, sim, 3)
        kwargs = dict(robot=robot, dyn_model=robot.get_dynamics(), q_arm_list=qa, q_head_list=qh,
            arm_idx=robot.model().right_arm_idx + robot.model().left_arm_idx,
            head_idx=robot.model().head_idx, q_nominal=robot.get_state().position,
            active_arms=['right', 'left'], ee_links={}, mount_to_cam_nom=[], head_base_to_cam_nom=[], ee_to_marker_nom={},
            simulation_model=sim, camera_position_noise_std_m=0, camera_orientation_noise_std_deg=0)
        a = generate_sim_measurements(**kwargs, optimize_arm=True, optimize_head=True, optimize_camera=True)
        b = generate_sim_measurements(**kwargs, optimize_arm=False, optimize_head=False, optimize_camera=False)
        np.testing.assert_array_equal(a, b)
        from core.calibration.CalibratorBase import BaseCalibrator
        base = BaseCalibrator.__new__(BaseCalibrator)
        base.robot, base.robot_version, base.marker_st = robot, '1.2', None
        q = robot.get_state().position
        np.testing.assert_allclose(base.get_simulated_marker_pose('right', noisy=False), sim.marker_pose(robot, q, 'right', noisy=False))

    def test_jacobian_against_independent_difference(self):
        robot, sim = OfflineRobot(), SimulationModel.create()
        opt = optimizer(robot, sim)
        qa, qh, _ = data(robot, sim, 1)
        a, h, x = np.zeros(14), np.zeros(2), np.array([.01, -.02, .015, .001, -.002, .003])
        jb, _, b, T = opt.evaluate_sample(qa[0], qh[0], 'right', a, h, x)
        J = opt.build_jacobian(qa[0], qh[0], 'right', a, h, x, jb, b, T)
        numeric = np.zeros_like(J)
        for i in range(J.shape[1]):
            step = np.zeros(J.shape[1]); step[i] = 1e-5
            aa, hh, xx = opt.unpack_params(step)
            plus = opt.evaluate_sample(qa[0], qh[0], 'right', a+aa, h+hh, x+xx)[3]
            minus = opt.evaluate_sample(qa[0], qh[0], 'right', a-aa, h-hh, x-xx)[3]
            delta = np.linalg.inv(T) @ ((plus-minus) / 2e-5)
            numeric[:, i] = [delta[2,1], delta[0,2], delta[1,0], *delta[:3,3]]
        np.testing.assert_allclose(J, numeric, atol=2e-7)

    def test_head_and_headless_recovery(self):
        for version in ('1.2', '1.3'):
            for head in (True, False):
                with self.subTest(version=version, head=head):
                    robot = OfflineRobot(version)
                    cfg = SimulationModel.create(version).config
                    cfg['camera_mount_mode'] = 'head' if head else 'fixed'
                    sim = SimulationModel.create(version, cfg)
                    qa, qh, obs = data(robot, sim)
                    opt = optimizer(robot, sim, head)
                    with contextlib.redirect_stdout(io.StringIO()):
                        arm, h, _, _, _ = opt.optimize(qa, qh if head else None, obs)
                    truth = np.r_[sim.arm_offsets('right'), sim.arm_offsets('left')]
                    self.assertLess(np.max(np.abs(np.rad2deg(arm-truth))), .001)
                    self.assertTrue(opt.last_diagnostics['converged'])
                    self.assertTrue(opt.last_diagnostics['observable'])
                    if head:
                        self.assertEqual(h[1], 0)
                        self.assertEqual(opt.last_diagnostics['data_rank_before_gauge'], 21)
                    print(json.dumps({'version': version, 'head': head,
                        'max_arm_error_deg': float(np.max(np.abs(np.rad2deg(arm-truth)))), **opt.last_diagnostics}))

    def test_failed_qp_never_falls_back(self):
        robot, sim = OfflineRobot(), SimulationModel.create()
        opt = optimizer(robot, sim)
        qa, qh, obs = data(robot, sim, 2)
        with patch('core.calibration_optimizer.qpsolvers', None), self.assertRaises(RuntimeError):
            opt.compute_step(qa, qh, obs, np.zeros(14), np.zeros(2), np.zeros(6))

    def test_head_sweep_uses_encoder_fit(self):
        from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
        robot, sim = OfflineRobot(), SimulationModel.create()
        solver = HeadCameraCalibrator.__new__(HeadCameraCalibrator)
        solver.robot, solver.camera_config = robot, {}
        t, p = np.linspace(-10, 10, 11), np.linspace(-15, 15, 11)
        def points(angles, axis):
            values = []
            for angle in angles:
                q = robot.get_state().position.copy()
                q[robot.model().head_idx[axis]] = np.deg2rad(angle)
                values.append(sim.marker_pose(robot, q, 'right', noisy=False)[:3, 3])
            return values
        result = solver._compute_head_camera_solution(points(t, 1), points(p, 0), t, p,
            sim.config['mount_to_cam'], make_transform(sim.config['mount_to_cam'])[:3, :3])
        self.assertTrue(result['success'], result)
        self.assertFalse(result['quality']['decoupled'])
        self.assertLess(result['quality']['rmse_3d_marker_mm'], 1e-5)
        self.assertEqual(result['head_offsets_deg']['tilt'], 0)

    def test_step1_bracket_to_step2_pipeline(self):
        from core.calibration.JointCalibrator import JointCalibrator
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        for version in ('1.2', '1.3'):
            robot, sim = OfflineRobot(version), SimulationModel.create(version)
            joint, marker = JointCalibrator(robot=robot), MarkerCalibrator(robot=robot)
            joint.robot_version = marker.robot_version = version
            effective_brackets, j6_references = {}, {}
            for side in ('right', 'left'):
                idx = getattr(robot.model(), f'{side}_arm_idx')
                mode = 'wrist_roll_v13' if version == '1.3' else 'wrist_yaw2'
                ready = joint.get_ready_pose(f'v{version}', 'joint', mode, side)
                def sweep(axis, center):
                    samples = []
                    for angle in np.linspace(-15, 15, 25):
                        q = robot.get_state().position.copy()
                        q[idx] = center
                        q[idx[axis]] += np.deg2rad(angle)
                        samples.append((q, sim.marker_pose(robot, q, side, noisy=False)))
                    return samples
                with contextlib.redirect_stdout(io.StringIO()):
                    j6 = joint.compute_calibration_results(side, mode, sweep(6, ready), sweep(5, ready), ready)
                correction = j6['recommended_joint_offset']
                j6_references[side] = -correction
                marker.joint_offsets[side]['wrist_roll' if version == '1.3' else 'wrist_yaw2'] = correction
                center = marker.get_ready_pose(f'v{version}', 'marker', '', side)
                sweeps = []
                for axis in (4, 5, 6):
                    samples = sweep(axis, center)
                    sweeps.append({'captured_poses': [p for _, p in samples], 'captured_q_full': [q for q, _ in samples]})
                # Joint estimation is separate from bracket fitting. This
                # algebraic test supplies known J5; connected tests measure it.
                j5 = np.rad2deg(sim.arm_offsets(side)[5])
                fitted = marker.fit_encoder_bracket(*sweeps, side, correction, -j5)
                effective_brackets[side] = [fitted[k]/1000 for k in ('x_e', 'y_e', 'z_e')] + [fitted[k] for k in ('roll_e', 'pitch_e', 'yaw_e')]
                self.assertEqual(fitted['data_rank'], 6)
                self.assertNotIn('opt_delta_5', fitted)
                self.assertEqual(fitted['fixed_j5_correction_deg'], -j5)
            qa, qh, obs = data(robot, sim)
            opt = optimizer(robot, sim)
            opt.ee_to_marker_nom = effective_brackets
            from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
            head_solver = HeadCameraCalibrator(robot=robot)
            tilt_angles, pan_angles = np.linspace(-10, 10, 11), np.linspace(-15, 15, 11)
            head_points = []
            for axis, angles in ((1, tilt_angles), (0, pan_angles)):
                points = []
                for angle in angles:
                    q = robot.get_state().position.copy()
                    q[robot.model().head_idx[axis]] = np.deg2rad(angle)
                    points.append(sim.marker_pose(robot, q, 'right', noisy=False)[:3, 3])
                head_points.append(points)
            res15 = head_solver._compute_head_camera_solution(*head_points, tilt_angles, pan_angles,
                sim.config['mount_to_cam'], make_transform(sim.config['mount_to_cam'])[:3, :3])
            self.assertTrue(res15['success'])
            opt.T_mount_to_cam_nom = make_transform(res15['calibrated_mount_to_cam'])
            with contextlib.redirect_stdout(io.StringIO()):
                arm, *_ = opt.optimize(qa, qh, obs)
            truth = np.r_[sim.arm_offsets('right'), sim.arm_offsets('left')]
            errors = np.rad2deg(arm-truth)
            self.assertLess(np.max(np.abs(np.delete(errors, [6, 13]))), .01)
            self.assertLess(np.max(np.abs(errors[[6, 13]])), .3)
            print(json.dumps({'pipeline': 'step1_j6_and_bracket_step1_5_step2', 'version': version,
                              'arm_error_deg': errors.tolist(), 'j6_effective_reference_deg': j6_references}))

    def test_head_disabled_camera_and_noise_stress(self):
        for version in ('1.2', '1.3'):
            for bracket_deg in (0, .1, 1, 3):
                robot = OfflineRobot(version)
                cfg = SimulationModel.create(version).config
                for side in ('right', 'left'):
                    cfg['offsets'][side]['bracket_rpy'] = [bracket_deg, 0, 0]
                # Keep large joint offsets (including 5.4 deg) in every case.
                sim = SimulationModel.create(version, cfg)
                qa, qh, _ = data(robot, sim)
                qh[:] = [.1, -.2]  # physical head exists, but never moves
                rng = np.random.default_rng(700)
                obs = []
                for a, h in zip(qa, qh):
                    q = robot.get_state().position.copy()
                    q[robot.model().right_arm_idx + robot.model().left_arm_idx] = a
                    q[robot.model().head_idx] = h
                    obs.append([sim.marker_pose(robot, q, s, rng) for s in ('right', 'left')])
                opt = optimizer(robot, sim)
                opt.optimize_head = False
                opt.head_tilt_gauge = False
                opt.joint_offset_lower, opt.joint_offset_upper = opt.get_joint_offset_limits()
                with contextlib.redirect_stdout(io.StringIO()):
                    arm, head, *_ = opt.optimize(qa, qh, np.array(obs))
                self.assertIsNone(head)
                truth = np.r_[sim.arm_offsets('right'), sim.arm_offsets('left')]
                self.assertLess(np.max(np.abs(np.rad2deg(arm-truth))), .1)
                self.assertTrue(opt.last_diagnostics['observable'])

    def test_unexcited_data_is_not_observable(self):
        robot, sim = OfflineRobot(), SimulationModel.create()
        qa, qh, obs = data(robot, sim, 1)
        opt = optimizer(robot, sim)
        report = opt.observability_report(qa, qh, obs, np.zeros(14), np.zeros(2), np.zeros(6))
        self.assertFalse(report['observable'])

    def test_fixed_camera_mount_error_is_recovered(self):
        for version in ('1.2', '1.3'):
            robot = OfflineRobot(version)
            cfg = SimulationModel.create(version).config
            cfg['camera_mount_mode'] = 'fixed'
            sim = SimulationModel.create(version, cfg)
            qa, _, obs = data(robot, sim, noise=True)
            opt = optimizer(robot, sim, head=False)
            opt.T_mount_to_cam_nom[0, 3] += .002
            with contextlib.redirect_stdout(io.StringIO()):
                arm, _, _, _, cam = opt.optimize(qa, None, obs)
            truth = np.r_[sim.arm_offsets('right'), sim.arm_offsets('left')]
            self.assertLess(np.max(np.abs(np.rad2deg(arm-truth))), .1)
            self.assertLess(np.linalg.norm(np.array(cam[:3]) - cfg['head_base_to_cam'][:3]), .0001)

    def test_headless_move_strips_head_command(self):
        from core.calibration.CalibratorBase import BaseCalibrator
        from unittest.mock import MagicMock
        base = BaseCalibrator.__new__(BaseCalibrator)
        base.include_head_motion = False
        base.joint_offsets = None
        fake_robot, component = MagicMock(), MagicMock()
        with patch('core.calibration.CalibratorBase.rby.ComponentBasedCommandBuilder', return_value=component):
            base.movej(fake_robot, head=[.1, .2], apply_offsets=False)
        component.set_head_command.assert_not_called()

    @unittest.skipUnless(os.environ.get('CALIBRATION_ROBOT_ADDRESS'), 'opt-in connected SDK read-only test')
    def test_connected_robot(self):
        import rby1_sdk as rby
        robot = rby.create_robot(os.environ['CALIBRATION_ROBOT_ADDRESS'], 'm')
        self.assertTrue(robot.connect(3))
        version = robot.get_robot_info().robot_model_version.removeprefix('v')
        sim = SimulationModel.create(version)
        qa, qh, obs = data(robot, sim, 24, noise=True)
        opt = optimizer(robot, sim, reference=np.deg2rad(-1.5))
        with contextlib.redirect_stdout(io.StringIO()):
            arm, head, *_ = opt.optimize(qa, qh, obs)
        truth = np.r_[sim.arm_offsets('right'), sim.arm_offsets('left')]
        error = float(np.max(np.abs(np.rad2deg(arm-truth))))
        print(json.dumps({'connected_address': os.environ['CALIBRATION_ROBOT_ADDRESS'],
            'robot_version': version, 'noise_std_mm': .1, 'max_arm_error_deg': error,
            'head_error_deg': np.rad2deg(head - [.8*np.pi/180, -1.5*np.pi/180]).tolist(),
            **opt.last_diagnostics}))
        self.assertLess(error, .1)
        self.assertTrue(opt.last_diagnostics['converged'])


if __name__ == '__main__':
    unittest.main()
