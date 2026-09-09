"""Read-only regression suite. Optional CALIBRATION_ROBOT_ADDRESS connects SDK
kinematics only; no power, servo, motion, or home-offset writes are issued.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from core.calibration_optimizer import QPCalibrationOptimizer, make_transform, se3_exp, se3_log
from core.calibration_core import load_npz_dataset, save_npz_dataset
from core.marker_detection import SimulationModel
from core.marker_detection import Marker_Transform
from core.homeoffset_core import load_offset_from_json
from calibration_support import OfflineRobot, data, transform_vector, optimizer








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
        provider = Marker_Transform(sim=True)
        metadata = provider.intrinsics_metadata
        self.assertEqual(metadata['calibration_temperature_c'], 38)
        self.assertNotIn('connected_serial', metadata)
        self.assertEqual(metadata['file'], str(path))
        self.assertEqual(metadata['source'], 'config/camera_intrinsics.yaml')
        provider.width = 640
        with self.assertRaises(ValueError):
            provider.reload_intrinsics()

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
        a = data(robot, sim, 3)[2]
        # Estimator switches cannot mutate an immutable sensor session.
        optimizer(robot, sim, head=True, free_camera=True)
        optimizer(robot, sim, head=False, free_camera=False)
        b = data(robot, sim, 3)[2]
        np.testing.assert_array_equal(a, b)
        from core.marker_detection import Marker_Transform
        provider = Marker_Transform(sim=True, robot=robot)
        q = robot.get_state().position
        np.testing.assert_allclose(provider.simulation_model.marker_pose(robot, q, 'right', noisy=False), sim.marker_pose(robot, q, 'right', noisy=False))

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
                        samples.append(sim.marker_pose(robot, q, side, noisy=False))
                    return samples
                with contextlib.redirect_stdout(io.StringIO()):
                    j6 = joint.compute_calibration_results(side, mode, sweep(6, ready), sweep(5, ready))
                correction = j6['optimal_offset']
                j6_references[side] = -correction
                marker.joint_offsets[side]['wrist_roll' if version == '1.3' else 'wrist_yaw2'] = correction
                center = marker.get_ready_pose(f'v{version}', 'marker', '', side)
                declared_j6 = float(np.rad2deg(center[6]))
                center[6] += np.deg2rad(correction)
                sweeps = []
                for axis in (4, 5, 6):
                    samples = sweep(axis, center)
                    observed = marker.fit_observed_circle(samples)
                    observed.update(captured_poses=samples, commanded_reference_j6_deg=declared_j6)
                    sweeps.append(observed)
                # Joint estimation is separate from bracket fitting. This
                # algebraic test supplies known J5; connected tests measure it.
                j5 = np.rad2deg(sim.arm_offsets(side)[5])
                fitted = marker.fit_observed_bracket(*sweeps, side)
                self.assertTrue(fitted['measurement_accepted'], fitted)
                effective_brackets[side] = [fitted[k]/1000 for k in ('x_e', 'y_e', 'z_e')] + [fitted[k] for k in ('roll_e', 'pitch_e', 'yaw_e')]
                self.assertEqual(fitted['data_rank'], 3)
                self.assertNotIn('opt_delta_5', fitted)
                self.assertLess(fitted['axis_intersection_rms_mm'], .0001)
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
