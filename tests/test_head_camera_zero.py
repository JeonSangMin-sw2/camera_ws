"""Camera-zero math and SDK kinematics tests; never command robot motion."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.calibration_optimizer import compute_fk, make_transform
from core.head_camera_zero import camera_forward_zero, camera_command_to_encoder
from core.homeoffset_core import load_offset_from_json
from core.simulation_model import SimulationModel
from test_calibration_regression import OfflineRobot, data, optimizer


def head_fk(robot):
    def fk(head):
        q = np.array(robot.get_state().position, copy=True)
        q[robot.model().head_idx] = head
        return compute_fk(robot, robot.get_dynamics(), q, 'link_head_2', 'link_torso_5')[1]
    return fk


class CameraZeroTests(unittest.TestCase):
    def test_full_se3_gauge_and_camera_yaw_command_zero(self):
        for version in ('1.2', '1.3'):
            with self.subTest(version=version):
                robot = OfflineRobot(version)
                fk = head_fk(robot)
                camera = make_transform([.049, .008, .059, -89.4, .7, -89.6])
                old_head = np.deg2rad([.8, -1.5])
                h, c, reference = camera_forward_zero(fk, camera, old_head, [-1., -1.], [1., 1.])
                zero = camera_command_to_encoder([0., 0.], reference)
                np.testing.assert_allclose((fk(zero + h) @ c)[:3, 2], [1, 0, 0], atol=1e-10)
                for enc in ([0., 0.], [.2, -.3], [-.18, .27]):
                    np.testing.assert_allclose(fk(enc + old_head) @ camera, fk(enc + h) @ c, atol=1e-10)
                self.assertEqual(h[0], old_head[0])  # Pan is NOT a free camera gauge.
                self.assertGreater(abs(zero[0] + old_head[0]), 1e-4)  # Camera yaw matters.
                self.assertGreater(np.linalg.norm(c[:3, 3] - camera[:3, 3]), 1e-5)
                self.assertFalse(reference['independent_physical_offset'])
                self.assertFalse(reference['image_roll_corrected'])

    def test_known_reference_is_not_redistributed(self):
        robot = OfflineRobot()
        c = make_transform([.047, .009, .057, -89.4, .7, -89.6])
        h = np.deg2rad([.8, -1.5])
        new_h, new_c, ref = camera_forward_zero(head_fk(robot), c, h, [-1., -1.], [1., 1.], False)
        np.testing.assert_array_equal(new_h, h)
        np.testing.assert_array_equal(new_c, c)
        self.assertTrue(ref['accepted'])

    def test_unreachable_or_degenerate_reference_is_rejected(self):
        fk = head_fk(OfflineRobot())
        for camera in (np.eye(4), make_transform([0, 0, 0, -90, 0, 90])):
            with self.assertRaises(ValueError):
                camera_forward_zero(fk, camera, np.zeros(2), [-1., -1.], [1., 1.])
        with self.assertRaises(ValueError):
            camera_forward_zero(lambda _: np.eye(4), np.eye(4), np.zeros(2), [-1., -1.], [1., 1.])

    def test_command_validation_and_sign(self):
        ref = {'accepted': True, 'reference_frame': 'link_torso_5', 'encoder_zero_deg': [-.8, 1.5]}
        np.testing.assert_allclose(camera_command_to_encoder(np.deg2rad([2., -3.]), ref), np.deg2rad([1.2, -1.5]))
        for cmd in ([np.nan, 0], [0], [0, 0, 0]):
            with self.assertRaises(ValueError):
                camera_command_to_encoder(cmd, ref)
        with self.assertRaises(ValueError):
            camera_command_to_encoder([0, 0], {**ref, 'accepted': False})

    def test_camera_gauge_never_exports_mechanical_head_home(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            # Also guard a missing or accidentally true legacy top-level flag.
            for flag in (None, True, False):
                result = {'joint_offset_deg': [0.]*14, 'head_joint_offset_deg': [.8, -1.5],
                          'diagnostics': {'converged': True, 'observable': True, 'head_tilt_mode': 'camera_forward_gauge'}}
                if flag is not None:
                    result['head_tilt_independent'] = flag
                with path.open('w') as f:
                    json.dump(result, f)
                self.assertIsNone(load_offset_from_json(path)[1])

    def check_optimizer(self, robot, version):
        cfg = SimulationModel.create(version).config
        cfg['mount_to_cam'] = [.048, .008, .058, -89.6, .3, -89.7]
        sim = SimulationModel.create(version, cfg)
        qa, qh, observations = data(robot, sim, 24, noise=False)
        old = optimizer(robot, sim)
        new = optimizer(robot, sim)
        new.head_zero_convention = 'camera_forward'
        # Both solves start from CAD, not from the injected camera assembly error.
        for opt in (old, new):
            opt.T_mount_to_cam_nom = make_transform([.047, .009, .057, -90, 0, -90])
        with contextlib.redirect_stdout(io.StringIO()):
            a0, h0, x0, _, _ = old.optimize(qa, qh, observations)
            a1, h1, x1, camera, _ = new.optimize(qa, qh, observations)
        np.testing.assert_allclose(a0, a1, atol=1e-12)
        self.assertEqual(new.last_diagnostics['head_tilt_mode'], 'camera_forward_gauge')
        self.assertFalse(new.last_diagnostics['head_tilt_independent'])
        self.assertGreater(abs(h1[1]), .005)
        for i in range(len(qa)):
            for side in ('right', 'left'):
                np.testing.assert_allclose(old.evaluate_sample(qa[i], qh[i], side, a0, h0, x0)[3],
                                           new.evaluate_sample(qa[i], qh[i], side, a1, h1, x1)[3], atol=1e-9)
        zero = camera_command_to_encoder([0, 0], new.camera_forward_zero)
        np.testing.assert_allclose((head_fk(robot)(zero + h1) @ make_transform(camera))[:3, 2], [1, 0, 0], atol=1e-9)
        # Independent ground-truth chain validates pointing, not merely the fitted model.
        physical = np.deg2rad([cfg['offsets']['head']['pan'], cfg['offsets']['head']['tilt']])
        direction = (head_fk(robot)(zero + physical) @ make_transform(cfg['mount_to_cam']))[:3, 2]
        np.testing.assert_allclose(direction, [1, 0, 0], atol=1e-7)
        print(json.dumps({'camera_zero_version': version, 'encoder_zero_deg': new.camera_forward_zero['encoder_zero_deg'],
                          'effective_head_deg': np.rad2deg(h1).tolist(), 'truth_optical_direction': direction.tolist()}))

    def test_optimizer_output_invariance(self):
        for version in ('1.2', '1.3'):
            with self.subTest(version=version):
                self.check_optimizer(OfflineRobot(version), version)

    def test_no_head_output_or_reference_overwrite(self):
        for head in (False, True):
            sim, robot = SimulationModel.create(), OfflineRobot()
            if not head:
                cfg = sim.config
                cfg['camera_mount_mode'] = 'fixed'
                sim = SimulationModel.create(config=cfg)
            qa, qh, obs = data(robot, sim, 24)
            opt = optimizer(robot, sim, head=head, reference=np.deg2rad(-1.5) if head else None)
            opt.head_zero_convention = 'camera_forward'
            with contextlib.redirect_stdout(io.StringIO()):
                _, h, *_ = opt.optimize(qa, qh if head else None, obs)
            if head:
                self.assertEqual(h[1], np.deg2rad(-1.5))
                self.assertTrue(opt.last_diagnostics['head_tilt_independent'])
                self.assertEqual(opt.last_diagnostics['head_tilt_mode'], 'independent_reference')
            else:
                self.assertIsNone(h)
                self.assertIsNone(opt.camera_forward_zero)

    @unittest.skipUnless(os.environ.get('CALIBRATION_ROBOT_ADDRESS'), 'opt-in read-only SDK connection')
    def test_connected_sdk_camera_zero(self):
        import rby1_sdk as rby
        robot = rby.create_robot(os.environ['CALIBRATION_ROBOT_ADDRESS'], 'm')
        self.assertTrue(robot.connect(3))
        self.check_optimizer(robot, robot.get_robot_info().robot_model_version.removeprefix('v'))


if __name__ == '__main__':
    unittest.main()
