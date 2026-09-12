"""Two offline regression contracts, adapted from backup ccfd791 tests.

Intentionally expose unresolved production defects; no expectedFailure/skip.
No robot client, camera, UI, motion, or production configuration writes.
"""
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.paths import CONFIG_PATHS


class StepContracts(unittest.TestCase):
    def test_head_solution_preserves_stationary_marker_in_3d(self):
        # Independent intersecting Z-pan/Y-tilt chain from v1.2 URDF.
        # An unknown stationary marker avoids assuming calibrated arm FK.
        nominal = [.047, .009, .057, -90., 0., -90.]
        mount_r = Rotation.from_euler('xyz', nominal[3:], degrees=True).as_matrix()
        mount_t = np.array(nominal[:3])
        marker = np.array([.30, -.04, .02])
        tilt_zero_deg = -2.
        tilt = np.linspace(-10., 10., 11)
        pan = np.linspace(-15., 15., 11)
        angles = [(0., t) for t in tilt] + [(p, 0.) for p in pan]

        def head_r(p, t):
            return (Rotation.from_euler('z', p, degrees=True).as_matrix()
                    @ Rotation.from_euler('y', t, degrees=True).as_matrix())

        observations = np.array([
            mount_r.T @ (head_r(p, t + tilt_zero_deg).T @ marker - mount_t)
            for p, t in angles
        ])
        solver = HeadCameraCalibrator.__new__(HeadCameraCalibrator)
        result = solver._compute_head_camera_solution(
            observations[:11], observations[11:], tilt, pan, nominal, mount_r)
        fitted = result['calibrated_mount_to_cam']
        fitted_r = Rotation.from_euler('xyz', fitted[3:], degrees=True).as_matrix()
        fitted_t = np.array(fitted[:3])
        offsets = result['head_offsets_deg']
        reconstructed = np.array([
            head_r(p + offsets['pan'], t + offsets['tilt']) @ (fitted_r @ obs + fitted_t)
            for (p, t), obs in zip(angles, observations)
        ])
        rms_mm = float(np.sqrt(np.mean(np.sum(
            (reconstructed - reconstructed.mean(axis=0)) ** 2, axis=1))) * 1000)
        effective_t = head_r(0., tilt_zero_deg) @ mount_t
        print(json.dumps({'head_stationary_rms_mm': rms_mm,
                          'reported_quality': result['quality'],
                          'head_offsets_deg': offsets,
                          'effective_translation_m': effective_t.tolist(),
                          'returned_translation_m': fitted_t.tolist()}), flush=True)
        self.assertTrue(result['success'])
        # 0.01 mm is a numerical contract for noiseless inputs, not a sensor specification.
        self.assertLess(rms_mm, .01,
                        'Head/camera gauge must preserve full SE(3), including translation')

    def test_step1_constant_absolute_j6_target_converges_to_target(self):
        # Backup tests/test_joint_retry.py isolates the iterator from hardware.
        # Here the CURRENT API returns an absolute J6 target, not a residual.
        cal = JointCalibrator.__new__(JointCalibrator)
        cal.robot = None
        cal.joint_offsets = {'right': {'wrist_yaw2': -2.}}
        cal.use_angle_based_fitting = True
        stages = []

        def measure(*args, **kwargs):
            stages.append(float(kwargs['current_offset_deg']))
            return {'optimal_offset': -1., 'angle_between_normals': 90.,
                    'r_A': 50., 'r_B': 50., 'center_dist': 0.}

        cal.perform_calibration_sweep_continuous = measure
        cal.save_calibration_comparison_plot = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory() as folder, patch.dict(CONFIG_PATHS, txt_dir=folder):
            result = cal.perform_joint_calibration('right', 'wrist_yaw2', current_offset_deg=-2.)
        print(json.dumps({'j6_staged_offsets_deg': stages,
                          'returned_offset_deg': result['recommended_joint_offset'],
                          'converged': result['converged']}), flush=True)
        self.assertAlmostEqual(result['recommended_joint_offset'], -1., places=10,
                               msg='Damp target minus current, never the absolute target itself')
        self.assertTrue(result['converged'])


if __name__ == '__main__':
    unittest.main()
