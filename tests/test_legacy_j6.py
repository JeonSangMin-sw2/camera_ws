"""September 2 J6 restoration: offline capture and numerical contracts."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation


class LegacyJ6Tests(unittest.TestCase):
    def data(self, staged=-1., physical=2.):
        point = np.array([.06, .03, .18])
        a, b, qa, qb = [], [], [], []
        for angle in np.linspace(-15., 15., 61):
            q = np.zeros(7); q[6] = np.deg2rad(angle + staged)
            r = Rotation.from_euler('z', angle+staged+physical, degrees=True).as_matrix()
            t = np.eye(4); t[:3,:3] = r; t[:3,3] = r @ point
            a.append(t); qa.append(q)
        for angle in np.linspace(-10., 10., 61):
            q = np.zeros(7); q[5] = np.deg2rad(angle); q[6] = np.deg2rad(staged)
            r = (Rotation.from_euler('y', angle, degrees=True).as_matrix()
                 @ Rotation.from_euler('z', staged+physical, degrees=True).as_matrix())
            t = np.eye(4); t[:3,:3] = r; t[:3,3] = r @ point
            b.append(t); qb.append(q)
        return np.array(a), np.array(qa), np.array(b), np.array(qb)

    def calculate(self, data=None):
        from core.calibration.legacy_j6 import estimate_legacy_j6
        return estimate_legacy_j6(*(self.data() if data is None else data),
            arm_indices=list(range(7)), initial_arm=np.deg2rad([0,0,0,0,0,0,-1]),
            axis_a=np.array([0.,0.,1.]), axis_b=np.array([0.,1.,0.]),
            reference_rotation=np.eye(3), staged_offset_deg=-1.)

    def test_encoder_fit_and_legacy_absolute_to_relative_conversion(self):
        result = self.calculate()
        self.assertTrue(result['measurement_accepted'], result)
        self.assertAlmostEqual(result['raw_diff_deg'], -1., places=3)
        self.assertAlmostEqual(result['legacy_absolute_offset_deg'], -1.8, places=3)
        self.assertAlmostEqual(result['optimal_offset'], -.8, places=3)
        self.assertFalse(result['converged'])

    def test_old_midpoint_policy_not_new_frame_consistency_gate(self):
        data = list(self.data())
        # Keep positions and midpoint unchanged, corrupt other rotations.
        for index in range(len(data[2])):
            if index != len(data[2])//2:
                data[2][index,:3,:3] @= Rotation.from_euler('x', 3., degrees=True).as_matrix()
        result = self.calculate(data)
        self.assertTrue(result['measurement_accepted'], result)
        self.assertAlmostEqual(result['optimal_offset'], -.8, places=3)

    def test_missing_or_invalid_encoders_rejected_without_zero_offset(self):
        for invalid in (np.zeros((3,7)), np.full((61,7), np.nan)):
            data = list(self.data()); data[1] = invalid
            result = self.calculate(data)
            self.assertFalse(result['measurement_accepted'])
            self.assertNotIn('optimal_offset', result)

    def test_stationary_observations_request_recovery_without_offset(self):
        poses = np.tile(np.eye(4), (20,1,1))
        encoders = np.zeros((20,7))
        result = self.calculate((poses, encoders, poses, encoders))
        self.assertFalse(result['measurement_accepted'])
        self.assertTrue(result['retryable_observation'])
        self.assertNotIn('optimal_offset', result)

    def test_quality_uses_geometric_point_rms_not_solver_coordinate_rms(self):
        from unittest.mock import patch
        from core.calibration.legacy_j6 import LegacyCircleFit
        a,qa,b,qb = self.data()
        a[:,2,3] += .0007 * np.where(np.arange(len(a)) % 2, 1., -1.)
        fits = [dict(c_opt=np.array([0.,0.,180.]), axis_opt=np.array([0.,0.,1.]),
                     radius=np.hypot(60.,30.), rmse=.4),
                dict(c_opt=np.zeros(3), axis_opt=np.array([0.,1.,0.]), radius=190., rmse=.1)]
        with patch.object(LegacyCircleFit, 'fit_circle_3d_and_6dof_misalignment', side_effect=fits):
            result = self.calculate((a,qa,b,qb))
        self.assertFalse(result['measurement_accepted'])
        self.assertTrue(result['retryable_observation'])
        self.assertIn('RMS=0.700', result['failure_reason'])

    def test_fk_adapter_uses_saved_reference_and_relative_contract(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from core.calibration.JointCalibrator import JointCalibrator
        cal = JointCalibrator.__new__(JointCalibrator)
        cal.robot = SimpleNamespace(model=lambda: SimpleNamespace(right_arm_idx=list(range(7))),
                                    get_dynamics=lambda: object())
        cal.get_ready_pose = lambda *args: np.zeros(7)
        cal.is_head_active = lambda: False
        cal.camera_config = {'head_base_to_cam': [0.]*6,
                             'Tf_to_marker_right_v12': [0.]*6}
        cal.compute_fk = Mock(return_value=np.eye(4))
        data = self.data()
        result = cal.compute_legacy_j6_results('right', *data,
                    np.deg2rad([0,0,0,0,0,0,-1]), -1.)
        self.assertTrue(result['measurement_accepted'], result)
        self.assertAlmostEqual(result['optimal_offset'], -.8, places=3)
        self.assertEqual(result['bracket_reference_source'], 'Tf_to_marker_right_v12')
        self.assertEqual([call.args[3] for call in cal.compute_fk.call_args_list],
                         ['link_right_arm_5', 'link_right_arm_4', 'link_head_0'])


if __name__ == '__main__': unittest.main()
