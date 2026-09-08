"""No robot connection or configuration writes; recorded and analytic trajectories."""
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from core.calibration.joint_reference import estimate_j6_reference


def sweep(axis, orientation, n=61):
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i, angle in enumerate(np.linspace(-15, 15, n)):
        poses[i, :3, :3] = Rotation.from_rotvec(np.deg2rad(angle)*np.array(axis)).as_matrix() @ orientation
    return poses


class J6ReferenceTests(unittest.TestCase):
    def calculate(self, a=None, b=None, encoder=-1.):
        # Physical J6 = encoder(-1) + unknown error(+2) = +1 degree.
        R0 = Rotation.from_euler('z', 1., degrees=True).as_matrix()
        return estimate_j6_reference(
            sweep([0,0,1], R0) if a is None else a, [0,0,1],
            sweep([0,1,0], R0) if b is None else b, [0,1,0],
            np.eye(3), encoder)

    def test_recovers_absolute_correction_not_staged_or_zero(self):
        result = self.calculate()
        self.assertTrue(result['measurement_accepted'])
        self.assertAlmostEqual(result['optimal_offset'], -2., places=6)

    def test_one_bad_middle_rotation_does_not_determine_offset(self):
        R0 = Rotation.from_euler('z', 1., degrees=True).as_matrix()
        b = sweep([0,1,0], R0)
        b[len(b)//2,:3,:3] = b[len(b)//2,:3,:3] @ Rotation.from_euler('z', 8., degrees=True).as_matrix()
        result = self.calculate(b=b)
        self.assertTrue(result['measurement_accepted'])
        self.assertAlmostEqual(result['optimal_offset'], -2., places=6)

    def test_competing_branches_are_not_averaged_into_success(self):
        R0 = Rotation.from_euler('z', 1., degrees=True).as_matrix()
        b = sweep([0,1,0], R0)
        b[30:,:3,:3] = b[30:,:3,:3] @ Rotation.from_euler('z', 8., degrees=True).as_matrix()
        result = self.calculate(b=b)
        self.assertFalse(result['measurement_accepted'])
        self.assertNotIn('optimal_offset', result)

    def test_constant_bracket_gauge_is_not_claimed_as_physical_truth(self):
        # Constant coaxial +3 degree bracket rotation is indistinguishable
        # from J6 rotation. Effective correction must be -5, not GT -2.
        R0 = Rotation.from_euler('z', 4., degrees=True).as_matrix()
        result = self.calculate(a=sweep([0,0,1], R0), b=sweep([0,1,0], R0))
        self.assertTrue(result['measurement_accepted'])
        self.assertAlmostEqual(result['optimal_offset'], -5., places=6)
        self.assertEqual(result['j6_mode'], 'effective_bracket_reference')

    def test_invalid_and_degenerate_inputs_never_return_an_offset(self):
        for value in (np.empty((0,4,4)), np.full((20,4,4), np.nan)):
            result = self.calculate(b=value)
            self.assertFalse(result['measurement_accepted'])
            self.assertNotIn('optimal_offset', result)
        result = estimate_j6_reference(sweep([0,0,1], np.eye(3)), [0,0,0],
            sweep([0,1,0], np.eye(3)), [0,1,0], np.eye(3), 0.)
        self.assertFalse(result['measurement_accepted'])

    def test_saved_passes_reject_inconsistent_raw_rotation_without_encoders(self):
        data = np.loadtxt(Path(__file__).parent/'fixtures/j6_rotation_discontinuity.csv', delimiter=',')
        nominal = Rotation.from_euler('xyz', [90,0,180], degrees=True).as_matrix()
        for pass_index in (1,2,3):
            with self.subTest(pass_index=pass_index):
                sweeps=[]
                for axis in (6,5):
                    rows=data[(data[:,0]==axis)&(data[:,1]==pass_index)]
                    poses=np.tile(np.eye(4),(len(rows),1,1))
                    poses[:,:3,:4]=rows[:,3:].reshape(-1,3,4)
                    # Camera POSITION plane normal only, no encoder angles.
                    points=poses[:,:3,3]
                    normal=np.linalg.svd(points-points.mean(axis=0))[2][-1]
                    sweeps.extend([poses,normal])
                result=estimate_j6_reference(*sweeps, nominal, 0.)
                self.assertFalse(result['measurement_accepted'])
                self.assertNotIn('optimal_offset',result)

    def test_actual_joint_calculator_with_sdk_kinematics_and_known_j6(self):
        # Actual compute_calibration_results and actual circle fitting, not
        # mocked axes. Only observations are synthetic; no SDK connection.
        import yaml
        from test_calibration_regression import OfflineRobot
        from core.calibration.JointCalibrator import JointCalibrator
        from core.simulation_model import SimulationModel
        from core.paths import CONFIG_PATHS
        for version in ('1.2', '1.3'):
            for mount in ('head', 'fixed'):
                for side in ('right', 'left'):
                    with self.subTest(version=version, mount=mount, side=side):
                        robot = OfflineRobot(version)
                        config = SimulationModel.create(version).config
                        config['camera_mount_mode'] = mount
                        config['offsets'][side]['bracket_rpy'] = [0.,0.,0.]
                        config['offsets'][side]['bracket_pos'] = [0.,0.,0.]
                        simulation = SimulationModel.create(version, config)
                        cal = JointCalibrator.__new__(JointCalibrator)
                        cal.robot, cal.robot_version = robot, version
                        cal.marker_st = None
                        cal.include_head_motion = mount == 'head'
                        cal.camera_config = dict(config)
                        cal.camera_config[f'Tf_to_marker_{side}_v{version.replace(".", "")}'] = config['brackets'][version][side]
                        cal.joint_offsets = {side: {}}
                        with open(CONFIG_PATHS['ready_poses_yaml']) as stream:
                            cal.ready_poses = yaml.safe_load(stream)
                        mode = 'wrist_yaw2' if version == '1.2' else 'wrist_roll_v13'
                        center = cal.get_ready_pose('v'+version, 'joint', mode, side)
                        indices = getattr(robot.model(), side+'_arm_idx')
                        datasets = []
                        for joint in (6,5):
                            samples = []
                            for angle in np.linspace(-15.,15.,41):
                                q = robot.get_state().position.copy()
                                q[indices] = center
                                q[indices[joint]] += np.deg2rad(angle)
                                samples.append((q, simulation.marker_pose(robot, q, side, noisy=False)))
                            datasets.append(samples)
                        result = cal.compute_calibration_results(side, mode, *datasets, center)
                        self.assertTrue(result['measurement_accepted'], result)
                        self.assertAlmostEqual(result['optimal_offset'],
                            -config['offsets'][side]['joint6'], places=3)


if __name__ == '__main__':
    unittest.main()
