"""No robot connection or configuration writes; recorded and analytic trajectories."""
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from core.calibration.JointCalibrator import estimate_j6_reference


def sweep(axis, orientation, n=61):
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i, angle in enumerate(np.linspace(-15, 15, n)):
        poses[i, :3, :3] = Rotation.from_rotvec(np.deg2rad(angle)*np.array(axis)).as_matrix() @ orientation
    return poses


class ObservedCircleTests(unittest.TestCase):
    def trajectory(self, direction=1):
        theta = np.linspace(-.4, .4, 81)[::direction]
        poses = np.tile(np.eye(4), (len(theta), 1, 1))
        poses[:, :3, 3] = np.c_[.08*np.cos(theta), .08*np.sin(theta), np.full(len(theta), .4)]
        return poses

    def test_time_order_and_command_sign_determine_same_positive_axis(self):
        from core.calibration.CalibratorBase import BaseCalibrator
        self.assertTrue(hasattr(BaseCalibrator, 'fit_observed_circle'),
                        'Step1 needs a marker-only circle fitter')
        for direction in (1, -1):
            result = BaseCalibrator.fit_observed_circle(self.trajectory(direction), direction)
            np.testing.assert_allclose(result['center_m'], [0, 0, .4], atol=1e-6)
            np.testing.assert_allclose(result['axis'], [0, 0, 1], atol=1e-7)
            self.assertAlmostEqual(result['radius_m'], .08, places=6)

    def test_rigid_coordinate_change_and_noisy_points(self):
        from core.calibration.CalibratorBase import BaseCalibrator
        self.assertTrue(hasattr(BaseCalibrator, 'fit_observed_circle'))
        poses = self.trajectory()
        rotation = Rotation.from_euler('xyz', [47, -30, 81], degrees=True).as_matrix()
        shift = np.array([.3, -.2, .6])
        poses[:, :3, 3] = poses[:, :3, 3] @ rotation.T + shift
        poses[:, :3, 3] += np.random.default_rng(3).normal(0, .00003, (len(poses), 3))
        result = BaseCalibrator.fit_observed_circle(poses, 1)
        np.testing.assert_allclose(result['center_m'], rotation @ [0, 0, .4] + shift, atol=.0005)
        self.assertGreater(np.dot(result['axis'], rotation[:, 2]), .999)

    def test_degenerate_or_stationary_samples_are_rejected(self):
        from core.calibration.CalibratorBase import BaseCalibrator
        self.assertTrue(hasattr(BaseCalibrator, 'fit_observed_circle'))
        for poses in (np.tile(np.eye(4), (30, 1, 1)), self.trajectory()[:2],
                      np.full((30, 4, 4), np.nan)):
            with self.assertRaises(ValueError):
                BaseCalibrator.fit_observed_circle(poses, 1)

    def test_parallel_but_noncoincident_circles_cannot_converge(self):
        from core.calibration.JointCalibrator import JointCalibrator
        cal = JointCalibrator.__new__(JointCalibrator)
        a, b, c = self.trajectory(), self.trajectory(), self.trajectory()
        b[:, 0, 3] += .1
        c[:, :3, 3] = c[:, :3, 3] @ Rotation.from_euler('x', -90, degrees=True).as_matrix().T
        result = cal.compute_calibration_results('right', 'wrist_pitch', a, b, dataset_C=c)
        self.assertFalse(result['measurement_accepted'])
        b = a.copy()
        b[:, 2, 3] += .1
        result = cal.compute_calibration_results('right', 'wrist_pitch', a, b, dataset_C=c)
        self.assertFalse(result['measurement_accepted'])
        b = a.copy()
        b[:, :2, 3] *= 1.5
        result = cal.compute_calibration_results('right', 'wrist_pitch', a, b, dataset_C=c)
        self.assertFalse(result['measurement_accepted'])

    def test_short_arc_centers_use_the_same_measured_axis_constraints_as_angles(self):
        from core.calibration.JointCalibrator import JointCalibrator
        cal = JointCalibrator.__new__(JointCalibrator)
        theta = np.deg2rad(np.linspace(-15., 15., 161))
        original = np.tile(np.eye(4), (len(theta), 1, 1))
        original[:, :3, 3] = np.c_[.18*np.cos(theta), .18*np.sin(theta), np.full(len(theta), .4)]
        a, b, c = original.copy(), original.copy(), original.copy()
        # Tiny, opposing out-of-plane observation errors on a short arc can
        # move extrapolated independent circle centers by over a millimetre.
        pivot = np.array([.18, 0., .4])
        for poses, tilt in ((a, .2), (b, -.2)):
            poses[:, :3, 3] = (poses[:, :3, 3]-pivot) @ Rotation.from_euler('y', tilt, degrees=True).as_matrix().T + pivot
            self.assertLess(np.max(np.linalg.norm(poses[:, :3, 3]-original[:, :3, 3], axis=1)), .00003)
        c[:, :3, 3] = c[:, :3, 3] @ Rotation.from_euler('y', 90, degrees=True).as_matrix().T
        result = cal.compute_calibration_results('left', 'elbow', a, b, dataset_C=c)
        self.assertTrue(result['measurement_accepted'], result)
        self.assertLess(abs(result['optimal_offset']), .06)
        self.assertLess(result['center_dist'], .5)

    def test_bracket_rejects_inconsistent_marker_frame_axis_origins(self):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        data = {'captured_poses': self.trajectory(), 'axis': np.array([0., 0., 1.]),
                'center_m': np.array([0., 0., .4])}
        angles = np.linspace(-.4, .4, 81)
        for pose, angle in zip(data['captured_poses'], angles):
            pose[:3,:3] = Rotation.from_rotvec([0.,0.,angle]).as_matrix()
        MarkerCalibrator.observed_axis_line(data)
        for pose in data['captured_poses'][40:]:
            pose[:3,:3] = pose[:3,:3] @ Rotation.from_euler('z', 180, degrees=True).as_matrix()
        with self.assertRaises(ValueError):
            MarkerCalibrator.observed_axis_line(data)


class J6ReferenceTests(unittest.TestCase):
    def test_j3_j5_use_measured_candidate_axis_with_unknown_upstream_zeros(self):
        from test_calibration_regression import OfflineRobot
        from core.calibration.JointCalibrator import JointCalibrator
        from core.marker_detection import SimulationModel
        for version in ('1.2', '1.3'):
            robot = OfflineRobot(version)
            config = SimulationModel.create(version).config
            for side in ('right', 'left'):
                config['offsets'][side]['joint0'] = 13.
                config['offsets'][side]['joint2'] = -11.
            simulation = SimulationModel.create(version, config)
            cal = JointCalibrator(robot=robot)
            cal.robot_version = version
            for side in ('right', 'left'):
                for mode in ('elbow', 'wrist_pitch_v13' if version == '1.3' else 'wrist_pitch'):
                    with self.subTest(version=version, side=side, mode=mode):
                        cfg = cal.JOINT_CONFIGS[mode]
                        center = cal.get_ready_pose('v'+version, 'joint', mode, side)
                        staged = -.2
                        center[cfg['cand_joint']] += np.deg2rad(staged)
                        indices = getattr(robot.model(), side+'_arm_idx')
                        datasets = []
                        for joint in (cfg['sweep_joint_A'], cfg['sweep_joint_B'], cfg['cand_joint']):
                            poses = []
                            for angle in np.linspace(-15., 0., 101):
                                q = robot.get_state().position.copy()
                                q[indices] = center
                                q[indices[joint]] += np.deg2rad(angle)
                                poses.append(simulation.marker_pose(robot, q, side, noisy=False))
                            datasets.append(poses)
                        cal.robot = None  # Estimator cannot access an encoder or dynamics model.
                        result = cal.compute_calibration_results(side, mode, *datasets[:2], dataset_C=datasets[2])
                        cal.robot = robot
                        self.assertTrue(result['measurement_accepted'], result)
                        expected = -np.rad2deg(simulation.arm_offsets(side)[cfg['cand_joint']])-staged
                        self.assertAlmostEqual(result['optimal_offset'], expected, places=4)

    def calculate(self, a=None, b=None):
        # Physical J6 = encoder(-1) + unknown error(+2) = +1 degree.
        R0 = Rotation.from_euler('z', 1., degrees=True).as_matrix()
        return estimate_j6_reference(
            sweep([0,0,1], R0) if a is None else a, [0,0,1],
            sweep([0,1,0], R0) if b is None else b, [0,1,0],
            np.eye(3))

    def test_recovers_measured_delta_without_encoder_reference(self):
        result = self.calculate()
        self.assertTrue(result['measurement_accepted'])
        self.assertAlmostEqual(result['optimal_offset'], -1., places=6)

    def test_one_bad_middle_rotation_does_not_determine_offset(self):
        R0 = Rotation.from_euler('z', 1., degrees=True).as_matrix()
        b = sweep([0,1,0], R0)
        b[len(b)//2,:3,:3] = b[len(b)//2,:3,:3] @ Rotation.from_euler('z', 8., degrees=True).as_matrix()
        result = self.calculate(b=b)
        self.assertTrue(result['measurement_accepted'])
        self.assertAlmostEqual(result['optimal_offset'], -1., places=6)

    def test_competing_branches_are_not_averaged_into_success(self):
        R0 = Rotation.from_euler('z', 1., degrees=True).as_matrix()
        b = sweep([0,1,0], R0)
        b[30:,:3,:3] = b[30:,:3,:3] @ Rotation.from_euler('z', 8., degrees=True).as_matrix()
        result = self.calculate(b=b)
        self.assertFalse(result['measurement_accepted'])
        self.assertNotIn('optimal_offset', result)

    def test_constant_bracket_gauge_is_not_claimed_as_physical_truth(self):
        # Constant coaxial +3 degree bracket rotation is indistinguishable
        # from J6 rotation. Observed delta is -4, not physical delta -1.
        R0 = Rotation.from_euler('z', 4., degrees=True).as_matrix()
        result = self.calculate(a=sweep([0,0,1], R0), b=sweep([0,1,0], R0))
        self.assertTrue(result['measurement_accepted'])
        self.assertAlmostEqual(result['optimal_offset'], -4., places=6)
        self.assertEqual(result['j6_mode'], 'effective_bracket_reference')

    def test_invalid_and_degenerate_inputs_never_return_an_offset(self):
        for value in (np.empty((0,4,4)), np.full((20,4,4), np.nan)):
            result = self.calculate(b=value)
            self.assertFalse(result['measurement_accepted'])
            self.assertNotIn('optimal_offset', result)
        result = estimate_j6_reference(sweep([0,0,1], np.eye(3)), [0,0,0],
            sweep([0,1,0], np.eye(3)), [0,1,0], np.eye(3))
        self.assertFalse(result['measurement_accepted'])

    def test_incompatible_j5_j6_axes_are_not_projected_into_success(self):
        tilted = np.array([0., np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))])
        result = estimate_j6_reference(sweep([0,0,1], np.eye(3)), [0,0,1],
                                      sweep(tilted, np.eye(3)), tilted, np.eye(3))
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
                result=estimate_j6_reference(*sweeps, nominal)
                self.assertFalse(result['measurement_accepted'])
                self.assertNotIn('optimal_offset',result)

    def test_actual_joint_calculator_with_sdk_kinematics_and_known_j6(self):
        # Actual compute_calibration_results and actual circle fitting, not
        # mocked axes. Only observations are synthetic; no SDK connection.
        import yaml
        from test_calibration_regression import OfflineRobot
        from core.calibration.JointCalibrator import JointCalibrator
        from core.marker_detection import SimulationModel
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
                                samples.append(simulation.marker_pose(robot, q, side, noisy=False))
                            datasets.append(samples)
                        result = cal.compute_calibration_results(side, mode, *datasets)
                        self.assertTrue(result['measurement_accepted'], result)
                        self.assertAlmostEqual(result['optimal_offset'],
                            -config['offsets'][side]['joint6'], places=3)


if __name__ == '__main__':
    unittest.main()
