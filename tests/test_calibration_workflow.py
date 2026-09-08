"""Regression coverage for stage ownership, actual worker order and UI storage."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading
import json
import tempfile
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from PySide6.QtWidgets import QApplication, QLineEdit
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.bracket_fitting import fit_bracket_sweeps
from core.numeric_fields import set_numeric_field, read_numeric_field
from core.simulation_model import SimulationModel
from main_ui import FullAutoWorker, UnifiedCalibrationApp
from test_calibration_regression import OfflineRobot, data, transform_vector


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_compact_fields_preserve_unchanged_full_precision(self):
        field = QLineEdit()
        value = .0545123456789
        set_numeric_field(field, value)
        self.assertEqual(field.text(), '0.0545')
        self.assertEqual(read_numeric_field(field), value)
        field.setText('0.0546')
        self.assertEqual(read_numeric_field(field), .0546)
        set_numeric_field(field, -0.0000001)
        self.assertEqual(field.text(), '0')
        set_numeric_field(field, 90.)
        self.assertEqual(field.text(), '90')

    def test_ui_exports_camera_zero_separately_from_physical_homes(self):
        from core.homeoffset_core import load_offset_from_json
        robot, sim = OfflineRobot(), SimulationModel.create()
        config = sim.config
        config['head_zero_convention'] = 'camera_forward'
        for side in ('right', 'left'):
            config[f'Tf_to_marker_{side}'] = transform_vector(sim.bracket_transform(side))
        qa, qh, observations = data(robot, sim, 24)
        logs = []
        fake = SimpleNamespace(robot=robot, model=robot.model(), include_head_motion=True,
            marker_calibrator=SimpleNamespace(camera_config=config),
            get_robot_version=lambda: '1.2', last_home_reset_path=None,
            log_msg=logs.append, _loaded_dataset_metadata=sim.metadata())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            with contextlib.redirect_stdout(io.StringIO()):
                UnifiedCalibrationApp.run_optimizer(fake, ['right', 'left'], True, True,
                    qa, qh, observations, str(path), lambda_cam_pos=0, lambda_cam_rot=0)
            with path.open() as f:
                result = json.load(f)
            self.assertFalse(result['head_tilt_independent'])
            self.assertEqual(result['diagnostics']['head_tilt_mode'], 'camera_forward_gauge')
            self.assertTrue(result['camera_forward_zero']['accepted'])
            self.assertIsNone(load_offset_from_json(path)[1])
            np.testing.assert_allclose(result['camera_forward_zero']['encoder_zero_deg'], [-.8, 1.5], atol=1e-6)
        self.assertTrue(any('[CAMERA ZERO]' in line for line in logs))

    def test_yaml_save_preserves_precision_and_legacy_lists(self):
        values = [.0545123456789, -.003456789123, .06, 90.123456789, .234567891, 180.]
        for section, method in [('marker', '_update_marker_key_in_lines'), ('camera', '_update_camera_key_in_lines')]:
            cfg = {section: {'pose': [0.]*6, 'keep': 7}, 'other': {'keep': 8}}
            lines = yaml.safe_dump(cfg).splitlines(keepends=True)
            getattr(UnifiedCalibrationApp, method)(None, lines, 'pose', values)
            result = yaml.safe_load(''.join(lines))
            self.assertEqual(result[section]['pose'], values)
            self.assertEqual(result[section]['keep'], 7)
            self.assertEqual(result['other']['keep'], 8)

    def test_bracket_callback_does_not_stage_any_joint(self):
        fake = SimpleNamespace(joint_offsets_store={'left': {'joint5': 2., 'joint6': -3.5}}, log_msg=lambda _: None)
        for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
            setattr(fake, f'txt_bracket_l_{axis}', QLineEdit())
        result = dict(arm_side='left', x_e=0., y_e=54., z_e=-48., roll_e=90., pitch_e=0., yaw_e=0.,
                      opt_delta_5=100., opt_delta_6=100.)
        before = deepcopy(fake.joint_offsets_store)
        UnifiedCalibrationApp.handle_full_auto_bracket_finished(fake, result)
        self.assertEqual(fake.joint_offsets_store, before)

    def test_sim_capture_uses_encoder_feedback_not_motion_target(self):
        robot, sim = OfflineRobot(), SimulationModel.create()
        fake = SimpleNamespace(robot=robot, model=robot.model(), ui_only=False,
            log_msg=lambda _: None, _write_step2_log=lambda _: None, shared_arm_q_list=[],
            step2_mode_sel=SimpleNamespace(currentText=lambda: 'sim'),
            get_robot_version=lambda: '1.2',
            get_capture_head_idx=lambda: robot.model().head_idx,
            marker_calibrator=SimpleNamespace(get_simulation_model=lambda: sim),
            marker_st=SimpleNamespace(rng=np.random.default_rng(7)))
        qa, qh, _ = UnifiedCalibrationApp.capture_one_sample(fake,
            motion_plan_step={'q_arm': np.ones(14), 'q_head': np.ones(2)})
        np.testing.assert_array_equal(qa, np.zeros(14))
        np.testing.assert_array_equal(qh, np.zeros(2))

    def test_full_auto_calibrates_j6_before_bounded_bracket(self):
        events = []
        store = {s: dict(joint3=0., joint5=0., joint6=0.) for s in ('right', 'left')}
        class Calibrator:
            robot = 'mock_robot'
            NOMINAL_BRACKET_TEMPLATES = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES
            def __init__(self):
                self.joint_offsets = {s: {} for s in store}
                self.camera_config = {}
            def get_robot_version(self): return '1.2'
            def perform_move_to_ready_pose(self, *a, **kw): return True
            def perform_joint_calibration(self, side, mode, **kw):
                events.append((side, mode))
                return dict(recommended_joint_offset=-3.5 if mode == 'wrist_yaw2' else 0., converged=True)
            def save_calibration_comparison_plot(self, *a, **kw): return None
            def perform_calibration_sweep(self, *a, **kw): return dict(axis_opt=np.array([0., 0., 1.]))
            def compute_unified_bracket_calibration(self, d5, d6, side, **kw):
                if kw['calib_roll_or_yaw_deg'] != -3.5:
                    raise RuntimeError('Regression: bounded bracket ran before J6 calibration')
                events.append((side, 'bracket'))
                n = self.NOMINAL_BRACKET_TEMPLATES['1.2'][side]
                return dict(x_e=n[0]*1000, y_e=n[1]*1000, z_e=n[2]*1000,
                            roll_e=n[3], pitch_e=n[4], yaw_e=n[5])
            def generate_marker_plot(self, *a, **kw): return False
            def clear_user_taught_ready_poses(self): pass
        worker = FullAutoWorker(Calibrator(), Calibrator(), stop_event=threading.Event(), joint_offsets_store=store)
        with patch('main_ui.time.sleep'):
            worker.run()
        self.assertIsNone(worker.error_msg)
        self.assertTrue(all(worker.arm_convergence.values()))
        for side in store:
            self.assertLess(events.index((side, 'wrist_yaw2')), events.index((side, 'bracket')))
            # Passed joints are not re-swept just because brackets change later.
            for mode in ('wrist_pitch', 'wrist_yaw2', 'elbow'):
                self.assertEqual(events.count((side, mode)), 1)

    def test_left_j6_uncalibrated_bound_failure_and_fixed_input_recovery(self):
        robot, sim = OfflineRobot(), SimulationModel.create()
        idx = robot.model().left_arm_idx
        sweeps = []
        for axis in (4, 5, 6):
            poses, qs = [], []
            for angle in np.linspace(-15, 15, 25):
                q = robot.get_state().position.copy()
                q[idx] = np.deg2rad([0., 30., 0., -90., 0., 0., 0.])
                q[idx[axis]] += np.deg2rad(angle)
                qs.append(q)
                poses.append(sim.marker_pose(robot, q, 'left', noisy=False))
            sweeps.append(dict(captured_poses=poses, captured_q_full=qs))
        nominal = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES['1.2']['left']
        gt = np.rad2deg(sim.arm_offsets('left'))
        with self.assertRaisesRegex(RuntimeError, 'at_bounds=True'):
            fit_bracket_sweeps(robot, 'left', sweeps, nominal, 0., gt[5])
        fitted = fit_bracket_sweeps(robot, 'left', sweeps, nominal, gt[6], gt[5])
        self.assertEqual(fitted['data_rank'], 6)
        self.assertEqual(fitted['fit_scope'], 'bracket_only')
        self.assertNotIn('opt_delta_5', fitted)
        self.assertNotIn('opt_delta_6', fitted)
        self.assertLess(fitted['normalized_residual_rms'], 1e-5)

    def test_head_fit_uses_both_encoders_instead_of_assuming_idle_axis_zero(self):
        from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
        robot, sim = OfflineRobot(), SimulationModel.create()
        cal = HeadCameraCalibrator(robot=robot)
        ta, pa = np.linspace(-10, 10, 11), np.linspace(-15, 15, 11)
        tilt_head = np.column_stack((.2*np.sin(np.arange(11)), ta))
        pan_head = np.column_stack((pa, .3*np.cos(np.arange(11))))
        points = []
        for heads in (tilt_head, pan_head):
            poses = []
            for h in heads:
                q = robot.get_state().position.copy()
                q[robot.model().head_idx] = np.deg2rad(h)
                poses.append(sim.marker_pose(robot, q, 'right', noisy=False)[:3, 3])
            points.append(poses)
        result = cal._compute_head_camera_solution(*points, ta, pa,
            sim.config['mount_to_cam'], np.eye(3), tilt_head_deg=tilt_head, pan_head_deg=pan_head)
        self.assertTrue(result['success'])
        self.assertLess(result['quality']['rmse_3d_marker_mm'], 1e-5)

    def test_empty_motion_plan_is_not_reported_as_success(self):
        from main_ui import Step2AutoMotionWorker
        worker = Step2AutoMotionWorker(SimpleNamespace(get_auto_pose_target_count=lambda: 0))
        done = []
        worker.finished_signal.connect(lambda ok, msg: done.append((ok, msg)))
        worker.run()
        self.assertFalse(done[0][0])
        self.assertIn('empty', done[0][1])

    def test_camera_temperature_and_legacy_serial_are_not_restrictions(self):
        import tempfile
        from core.camera_intrinsics import select_intrinsics
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/camera_intrinsics.yaml').read_text())
        cfg['serial_number'] = 'a-different-camera'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'intrinsics.yaml'
            outputs = []
            for temperature in (None, 25., 38., 50.):
                cfg['calibration_temperature_c'] = temperature
                path.write_text(yaml.safe_dump(cfg))
                intr, dist, metadata = select_intrinsics('per_device', [], [], 1280, 720, path)
                self.assertNotIn('calibration_serial', metadata)
                self.assertNotIn('temperature_source', metadata)
                outputs.append((intr, dist))
            for intr, dist in outputs[1:]:
                np.testing.assert_array_equal(intr, outputs[0][0])
                np.testing.assert_array_equal(dist, outputs[0][1])

    def test_no_head_servo_rule_also_filters_explicit_patterns(self):
        import re
        from unittest.mock import MagicMock
        import rby1_sdk as rby
        for servo in (None, 'right_arm.*|head.*'):
            robot = MagicMock()
            robot.get_robot_info.return_value.robot_model_name = 'm'
            robot.get_control_manager_state.return_value.state = rby.ControlManagerState.State.Enabled
            robot.is_servo_on.return_value = False
            with patch('core.calibration.CalibratorBase.rby.create_robot', return_value=robot), patch('core.calibration.CalibratorBase.time.sleep'):
                self.assertIs(BaseCalibrator.initialize_robot('127.0.0.1:50051', 'm', servo=servo, include_head=False), robot)
            pattern = robot.servo_on.call_args.args[0]
            self.assertIsNone(re.fullmatch(pattern, 'head_0'))
            self.assertIsNotNone(re.fullmatch(pattern, 'right_arm_0'))

    def test_full_auto_ui_does_not_call_pass_limit_success(self):
        messages = []
        fake = SimpleNamespace(set_controls_enabled=lambda _: None, log_msg=messages.append,
            active_worker=SimpleNamespace(error_msg=None, arm_convergence={'right': True, 'left': False}, wait=lambda: None),
            left_tabs=SimpleNamespace(currentIndex=lambda: 0),
            poll_timer=SimpleNamespace(isActive=lambda: True))
        UnifiedCalibrationApp.on_full_auto_finished(fake)
        self.assertFalse(fake.last_full_auto_converged)
        self.assertTrue(any('[WARNING]' in m for m in messages))
        self.assertFalse(any('[SUCCESS]' in m for m in messages))


if __name__ == '__main__':
    unittest.main()
