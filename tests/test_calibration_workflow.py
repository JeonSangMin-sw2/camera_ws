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
from scipy.spatial.transform import Rotation
import yaml
from PySide6.QtWidgets import QApplication, QLineEdit
from core.calibration.CalibratorBase import BaseCalibrator
from main_ui import set_numeric_field, read_numeric_field
from core.marker_detection import SimulationModel, Marker_Transform
from main_ui import FullAutoWorker, UnifiedCalibrationApp
from core.config_store import CONFIG_PATHS
from core import calibration_core
from calibration_support import OfflineRobot, data, transform_vector


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_actual_ui_language_switch_and_destroyed_widget_unsubscribe(self):
        import shutil
        import shiboken6
        from PySide6.QtCore import QTimer
        from core.config_store import Language
        language = Language.instance()
        original_lang = language.current_lang
        with tempfile.TemporaryDirectory() as directory:
            redirected = {}
            for key, original in CONFIG_PATHS.items():
                target = Path(directory) / key
                if Path(original).is_file():
                    shutil.copy2(original, target)
                else:
                    target.mkdir()
                redirected[key] = str(target)
            with patch.dict(CONFIG_PATHS, redirected), contextlib.redirect_stdout(io.StringIO()):
                window = UnifiedCalibrationApp(Marker_Transform(sim=True), None, sim=True)
                for timer in window.findChildren(QTimer):
                    timer.stop()
                language.set_language('en')
                english = window.wizard_widget.t0.text()
                language.set_language('ko')
                self.assertNotEqual(window.wizard_widget.t0.text(), english)
                wrapper = window.wizard_widget
                window.close()
                shiboken6.delete(window)
                self.assertFalse(shiboken6.isValid(wrapper))
                language.set_language('en')  # Must not call the deleted C++ widget.
        language.set_language(original_lang)

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

    def test_ui_baseline_lookup_respects_central_path_override(self):
        stub = SimpleNamespace()
        stub.get_latest_home_reset_path = lambda required=True: UnifiedCalibrationApp.get_latest_home_reset_path(stub, required)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'baseline.json'
            with patch.dict(CONFIG_PATHS, home_reset_baseline=str(target)):
                self.assertIsNone(UnifiedCalibrationApp.get_home_reset_path_for_result(stub, 'unused'))
                target.write_text('{}')
                self.assertEqual(UnifiedCalibrationApp.get_latest_home_reset_path(stub), target)
                self.assertEqual(UnifiedCalibrationApp.get_home_reset_path_for_result(stub, 'unused'), target)

    def test_ui_exports_camera_zero_separately_from_physical_homes(self):
        from core.homeoffset_core import load_offset_from_json
        robot, sim = OfflineRobot(), SimulationModel.create()
        config = sim.config
        config['head_zero_convention'] = 'camera_forward'
        for side in ('right', 'left'):
            config[f'Tf_to_marker_{side}'] = transform_vector(sim.bracket_transform(side))
        qa, qh, observations = data(robot, sim, 24)
        logs = []
        context = calibration_core.OptimizerContext(robot=robot, model=robot.model(),
            include_head_motion=True, camera_config=config, robot_version='1.2',
            capture_metadata=sim.metadata())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            with contextlib.redirect_stdout(io.StringIO()):
                calibration_core.run_calibration_optimizer(context, ['right', 'left'], True, True,
                    qa, qh, observations, str(path), lambda_cam_pos=0, lambda_cam_rot=0,
                    log_callback=logs.append)
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
                      opt_delta_5=100., opt_delta_6=100., measurement_accepted=True, success=True)
        before = deepcopy(fake.joint_offsets_store)
        UnifiedCalibrationApp.handle_full_auto_bracket_finished(fake, result)
        self.assertEqual(fake.joint_offsets_store, before)
        self.assertEqual(read_numeric_field(fake.txt_bracket_l_y), .054)

    def test_sim_capture_uses_encoder_feedback_not_motion_target(self):
        robot = OfflineRobot()
        provider = Marker_Transform(sim=True, robot=robot)
        provider.set_marker_type('plate')
        qa, qh, _ = calibration_core.capture_calibration_sample(
            robot, robot.model(), provider, robot_version='1.2',
            head_idx=robot.model().head_idx)
        np.testing.assert_array_equal(qa, np.zeros(14))
        np.testing.assert_array_equal(qh, np.zeros(2))

    def test_queued_capture_logs_keep_capture_time_sample_ordinals(self):
        from PySide6.QtCore import QObject, Signal, Qt
        from core.robot_motion import AutoCollectionConfig
        class Emitter(QObject):
            captured = Signal(object)
        class LogReceiver(QObject):
            log_captured_sample = UnifiedCalibrationApp.log_captured_sample
        state = calibration_core.CollectionState(
            motion_plan=[{'desc': 'offline pose'}] * 2, ready=True)
        logs = []
        receiver = LogReceiver()
        receiver.shared_arm_q_list = state.arm_samples
        receiver.log_msg = logs.append
        receiver._write_step2_log = lambda message: None
        emitter = Emitter()
        emitter.captured.connect(receiver.log_captured_sample, Qt.QueuedConnection)
        sample = (np.zeros(14), None, np.stack([np.eye(4), np.eye(4)]))
        service = calibration_core.AutoCollectionService(
            object(), object(), None, None, AutoCollectionConfig(), state,
            include_head_motion=False, sample_callback=emitter.captured.emit)
        with patch('core.robot_motion.build_incremental_motion_plan', return_value=state.motion_plan), \
             patch('core.robot_motion.execute_auto_motion_step'), \
             patch('core.calibration_core.capture_calibration_sample', return_value=sample), \
             patch('core.calibration_core.time.sleep'):
            self.assertTrue(service.run())
        self.assertEqual(logs, [])
        state.arm_samples.append(np.ones(14))
        self.qt.processEvents()
        self.assertEqual([line.split(']')[0] for line in logs], ['[Sample 1', '[Sample 2'])

    def test_manual_capture_logs_next_sample_and_preserves_return_tuple(self):
        class ManualCaptureAdapter:
            capture_one_sample = UnifiedCalibrationApp.capture_one_sample
            log_captured_sample = UnifiedCalibrationApp.log_captured_sample
        adapter = ManualCaptureAdapter()
        adapter.robot, adapter.model, adapter.marker_st = object(), object(), object()
        adapter.get_robot_version = lambda: '1.2'
        adapter.get_capture_head_idx = lambda: None
        adapter.shared_arm_q_list = [np.zeros(14)]
        logs = []
        adapter.log_msg = logs.append
        adapter._write_step2_log = lambda message: None
        sample = (np.ones(14), None, np.stack([np.eye(4), np.eye(4)]))
        with patch('main_ui.capture_calibration_sample', return_value=sample):
            captured = adapter.capture_one_sample()
        self.assertIs(captured[0], sample[0])
        self.assertIs(captured[2], sample[2])
        self.assertTrue(logs[0].startswith('[Sample 2]'))
        self.assertEqual(len(adapter.shared_arm_q_list), 1)

    def test_full_auto_fits_bracket_before_j6_and_checks_it_with_fresh_sweeps(self):
        for version in ('1.2', '1.3'):
            for rejected in (None, 'joint', 'bracket', 'changed_after_j6', 'j6_disagrees', 'j6_nonfinite'):
                events = []
                store = {s: dict(joint3=0., joint5=0., joint6=0.) for s in ('right', 'left')}
                mode5 = 'wrist_pitch_v13' if version == '1.3' else 'wrist_pitch'
                mode6 = 'wrist_roll_v13' if version == '1.3' else 'wrist_yaw2'
                key6 = 'wrist_roll' if version == '1.3' else 'wrist_yaw2'
                class Calibrator:
                    robot = OfflineRobot(version)
                    NOMINAL_BRACKET_TEMPLATES = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES
                    def __init__(self):
                        self.joint_offsets = {s: {} for s in store}
                        self.camera_config = {}
                    def get_robot_version(self): return version
                    def perform_move_to_ready_pose(self, *a, **kw): return True
                    def perform_joint_calibration(self, side, mode, **kw):
                        events.append((side, mode))
                        return dict(recommended_joint_offset=-3.5 if mode == mode6 else 0.,
                                    converged=rejected != 'joint', failure_reason='rejected observation')
                    def save_calibration_comparison_plot(self, *a, **kw): return None
                    def perform_calibration_sweep(self, *a, **kw): return dict(axis_opt=np.array([0., 0., 1.]))
                    def fit_observed_bracket(self, d4, d5, d6, side):
                        events.append((side, 'bracket'))
                        if rejected == 'bracket':
                            return dict(success=False, measurement_accepted=False, failure_reason='rejected bracket',
                                        x_e=0., y_e=0., z_e=0., roll_e=0., pitch_e=0., yaw_e=0.)
                        n = self.NOMINAL_BRACKET_TEMPLATES[version][side]
                        drift = 2. if rejected == 'changed_after_j6' and events.count((side, 'bracket')) == 2 else 0.
                        twist = (1. if rejected == 'j6_disagrees' else float('nan')) if rejected in ('j6_disagrees', 'j6_nonfinite') else 0.
                        return dict(x_e=n[0]*1000 + drift, y_e=n[1]*1000, z_e=n[2]*1000,
                                    roll_e=n[3], pitch_e=n[4], yaw_e=n[5],
                                    removed_j6_twist_deg=twist,
                                    success=True, measurement_accepted=True)
                    def generate_marker_plot(self, *a, **kw): return False
                    def clear_user_taught_ready_poses(self): pass
                worker = calibration_core.FullAutoCalibrationService(Calibrator(), Calibrator(),
                    stop_event=threading.Event(), joint_offsets_store=store)
                with patch('main_ui.time.sleep'):
                    worker.run()
                if rejected:
                    self.assertIsNotNone(worker.error_msg)
                    self.assertFalse(all(worker.arm_convergence.values()))
                    self.assertNotIn(('right', 'elbow'), events)
                    if rejected == 'joint':
                        self.assertNotIn(('right', 'bracket'), events)
                    if rejected in ('changed_after_j6', 'j6_disagrees', 'j6_nonfinite'):
                        self.assertEqual(worker.marker_calibrator.camera_config, {})
                        self.assertFalse(worker.stage_results['right']['bracket_verification']['accepted'])
                    continue
                self.assertIsNone(worker.error_msg)
                self.assertTrue(all(worker.arm_convergence.values()))
                for side in store:
                    self.assertLess(events.index((side, mode5)), events.index((side, mode6)))
                    self.assertLess(events.index((side, 'bracket')), events.index((side, mode6)))
                    brackets = [i for i, event in enumerate(events) if event == (side, 'bracket')]
                    self.assertEqual(len(brackets), 2)
                    self.assertLess(events.index((side, mode6)), brackets[1])
                    self.assertLess(brackets[1], events.index((side, 'elbow')))
                    for mode in (mode5, mode6, 'elbow'):
                        self.assertEqual(events.count((side, mode)), 1)

    def test_observed_bracket_fit_ignores_encoder_payload(self):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        robot, sim = OfflineRobot(), SimulationModel.create()
        cal = MarkerCalibrator()
        idx = robot.model().left_arm_idx
        sweeps = []
        for axis in (4, 5, 6):
            poses = []
            for angle in np.linspace(-15, 15, 25):
                q = robot.get_state().position.copy()
                q[idx] = np.deg2rad([0., 30., 0., -90., 0., 0., 0.]) - sim.arm_offsets('left')
                q[idx[axis]] += np.deg2rad(angle)
                poses.append(sim.marker_pose(robot, q, 'left', noisy=False))
            measured = cal.fit_observed_circle(poses)
            measured['captured_poses'] = poses
            measured['commanded_reference_j6_deg'] = float(np.rad2deg(q[idx[6]]) - (15 if axis == 6 else 0))
            measured['captured_q_full'] = 'not encoder data'
            sweeps.append(measured)
        fitted = cal.fit_observed_bracket(*sweeps, 'left')
        self.assertTrue(fitted['measurement_accepted'], fitted)
        self.assertLess(fitted['axis_intersection_rms_mm'], 1e-5)
        self.assertNotIn('opt_delta_5', fitted)
        self.assertNotIn('opt_delta_6', fitted)
        nominal = cal.make_transform(cal.NOMINAL_BRACKET_TEMPLATES['1.2']['left'])
        measured = Rotation.from_euler('xyz', [fitted[k] for k in ('roll_e', 'pitch_e', 'yaw_e')], degrees=True)
        relative = measured * Rotation.from_matrix(nominal[:3, :3]).inv()
        self.assertAlmostEqual(relative.as_quat()[2], 0., places=10)

    def test_bracket_twist_lock_moves_translation_about_the_wrist_pivot(self):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        cal = MarkerCalibrator()
        cal.robot_version = '1.2'
        nominal = cal.make_transform(cal.NOMINAL_BRACKET_TEMPLATES['1.2']['left'])[:3, :3]
        swing = Rotation.from_euler('x', 7., degrees=True).as_matrix()
        twist = Rotation.from_euler('z', 30., degrees=True).as_matrix()
        pivot = np.array([.03, .04, .1])
        locked, position, angle = cal.lock_bracket_j6_twist(twist @ swing @ nominal, pivot, 'left')
        np.testing.assert_allclose(locked, swing @ nominal, atol=1e-12)
        self.assertAlmostEqual(angle, 30., places=10)
        wrist = np.array([0., 0., cal.robot_parameters.tool_lengths['1.2']])
        # Full rigid transform is preserved when the removed twist is assigned
        # to J6: rotation-only editing would fail this translation assertion.
        np.testing.assert_allclose(twist @ (position-wrist), -(twist @ swing @ nominal) @ pivot, atol=1e-12)

    def test_common_wrist_pivot_uses_all_pose_observations_without_short_arc_center_bias(self):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        rng = np.random.default_rng(103)
        datasets = []
        for axis in ('x', 'y', 'z'):
            poses = np.tile(np.eye(4), (301, 1, 1))
            poses[:, :3, :3] = Rotation.from_euler(axis, np.linspace(-15.,15.,301), degrees=True).as_matrix()
            poses[:, :3, 3] = [0.1,0.2,0.3] - poses[:, :3, :3] @ np.array([.01,.17,.05])
            poses[:, :3, 3] += rng.normal(0., .0001, (301,3))
            datasets.append(poses)
        pivot, rms = MarkerCalibrator.fit_common_wrist_pivot(datasets)
        np.testing.assert_allclose(pivot, [.01,.17,.05], atol=.00003)
        self.assertLess(rms, .0002)
        datasets[2][:,0,3] += .005
        with self.assertRaisesRegex(ValueError, 'pivot'):
            MarkerCalibrator.fit_common_wrist_pivot(datasets)

    def test_marker_report_uses_observed_geometry_metrics(self):
        logs = []
        app = SimpleNamespace(log_msg=logs.append, arm_side='left')
        result = dict(success=True, measurement_accepted=True, axis_intersection_rms_mm=.01,
                      x_e=1., y_e=54., z_e=-48., roll_e=90., pitch_e=0., yaw_e=0.)
        self.assertTrue(UnifiedCalibrationApp.show_unified_result_marker_direct(app, result))
        self.assertTrue(any('0.0100 mm' in line for line in logs))

    def test_bracket_refines_weak_short_arc_axis_with_observed_wrist_geometry(self):
        from core.calibration.MarkerCalibrator import MarkerCalibrator
        robot, sim = OfflineRobot(), SimulationModel.create()
        cal = MarkerCalibrator()
        idx = robot.model().left_arm_idx
        sweeps = []
        for axis in (4, 5, 6):
            direction = -1 if axis == 5 else 1
            poses = []
            for angle in np.linspace(-15, 15, 161)[::direction]:
                q = robot.get_state().position.copy()
                q[idx] = np.deg2rad([0., 30., 0., -90., 0., 90., 0.]) - sim.arm_offsets('left')
                q[idx[axis]] += np.deg2rad(angle)
                poses.append(sim.marker_pose(robot, q, 'left', noisy=False))
            measured = cal.fit_observed_circle(poses, direction)
            measured.update(captured_poses=np.asarray(poses), commanded_reference_j6_deg=0., sweep_direction=direction)
            sweeps.append(measured)
        weak = sweeps[2]
        tilt_axis = np.cross(weak['axis'], sweeps[1]['axis'])
        tilt_axis /= np.linalg.norm(tilt_axis)
        poses = weak['captured_poses'].copy()
        original_points = poses[:, :3, 3].copy()
        pivot = poses[len(poses)//2, :3, 3].copy()
        rotation = Rotation.from_rotvec(np.deg2rad(1.)*tilt_axis).as_matrix()
        poses[:, :3, 3] = (original_points-pivot) @ rotation.T + pivot
        self.assertLess(np.max(np.linalg.norm(poses[:, :3, 3]-original_points, axis=1)), .0005)
        weak.update(cal.fit_observed_circle(poses), captured_poses=poses)
        fitted = cal.fit_observed_bracket(*sweeps, 'left')
        self.assertTrue(fitted['measurement_accepted'], fitted)
        self.assertLess(fitted['axis_intersection_rms_mm'], .5)

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
        from core.robot_motion import AutoCollectionConfig
        service = calibration_core.AutoCollectionService(None, None, None, None,
            AutoCollectionConfig(), calibration_core.CollectionState(motion_plan=[]))
        worker = Step2AutoMotionWorker(service)
        done = []
        worker.finished_signal.connect(lambda ok, msg: done.append((ok, msg)))
        worker.run()
        self.assertFalse(done[0][0])
        self.assertIn('empty', done[0][1])

    def test_camera_temperature_and_legacy_serial_are_not_restrictions(self):
        import tempfile
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/camera_intrinsics.yaml').read_text())
        cfg['serial_number'] = 'a-different-camera'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'intrinsics.yaml'
            outputs = []
            for temperature in (None, 25., 38., 50.):
                cfg['calibration_temperature_c'] = temperature
                path.write_text(yaml.safe_dump(cfg))
                with patch.dict(CONFIG_PATHS, camera_intrinsics=str(path)):
                    provider = Marker_Transform(sim=True)
                metadata = provider.intrinsics_metadata
                intr, dist = np.array(metadata['camera_matrix']), np.array(metadata['dist_coeffs'])
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

    def test_wizard_failed_step1_save_stops_before_following_stages(self):
        from core.wizard_widget import CalibrationWizardWidget
        for has_head in (False, True):
            events = []
            parent = SimpleNamespace(last_full_auto_error=None, last_full_auto_converged=True,
                include_head_motion=has_head, head_camera_calibrator=object(),
                apply_full_auto_results=lambda **kw: False)
            wizard = SimpleNamespace(parent_app=parent,
                start_unified_step1_5=lambda: events.append('head'),
                start_unified_step2=lambda: events.append('step2'),
                stop_unified_calibration_error=lambda message: events.append('stopped'))
            CalibrationWizardWidget.on_unified_step1_finished(wizard)
            self.assertEqual(events, ['stopped'])


if __name__ == '__main__':
    unittest.main()
