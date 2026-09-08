"""No hardware commands: real iteration/worker with deterministic measurements."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from main_ui import FullAutoWorker, UnifiedCalibrationApp
from core.paths import CONFIG_PATHS
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator


class RetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_constant_absolute_target_does_not_drift_to_zero(self):
        seen = []
        def sweep(*args, **kw):
            seen.append(kw['current_offset_deg'])
            return {'optimal_offset': -1.0}
        cal = SimpleNamespace(robot=None, use_angle_based_fitting=True,
            JOINT_CONFIGS=BaseCalibrator.JOINT_CONFIGS, joint_offsets={'right': {}},
            perform_calibration_sweep_continuous=sweep,
            save_calibration_comparison_plot=lambda *a, **kw: None)
        with tempfile.TemporaryDirectory() as folder, patch.dict(CONFIG_PATHS, txt_dir=folder):
            result = JointCalibrator.perform_joint_calibration(cal, 'right', 'wrist_yaw2', current_offset_deg=-2.)
        self.assertTrue(result['converged'])
        self.assertEqual(result['recommended_joint_offset'], -1.)
        self.assertEqual(seen, [-2., -1.])

    def run_worker(self, fail_mode, recovery_pass, recovered_offset=0.):
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
                events.append((side, mode, kw['pass_idx']))
                return dict(recommended_joint_offset=recovered_offset if mode == fail_mode and kw['pass_idx'] >= recovery_pass else 0.,
                    converged=mode != fail_mode or kw['pass_idx'] >= recovery_pass)
            def save_calibration_comparison_plot(self, *a, **kw): return None
            def perform_calibration_sweep(self, side, axis, **kw):
                events.append((side, 'sweep', kw['pass_idx']))
                return dict(axis_opt=np.array([0., 0., 1.]))
            def compute_unified_bracket_calibration(self, d5, d6, side, **kw):
                events.append((side, 'bracket', None))
                n = self.NOMINAL_BRACKET_TEMPLATES['1.2'][side]
                return dict(x_e=n[0]*1000, y_e=n[1]*1000, z_e=n[2]*1000,
                    roll_e=n[3], pitch_e=n[4], yaw_e=n[5])
            def generate_marker_plot(self, *a, **kw): return False
            def clear_user_taught_ready_poses(self): pass
        worker = FullAutoWorker(Calibrator(), Calibrator(), stop_event=threading.Event(), joint_offsets_store=store)
        with patch('main_ui.time.sleep'):
            worker.run()
        return worker, events

    def test_damping_cannot_turn_large_measured_error_into_success(self):
        seen = []
        def sweep(*args, **kw):
            delta = .1 if len(seen) % 2 == 0 else -.1
            seen.append(delta)
            return {'optimal_offset': kw['current_offset_deg'] + delta}
        cal = SimpleNamespace(robot=None, use_angle_based_fitting=True,
            JOINT_CONFIGS=BaseCalibrator.JOINT_CONFIGS, joint_offsets={'right': {}},
            perform_calibration_sweep_continuous=sweep,
            save_calibration_comparison_plot=lambda *a, **kw: None)
        with tempfile.TemporaryDirectory() as folder, patch.dict(CONFIG_PATHS, txt_dir=folder):
            result = JointCalibrator.perform_joint_calibration(cal, 'right', 'wrist_yaw2')
        self.assertFalse(result['converged'])
        self.assertEqual(len(seen), 6)

    def test_retry_failed_j6_without_bracket_or_passed_joint_resweep(self):
        worker, events = self.run_worker('wrist_yaw2', 2, recovered_offset=2.)
        self.assertIsNone(worker.error_msg)
        self.assertTrue(all(worker.arm_convergence.values()))
        for side in ('right', 'left'):
            self.assertEqual([p for s,m,p in events if s==side and m=='wrist_pitch'], [1])
            self.assertEqual([p for s,m,p in events if s==side and m=='wrist_yaw2'], [1,2])
            self.assertEqual([p for s,m,p in events if s==side and m=='sweep'], [2,2,2])

    def test_persistent_failed_j6_never_enters_bracket(self):
        worker, events = self.run_worker('wrist_yaw2', 99)
        self.assertIsNotNone(worker.error_msg)
        self.assertFalse(any(m in ('sweep', 'bracket', 'elbow') for s,m,p in events))
        self.assertEqual([p for s,m,p in events if m=='wrist_yaw2'], [1,2])

    def test_persistent_failed_j5_defers_all_dependent_stages(self):
        worker, events = self.run_worker('wrist_pitch', 99)
        self.assertIsNotNone(worker.error_msg)
        self.assertTrue(all(m == 'wrist_pitch' for s,m,p in events))
        self.assertEqual([p for s,m,p in events], [1,2])

    def test_unconverged_elbow_cannot_pass_on_small_parameter_changes(self):
        worker, events = self.run_worker('elbow', 99)
        self.assertFalse(any(worker.arm_convergence.values()))
        self.assertEqual([p for s,m,p in events if s=='right' and m=='elbow'], [1,2])

    def test_pass_two_recovery_finishes_without_pass_three(self):
        worker, events = self.run_worker('elbow', 2)
        self.assertIsNone(worker.error_msg)
        self.assertTrue(all(worker.arm_convergence.values()))
        self.assertEqual([p for s,m,p in events if s=='right' and m=='elbow'], [1,2])

    def test_rejected_measurement_does_not_change_staging_or_trigger_inner_motion(self):
        seen = []
        def sweep(*args, **kw):
            seen.append(kw['current_offset_deg'])
            return dict(measurement_accepted=False, optimal_offset=8.,
                        failure_reason='inconsistent marker-frame axis')
        cal = SimpleNamespace(robot=None, use_angle_based_fitting=True,
            JOINT_CONFIGS=BaseCalibrator.JOINT_CONFIGS, joint_offsets={'right': {}},
            perform_calibration_sweep_continuous=sweep,
            save_calibration_comparison_plot=lambda *a, **kw: None)
        with tempfile.TemporaryDirectory() as folder, patch.dict(CONFIG_PATHS, txt_dir=folder):
            result = JointCalibrator.perform_joint_calibration(cal, 'right', 'wrist_yaw2', current_offset_deg=-2.)
        self.assertFalse(result['converged'])
        self.assertEqual(result['recommended_joint_offset'], -2.)
        self.assertEqual(seen, [-2.])

    def test_successful_first_pass_joints_are_not_remeasured(self):
        worker, events = self.run_worker(None, 1)
        self.assertIsNone(worker.error_msg)
        self.assertTrue(all(worker.arm_convergence.values()))
        for side in ('right', 'left'):
            for mode in ('wrist_pitch', 'wrist_yaw2', 'elbow'):
                self.assertEqual([p for s,m,p in events if s==side and m==mode], [1])

    def test_sweep_defaults_and_explicit_override_reach_every_iteration(self):
        for mode, override, expected in [('wrist_yaw2', None, 20.),
                                         ('wrist_pitch', None, 15.),
                                         ('elbow', None, 20.),
                                         ('wrist_yaw2', 1.5, 1.5)]:
            with self.subTest(mode=mode, override=override):
                seen=[]
                def sweep(*args, **kw):
                    seen.append((kw['sweep_duration'], kw['save_debug']))
                    return {'optimal_offset': 1. if mode=='wrist_yaw2' else (1. if len(seen)==1 else 0.)}
                cal=SimpleNamespace(robot=None, use_angle_based_fitting=True,
                    JOINT_CONFIGS=BaseCalibrator.JOINT_CONFIGS, joint_offsets={'right': {}},
                    perform_calibration_sweep_continuous=sweep,
                    save_calibration_comparison_plot=lambda *a, **kw: None)
                with tempfile.TemporaryDirectory() as folder, patch.dict(CONFIG_PATHS, txt_dir=folder):
                    kwargs={} if override is None else {'sweep_duration': override}
                    JointCalibrator.perform_joint_calibration(cal, 'right', mode, save_debug=True, **kwargs)
                self.assertTrue(seen)
                self.assertTrue(all(item==(expected, True) for item in seen))

    def test_rejected_full_auto_result_does_not_overwrite_ui_store(self):
        store={'right': {'joint6': -2.}}
        fake=SimpleNamespace(joint_offsets_store=store, log_msg=lambda msg: None,
            update_applied_offset_label=lambda: None,
            left_tabs=SimpleNamespace(currentIndex=lambda: 0), on_left_tab_changed=lambda index: None)
        UnifiedCalibrationApp.handle_full_auto_joint_finished(fake, dict(
            arm_side='right', mode='wrist_yaw2', recommended_joint_offset=8.,
            measurement_accepted=False, failure_reason='inconsistent axis'))
        self.assertEqual(store['right']['joint6'], -2.)

    def test_rejected_manual_result_is_not_presented_as_apply_candidate(self):
        offsets={'right': {'wrist_yaw2': 8.}}
        finalized=[]
        fake=SimpleNamespace(arm_side='right', original_joint_offset=-2.,
            joint_offsets=offsets, joint_calibrator=SimpleNamespace(joint_offsets=offsets),
            marker_calibrator=SimpleNamespace(joint_offsets=offsets),
            log_msg=lambda msg: None, on_action_finished=lambda: None,
            update_applied_offset_label=lambda: None,
            finalize_joint_calibration_run=lambda *a, **kw: finalized.append(kw),
            get_offset_key_for_mode=lambda mode: 'wrist_yaw2')
        UnifiedCalibrationApp.on_calibration_finished_joint(fake, dict(
            mode='wrist_yaw2', recommended_joint_offset=8., optimal_offset=8.,
            measurement_accepted=False, failure_reason='inconsistent axis'))
        self.assertEqual(offsets['right']['wrist_yaw2'], -2.)
        self.assertEqual(finalized, [])
        self.assertIsNone(fake.joint_sweep_data)

    def test_failed_full_auto_cannot_be_saved_even_in_silent_mode(self):
        for error, accepted in [('J6 failed', False), ('J6 failed', True), (None, False)]:
            writes=[]
            fake=SimpleNamespace(last_full_auto_error=error, last_full_auto_converged=accepted,
                log_msg=lambda msg: None,
                _joint_offset_patch=lambda: {'joint_offset': {}},
                _bracket_patch=lambda: {'marker': {}},
                _publish_joint_offsets=lambda values: None,
                _publish_brackets=lambda values: None)
            def save(*args):
                writes.append(args)
                return {'joint_offset': {}, 'marker': {}}
            with patch('core.config_store.update_yaml', side_effect=save):
                result=UnifiedCalibrationApp.apply_full_auto_results(fake, silent=True)
            self.assertFalse(result)
            self.assertEqual(writes, [])
