"""Opt-in integration test: real SDK commands to a confirmed LOCAL simulator.

Imports the production GUI, workers and calibrators. Marker observations come
from SimulationModel, NOT real D405 images. Never writes robot home offsets.
All settings/results are copied into --output. Do not run alongside another
controller. The caller must confirm that port 50051 belongs to rby1-sim.
"""
import argparse
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
import threading

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from core.paths import CONFIG_PATHS


def run(args):
    if not args.confirm_local_simulator:
        raise RuntimeError('Confirm the local simulator before enabling SDK motion.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Redirect before importing classes that cache paths or load settings.
    for key, original in list(CONFIG_PATHS.items()):
        target = output / (Path(original).name if key.endswith('yaml') or key in ('camera_info', 'camera_intrinsics') else key)
        if Path(original).is_file():
            shutil.copy2(original, target)
        else:
            target.mkdir(exist_ok=True)
        CONFIG_PATHS[key] = str(target)
    setting = Path(CONFIG_PATHS['setting_yaml'])
    cfg = yaml.safe_load(setting.read_text())
    # Cold start: no previous estimates or ground truth supplied to the solver.
    cfg['joint_offset'] = {side: dict(joint3=0., joint5=0., joint6=0.) for side in ('right', 'left')}
    for side in ('right', 'left'):
        cfg['marker'][f'Tf_to_marker_{side}'] = cfg['marker'][f'Tf_to_marker_{side}_v12']
    setting.write_text(yaml.safe_dump(cfg))

    from core.calibration.CalibratorBase import BaseCalibrator
    from main_ui import (UnifiedCalibrationApp, SimulatedMarkerTransform, FullAutoWorker,
                         HeadCamSweepWorker, Step2InitPoseWorker, Step2AutoMotionWorker)
    qt = QApplication.instance() or QApplication([])
    robot = BaseCalibrator.initialize_robot('127.0.0.1:50051', 'm', include_head=not args.no_head)
    if robot is None:
        raise RuntimeError('SDK connection/initialization failed')
    info = robot.get_robot_info()
    print('CONNECTED', info, flush=True)
    try:
        if info.robot_model_version.removeprefix('v') != '1.2':
            raise RuntimeError('This connected regression profile requires the v1.2 simulator')
        detector = SimulatedMarkerTransform(robot, cfg['camera'], '1.2', not args.no_head)
        app = UnifiedCalibrationApp(detector, robot, ui_only=False)
        # Run workers synchronously with the same production slots and classes.
        for timer in app.findChildren(QTimer):
            timer.stop()
        app.log_msg = lambda msg: print(msg, flush=True)
        app.include_head_motion = not args.no_head
        app.chk_servo_head.setChecked(not args.no_head)
        app.step2_mode_sel.setCurrentText('sim')
        app.joint_offsets_store = deepcopy(cfg['joint_offset'])
        for calibrator in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
            calibrator.include_head_motion = not args.no_head
            calibrator.robot_version = '1.2'
            # Shorter simulator sweeps, still real SDK trajectories and encoder capture.
        app.joint_calibrator.perform_joint_calibration = partial(app.joint_calibrator.perform_joint_calibration, sweep_duration=args.sweep_seconds)
        app.marker_calibrator.perform_calibration_sweep = partial(app.marker_calibrator.perform_calibration_sweep, sweep_duration=args.sweep_seconds)
        stop = threading.Event()
        worker = FullAutoWorker(app.joint_calibrator, app.marker_calibrator,
                               stop_event=stop, joint_offsets_store=app.joint_offsets_store)
        worker.log_msg.connect(app.log_msg, Qt.DirectConnection)
        worker.joint_finished_signal.connect(app.handle_full_auto_joint_finished, Qt.DirectConnection)
        worker.bracket_finished_signal.connect(app.handle_full_auto_bracket_finished, Qt.DirectConnection)
        summary = {'endpoint': '127.0.0.1:50051', 'sdk_motion': True,
                   'real_camera_detection': False, 'include_head': not args.no_head,
                   'sweep_seconds': args.sweep_seconds}
        if args.resume_step2:
            checkpoint = json.loads(Path(args.resume_step2).read_text())
            app.joint_offsets_store = checkpoint['joint_offsets']
            for side, vec in checkpoint['brackets'].items():
                for cal in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
                    cal.camera_config[f'Tf_to_marker_{side}'] = vec
                bracket = dict(arm_side=side, x_e=vec[0]*1000, y_e=vec[1]*1000, z_e=vec[2]*1000,
                               roll_e=vec[3], pitch_e=vec[4], yaw_e=vec[5])
                app.handle_full_auto_bracket_finished(bracket)
        else:
            worker.run()
            if worker.error_msg:
                raise RuntimeError(worker.error_msg)
            summary['step1'] = 'passed'
            summary['step1_parameter_stability'] = worker.arm_convergence
        summary['joint_offsets'] = deepcopy(app.joint_offsets_store)
        summary['brackets'] = {side: app.marker_calibrator.camera_config[f'Tf_to_marker_{side}'] for side in ('right', 'left')}
        (output / 'step1.json').write_text(json.dumps(summary, indent=2))
        if args.only_step1:
            return
        # Save only the isolated configuration, never the user's setting.yaml.
        app.apply_bracket_design_values(silent=True)
        headcal = app.head_camera_calibrator
        if not args.no_head:
            if not headcal.perform_move_to_ready_pose(log_callback=app.log_msg, stop_event=stop):
                raise RuntimeError('Step 1.5 ready pose failed')
        head_worker = HeadCamSweepWorker(headcal, num_steps=11, stop_event=stop)
        head_result = []
        head_worker.log_signal.connect(app.log_msg, Qt.DirectConnection)
        head_worker.finished_signal.connect(lambda ok, res: head_result.append((ok, res)), Qt.DirectConnection)
        head_worker.run()
        if not head_result or not head_result[0][0]:
            raise RuntimeError(f'Step 1.5 failed: {head_result}')
        headcal.calibrated_results = head_result[0][1]
        (output / 'head_sweep.json').write_text(json.dumps(head_result[0][1], indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        summary['step1_5'] = 'skipped_head_disabled' if args.no_head else 'passed'

        init = Step2InitPoseWorker(robot, ['right', 'left'], priority=10,
                                  include_head_motion=not args.no_head, parent=app)
        done = []
        init.log_signal.connect(app.log_msg, Qt.DirectConnection)
        init.finished_signal.connect(lambda ok, msg: done.append((ok, msg)), Qt.DirectConnection)
        init.run()
        if not done or not done[0][0]:
            raise RuntimeError(f'Step 2 init failed: {done}')
        app.auto_ready_done = True
        # The UI builds this before starting Step2AutoMotionWorker as well.
        from core.robot_motion import build_incremental_motion_plan
        app.auto_motion_plan = build_incremental_motion_plan(robot, app.dyn_model,
            app.auto_config, ['right', 'left'], include_head_motion=not args.no_head)
        print('STEP 2 PLANNED POSES', len(app.auto_motion_plan), flush=True)
        collect = Step2AutoMotionWorker(app)
        done = []
        collect.log_signal.connect(app.log_msg, Qt.DirectConnection)
        collect.finished_signal.connect(lambda ok, msg: done.append((ok, msg)), Qt.DirectConnection)
        collect.run()
        if not done or not done[0][0]:
            raise RuntimeError(f'Step 2 collection failed: {done}')
        expected = len(app.auto_motion_plan)
        if len(app.shared_arm_q_list) != expected:
            raise RuntimeError(f'Only {len(app.shared_arm_q_list)}/{expected} poses captured')
        summary['samples'] = expected
        summary['head_encoder_span_deg'] = np.rad2deg(np.ptp(np.array(app.shared_head_q_list), axis=0)).tolist()
        app.auto_save_current_dataset()
        app.run_optimizer(['right', 'left'], not args.no_head, True,
            np.array(app.shared_arm_q_list), np.array(app.shared_head_q_list),
            np.array(app.shared_T_list), str(output / 'optimizer.json'),
            lambda_cam_pos=0., lambda_cam_rot=0.)
        result = json.loads((output / 'optimizer.json').read_text())
        diagnostics = result['diagnostics']
        if not diagnostics['converged'] or not diagnostics['observable']:
            raise RuntimeError(f'Step 2 rejected: {diagnostics}')
        errors = {}
        for side in ('right', 'left'):
            gt = np.rad2deg(detector.simulation_model.arm_offsets(side))
            errors[side] = (np.array(result[f'{side}_arm_joint_offset_deg']) - gt).tolist()
        summary['arm_gt_error_deg'] = errors
        summary['max_non_j6_gt_error_deg'] = max(abs(x) for e in errors.values() for x in e[:6])
        summary['step2'] = 'passed'
        summary['result'] = result
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        print('CONNECTED ALL-STEPS PASS', output, flush=True)
    except BaseException:
        robot.cancel_control()
        raise
    finally:
        robot.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-local-simulator', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sweep-seconds', type=float, default=2.)
    parser.add_argument('--no-head', action='store_true')
    parser.add_argument('--only-step1', action='store_true')
    parser.add_argument('--resume-step2')
    run(parser.parse_args())
