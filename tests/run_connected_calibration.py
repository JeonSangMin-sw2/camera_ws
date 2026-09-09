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
import hashlib
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
from core.config_store import CONFIG_PATHS


def make_head_sweep_worker(app, stop_event):
    """Exercise the UI's configured range, not the worker's wider defaults."""
    from main_ui import HeadCamSweepWorker
    return HeadCamSweepWorker(app.head_camera_calibrator,
        pan_range=float(app.step1_5_pan_range.text()),
        tilt_range=float(app.step1_5_tilt_range.text()),
        num_steps=int(app.step1_5_num_steps.text()), stop_event=stop_event)


def checkpoint_input_context(paths, include_head=True):
    """Inputs that must match to reuse Step 1 before a NEW head/camera solve.

    The runner cold-starts joint/bracket estimates, and Step 1.5 replaces
    mount_to_cam. Those fitted outputs are not independent checkpoint inputs.
    All other settings, design parameters and immutable sensor truth must match.
    """
    context = {key: hashlib.sha256(Path(paths[key]).read_bytes()).hexdigest()
               for key in ('simulation_yaml', 'ready_poses_yaml', 'camera_intrinsics')}
    settings = yaml.safe_load(Path(paths['setting_yaml']).read_text())
    settings.pop('joint_offset', None)
    if include_head:
        settings.get('camera', {}).pop('mount_to_cam', None)
    for side in ('right', 'left'):
        settings.get('marker', {}).pop(f'Tf_to_marker_{side}', None)
    context['cold_start_settings'] = settings
    return context


def load_step1_checkpoint(path, version, include_head, source_hashes, input_context, sweep_seconds):
    """A partial verification is explicit and requires unchanged production code."""
    path = Path(path)
    checkpoint = json.loads(path.read_text())
    if (checkpoint.get('step1') != 'passed' or checkpoint.get('robot_version') != version
            or checkpoint.get('include_head') is not include_head):
        raise ValueError('Step 1 checkpoint version/mode/success does not match this run')
    previous = json.loads((path.parent / 'source_manifest.json').read_text())
    for name, value in source_hashes.items():
        if (name.startswith('core/') or name == 'main_ui.py') and previous.get(name) != value:
            raise ValueError(f'Production source changed since Step 1 checkpoint: {name}')
    input_manifest = path.parent / 'checkpoint_inputs.json'
    if input_manifest.exists():
        previous_inputs = json.loads(input_manifest.read_text())
    else:
        # Legacy full runs copied these files before executing and only rewrote
        # the fitted setting fields explicitly excluded by the helper above.
        previous_inputs = checkpoint_input_context({key: path.parent / filename
            for key, filename in (('simulation_yaml', 'simulation.yaml'),
                ('ready_poses_yaml', 'ready_poses.yaml'), ('camera_intrinsics', 'camera_intrinsics.yaml'),
                ('setting_yaml', 'setting.yaml'))}, include_head=include_head)
    if previous_inputs != input_context:
        raise ValueError('Step 1 checkpoint sensor truth/design/input settings changed')
    if checkpoint.get('sweep_seconds') != (sweep_seconds or 'production_defaults'):
        raise ValueError('Step 1 checkpoint sweep duration policy does not match this run')
    return checkpoint


def restore_step1_checkpoint(app, checkpoint):
    """Restore only a verified successful run using the production UI contract."""
    from main_ui import read_numeric_field
    if checkpoint.get('step1') != 'passed':
        raise ValueError('Only a successful Step 1 checkpoint can be restored')
    for side in ('right', 'left'):
        vector = np.asarray(checkpoint['brackets'][side], dtype=float)
        offsets = [checkpoint['joint_offsets'][side][key] for key in ('joint3', 'joint5', 'joint6')]
        if vector.shape != (6,) or not np.all(np.isfinite(vector)) or not np.all(np.isfinite(offsets)):
            raise ValueError(f'Invalid Step 1 checkpoint values for {side}')
    app.joint_offsets_store = deepcopy(checkpoint['joint_offsets'])
    for side, short in (('right', 'r'), ('left', 'l')):
        vec = checkpoint['brackets'][side]
        # Acceptance was established by the completed source-matched run above,
        # not inferred again from numeric residuals or fabricated observations.
        bracket = dict(arm_side=side, measurement_accepted=True,
                       source='verified_step1_checkpoint',
                       x_e=vec[0]*1000, y_e=vec[1]*1000, z_e=vec[2]*1000,
                       roll_e=vec[3], pitch_e=vec[4], yaw_e=vec[5])
        app.handle_full_auto_bracket_finished(bracket)
        restored = [read_numeric_field(getattr(app, f'txt_bracket_{short}_{axis}'))
                    for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')]
        if not np.allclose(restored, vec, rtol=0., atol=1e-12):
            raise RuntimeError(f'UI rejected or changed checkpoint bracket for {side}')
        for cal in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
            cal.camera_config[f'Tf_to_marker_{side}'] = deepcopy(vec)
    app.update_applied_offset_label()


def run(args):
    if not args.confirm_local_simulator:
        raise RuntimeError('Confirm the local simulator before enabling SDK motion.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    source_files = sorted((project/'core').rglob('*.py')) + [project/'main_ui.py', Path(__file__).resolve()]
    source_hashes = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    protected_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (project/'config').iterdir() if path.is_file()}
    (output/'source_manifest.json').write_text(json.dumps(source_hashes, indent=2))
    input_context = checkpoint_input_context(CONFIG_PATHS, include_head=not args.no_head)
    (output/'checkpoint_inputs.json').write_text(json.dumps(input_context, indent=2))
    # Redirect before importing classes that cache paths or load settings.
    for key, original in list(CONFIG_PATHS.items()):
        is_directory = key.endswith('_dir')
        target = output / (key if is_directory else Path(original).name)
        if Path(original).is_file():
            shutil.copy2(original, target)
        elif is_directory:
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
    from core.marker_detection import Marker_Transform
    from main_ui import (UnifiedCalibrationApp, FullAutoWorker,
                         Step2InitPoseWorker, Step2AutoMotionWorker)
    qt = QApplication.instance() or QApplication([])
    robot = BaseCalibrator.initialize_robot('127.0.0.1:50051', 'm', include_head=not args.no_head)
    if robot is None:
        raise RuntimeError('SDK connection/initialization failed')
    info = robot.get_robot_info()
    print('CONNECTED', info, flush=True)
    try:
        version = info.robot_model_version.removeprefix('v')
        if version not in ('1.2', '1.3'):
            raise RuntimeError(f'Unsupported robot version: {version}')
        for side in ('right', 'left'):
            cfg['marker'][f'Tf_to_marker_{side}'] = cfg['marker'][f'Tf_to_marker_{side}_v{version.replace(".", "")}']
        setting.write_text(yaml.safe_dump(cfg))
        detector = Marker_Transform(sim=True, robot=robot, robot_version=version)
        app = UnifiedCalibrationApp(detector, robot, sim=True)
        # Run workers synchronously with the same production slots and classes.
        for timer in app.findChildren(QTimer):
            timer.stop()
        app.log_msg = lambda msg: print(msg, flush=True)
        app.include_head_motion = not args.no_head
        app.chk_servo_head.setChecked(not args.no_head)
        app.step2_mode_sel.setCurrentText('live')
        app.joint_offsets_store = deepcopy(cfg['joint_offset'])
        for calibrator in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
            calibrator.include_head_motion = not args.no_head
            calibrator.robot_version = version
        # Default is exactly the production timing, for either marker source.
        if args.sweep_seconds is not None:
            app.joint_calibrator.perform_joint_calibration = partial(app.joint_calibrator.perform_joint_calibration, sweep_duration=args.sweep_seconds)
            app.marker_calibrator.perform_calibration_sweep = partial(app.marker_calibrator.perform_calibration_sweep, sweep_duration=args.sweep_seconds)
        stop = threading.Event()
        worker = FullAutoWorker(app.joint_calibrator, app.marker_calibrator,
                               stop_event=stop, joint_offsets_store=app.joint_offsets_store,
                               save_debug=True)
        worker.log_msg.connect(app.log_msg, Qt.DirectConnection)
        worker.joint_finished_signal.connect(app.handle_full_auto_joint_finished, Qt.DirectConnection)
        worker.bracket_finished_signal.connect(app.handle_full_auto_bracket_finished, Qt.DirectConnection)
        summary = {'endpoint': '127.0.0.1:50051', 'sdk_motion': True,
                   'real_camera_detection': False, 'include_head': not args.no_head,
                   'sweep_seconds': args.sweep_seconds or 'production_defaults', 'robot_version': version}
        if args.resume_step2:
            checkpoint = load_step1_checkpoint(args.resume_step2, version, not args.no_head,
                                              source_hashes, input_context, args.sweep_seconds)
            summary['step1'] = 'reused_verified_checkpoint'
            summary['step1_checkpoint'] = str(Path(args.resume_step2).resolve())
            summary['step1_checkpoint_sha256'] = hashlib.sha256(Path(args.resume_step2).read_bytes()).hexdigest()
            restore_step1_checkpoint(app, checkpoint)
        else:
            worker.run()
            if worker.error_msg:
                raise RuntimeError(worker.error_msg)
            if not all(worker.arm_convergence.values()):
                raise RuntimeError(f'Step 1 did not converge: {worker.arm_convergence}')
            summary['step1'] = 'passed'
            summary['step1_parameter_stability'] = worker.arm_convergence
        summary['joint_offsets'] = deepcopy(app.joint_offsets_store)
        summary['brackets'] = {side: app.marker_calibrator.camera_config[f'Tf_to_marker_{side}'] for side in ('right', 'left')}
        (output / 'step1.json').write_text(json.dumps(summary, indent=2))
        if args.only_step1:
            return
        # Save only the isolated configuration, never the user's setting.yaml.
        app.last_full_auto_error = None
        app.last_full_auto_converged = True
        if not app.apply_full_auto_results(silent=True):
            raise RuntimeError('Step 1 isolated save/apply failed')
        headcal = app.head_camera_calibrator
        if not args.no_head:
            if not headcal.perform_move_to_ready_pose(log_callback=app.log_msg, stop_event=stop):
                raise RuntimeError('Step 1.5 ready pose failed')
        head_worker = make_head_sweep_worker(app, stop)
        summary['head_sweep_configuration'] = {
            'pan_range_deg': head_worker.pan_range, 'tilt_range_deg': head_worker.tilt_range,
            'num_steps': head_worker.num_steps, 'source': 'production_UI_fields'}
        head_result = []
        head_worker.log_signal.connect(app.log_msg, Qt.DirectConnection)
        head_worker.finished_signal.connect(lambda ok, res: head_result.append((ok, res)), Qt.DirectConnection)
        head_worker.run()
        if not head_result or not head_result[0][0]:
            raise RuntimeError(f'Step 1.5 failed: {head_result}')
        headcal.calibrated_results = head_result[0][1]
        if not headcal.apply_calibration_results(log_callback=app.log_msg):
            raise RuntimeError('Step 1.5 isolated save/apply failed')
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
        collect = Step2AutoMotionWorker(app.collection_service(), parent=app)
        collect.state_signal.connect(app.publish_collection_state, Qt.DirectConnection)
        collect.captured_signal.connect(app.log_captured_sample, Qt.DirectConnection)
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
            # Same free-camera settings used by the production Run action.
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
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path.relative_to(project))]
               for path in source_files):
            raise RuntimeError('Source code changed during verification; repeat the full run')
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in protected_hashes.items()):
            raise RuntimeError('A protected user configuration changed during verification')
        summary['verified_source_manifest'] = str(output/'source_manifest.json')
        summary['user_configs_preserved'] = True
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
    parser.add_argument('--sweep-seconds', type=float, default=None,
                        help='Optional common sweep-time override; default uses production timings')
    parser.add_argument('--no-head', action='store_true')
    parser.add_argument('--only-step1', action='store_true')
    parser.add_argument('--resume-step2')
    run(parser.parse_args())
