"""Supervised Step 1 -> 1.5 -> 2 using production workers and a REAL camera.

Unlike run_connected_calibration.py, this connects to the user's real A@v1.2.
Never enables power/servos, resets faults or writes motor home offsets. Results
and intermediate configuration are isolated in --output. A STOP file in that
directory cancels motion. Requires the on-site supervision approved in chat.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import numpy as np
import rby1_sdk as rby
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from core.config_store import CONFIG_PATHS


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class SupervisedRobot:
    """Latch cancellation at command submission, including initialization moves."""
    def __init__(self, sdk, stop):
        self.sdk, self.stop = sdk, stop
        self.gate = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.sdk, name)

    def send_command(self, *args, **kwargs):
        with self.gate:
            if self.stop.is_set():
                raise RuntimeError('Motion prohibited after stop request')
            return self.sdk.send_command(*args, **kwargs)

    def halt(self):
        with self.gate:
            self.stop.set()
            self.sdk.cancel_control()


def run(args):
    if not args.supervised_motion:
        raise RuntimeError('On-site supervision is required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / 'core').rglob('*.py')) + [ROOT / 'main_ui.py', Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'config').iterdir() if p.is_file()}
    def save(name, data):
        (output / name).write_text(json.dumps(data, indent=2, default=json_value), encoding='utf-8')
    def log(message):
        print(message, flush=True)
        with (output / 'run.log').open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')
    save('source_manifest.json', source_hashes)
    for key, original in list(CONFIG_PATHS.items()):
        target = output / (key if key.endswith('_dir') else Path(original).name)
        if Path(original).is_file():
            shutil.copy2(original, target)
        elif key.endswith('_dir'):
            target.mkdir()
        CONFIG_PATHS[key] = str(target)

    from main_ui import UnifiedCalibrationApp, FullAutoWorker, Step2InitPoseWorker, Step2AutoMotionWorker
    from core.marker_detection import Marker_Transform
    from core.calibration.JointCalibrator import J5_CIRCLE_TOLERANCE_MM
    from core.robot_motion import build_incremental_motion_plan
    from run_connected_calibration import make_head_sweep_worker
    qt = QApplication.instance() or QApplication([])
    marker, app = None, None
    stop = threading.Event()
    robot = SupervisedRobot(rby.create_robot(args.address, 'a'), stop)
    watcher_done = threading.Event()
    summary = dict(endpoint=args.address, real_camera=True, include_head=True,
                   exposure_us=6000, auto_exposure=False, stage='connecting',
                   j5_sweep_seconds=15.,
                   j5_circle_tolerance_mm=J5_CIRCLE_TOLERANCE_MM,
                   step1='not_run', step1_5='not_run', step2='not_run')
    def verify_exposure():
        if marker.sim or marker.camera is None:
            raise RuntimeError('Real camera required; no simulated fallback')
        # Session-only diagnostic setting; production defaults stay unchanged.
        marker.set_camera_exposure(6000., auto_exposure=False)
        auto_mode, configured = marker.get_camera_exposure()
        for _ in range(5):
            marker.camera.capture_image()
        actual = marker.get_actual_exposure()
        log(f'[EXPOSURE] auto={auto_mode}, configured={configured}us, actual={actual}us')
        if auto_mode or abs(configured - 6000.) > 1. or abs(actual - 6000.) > 100.:
            raise RuntimeError('Manual 6000us exposure verification failed')
        summary['exposure_readback'] = dict(auto=auto_mode, configured_us=configured, actual_us=actual)
    def halt_on_teaching(*unused):
        raise RuntimeError('Marker visibility requires manual teaching; stopped for inspection')
    def watch_stop():
        deadline = time.monotonic() + 3600.
        while not watcher_done.wait(.2):
            if (output / 'STOP').exists() or time.monotonic() > deadline:
                robot.halt()
                if app is not None:
                    app.auto_stop_requested = True
                    if hasattr(app, '_auto_collection_stop_event'):
                        app._auto_collection_stop_event.set()
                    for cal in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
                        cal.stop_requested = True
                return
    watcher = None
    try:
        if not robot.connect(max_retries=1, timeout_ms=3000):
            raise RuntimeError('Robot connection failed')
        info = robot.get_robot_info()
        if info.robot_model_name.lower() != 'a' or info.robot_model_version.removeprefix('v') != '1.2':
            raise RuntimeError(f'Unexpected robot model: {info}')
        if robot.get_control_manager_state().state != rby.ControlManagerState.State.Enabled:
            raise RuntimeError('Control manager is not enabled; no automatic fault reset')
        if not robot.is_servo_on('^(?!.*wheel).*$'):
            raise RuntimeError('Body/arm/head servos are not all enabled')
        if np.max(np.abs(robot.get_state().velocity)) > .02:
            raise RuntimeError('Another controller appears to be moving the robot')
        log(f'CONNECTED REAL ROBOT {info}; output={output}')
        summary['robot'] = str(info)
        marker = Marker_Transform(sim=False, robot=robot, robot_version='1.2')
        if marker.sim:
            raise RuntimeError('No real camera; robot will not move')
        marker.set_marker_type('plate')
        app = UnifiedCalibrationApp(marker, robot, sim=False)
        for timer in app.findChildren(QTimer):
            timer.stop()
        app.log_msg = log
        app.include_head_motion = True
        app.chk_servo_head.setChecked(True)
        app.step2_mode_sel.setCurrentText('live')
        for cal in (app.joint_calibrator, app.marker_calibrator, app.head_camera_calibrator):
            cal.include_head_motion = True
            cal.robot_version = '1.2'
            cal.marker_problem_callback = halt_on_teaching
        app.show_message_box = halt_on_teaching
        app.prompt_marker_problem_teaching = halt_on_teaching
        verify_exposure()
        save('camera_metadata.json', marker.intrinsics_metadata)
        watcher = threading.Thread(target=watch_stop, daemon=True)
        watcher.start()

        summary['stage'] = 'step1'
        save('status.json', summary)
        worker = FullAutoWorker(app.joint_calibrator, app.marker_calibrator,
            stop_event=stop, joint_offsets_store=app.joint_offsets_store,
            save_debug=True, reset_initial_state=True, head_camera_calibrator=app.head_camera_calibrator)
        worker.log_msg.connect(log, Qt.DirectConnection)
        worker.joint_finished_signal.connect(app.handle_full_auto_joint_finished, Qt.DirectConnection)
        worker.bracket_finished_signal.connect(app.handle_full_auto_bracket_finished, Qt.DirectConnection)
        worker.run()
        save('step1_measurements.json', worker.service.stage_results)
        if stop.is_set() or worker.error_msg or not all(worker.arm_convergence.get(s, False) for s in ('right', 'left')):
            raise RuntimeError(f'Step 1 failed: {worker.error_msg}; convergence={worker.arm_convergence}')
        summary['step1'] = 'passed'
        summary['joint_offsets'] = deepcopy(app.joint_offsets_store)
        summary['brackets'] = {s: app.marker_calibrator.camera_config[f'Tf_to_marker_{s}'] for s in ('right', 'left')}
        save('step1.json', summary)
        app.last_full_auto_error = None
        app.last_full_auto_converged = True
        if not app.apply_full_auto_results(silent=True):
            raise RuntimeError('Step 1 isolated save/apply failed')

        summary['stage'] = 'step1_5'
        save('status.json', summary)
        verify_exposure()
        headcal = app.head_camera_calibrator
        if not headcal.perform_move_to_ready_pose(log_callback=log, stop_event=stop):
            raise RuntimeError('Step 1.5 ready pose failed')
        head_worker = make_head_sweep_worker(app, stop)
        summary['head_sweep'] = dict(pan_range=head_worker.pan_range,
            tilt_range=head_worker.tilt_range, num_steps=head_worker.num_steps)
        head_results = []
        head_worker.log_signal.connect(log, Qt.DirectConnection)
        head_worker.finished_signal.connect(lambda ok, res: head_results.append((ok, res)), Qt.DirectConnection)
        head_worker.run()
        save('head_sweep.json', head_results)
        if stop.is_set() or not head_results or not head_results[0][0]:
            raise RuntimeError(f'Step 1.5 failed: {head_results}')
        headcal.calibrated_results = head_results[0][1]
        if not headcal.apply_calibration_results(log_callback=log):
            raise RuntimeError('Step 1.5 isolated save/apply failed')
        summary['step1_5'] = 'passed'

        summary['stage'] = 'step2'
        save('status.json', summary)
        verify_exposure()
        init = Step2InitPoseWorker(robot, ['right', 'left'], priority=10, include_head_motion=True, parent=app)
        init.head_pose_signal.disconnect()
        init.head_pose_signal.connect(app.on_capture_head_centered, Qt.DirectConnection)
        outcomes = []
        init.log_signal.connect(log, Qt.DirectConnection)
        init.finished_signal.connect(lambda ok, msg: outcomes.append((ok, msg)), Qt.DirectConnection)
        init.run()
        if stop.is_set() or not outcomes or not outcomes[0][0]:
            raise RuntimeError(f'Step 2 ready pose failed: {outcomes}')
        app.auto_ready_done = True
        app.auto_motion_plan = build_incremental_motion_plan(robot, app.dyn_model,
            app.auto_config, ['right', 'left'], include_head_motion=True)
        log(f'STEP 2 PLANNED POSES: {len(app.auto_motion_plan)}')
        collect = Step2AutoMotionWorker(app.collection_service(), parent=app)
        collect.state_signal.connect(app.publish_collection_state, Qt.DirectConnection)
        collect.captured_signal.connect(app.log_captured_sample, Qt.DirectConnection)
        outcomes = []
        collect.log_signal.connect(log, Qt.DirectConnection)
        collect.finished_signal.connect(lambda ok, msg: outcomes.append((ok, msg)), Qt.DirectConnection)
        collect.run()
        if app.shared_arm_q_list:
            app.auto_save_current_dataset()
        if stop.is_set() or not outcomes or not outcomes[0][0]:
            raise RuntimeError(f'Step 2 collection failed: {outcomes}')
        expected = len(app.auto_motion_plan)
        if len(app.shared_arm_q_list) != expected:
            raise RuntimeError(f'Only {len(app.shared_arm_q_list)}/{expected} poses captured')
        summary['samples'] = expected
        app.run_optimizer(['right', 'left'], True, True,
            np.asarray(app.shared_arm_q_list), np.asarray(app.shared_head_q_list),
            np.asarray(app.shared_T_list), str(output / 'optimizer.json'),
            lambda_cam_pos=0., lambda_cam_rot=0.)
        result = json.loads((output / 'optimizer.json').read_text(encoding='utf-8'))
        if not result['diagnostics']['converged'] or not result['diagnostics']['observable']:
            raise RuntimeError(f'Step 2 optimizer rejected: {result["diagnostics"]}')
        summary.update(step2='passed', stage='complete', diagnostics=result['diagnostics'])
        log('REAL ALL-STEPS PASS')
    except BaseException as error:
        if robot.is_connected():
            robot.halt()
        else:
            stop.set()
        summary['error'] = str(error)
        if summary['stage'] in ('step1', 'step1_5', 'step2'):
            summary[summary['stage']] = 'failed'
        log(traceback.format_exc())
        raise
    finally:
        watcher_done.set()
        if watcher is not None:
            watcher.join(timeout=2.)
        if marker is not None and marker.camera is not None:
            marker.camera.stream_off()
        if robot.is_connected():
            summary['final_position_deg'] = np.rad2deg(robot.get_state().position).tolist()
        robot.disconnect()
        summary['user_configs_preserved'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in protected.items())
        summary['source_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == source_hashes[str(p.relative_to(ROOT))] for p in sources)
        save('status.json', summary)
        log(f'FINAL STATUS {json.dumps(summary, default=json_value)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervised-motion', action='store_true')
    parser.add_argument('--address', default='192.168.1.40:50051')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
