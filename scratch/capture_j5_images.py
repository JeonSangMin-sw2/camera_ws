"""Supervised, fixed-command J5 diagnostic: three 15 s sweeps, raw images.

Keeps the previously reviewed right-arm posture and sweep ranges. No offset
updates, power/servo changes, or fallback to a simulated camera. STOP file
latches command cancellation. Images are written only after each sweep ends.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import rby1_sdk as rby

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_real_calibration import SupervisedRobot
from core.config_store import CONFIG_PATHS
from core.marker_detection import Marker_Transform
from core.calibration.JointCalibrator import JointCalibrator


def run(args):
    if not args.supervised_motion:
        raise RuntimeError('Requires previously approved on-site supervision')
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=False, parents=True)
    CONFIG_PATHS['txt_dir'] = str(output)
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'config').iterdir() if p.is_file()}
    stop, done = threading.Event(), threading.Event()
    robot = SupervisedRobot(rby.create_robot('192.168.1.40:50051', 'a'), stop)
    marker, cal, watcher = None, None, None
    report = {'sweep_seconds': 15., 'exposure_us': 6000., 'completed': False}
    def log(message):
        print(message, flush=True)
        with (output / 'run.log').open('a', encoding='utf-8') as stream:
            stream.write(str(message) + '\n')
    def watch():
        deadline = time.monotonic() + 300.
        while not done.wait(.2):
            if (output / 'STOP').exists() or time.monotonic() > deadline:
                robot.halt()
                if cal is not None:
                    cal.stop_requested = True
                return
    try:
        if not robot.connect(max_retries=1, timeout_ms=3000):
            raise RuntimeError('Robot connection failed')
        info = robot.get_robot_info()
        if info.robot_model_name.lower() != 'a' or info.robot_model_version.removeprefix('v') != '1.2':
            raise RuntimeError(f'Unexpected robot: {info}')
        if robot.get_control_manager_state().state != rby.ControlManagerState.State.Enabled:
            raise RuntimeError('Control manager is not enabled')
        if not robot.is_servo_on('^(?!.*wheel).*$'):
            raise RuntimeError('Body/arm/head servos are not all enabled')
        state = robot.get_state()
        baseline = np.array(state.position[robot.model().right_arm_idx], copy=True)
        expected = [-55., -45., 25., -127., 90., -.26235, 0.]
        if np.max(np.abs(np.rad2deg(baseline) - expected)) > .5:
            raise RuntimeError('Robot is outside the reviewed starting posture')
        if np.max(np.abs(state.velocity)) > .01:
            raise RuntimeError('Robot is already moving')
        baseline[6] = 0.  # Restore the reviewed J6 center after cancellation.
        report['baseline_deg'] = np.rad2deg(baseline).tolist()
        marker = Marker_Transform(sim=False, robot=robot, robot_version='1.2')
        if marker.sim or marker.camera is None:
            raise RuntimeError('Real camera required')
        marker.set_marker_type('plate')
        marker.set_camera_exposure(6000., auto_exposure=False)
        auto, configured = marker.get_camera_exposure()
        for _ in range(5):
            marker.camera.capture_image()
        actual = marker.get_actual_exposure()
        log(f'[EXPOSURE] auto={auto}, configured={configured}, actual={actual}')
        if auto or abs(configured - 6000.) > 1. or abs(actual - 6000.) > 100.:
            raise RuntimeError('Manual 6000us exposure not verified')
        report['intrinsics'] = marker.intrinsics_metadata
        cal = JointCalibrator(marker, robot)
        cal.include_head_motion = False
        def teaching(*unused):
            raise RuntimeError('Marker not visible; no automatic teaching or retry')
        cal.marker_problem_callback = teaching
        if not marker.get_marker_transform(sampling_time=1., side='right'):
            raise RuntimeError('Right marker is not visible')
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        original_detect = marker.marker_detection.detect
        images, observations = [], []
        def record(image, *positional, **kwargs):
            result = original_detect(image, *positional, **kwargs)
            if kwargs.get('use_filter') is False:
                if len(images) >= 600:
                    raise RuntimeError('Capture exceeded expected 15s frame count')
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                observations.append([(str(mid), np.asarray(pose).tolist()) for mid, pose in result])
            return result
        marker.marker_detection.detect = record
        datasets = []
        for tag, axis, start, end in [('A', 4, -15., 15.), ('B', 6, -15., 15.), ('C', 5, -15., 0.)]:
            images.clear()
            observations.clear()
            poses = cal.perform_single_joint_sweep('right', axis, baseline,
                start, end, 15., label=f'Raw image {tag}', log_callback=log, mode='wrist_pitch')
            if poses is None or stop.is_set():
                raise RuntimeError('Sweep failed or stopped')
            cal.save_observed_points('right', axis, poses, 'joint_' + tag)
            datasets.append(poses)
            np.savez_compressed(output / f'images_{tag}.npz', gray=np.stack(images))
            (output / f'observations_{tag}.json').write_text(json.dumps(observations), encoding='utf-8')
            log(f'[SAVED] {tag}: {len(images)} raw grayscale images, {len(poses)} poses')
        result = cal.compute_calibration_results('right', 'wrist_pitch',
            datasets[0], datasets[1], dataset_C=datasets[2], log_callback=log)
        report['measurement'] = {k: v for k, v in result.items() if k != '_plot_data'}
        report['completed'] = True
    except BaseException as error:
        report['error'] = str(error)
        if robot.is_connected():
            robot.halt()
        raise
    finally:
        done.set()
        if watcher is not None:
            watcher.join(timeout=2.)
        if marker is not None and marker.camera is not None:
            marker.camera.stream_off()
        robot.disconnect()
        report['configs_preserved'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in protected.items())
        (output / 'status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        log(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervised-motion', action='store_true')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
