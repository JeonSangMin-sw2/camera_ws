"""Supervised real-camera repeat at the existing right-arm J5 posture.

No initialization of power/servos/control manager and no home-offset writes.
Uses production acquisition and fitting, holding the measured starting J5
command fixed across three repetitions. Never falls back to simulated markers.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import rby1_sdk as rby

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.config_store import CONFIG_PATHS
from core.marker_detection import Marker_Transform
from core.calibration.JointCalibrator import JointCalibrator


if __name__ == '__main__':
    if sys.argv[1:] != ['--supervised-motion']:
        raise SystemExit('Requires --supervised-motion and an on-site operator.')
    output = ROOT / 'result' / ('j5_repeat_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    CONFIG_PATHS['txt_dir'] = str(output)
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'config').iterdir() if p.is_file()}
    robot = rby.create_robot('192.168.1.40:50051', 'a')
    marker = None
    def log(message):
        print(message, flush=True)
        with (output / 'capture.log').open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')
    try:
        if not robot.connect(max_retries=1, timeout_ms=3000):
            raise RuntimeError('Robot connection failed')
        info = robot.get_robot_info()
        if str(info.robot_model_version).removeprefix('v') != '1.2':
            raise RuntimeError('Expected previously verified A@v1.2')
        state = robot.get_state()
        indices = robot.model().right_arm_idx
        baseline = np.array(state.position[indices], copy=True)
        expected = [-55., -45., 25., -127., 90., -.30221, 0.]
        if np.max(np.abs(np.rad2deg(baseline) - expected)) > .5:
            raise RuntimeError('Right arm moved from the reviewed starting posture')
        if np.max(np.abs(state.velocity)) > .01:
            raise RuntimeError('Robot is already moving')
        log(f'Output: {output}; robot={info}; fixed center deg={np.rad2deg(baseline)}')
        marker = Marker_Transform(sim=False, robot=robot, robot_version='1.2')
        if marker.sim or marker.camera is None:
            raise RuntimeError('A real camera is required; simulated fallback rejected')
        marker.set_marker_type('plate')
        marker.set_camera_exposure(6000., auto_exposure=False)
        cal = JointCalibrator(marker, robot)
        cal.include_head_motion = False
        if not marker.get_marker_transform(sampling_time=2., side='right'):
            raise RuntimeError('Right marker not visible; no movement sent')
        (output / 'metadata.json').write_text(json.dumps({
            'robot': str(info), 'fixed_center_deg': np.rad2deg(baseline).tolist(),
            'intrinsics': marker.intrinsics_metadata,
            'scope': 'fixed-command real-camera repeat; no offset updates'}, indent=2), encoding='utf-8')
        for repeat in range(1, 4):
            log(f'REPEAT {repeat}/3: same fixed J5 command')
            datasets = []
            for tag, axis, start, end in [('A', 4, -15., 15.),
                                          ('B', 6, -15., 15.), ('C', 5, -15., 0.)]:
                poses = cal.perform_single_joint_sweep('right', axis, baseline,
                    start, end, 15., label=f'J5 repeat {repeat} {tag}',
                    log_callback=log, mode='wrist_pitch')
                if poses is None:
                    raise RuntimeError('Capture failed; no automatic retry')
                cal.save_observed_points('right', axis, poses, 'joint_' + tag)
                datasets.append(poses)
            result = cal.compute_calibration_results('right', 'wrist_pitch',
                datasets[0], datasets[1], dataset_C=datasets[2], log_callback=log)
            report = {k: v for k, v in result.items() if k != '_plot_data'}
            log(json.dumps(report))
            (output / f'repeat_{repeat}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        if marker is not None and marker.camera is not None:
            marker.camera.stream_off()
        robot.disconnect()
        changed = [str(p) for p, digest in protected.items()
                   if hashlib.sha256(p.read_bytes()).hexdigest() != digest]
        log(f'Configuration files changed: {changed}')
