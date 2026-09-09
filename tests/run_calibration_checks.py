"""Supported checks: unit (no motion), replay (no connection), connected (opt-in).

Examples:
  .venv/bin/python tests/run_calibration_checks.py unit
  .venv/bin/python tests/run_calibration_checks.py replay result/result_txt --side left
  .venv/bin/python tests/run_calibration_checks.py replay data.npz --version 1.2 --output /tmp/new-replay
  .venv/bin/python tests/run_calibration_checks.py connected --confirm-local-simulator --output /tmp/new-run
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest='command', required=True)
    unit = commands.add_parser('unit', help='offline regression suite; no power, servo, motion or home writes')
    unit.add_argument('--robot-address', help='optional read-only SDK connection checks')
    replay = commands.add_parser('replay', help='TXT folder or NPZ through current core; never robot connection')
    replay.add_argument('input', type=Path)
    replay.add_argument('--side', choices=('right', 'left'), default='right', help='TXT arm; NPZ needs --arms')
    replay.add_argument('--arms', nargs='+', choices=('right', 'left'))
    replay.add_argument('--version', choices=('1.2', '1.3'))
    replay.add_argument('--output', type=Path, help='new directory, required for NPZ')
    replay.add_argument('--settings', type=Path, help='explicit settings for legacy NPZ without estimation snapshot')
    replay.add_argument('--no-head', action='store_true', help='disable head estimation, not head-camera kinematics')
    connected = commands.add_parser('connected', help='production SDK full sequence, confirmed local simulator only')
    connected.add_argument('--confirm-local-simulator', action='store_true')
    connected.add_argument('--output', required=True, type=Path)
    connected.add_argument('--no-head', action='store_true',
                           help='disable head servo/motion; does not remove an installed head or relocate its camera')
    return result


def _unit(args):
    if args.robot_address:
        os.environ['CALIBRATION_ROBOT_ADDRESS'] = args.robot_address
    else:
        os.environ.pop('CALIBRATION_ROBOT_ADDRESS', None)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'tests': result.testsRun, 'failures': len(result.failures),
                      'errors': len(result.errors), 'skipped': len(result.skipped),
                      'sdk_motion': False}))
    return 0 if result.wasSuccessful() else 1


def _new_output(path):
    if path is None:
        raise ValueError('--output must name a new evidence directory')
    path = path.resolve()
    if path.exists():
        raise ValueError(f'Output already exists; refusing overwrite: {path}')
    return path


def _replay_npz(args):
    import yaml
    import numpy as np
    from calibration_support import OfflineRobot
    from core.config_store import RobotConfig
    from core.calibration_core import load_npz_dataset, validate_dataset, OptimizerContext, run_calibration_optimizer
    output = _new_output(args.output)
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    qa, qh, marker, metadata = load_npz_dataset(args.input, return_metadata=True)
    if qa.ndim != 2 or qa.shape[1] not in (7, 14):
        raise ValueError('NPZ arm joint data must be a matrix with 7 or 14 columns')
    recorded_version = metadata.get('robot_version')
    version = args.version or recorded_version
    if version not in ('1.2', '1.3'):
        raise ValueError('NPZ has no supported robot_version; specify --version')
    if recorded_version and args.version and recorded_version != args.version:
        raise ValueError('Requested version conflicts with NPZ metadata')
    arms = args.arms or (['right', 'left'] if qa.shape[1] == 14 else None)
    if arms is None or len(set(arms)) != len(arms):
        raise ValueError('Specify --arms right or --arms left for single-arm NPZ')
    if len(arms) == 2 and arms != ['right', 'left']:
        raise ValueError('Dual-arm NPZ uses right then left ordering')
    snapshot = metadata.get('estimation_camera_snapshot')
    settings_path = args.settings
    if settings_path:
        settings = yaml.safe_load(settings_path.read_text())
        camera = {**settings['camera'], **settings.get('marker', {})}
        provenance = {'settings_file': str(settings_path.resolve()),
                      'settings_sha256': hashlib.sha256(settings_path.read_bytes()).hexdigest()}
    elif isinstance(snapshot, dict):
        camera = dict(snapshot)
        settings = {}
        provenance = {'settings_source': 'NPZ estimation_camera_snapshot'}
    else:
        raise ValueError('Legacy NPZ has no estimation snapshot: provide --settings explicitly; no current-config guessing')
    head = not args.no_head and metadata.get('head_motion_enabled', True)
    validate_dataset(qa, qh, marker, head and qh is not None, arms)
    robot = OfflineRobot(version)
    context = OptimizerContext(robot=robot, model=robot.model(), camera_config=camera,
        robot_version=version, include_head_motion=head,
        joint_offsets_store=settings.get('joint_offset', {}), capture_metadata=metadata,
        nominal_brackets=RobotConfig.load().nominal_brackets)
    output.mkdir(parents=True, exist_ok=False)
    result = run_calibration_optimizer(context, arms, head, True, qa, qh, marker,
        str(output / 'optimizer.json'), lambda_cam_pos=0., lambda_cam_rot=0., log_callback=print)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Input changed during replay')
    summary = {'input': str(args.input.resolve()), 'input_sha256': source_hash,
               'robot_version': version, 'sdk_motion': False, 'sdk_connection': False,
               'provenance': provenance, 'diagnostics': result['diagnostics']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0 if result['diagnostics'].get('converged') and result['diagnostics'].get('observable') else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'unit':
            return _unit(args)
        if args.command == 'replay':
            if args.input.is_dir():
                from replay_j6_sweeps import replay
                result = replay(args.input, args.side, args.version or '1.2')
                print(json.dumps(result, indent=2))
                return 0 if all(block['measurement_accepted'] for block in result['blocks']) else 1
            return _replay_npz(args)
        if not args.confirm_local_simulator:
            raise ValueError('SDK motion requires --confirm-local-simulator and no competing controller')
        args.output = str(_new_output(args.output))
        # Optional partial/shortened runs belong to the specialist runner,
        # not this default-process verification entry point.
        args.sweep_seconds = None
        args.only_step1 = False
        args.resume_step2 = None
        from run_connected_calibration import run
        run(args)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f'CHECK FAILED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
