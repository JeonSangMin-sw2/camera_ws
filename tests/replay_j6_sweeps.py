"""Read-only replay of legacy J6/J5 TXT sweeps; no robot connection.

python tests/replay_j6_sweeps.py --input-dir result/result_txt
Legacy TXT lacks full encoders/timestamps. This checks raw camera-axis
consistency, not physical J6 accuracy or camera/encoder latency.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import estimate_j6_reference


def read_blocks(path):
    blocks = []
    for block in path.read_text().split('=== NEW ITERATION ==='):
        rows = []
        for line in block.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            row = [float(value) for value in line.split(',')]
            if len(row) != 58:
                raise ValueError(f'Expected legacy 58-column TXT: {path}')
            rows.append(row)
        if rows:
            blocks.append(np.asarray(rows))
    return blocks


def replay(folder, side='right', version='1.2'):
    paired = [read_blocks(folder/f'sweep_points_{side}_joint_{tag}_axis_{axis}.txt')
              for tag, axis in [('A',6),('B',5)]]
    if not paired[0] or len(paired[0]) != len(paired[1]):
        raise ValueError('Matching nonempty A/B sweep blocks are required')
    nominal = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES[version][side]
    R0 = Rotation.from_euler('xyz', nominal[3:], degrees=True).as_matrix()
    reports = []
    for i, pair in enumerate(zip(*paired),1):
        fitted, metrics = [], []
        for rows in pair:
            poses = rows[:,10:26].reshape(-1,4,4)
            # Joint calibration commands both sweeps in increasing order.
            # Legacy angle/encoder columns are deliberately ignored.
            fit = BaseCalibrator.fit_observed_circle(poses, 1)
            fitted.extend([poses, fit['axis']])
            rotations = Rotation.from_matrix(poses[:,:3,:3])
            jumps = np.rad2deg((rotations[:-1].inv()*rotations[1:]).magnitude())
            metrics.append(dict(frames=len(rows), position_fit_rmse_mm=fit['rmse'],
                                largest_raw_rotation_step_deg=float(np.max(jumps))))
        result = estimate_j6_reference(*fitted, R0)
        reports.append(dict(block=i, sweeps=metrics,
            measurement_accepted=result['measurement_accepted'],
            failure_reason=result.get('failure_reason'),
            quality_diagnostics=result['quality_diagnostics']))
    return dict(scope='offline raw-camera sweep consistency; not absolute J6 recovery',
                legacy_missing=['full_encoder_vector','frame_timestamp','encoder_timestamp','image_corners'],
                blocks=reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--side', choices=['right','left'], default='right')
    parser.add_argument('--version', choices=['1.2','1.3'], default='1.2')
    args = parser.parse_args()
    print(json.dumps(replay(args.input_dir, args.side, args.version), indent=2))
