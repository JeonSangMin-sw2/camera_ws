"""Read-only marker/log audit; no calibration, robot, camera, or optimization.

Usage: python tests/analyze_calibration_results.py --root SNAPSHOT --output audit.json
The snapshot contains result/, config/, core/, and main_ui.py for provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


def blocks(path):
    rows = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if '===' in line:
            if rows:
                yield np.array(rows)
                rows = []
        elif line.strip() and not line.startswith('#'):
            row = [float(v) for v in line.split(',')]
            if len(row) != 58:
                raise ValueError(f'{path}: expected 58 columns, got {len(row)}')
            rows.append(row)
    if rows:
        yield np.array(rows)


def audit(root):
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    result = {'input_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths}, 'sweeps': [], 'datasets': [], 'joint_logs': []}
    for path in sorted((root / 'result/result_txt').glob('sweep_points_*.txt')):
        for block_id, rows in enumerate(blocks(path), 1):
            poses = rows[:, 10:26].reshape(-1, 4, 4)
            rotations = poses[:, :3, :3]
            dr = np.rad2deg(Rotation.from_matrix(
                rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).magnitude())
            dq = np.abs(np.diff(rows[:, 0]))
            mismatch = np.abs(dr - dq)
            outliers = np.flatnonzero(mismatch > 3.)
            result['sweeps'].append({
                'file': path.name, 'block': block_id, 'frames': len(rows),
                'finite': bool(np.isfinite(rows).all()),
                'encoder_arc_deg': float(np.ptp(rows[:, 0])),
                'depth_range_mm': [float(rows[:, 3].min()), float(rows[:, 3].max())],
                'mismatch_p50_p95_max_deg': np.percentile(mismatch, [50, 95, 100]).tolist(),
                'outlier_pairs': [{'rows_1based': [int(i + 1), int(i + 2)],
                                   'marker_rotation_deg': float(dr[i]), 'encoder_delta_deg': float(dq[i]),
                                   'position_step_mm': float(np.linalg.norm(poses[i+1, :3, 3] - poses[i, :3, 3])*1000)}
                                  for i in outliers]})
    for path in sorted((root / 'result/result_step2').glob('dataset_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            qa, qh, obs = data['q_arm'], data['q_head'], data['marker']
        rot = obs[:, :, :3, :3]
        rel = np.linalg.inv(obs[:, 0]) @ obs[:, 1]
        pairs = []
        for i in range(len(qa)):
            for j in range(i+1, len(qa)):
                if np.max(np.abs(np.rad2deg(qa[i] - qa[j]))) < .03:
                    pairs.append({'samples_1based': [i+1, j+1],
                                  'position_mm': float(np.linalg.norm(rel[i, :3, 3] - rel[j, :3, 3])*1000),
                                  'rotation_deg': float(np.rad2deg(Rotation.from_matrix(
                                      rel[i, :3, :3].T @ rel[j, :3, :3]).magnitude()))})
        result['datasets'].append({'file': path.name, 'samples': len(qa),
            'finite': bool(np.isfinite(obs).all() and np.isfinite(qa).all() and np.isfinite(qh).all()),
            'orthogonality_max': float(np.max(np.abs(rot.transpose(0, 1, 3, 2) @ rot - np.eye(3)))),
            'homogeneous_row_error': float(np.max(np.abs(obs[:, :, 3, :] - [0, 0, 0, 1]))),
            'same_arm_pairs': pairs})
    for path in sorted((root / 'result/result_txt').glob('joint_calib_debug_*.txt')):
        text = path.read_text(encoding='utf-8-sig')
        result['joint_logs'].append({'file': path.name, 'evidence_lines': [line for line in text.splitlines()
            if re.search(r'ITERATION|Calculated Offset Correction|Step Correction:|Updated Absolute Offset|Recommended Absolute Offset|noise floor|Damping fallback', line)]})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Sweeps: {len(report['sweeps'])}, frames: {sum(s['frames'] for s in report['sweeps'])}, "
          f"blocks with >3deg mismatch: {sum(bool(s['outlier_pairs']) for s in report['sweeps'])}, "
          f"pairs: {sum(len(s['outlier_pairs']) for s in report['sweeps'])}")
    for dataset in report['datasets']:
        pairs = dataset['same_arm_pairs']
        print(dataset['file'], 'finite=', dataset['finite'], 'repeat pairs=', len(pairs))
        if pairs:
            print('  relative marker position median/max mm=', np.median([p['position_mm'] for p in pairs]),
                  max(p['position_mm'] for p in pairs), 'rotation median/max deg=',
                  np.median([p['rotation_deg'] for p in pairs]), max(p['rotation_deg'] for p in pairs))
