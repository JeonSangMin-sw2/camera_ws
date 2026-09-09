"""Offline diagnostic for the 2026-09-09 J5 capture; never connects to a robot.

Run from the repository root with .venv/Scripts/python.exe.
Bootstrap ranges describe sensitivity to resampling recorded frames, not
physical accuracy: systematic camera error and temporal correlation remain.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from replay_j6_sweeps import read_blocks
from core.calibration.JointCalibrator import JointCalibrator


if __name__ == '__main__':
    folder = ROOT / 'result' / 'result_txt'
    blocks = [read_blocks(folder / f'sweep_points_right_joint_{tag}_axis_{axis}.txt')
              for tag, axis in [('A', 4), ('B', 6), ('C', 5)]]
    if not blocks[0] or len({len(b) for b in blocks}) != 1:
        raise ValueError('Matching nonempty A/B/C blocks are required')
    cal = JointCalibrator.__new__(JointCalibrator)
    rng = np.random.default_rng(20260909)
    reports = []
    for number, rows in enumerate(zip(*blocks), 1):
        datasets = [r.reshape(-1, 4, 4) for r in rows]
        result = cal.compute_calibration_results('right', 'wrist_pitch',
            datasets[0], datasets[1], dataset_C=datasets[2])
        report = dict(block=number, accepted=result['measurement_accepted'],
            reason=result.get('failure_reason'), quality=result['quality_diagnostics'])
        if not result['measurement_accepted']:
            samples = []
            for repeat in range(40):
                # Sorted indices preserve the commanded direction of each sweep.
                ds = [d[np.sort(rng.integers(0, len(d), len(d)))] for d in datasets]
                res = cal.compute_calibration_results('right', 'wrist_pitch',
                    ds[0], ds[1], dataset_C=ds[2])
                q = res['quality_diagnostics']
                if 'center_distance_mm' in q:
                    samples.append([q['relative_correction_deg'],
                                    q['center_distance_mm'], q['radius_difference_mm']])
            report['resampling_valid_count'] = len(samples)
            report['resampling_columns'] = ['delta_deg', 'center_mm', 'radius_difference_mm']
            report['resampling_percentiles_2_5_50_97_5'] = (
                np.percentile(samples, [2.5, 50, 97.5], axis=0).tolist() if samples else [])
        reports.append(report)
        print(json.dumps(report), flush=True)
    output = ROOT / 'result' / 'j5_capture_analysis.json'
    output.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(f'Saved {output}', flush=True)
