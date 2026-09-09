"""Offline tests of fixed bracket/offset hypothesis; no camera or robot I/O."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from replay_j6_sweeps import read_blocks
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator, _marker_axis


def transform(rpy, xyz=(0., 0., 0.)):
    t = np.eye(4)
    t[:3,:3] = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
    t[:3,3] = xyz
    return t


def run():
    files = [*sorted((ROOT/'core').rglob('*.py')), *sorted((ROOT/'config').glob('*.yaml')), ROOT/'main_ui.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    cal = JointCalibrator.__new__(JointCalibrator)
    cal.robot, cal.robot_version = None, '1.2'
    nominal = cal._default_parameters.nominal_brackets['1.2']['right']
    bracket_nominal = transform(nominal[3:], (.05, .02, .08))
    report = {'synthetic_cases': [], 'recorded_coordinate_changes': [], 'legacy_midpoint_comparison': []}
    scenarios = [([0,0,0], [0,0,0], 0., 0.),
                 ([12,-8,17], [0,0,0], 0., 0.),
                 ([0,0,0], [35,-22,41], 4., -3.),
                 ([12,-8,17], [35,-22,41], 4., -3.),
                 ([-20,15,-10], [-30,25,60], -5., 6.)]
    for bracket_error, upstream_error, off5, off6 in scenarios:
        upstream = transform(upstream_error, (.12, -.08, .4))
        bracket = bracket_nominal @ transform(bracket_error, (.003, -.004, .002))
        a = np.array([upstream @ transform([0,off5,0]) @ transform([0,0,q+off6]) @ bracket
                      for q in np.linspace(-15,15,121)])
        b = np.array([upstream @ transform([0,q+off5,0]) @ transform([0,0,off6]) @ bracket
                      for q in np.linspace(-10,10,121)])
        result = cal.compute_calibration_results('right', 'wrist_yaw2', a, b)
        report['synthetic_cases'].append(dict(bracket_rotation_error_deg=bracket_error,
            upstream_rotation_deg=upstream_error, j5_offset_deg=off5, j6_offset_deg=off6,
            accepted=result['measurement_accepted'], quality=result['quality_diagnostics'],
            correction=result.get('optimal_offset'), reason=result.get('failure_reason')))
        assert result['measurement_accepted'], result
    # Negative control: a frame-varying orientation error must still be rejected.
    bad = b.copy()
    for index, error in enumerate(np.linspace(-4., 4., len(bad))):
        bad[index,:3,:3] = bad[index,:3,:3] @ Rotation.from_euler('x', error, degrees=True).as_matrix()
    result = cal.compute_calibration_results('right', 'wrist_yaw2', a, bad)
    report['varying_orientation_negative_control'] = dict(accepted=result['measurement_accepted'],
        quality=result['quality_diagnostics'])
    assert not result['measurement_accepted'], result
    blocks = [read_blocks(ROOT/f'result/result_txt/sweep_points_right_joint_{tag}_axis_{axis}.txt')
              for tag,axis in [('A',6),('B',5)]]
    for tag, rows in zip(('A','B'), blocks):
        d = rows[-1].reshape(-1,4,4)
        axis = BaseCalibrator.fit_observed_circle(d)['axis']
        for rpy in ([0,0,0], [12,-8,17], [-20,15,-10], [90,0,0]):
            adjusted = d.copy()
            adjusted[:,:3,:3] = d[:,:3,:3] @ Rotation.from_euler('xyz',rpy,degrees=True).as_matrix()
            report['recorded_coordinate_changes'].append(dict(tag=tag, fixed_rotation_deg=rpy,
                diagnostics=_marker_axis(adjusted, axis)[1]))
    # Isolate only the historical midpoint projection using today's fitted axes.
    # This is NOT a replay of the old encoder/FK-assisted circle fitting.
    reference = bracket_nominal[:3,:3].T @ [0.,1.,0.]
    for number, (rows_a, rows_b) in enumerate(zip(*blocks), 1):
        a, b = rows_a.reshape(-1,4,4), rows_b.reshape(-1,4,4)
        na, nb = BaseCalibrator.fit_observed_circle(a)['axis'], BaseCalibrator.fit_observed_circle(b)['axis']
        values = []
        for shift in range(-5,6):
            n6 = a[len(a)//2+shift,:3,:3].T @ na
            n5 = b[len(b)//2+shift,:3,:3].T @ nb
            ideal6 = bracket_nominal[:3,:3].T @ [0.,0.,1.]
            n6 *= 1 if n6 @ ideal6 >= 0 else -1
            n5 *= 1 if n5 @ reference >= 0 else -1
            actual = n5 - (n5 @ n6)*n6
            ref = reference - (reference @ n6)*n6
            actual /= np.linalg.norm(actual)
            ref /= np.linalg.norm(ref)
            values.append(dict(frame_shift=shift, raw_angle_deg=float(np.rad2deg(
                np.arctan2(actual @ np.cross(n6,ref), actual @ ref)))))
        report['legacy_midpoint_comparison'].append(dict(block=number, values=values,
            caveat='Old projection only; current circles/nominal reference, not full old estimator'))
    report['production_files_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert report['production_files_unchanged']
    output = ROOT/'result/j6_fixed_offsets_diagnosis.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    run()
