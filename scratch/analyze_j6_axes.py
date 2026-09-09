"""Read-only offline diagnosis of recorded J6 poses; never controls hardware.

Rotation-only axes are diagnostic comparisons, not replacement calibration
results. Their self-consistency does not establish physical accuracy.
"""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import yaml
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from replay_j6_sweeps import read_blocks
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import _marker_axis


def angle(a, b):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


def rotation_axis(rotations, reference):
    # Maximize ||mean(R).T n|| for a unit camera-frame axis n. This minimizes
    # marker-frame axis variance, using rotations only, without encoders/FK.
    u, _, _ = np.linalg.svd(rotations.mean(axis=0))
    n = u[:, 0]
    return n if n @ reference >= 0 else -n


def run():
    output = ROOT / 'result/j6_axis_diagnosis_20260909'
    output.mkdir(exist_ok=True)
    protected = [*sorted((ROOT/'core').rglob('*.py')), ROOT/'main_ui.py',
                 *sorted((ROOT/'config').glob('*.yaml'))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report = {'thresholds_changed': False, 'hardware_used': False, 'sweeps': []}
    for tag, axis in [('A', 6), ('B', 5)]:
        source = ROOT/f'result/result_txt/sweep_points_right_joint_{tag}_axis_{axis}.txt'
        blocks = read_blocks(source)
        axes = []
        for number, rows in enumerate(blocks, 1):
            poses = rows.reshape(-1, 4, 4)
            r = poses[:, :3, :3]
            p = poses[:, :3, 3]
            circle = BaseCalibrator.fit_observed_circle(poses)
            n = circle['axis']
            nr = rotation_axis(r, n)
            axes.append((n, nr))
            dr = Rotation.from_matrix(r[1:] @ r[:-1].transpose(0, 2, 1)).magnitude()*180/np.pi
            dp = np.linalg.norm(np.diff(p, axis=0), axis=1)*1000.
            worst = np.argsort(dr)[-5:][::-1]
            comparison = dict(tag=tag, joint_index=axis, block=number, frames=len(poses),
                source=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                circle_rms_mm=circle['rmse'], circle_arc_deg=circle['arc_deg'],
                radius_mm=circle['radius'], position_axis=n.tolist(), rotation_axis=nr.tolist(),
                axis_disagreement_deg=angle(n, nr),
                original_gate=_marker_axis(poses, n)[1],
                rotation_only_axis_gate=_marker_axis(poses, nr)[1],
                rotation_step_percentiles_deg=np.percentile(dr, [50,90,99,100]).tolist(),
                steps_above_2deg=int(np.sum(dr > 2.)),
                worst_steps=[dict(frame=int(i), rotation_deg=float(dr[i]),
                    translation_mm=float(dp[i]),
                    position_chord_angle_deg=float(np.rad2deg(2*np.arcsin(
                        min(1., dp[i]/(2*circle['radius'])))))) for i in worst])
            # Hold the full position axis fixed. Removing only isolated jumps
            # cannot hide a broad systematic orientation disagreement.
            keep = np.ones(len(poses), dtype=bool)
            for i in np.flatnonzero(dr > 2.):
                keep[max(0, i-1):min(len(poses), i+3)] = False
            comparison['without_jump_neighbors'] = dict(frames=int(keep.sum()),
                original_axis_gate=_marker_axis(poses[keep], n)[1],
                rotation_only_axis_gate=_marker_axis(poses[keep], rotation_axis(r[keep], n))[1])
            # Contiguous-block deletion measures sensitivity, not physical CIs.
            jackknife = []
            for indices in np.array_split(np.arange(len(poses)), 10):
                mask = np.ones(len(poses), dtype=bool)
                mask[indices] = False
                fit = BaseCalibrator.fit_observed_circle(poses[mask])
                jackknife.append(angle(n, fit['axis']))
            comparison['position_axis_block_deletion_max_deg'] = max(jackknife)
            comparison['position_axis_block_deletion_angles_deg'] = jackknife
            report['sweeps'].append(comparison)
            if tag == 'B' and number == len(blocks):
                # Synthetic corner-noise experiment: this tests plausibility,
                # not the actual unsaved image corners of the real recording.
                index = int(worst[0])
                intrinsics = yaml.safe_load((ROOT/'config/camera_intrinsics.yaml').read_text(encoding='utf-8'))
                settings = yaml.safe_load((ROOT/'config/setting.yaml').read_text(encoding='utf-8'))
                half = settings['marker']['plate']['plate_size_mm'] * .8 / 2
                objects = np.array([[-half,-half,0],[half,-half,0],
                                    [half,half,0],[-half,half,0]], dtype=np.float32)
                matrix = np.array(intrinsics['camera_matrix'])
                distortion = np.array(intrinsics['dist_coeffs'])
                rotation = r[index]
                corners = cv2.projectPoints(objects, cv2.Rodrigues(rotation)[0],
                    p[index]*1000., matrix, distortion)[0].reshape(4,2)
                solutions = cv2.solvePnPGeneric(objects, corners, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
                experiment = dict(frame=index, synthetic=True, seed=42,
                    candidate_reprojection_rms_px=solutions[3].ravel().tolist(),
                    candidate_rotation_difference_deg=[float(Rotation.from_matrix(
                        cv2.Rodrigues(rv)[0] @ rotation.T).magnitude()*180/np.pi) for rv in solutions[1]],
                    trials=[])
                rng = np.random.default_rng(42)
                for sigma in (.05, .1, .2, .5, 1.):
                    errors = []
                    for _ in range(300):
                        _, rv, _ = cv2.solvePnP(objects,
                            (corners+rng.normal(0,sigma,(4,2))).astype(np.float32),
                            matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
                        errors.append(float(Rotation.from_matrix(
                            cv2.Rodrigues(rv)[0] @ rotation.T).magnitude()*180/np.pi))
                    experiment['trials'].append(dict(corner_noise_sigma_px=sigma, samples=300,
                        rotation_error_percentiles_deg=np.percentile(errors,[50,90,99,100]).tolist(),
                        errors_above_5_deg=int(np.sum(np.array(errors)>5.))))
                report['synthetic_pnp_experiment'] = experiment
        if len(axes) >= 2:
            report[f'{tag}_repeat_axis_difference_deg'] = dict(
                position=angle(axes[-1][0], axes[-2][0]),
                rotation=angle(axes[-1][1], axes[-2][1]))
    report['production_sources_and_config_unchanged'] = all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    (output/'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
