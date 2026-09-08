"""Camera-independent bracket fit from encoder-indexed wrist sweeps.

Within each sweep the camera must be stationary. Relative marker transforms
cancel the unknown camera and all joints upstream of the swept wrist joint.
J5 and J6 are fixed inputs from the joint stage; this module estimates ONLY
the six bracket pose parameters and never changes a joint offset. J6 remains
an effective convention, not an independently measured physical offset.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from core.calibration_optimizer import compute_fk, make_transform, se3_log, so3_exp


def fit_bracket_sweeps(robot, side, sweeps, nominal, j6_physical_deg, j5_physical_deg):
    model, dyn = robot.model(), robot.get_dynamics()
    indices = getattr(model, f'{side}_arm_idx')
    pairs = []
    for sweep in sweeps:
        poses, qs = np.asarray(sweep['captured_poses']), np.asarray(sweep['captured_q_full'])
        if len(poses) < 10 or len(poses) != len(qs):
            raise ValueError('Bracket fit requires at least 10 synchronized poses/encoders per sweep')
        head = getattr(model, 'head_idx', [])
        if len(head) and np.max(np.ptp(qs[:, head], axis=0)) > np.deg2rad(.02):
            raise ValueError('Camera moved during wrist sweep; reacquire with stationary head')
        selected = np.unique(np.linspace(0, len(qs)-1, min(24, len(qs)), dtype=int))
        ref = len(qs)//2
        for i in selected:
            if i != ref:
                pairs.append((qs[ref], qs[i], np.linalg.inv(poses[ref]) @ poses[i]))
    B0 = make_transform(nominal)
    # The fixed proximal frame removes irrelevant upstream encoder errors.
    def fk(q):
        adjusted = q.copy()
        adjusted[indices[5]] += np.deg2rad(j5_physical_deg)
        adjusted[indices[6]] += np.deg2rad(j6_physical_deg)
        return compute_fk(robot, dyn, adjusted, f'ee_{side}', base_link=f'link_{side}_arm_3')[1]
    relative_fk = [(np.linalg.inv(fk(q0)) @ fk(qi), observed) for q0, qi, observed in pairs]
    def bracket(x):
        B = B0.copy()
        B[:3, :3] = so3_exp(x[:3]) @ B0[:3, :3]
        B[:3, 3] += x[3:6]
        return B
    def residual(x):
        B = bracket(x)
        invB = np.linalg.inv(B)
        values = []
        for relative, observed in relative_fk:
            predicted = invB @ relative @ B
            e = se3_log(np.linalg.inv(predicted) @ observed)
            # Explicit sensor units, identical in simulation and real fitting.
            values.extend(e / np.r_[np.full(3, np.deg2rad(.1)), np.full(3, .0001)])
        return np.asarray(values)
    bound = np.r_[np.full(3, np.deg2rad(3)), np.full(3, .005)]
    fit = least_squares(residual, np.zeros(6),
                        bounds=(-bound, bound), jac='3-point', x_scale='jac',
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    saturated = bool(np.any(np.abs(fit.x) > .995 * bound))
    normalized_rms = float(np.sqrt(np.mean(fit.fun**2)))
    if not fit.success or rank != 6 or saturated or normalized_rms > 5.0:
        raise RuntimeError(
            f'Bracket-only fit rejected: converged={fit.success}, rank={rank}/6, '
            f'at_bounds={saturated}, normalized_rms={normalized_rms:.4f}. '
            'Calibrate J5/J6 before fitting the bracket; check fixed joint inputs and sweep quality.')
    B = bracket(fit.x)
    rpy = Rotation.from_matrix(B[:3, :3]).as_euler('xyz', degrees=True)
    return {'converged': True, 'x_e': B[0, 3]*1000, 'y_e': B[1, 3]*1000, 'z_e': B[2, 3]*1000,
            'roll_e': rpy[0], 'pitch_e': rpy[1], 'yaw_e': rpy[2],
            'fixed_j5_correction_deg': -float(j5_physical_deg),
            'fixed_j6_correction_deg': -float(j6_physical_deg),
            'fit_scope': 'bracket_only',
            'offset_convention': 'home_correction_minus_physical_error',
            'j6_mode': 'effective_reference; coaxial bracket rotation not independently separated',
            'data_rank': rank, 'normalized_residual_rms': normalized_rms,
            'rotation_error_deg': float(np.linalg.norm(np.rad2deg(fit.x[:3])))}
