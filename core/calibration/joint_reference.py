"""J6 effective reference from stationary-camera, single-joint sweeps.

R_camera_marker.T @ axis_camera is invariant during that axis's sweep.
Estimate it from all frames, never from one midpoint frame. This is a
measurement-consistency check, NOT a physical-offset accuracy certificate.
Coaxial bracket rotation remains indistinguishable from J6 zero error.
No FK, robot commands, bracket fitting, or configuration access occurs here.
"""
import numpy as np


# Gross pose-consistency limits, independent of the 0.06 degree joint
# convergence rule. A constant pose bias can pass these checks.
AXIS_CONSISTENCY_DEG = 0.5
MIN_INLIER_FRACTION = 0.8
MIN_FRAMES = 10


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm < 1e-8:
        raise ValueError('Invalid or degenerate sweep axis')
    return vector / norm


def _distance_deg(a, b):
    return np.rad2deg(np.arccos(np.clip(np.asarray(a) @ b, -1., 1.)))


def _marker_axis(poses, camera_axis, reference_axis):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4,4) or len(poses) < MIN_FRAMES:
        raise ValueError(f'At least {MIN_FRAMES} camera poses are required per sweep')
    if not np.all(np.isfinite(poses)):
        raise ValueError('Non-finite camera pose')
    rotations = poses[:, :3, :3]
    if (np.max(np.abs(rotations.transpose(0,2,1) @ rotations - np.eye(3))) > 1e-3
            or np.max(np.abs(np.linalg.det(rotations) - 1.)) > 1e-3):
        raise ValueError('Invalid camera rotation matrix')
    vectors = np.einsum('nji,j->ni', rotations, _unit(camera_axis))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    center = _unit(np.median(vectors, axis=0))
    distances = _distance_deg(vectors, center)
    inliers = distances <= AXIS_CONSISTENCY_DEG
    fraction = float(np.mean(inliers))
    # A smooth drift or two different branches must not be hidden by an
    # average, even if individual frames form locally tight groups.
    width = max(1, len(vectors)//3)
    early = _unit(np.median(vectors[:width], axis=0))
    late = _unit(np.median(vectors[-width:], axis=0))
    early_late = float(_distance_deg(early, late))
    accepted = fraction >= MIN_INLIER_FRACTION and early_late <= AXIS_CONSISTENCY_DEG
    diagnostics = dict(accepted=bool(accepted), frames=len(vectors),
        inlier_fraction=fraction, p90_deviation_deg=float(np.percentile(distances,90)),
        early_late_deg=early_late, consistency_limit_deg=AXIS_CONSISTENCY_DEG)
    if not accepted:
        return None, diagnostics
    axis = _unit(np.mean(vectors[inliers], axis=0))
    # Resolve only the plane-normal sign; do not pull the estimate toward
    # the nominal bracket direction or independently flip individual frames.
    if np.dot(axis, reference_axis) < 0:
        axis = -axis
    return axis, diagnostics


def estimate_j6_reference(poses_a, axis_a, poses_b, axis_b,
                          nominal_rotation, encoder_j6_deg, is_v13=False):
    """Return an effective absolute home correction, or a failed measurement.

Sweep A rotates J6; sweep B rotates J5 with J6 held stationary. Supplied
camera-frame circle axes may be derived using either positions alone or
encoder-indexed positions. No encoder is used in the consistency test.
"""
    diagnostics = {}
    try:
        nominal = np.asarray(nominal_rotation, dtype=float)
        if (nominal.shape != (3,3) or not np.all(np.isfinite(nominal))
                or not np.allclose(nominal.T @ nominal, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(nominal), 1., atol=1e-6)
                or not np.isfinite(encoder_j6_deg)):
            raise ValueError('Invalid nominal bracket rotation or J6 encoder')
        ref_j6 = nominal.T @ ([1.,0.,0.] if is_v13 else [0.,0.,1.])
        ref_j5 = nominal.T @ [0.,1.,0.]
        n6, diagnostics['j6_sweep'] = _marker_axis(poses_a, axis_a, ref_j6)
        n5, diagnostics['j5_sweep'] = _marker_axis(poses_b, axis_b, ref_j5)
        if n6 is None or n5 is None:
            raise ValueError('Inconsistent marker-frame sweep axes; no unique stable reference selected')
        actual = _unit(n5 - np.dot(n5,n6)*n6)
        reference = _unit(ref_j5 - np.dot(ref_j5,n6)*n6)
        cross = _unit(np.cross(n6,reference))
        raw = float(np.rad2deg(np.arctan2(np.dot(actual,cross), np.dot(actual,reference))))
        return dict(measurement_accepted=True, raw_diff_deg=raw,
                    optimal_offset=raw + float(encoder_j6_deg),
                    n6_marker_actual=n6, n5_marker_actual=n5,
                    quality_diagnostics=diagnostics, j6_mode='effective_bracket_reference')
    except ValueError as error:
        # Deliberately omit optimal_offset. The caller must not manufacture
        # a zero correction or issue another inner-iteration correction.
        return dict(measurement_accepted=False, failure_reason=str(error),
                    quality_diagnostics=diagnostics, j6_mode='effective_bracket_reference')
