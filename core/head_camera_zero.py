"""Camera-forward command coordinates, separate from mechanical joint homes.

Only the terminal tilt/camera gauge is redistributed. Pan's physical estimate
is retained; camera yaw assembly error belongs to the separate command zero.
"""
import numpy as np
from scipy.optimize import least_squares


def camera_command_to_encoder(command_rad, reference):
    """Map zero-relative head JOINT commands, not absolute optical Euler angles.

    The caller must validate the robot/session, head enable state and limits
    before motion. This helper never commands a robot or writes home offsets.
    """
    command = np.asarray(command_rad, dtype=float)
    zero = np.deg2rad(np.asarray(reference['encoder_zero_deg'], dtype=float))
    if command.shape != (2,) or zero.shape != (2,) or not np.all(np.isfinite([command, zero])):
        raise ValueError('Camera head command and encoder zero must be finite Pan/Tilt pairs')
    if not reference.get('accepted') or reference.get('reference_frame') != 'link_torso_5':
        raise ValueError('An accepted torso-frame camera zero is required')
    return command + zero


def camera_forward_zero(head_fk, mount_to_cam, head_offset, lower, upper,
                        redistribute_tilt=True):
    """Return an equivalent head/camera pair and the encoder pose facing torso +X.

    head_fk takes model Pan/Tilt radians and returns torso-to-head-mount SE(3).
    Bounds are encoder limits. The search is limited to +/-15 degrees around
    encoder zero, rejecting reversed/unreachable camera installations.
    """
    camera = np.asarray(mount_to_cam, dtype=float)
    offset = np.asarray(head_offset, dtype=float)
    lower = np.maximum(np.asarray(lower, dtype=float), np.deg2rad([-15., -15.]))
    upper = np.minimum(np.asarray(upper, dtype=float), np.deg2rad([15., 15.]))
    if (camera.shape != (4, 4) or offset.shape != (2,) or lower.shape != (2,)
            or upper.shape != (2,) or not np.all(np.isfinite(camera))
            or not np.all(np.isfinite([offset, lower, upper])) or np.any(lower >= upper)):
        raise ValueError('Invalid camera transform, head offsets or camera-zero search limits')
    if (not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-8)
            or np.linalg.det(camera[:3, :3]) < 0):
        raise ValueError('Camera transform must be a proper SE(3) transform')

    target = np.array([1., 0., 0.])

    def direction(encoder):
        return (head_fk(encoder + offset) @ camera)[:3, 2]

    fit = least_squares(lambda q: direction(q) - target,
                        np.clip(-offset, lower + 1e-10, upper - 1e-10),
                        bounds=(lower, upper), ftol=1e-13, xtol=1e-13, gtol=1e-13)
    error_deg = float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(direction(fit.x), target)),
                                           np.dot(direction(fit.x), target))))
    if not fit.success or error_deg > 1e-5 or np.linalg.matrix_rank(fit.jac, tol=1e-7) != 2:
        raise ValueError(f'Camera forward zero is unreachable or degenerate (error={error_deg:.4f} deg)')

    new_offset = offset.copy()
    new_camera = camera.copy()
    if redistribute_tilt:
        new_offset[1] = -fit.x[1]
        # Exact terminal-joint gauge: H(q+d_old) C_old = H(q+d_new) C_new.
        # Use FK, including the joint pivot translation; never copy an RPY scalar.
        delta = np.array([0., offset[1] - new_offset[1]])
        new_camera = np.linalg.inv(head_fk(np.zeros(2))) @ head_fk(delta) @ camera
        for q in (np.zeros(2), np.array([.12, -.09]), np.array([-.08, .13])):
            if not np.allclose(head_fk(q + offset) @ camera,
                               head_fk(q + new_offset) @ new_camera, atol=1e-9, rtol=0):
                raise ValueError('Selected camera link does not have a terminal Tilt gauge')

    reference = {
        'accepted': True,
        'reference_frame': 'link_torso_5',
        'forward_axis': [1., 0., 0.],
        'camera_optical_axis': [0., 0., 1.],
        'encoder_zero_deg': np.rad2deg(fit.x).tolist(),
        'command_offset_deg': np.rad2deg(-fit.x).tolist(),
        'command_convention': 'q_encoder = q_camera_command + encoder_zero',
        'independent_physical_offset': False,
        'image_roll_corrected': False,
        'predicted_alignment_error_deg': error_deg,
        'encoder_search_lower_deg': np.rad2deg(lower).tolist(),
        'encoder_search_upper_deg': np.rad2deg(upper).tolist(),
    }
    return new_offset, new_camera, reference
