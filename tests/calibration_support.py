"""Shared offline SDK kinematics fixture; cannot connect or command a robot."""
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rby1_sdk.dynamics as rd


class OfflineRobot:
    def __init__(self, version='1.2'):
        directory = Path(os.environ.get('RBY1_MODEL_DIR',
                         Path(__file__).resolve().parents[2] / 'sdk/rby1-sdk/models/rby1m/urdf'))
        self.urdf = directory / f'model_v{version}.urdf'
        if not self.urdf.is_file():
            raise FileNotFoundError(f'{self.urdf}: set RBY1_MODEL_DIR to the SDK URDF directory')
        self.dynamics = rd.Robot(rd.load_robot_from_urdf(str(self.urdf), 'base'))
        names = self.dynamics.get_joint_names()
        self.meta = SimpleNamespace(robot_joint_names=names,
            right_arm_idx=[names.index(f'right_arm_{i}') for i in range(7)],
            left_arm_idx=[names.index(f'left_arm_{i}') for i in range(7)],
            head_idx=[names.index(f'head_{i}') for i in range(2)])
        self.state = SimpleNamespace(position=np.zeros(len(names)))

    def model(self):
        return self.meta

    def get_state(self):
        return self.state

    def get_dynamics(self):
        return self.dynamics


from scipy.spatial.transform import Rotation
from core.calibration_optimizer import QPCalibrationOptimizer


def transform_vector(T):
    return [*T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)]


def data(robot, simulation, count=32, noise=False):
    rng = np.random.default_rng(991)
    model = robot.model()
    arms = list(model.right_arm_idx) + list(model.left_arm_idx)
    qa = rng.uniform(-1.4, 1.4, (count, 14))
    qh = rng.uniform(-0.3, 0.3, (count, 2))
    observations = []
    for a, h in zip(qa, qh):
        q = np.array(robot.get_state().position, copy=True)
        q[arms], q[model.head_idx] = a, h
        observations.append([simulation.marker_pose(robot, q, side, rng, noisy=noise) for side in ('right', 'left')])
    return qa, qh, np.asarray(observations)


def optimizer(robot, simulation, head=True, free_camera=True, reference=None):
    m, cfg = robot.model(), simulation.config
    return QPCalibrationOptimizer(robot=robot,
        arm_idx=list(m.right_arm_idx) + list(m.left_arm_idx),
        head_idx=m.head_idx, use_head_kinematics=head, optimize_head=head,
        optimize_camera=free_camera, head_tilt_reference_rad=reference,
        head_zero_convention='effective_zero',
        ee_links={s: f'ee_{s}' for s in ('right', 'left')},
        ee_to_marker_nom={s: transform_vector(simulation.bracket_transform(s)) for s in ('right', 'left')},
        mount_to_cam_nom=cfg['mount_to_cam'], head_base_to_cam_nom=cfg['head_base_to_cam'],
        lambda_cam_pos=0, lambda_cam_rot=0, estimate_measurement_noise=False,
        camera_pos_bound_m=.01, camera_rot_bound_rad=np.deg2rad(3), max_iter=80,
        qp_kwargs={'eps_abs': 1e-9, 'eps_rel': 1e-9, 'max_iter': 20000})
