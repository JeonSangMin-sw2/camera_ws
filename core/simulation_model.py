"""One immutable physical model for Steps 1, 1.5 and 2 (metres/radians).

Estimation flags and setting.yaml calibration results are deliberately absent.
Changing truth requires a new instance/session, never an optimizer toggle.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from core.calibration_optimizer import compute_fk, make_transform, so3_exp
from core.paths import CONFIG_PATHS


def load_truth_config():
    with open(CONFIG_PATHS['simulation_yaml'], encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def uses_head_camera(camera_config, model):
    mode = camera_config.get('camera_mount_mode', 'head')
    if mode not in ('head', 'fixed'):
        raise ValueError('camera_mount_mode must be head or fixed')
    return mode == 'head' and len(getattr(model, 'head_idx', [])) >= 2


@dataclass(frozen=True)
class SimulationModel:
    version: str
    config_json: str

    @classmethod
    def create(cls, version='1.2', config=None):
        config = load_truth_config() if config is None else config
        version = str(version).removeprefix('v')
        if version not in config['brackets']:
            raise ValueError(f'No simulation geometry for robot v{version}')
        return cls(version, json.dumps(config, sort_keys=True, allow_nan=False))

    @property
    def config(self):
        # Callers receive a detached copy, not writable truth storage.
        return json.loads(self.config_json)

    def arm_offsets(self, side):
        values = self.config['offsets'][side]
        return np.deg2rad([values[f'joint{i}' if i != 5 else
                                 ('joint5_v13' if self.version == '1.3' else 'joint5_v12')]
                          for i in range(7)])

    def bracket_transform(self, side):
        cfg = self.config
        nominal = make_transform(cfg['brackets'][self.version][side])
        error = cfg['offsets'][side]
        # Assembly translation is in flange axes; do not rotate nominal lever arm.
        nominal[:3, 3] += np.asarray(error['bracket_pos'])
        nominal[:3, :3] = make_transform([0, 0, 0] + error['bracket_rpy'])[:3, :3] @ nominal[:3, :3]
        return nominal

    def marker_pose(self, robot, q_encoder, side, rng=None, noisy=True):
        cfg, model = self.config, robot.model()
        q = np.array(q_encoder, dtype=float, copy=True)
        q[getattr(model, f'{side}_arm_idx')] += self.arm_offsets(side)
        head = uses_head_camera(cfg, model)
        if head:
            offsets = cfg['offsets']['head']
            q[model.head_idx] += np.deg2rad([offsets['pan'], offsets['tilt']])
        base = 'link_head_2' if head else 'link_head_0'
        camera = make_transform(cfg['mount_to_cam' if head else 'head_base_to_cam'])
        _, fk = compute_fk(robot, robot.get_dynamics(), q, f'ee_{side}', base_link=base)
        result = np.linalg.inv(camera) @ fk @ self.bracket_transform(side)
        if noisy:
            if rng is None:
                raise ValueError('A session RNG is required for reproducible sensor noise')
            result[:3, 3] += rng.normal(0, cfg['position_noise_std_m'], 3)
            result[:3, :3] = so3_exp(np.deg2rad(rng.normal(0, cfg['orientation_noise_std_deg'], 3))) @ result[:3, :3]
        return result

    def metadata(self, urdf_path=None):
        return {'schema_version': 1, 'source': 'simulation_pose_sensor',
                'robot_version': self.version, 'truth': self.config,
                'truth_sha256': hashlib.sha256(self.config_json.encode()).hexdigest(),
                'urdf_sha256': hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest() if urdf_path else None,
                'offset_convention': 'q_physical = q_encoder + delta; correction = -delta',
                'image_detection_verified': False}
