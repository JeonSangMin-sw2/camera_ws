"""Validated YAML read/merge/atomic replace shared by calibration writers.

Updates retain unknown data and float precision, but normalize YAML formatting
and comments. The lock serializes this process's writers, not external editors.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import stat
import shutil
import sys
import tempfile
import threading
import weakref

import numpy as np
import yaml


_LOCK = threading.RLock()


class Paths:
    """Runtime paths, independently redirectable for verification or packaging."""

    def __init__(self, root=None, *, bundle=None):
        frozen = getattr(sys, 'frozen', False)
        self.root = Path(root or (Path(sys.executable).parent if frozen else Path(__file__).resolve().parents[1])).resolve()
        self.bundle = Path(bundle).resolve() if bundle is not None else (Path(sys._MEIPASS).resolve() if frozen else None)
        files = {
            'home_reset_baseline': 'home_reset_baseline.json', 'simulation_yaml': 'simulation.yaml',
            'setting_yaml': 'setting.yaml', 'camera_info': 'camera_info.yaml',
            'ready_poses_yaml': 'ready_poses.yaml', 'camera_intrinsics': 'camera_intrinsics.yaml',
            'language_yaml': 'i18n.yaml',
        }
        self.config = {key: str(self.root / 'config' / filename) for key, filename in files.items()}
        self.config.update({key: str(self.root / 'result' / folder) for key, folder in
                           [('result_dir', 'result_step2'), ('plot_dir', 'result_img'), ('txt_dir', 'result_txt')]})

    def prepare_templates(self):
        """Install missing packaged defaults, never overwrite user configuration."""
        if self.bundle is None or not (self.bundle / 'config').is_dir():
            return
        target = self.root / 'config'
        target.mkdir(parents=True, exist_ok=True)
        for source in (self.bundle / 'config').iterdir():
            destination = target / source.name
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)

    def asset(self, relative_path):
        """Resolve bundled read-only assets, or source-tree assets when unfrozen."""
        base = self.bundle if self.bundle is not None else self.root
        return str((base / relative_path).resolve())


_paths = Paths()
_paths.prepare_templates()
CONFIG_PATHS = _paths.config


def get_asset_path(relative_path):
    return _paths.asset(relative_path)


class RobotConfig:
    """Validated motion/design snapshot. Estimated calibration is not stored here."""

    def __init__(self, document):
        self.document = deepcopy(document)
        try:
            section = self.document['calibration']
            self.joint_sweep_seconds = section['joint_sweep_seconds']
            self.joint_configs = section['joint_configs']
            self.marker_configs = section['marker_configs']
            self.nominal_brackets = section['nominal_brackets']
            self.head_disabled_adjustment = section['head_disabled_adjustment_deg']
            self.tool_lengths = section['tool_length_m']
            self.ready_poses = {key: self.document[key] for key in ('v1.2', 'v1.3')}
            required_modes = {'wrist_yaw2', 'wrist_roll_v13', 'wrist_pitch', 'wrist_pitch_v13', 'elbow'}
            expected_offset_keys = {
                'wrist_yaw2': 'wrist_yaw2',
                'wrist_roll_v13': 'wrist_roll',
                'wrist_pitch': 'wrist_pitch',
                'wrist_pitch_v13': 'wrist_pitch',
                'elbow': 'elbow',
            }
            required_ready_modes = {
                '1.2': {'wrist_pitch', 'elbow', 'wrist_yaw2'},
                '1.3': {'wrist_pitch', 'elbow', 'wrist_roll'},
            }
            if set(self.joint_configs) != required_modes or set(self.joint_sweep_seconds) != required_modes:
                raise ValueError('Missing or unsupported joint calibration modes')
            for mode, value in self.joint_configs.items():
                self._finite(self.joint_sweep_seconds[mode], positive=True)
                for key in ('cand_joint', 'sweep_joint_A', 'sweep_joint_B'):
                    self._joint(value[key])
                for key in ('sweep_range_A', 'sweep_range_B'):
                    self._finite(value[key], positive=True)
                low, high = value['offset_range']
                self._finite(low)
                self._finite(high)
                if low >= high or value['offset_key'] != expected_offset_keys[mode]:
                    raise ValueError('Invalid correction range or key')
            if set(self.marker_configs) != {'axis_4', 'axis_5', 'axis_6'}:
                raise ValueError('Missing marker sweep axes')
            for key, value in self.marker_configs.items():
                self._joint(value['joint_i'])
                if value['joint_i'] != int(key[-1]):
                    raise ValueError('Marker sweep name/index mismatch')
                self._finite(value['start_deg'])
                self._finite(value['end_deg'])
                if value['start_deg'] == value['end_deg']:
                    raise ValueError('Empty marker sweep')
                for version in ('12', '13'):
                    self._unit_vector(value[f'n_nom_v{version}'])
            for version in ('1.2', '1.3'):
                self._finite(self.tool_lengths[version], positive=True)
                ready = self.ready_poses['v' + version]
                if not required_ready_modes[version].issubset(set(ready['joint'])):
                    raise ValueError(f'Missing ready-pose modes for v{version}')
                for side in ('right', 'left'):
                    self._vector(self.nominal_brackets[version][side], 6)
                    for pose in ready['joint'].values():
                        self._vector(pose[side + '_arm'], 7)
                    for name in ('marker', 'check_calib'):
                        self._vector(ready[name][side + '_arm'], 7)
            for key in ('shoulder', 'elbow'):
                self._finite(self.head_disabled_adjustment[key])
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError(f'Invalid robot configuration: {error}') from error

    @staticmethod
    def _finite(value, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (positive and value <= 0):
            raise ValueError(f'Expected a finite {"positive " if positive else ""}number: {value!r}')

    @staticmethod
    def _joint(value):
        if isinstance(value, bool) or not isinstance(value, int) or value not in range(7):
            raise ValueError(f'Joint index must be J0..J6: {value!r}')

    @classmethod
    def _vector(cls, value, length):
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ValueError(f'Expected {length} configuration values')
        for item in value:
            cls._finite(item)

    @classmethod
    def _unit_vector(cls, value):
        cls._vector(value, 3)
        norm = math.sqrt(sum(float(item) ** 2 for item in value))
        if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('Nominal marker axis must be a unit vector')

    @classmethod
    def load(cls, path=None):
        return cls(_parse(Path(path or CONFIG_PATHS['ready_poses_yaml']).read_text(encoding='utf-8')))


class Language:
    """Translations and same-thread change subscriptions, without a Qt dependency.

    UI callers change language on their GUI thread. Bound subscribers are weak,
    so closing a window does not keep its widgets alive through this singleton.
    """
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, config_path=None):
        self.current_lang = 'en'
        self._listeners = []
        self.translations = {}
        self.load_translations(config_path)

    def load_translations(self, config_path=None):
        self.translations = _parse(Path(config_path or CONFIG_PATHS['language_yaml']).read_text(encoding='utf-8'))

    def subscribe(self, callback):
        try:
            listener = weakref.WeakMethod(callback)
        except TypeError:
            listener = callback
        self._listeners.append(listener)

    def unsubscribe(self, callback):
        self._listeners[:] = [item for item in self._listeners
                              if (item() if isinstance(item, weakref.WeakMethod) else item) not in (None, callback)]

    def set_language(self, lang):
        code = 'en' if lang.lower() in ('en', 'english') else 'ko'
        if code == self.current_lang:
            return
        self.current_lang = code
        for item in tuple(self._listeners):
            callback = item() if isinstance(item, weakref.WeakMethod) else item
            if callback is None:
                self._listeners.remove(item)
            else:
                callback(code)

    def get(self, key, lang=None, default=None):
        code = self.current_lang if lang is None else ('ko' if lang in ('ko', 'korean') else 'en')
        value = self.translations
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                return default if default is not None else key
            value = value[part]
        if isinstance(value, dict):
            for language in (code, 'en', 'ko'):
                if language in value:
                    return value[language]
        return default if default is not None else str(value)


def tr(key, lang=None, default=None):
    return Language.instance().get(key, lang=lang, default=default)


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ValueError(f'Duplicate YAML key: {key}')
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Configuration numbers must be finite')
    return value


def _parse(text):
    data = yaml.load(text, Loader=_UniqueLoader)
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a YAML mapping, not empty/scalar/list')
    for section in ('camera', 'marker', 'joint_offset'):
        if section in data and not isinstance(data[section], dict):
            raise ValueError(f'Configuration section {section} must be a mapping')
    return _plain(data)


def _merge(data, updates):
    for key, value in updates.items():
        if isinstance(value, dict):
            if key not in data:
                data[key] = {}
            if not isinstance(data[key], dict):
                raise ValueError(f'Cannot merge mapping into scalar configuration key {key}')
            _merge(data[key], value)
        else:
            data[key] = deepcopy(value)


def _serialize(data):
    plain = _plain(data)
    text = yaml.safe_dump(plain, allow_unicode=True, sort_keys=False, default_flow_style=False)
    if _parse(text) != plain:
        raise ValueError('Configuration did not survive YAML round-trip validation')
    return text


class YamlStore:
    """Validated atomic persistence, shared by all configuration writers."""
    @staticmethod
    def replace_yaml_values(lines, section, key, values):
        """Compatibility for existing line-list callers, parsed structurally."""
        data = _parse(''.join(lines))
        _merge(data, {section: {key: values}})
        lines[:] = _serialize(data).splitlines(keepends=True)


    @staticmethod
    def update_yaml(path, updates):
        """Merge dict updates (or call a mutator) and return the committed document.

        Existing missing/malformed files are errors, never implicit defaults.
        Before replace fails, the original file remains byte-for-byte unchanged.
        A snapshot check detects external modifications during preparation; external
        processes must still coordinate writes (this is not a cross-process lock).
        """
        path = Path(path).resolve()
        with _LOCK:
            original = path.read_bytes()
            data = _parse(original.decode('utf-8'))
            if callable(updates):
                updates(data)
            else:
                _merge(data, updates)
            text = _serialize(data)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                if path.read_bytes() != original:
                    raise RuntimeError(f'Configuration changed during save: {path}; retry after reloading')
                os.replace(temporary, path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return _parse(text)


# Public callables share the class implementation (not duplicated writers).
update_yaml = YamlStore.update_yaml
replace_yaml_values = YamlStore.replace_yaml_values
