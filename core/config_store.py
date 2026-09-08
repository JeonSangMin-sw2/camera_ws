"""Validated YAML read/merge/atomic replace shared by calibration writers.

Updates retain unknown data and float precision, but normalize YAML formatting
and comments. The lock serializes this process's writers, not external editors.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import stat
import tempfile
import threading

import numpy as np
import yaml


_LOCK = threading.RLock()


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


def replace_yaml_values(lines, section, key, values):
    """Compatibility for existing line-list callers, parsed structurally."""
    data = _parse(''.join(lines))
    _merge(data, {section: {key: values}})
    lines[:] = _serialize(data).splitlines(keepends=True)


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
