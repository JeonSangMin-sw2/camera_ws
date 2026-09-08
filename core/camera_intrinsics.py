"""User-selected intrinsics; temperature is reference only, not a restriction."""
import hashlib
from pathlib import Path

import numpy as np
import yaml


def select_intrinsics(source, factory, factory_dist, width, height, path):
    source = {'per_device': 'calibrated', 'transferred': 'calibrated'}.get(source, source)
    if source not in ('factory', 'calibrated'):
        raise ValueError('intrinsics_source must be factory or calibrated')
    metadata = {'source': source, 'width': width, 'height': height,
                'calibration_temperature_c': None}
    if source == 'factory':
        cx, cy, fx, fy = np.asarray(factory, dtype=float)
        metadata['camera_matrix'] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        metadata['dist_coeffs'] = np.asarray(factory_dist).ravel().tolist()
        return np.asarray(factory), np.asarray(factory_dist), metadata
    raw = Path(path).read_bytes()
    cfg = yaml.safe_load(raw)
    if cfg.get('width') != width or cfg.get('height') != height:
        raise ValueError('Intrinsics resolution mismatch; select parameters matching the capture resolution (no implicit scaling).')
    matrix = np.asarray(cfg['camera_matrix'], dtype=float)
    distortion = np.asarray(cfg['dist_coeffs'], dtype=float).ravel()
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
        raise ValueError('Invalid camera calibration matrix/distortion')
    metadata.update(calibration_temperature_c=cfg.get('calibration_temperature_c'),
                    file=str(Path(path).resolve()), file_sha256=hashlib.sha256(raw).hexdigest())
    metadata['camera_matrix'] = matrix.tolist()
    metadata['dist_coeffs'] = distortion.tolist()
    return np.array([matrix[0, 2], matrix[1, 2], matrix[0, 0], matrix[1, 1]]), distortion, metadata
