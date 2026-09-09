"""Inspect the real camera at its current posture, without robot connection."""
from pathlib import Path
import sys
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.marker_detection import Marker_Transform

if __name__ == '__main__':
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    marker = Marker_Transform(sim=False)
    try:
        if marker.sim or marker.camera is None:
            raise RuntimeError('Real camera required')
        marker.set_marker_type('plate')
        marker.set_camera_exposure(6000., auto_exposure=False)
        results = []
        for index in range(20):
            result = marker.get_marker_transform(sampling_time=0, side='right', use_filter=False)
            results.append(None if result is None else str(result))
            if index in (0, 10, 19):
                frame = marker.camera.get_color_image()
                if frame is not None:
                    cv2.imwrite(str(output / f'frame_{index}.png'), frame)
        (output / 'observations.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print({'manual_exposure': marker.get_camera_exposure(), 'actual_exposure': marker.get_actual_exposure(),
               'detected_frames': sum(x is not None for x in results), 'total_frames': len(results)}, flush=True)
    finally:
        if marker.camera is not None:
            marker.camera.stream_off()
