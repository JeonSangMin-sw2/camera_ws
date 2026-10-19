"""One-page check report of a Step 1 run: do the bracket sweeps and the J6 estimates hold together?

Reads the files a run leaves in result_txt (or a copy of them): the bracket sweeps
sweep_points_<arm>_marker_axis_<4|5|6>.txt and the joint_calib_debug_<arm>_*.txt logs. No robot,
no camera; writes one PNG and one JSON with the same numbers.

    .venv/Scripts/python.exe -X utf8 scratch/calibration_report.py result/result_txt
    .venv/Scripts/python.exe -X utf8 scratch/calibration_report.py archive/runs/20261002_headless/sweeps --label 1002_headless

What it shows is consistency, not accuracy: the J6 zero and the bracket's rotation about the J6 axis
cannot be told apart from these data, so there is no reference to grade J6 against. Circle fits are
the tool's own (BaseCalibrator.fit_circle_3d_and_6dof_misalignment). v1.2 wrist only: the J6 axis is
taken as the flange z axis when its direction is read as bracket pitch / roll.
"""
import argparse
import contextlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from matplotlib import colors, font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from core.calibration.CalibratorBase import BaseCalibrator
    from core.calibration.MarkerCalibrator import MarkerCalibrator
    from core.calibration.JointCalibrator import WRIST_YAW2_METHOD_DISAGREE_WARN_DEG
    from core.calibration.calibration_core import BRACKET_SYMMETRY_MAX_POS_MM, BRACKET_SYMMETRY_MAX_ROT_DEG

ARMS = ("right", "left")
ARM_KO = {"right": "오른팔", "left": "왼팔"}
ARM_COLOR = {"right": "#d32f2f", "left": "#1565c0"}
VERSION = "1.2"
# Earlier runs with the same camera (D405), intrinsics file and ready pose; the "previous range".
DEFAULT_HISTORY = [
    ("09-29 12:08", "archive/runs/20260929_d405/sweeps_1208"),
    ("09-29 15:00", "archive/runs/20260929_d405/sweeps_1500"),
    ("10-01 18:17", "archive/runs/20261001_after_fall_run2"),
]
# (label, key, format, compared with the previous runs?)
ROWS = [
    ("J6 스윕 프레임 수", "frames6", "{:.0f}", False),
    ("J6 원 맞춤 잔차 (mm)", "rmse6", "{:.3f}", True),
    ("J6 원 반지름 (mm)", "r6", "{:.2f}", True),
    ("J5 원 반지름 (mm)", "r5", "{:.2f}", True),
    ("J4 원 반지름 (mm)", "r4", "{:.2f}", True),
    ("J6 축 기울기 pitch (°)", "pitch", "{:+.2f}", True),
    ("J6 축 기울기 roll (°)", "roll", "{:.2f}", True),
    ("J5–J6 축 사이 각 (°)", "a56", "{:.2f}", True),
    ("J4–J5 축 사이 각 (°)", "a45", "{:.2f}", True),
    ("J4–J6 축 사이 각 (°)", "a46", "{:.2f}", True),
    ("마커 위치 가로 (°)", "off_x", "{:+.1f}", True),
    ("마커 위치 세로 (°)", "off_y", "{:+.1f}", True),
    ("J5 보정 뒤 J4·J6 원 크기 차 (mm)", "j5_size_err", "{:.2f}", True),
    # The J6 estimates are measured from the robot zero of that run, which differs between runs.
    ("J6 추정: 방향 방식 (°)", "j6_ori", "{:+.3f}", False),
    ("J6 추정: 위치 방식 (°)", "j6_pos", "{:+.3f}", False),
    ("J6 두 방식의 차 (°)", "j6_diff", "{:+.3f}", True),
]
RED = "#ffb4a8"
FULL_TINT_SPREADS = 5.0


def load_sweep(path):
    """Rows of one sweep file, one array per pass ("=== NEW ITERATION ===" starts a new pass)."""
    passes, rows = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "NEW ITERATION" in line:
            if rows:
                passes.append(np.array(rows))
            rows = []
        elif line and not line.startswith("#"):
            values = [float(x) for x in line.split(",") if x.strip()]
            if len(values) >= 26:
                rows.append(values[:26])
    if rows:
        passes.append(np.array(rows))
    return passes


def fit_sweep(rows):
    poses = [row[10:26].reshape(4, 4) for row in rows]
    with contextlib.redirect_stdout(io.StringIO()):
        fit = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, list(rows[:, 0]), robust=True)
    axis = np.asarray(fit["axis_opt"], dtype=float)
    return {"frames": len(rows), "radius": float(fit["radius"]), "rmse": float(fit["rmse"]),
            "axis": axis / np.linalg.norm(axis), "poses": poses}


def axis_angle_deg(a, b):
    angle = np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))
    return float(angle)


def nominal_bracket(arm):
    return BaseCalibrator.NOMINAL_BRACKET_TEMPLATES[VERSION][arm]


def analyze_sweeps(folder, arm):
    """Per pass: circle fits of the three bracket sweeps and the J6 axis as the marker sees it."""
    files = {axis: Path(folder) / f"sweep_points_{arm}_marker_axis_{axis}.txt" for axis in (4, 5, 6)}
    if not all(f.is_file() for f in files.values()):
        return []
    data = {axis: load_sweep(f) for axis, f in files.items()}
    roll, pitch, yaw = nominal_bracket(arm)[3:6]
    flange_z_in_marker = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix().T @ [0.0, 0.0, 1.0]
    passes = []
    for index in range(min(len(v) for v in data.values())):
        fits = {axis: fit_sweep(data[axis][index]) for axis in (4, 5, 6)}
        rows6 = data[6][index]
        n6 = MarkerCalibrator.axis_in_marker_frame_robust(fits[6]["axis"], fits[6]["poses"], flange_z_in_marker)
        off_x = np.degrees(np.arctan2(rows6[:, 1], rows6[:, 3]))
        off_y = np.degrees(np.arctan2(rows6[:, 2], rows6[:, 3]))
        passes.append({
            "frames6": fits[6]["frames"], "frames5": fits[5]["frames"], "rmse6": fits[6]["rmse"],
            "r4": fits[4]["radius"], "r5": fits[5]["radius"], "r6": fits[6]["radius"],
            # Third row of Rz(yaw) Ry(pitch) Rx(roll): the flange z axis in the marker frame.
            "pitch": float(-np.degrees(np.arcsin(n6[0]))), "roll": float(np.degrees(np.arctan2(n6[1], n6[2]))),
            "a45": axis_angle_deg(fits[4]["axis"], fits[5]["axis"]),
            "a56": axis_angle_deg(fits[5]["axis"], fits[6]["axis"]),
            "a46": axis_angle_deg(fits[4]["axis"], fits[6]["axis"]),
            "off_x": float(np.median(off_x)), "off_y": float(np.median(off_y)),
            "_track": np.column_stack((off_x, off_y)),
        })
    return passes


def log_blocks(path):
    path = Path(path)
    if not path.is_file():
        return []
    blocks = path.read_text(encoding="utf-8", errors="replace").split("=== NEW ITERATION ===")
    return [b for b in blocks if b.strip()]


def number(pattern, text, last=False):
    found = re.findall(pattern, text)
    return float(found[-1] if last else found[0]) if found else None


def parse_j6_log(path):
    """Per pass, the J6 estimates the tool logged from the bracket sweeps (both methods)."""
    passes = []
    for block in log_blocks(path):
        line = next((l for l in block.splitlines() if "from the bracket sweeps" in l), None)
        if line is None:
            continue
        ori = number(r"orientation ([+-]?[\d.]+)°", line)
        pos = number(r"position ([+-]?[\d.]+)°", line)
        passes.append({
            "j6_used": number(r"from the bracket sweeps: ([+-]?[\d.]+)°", line), "j6_ori": ori, "j6_pos": pos,
            "j6_diff": None if ori is None or pos is None else pos - ori,
            "log_frames6": number(r"J6 sweep (\d+) frames", line), "log_r6": number(r"J6 sweep \d+ frames, r ([\d.]+) mm", line),
            "log_frames5": number(r"J5 sweep (\d+) frames", line), "log_r5": number(r"J5 sweep \d+ frames, r ([\d.]+) mm", line),
        })
    return passes


def parse_size_error(path):
    """Per pass, |r_A - r_B| of the last sweep pair: what is left once the joint has converged."""
    return [number(r"Circle Size Error \(abs: r_A - r_B\)\s*:\s*([\d.]+) mm", block, last=True) for block in log_blocks(path)]


def analyze_run(folder):
    folder = Path(folder)
    arms = {}
    for arm in ARMS:
        passes = analyze_sweeps(folder, arm)
        logged = parse_j6_log(folder / f"joint_calib_debug_{arm}_wrist_yaw2.txt")
        size_err = parse_size_error(folder / f"joint_calib_debug_{arm}_wrist_pitch.txt")
        for index, item in enumerate(passes):
            item.update(logged[index] if index < len(logged) else {})
            item["j5_size_err"] = size_err[index] if index < len(size_err) else None
        arms[arm] = passes
    return arms


def recorded_brackets(folder):
    """Bracket vectors the run saved to setting.yaml (a copy next to the sweeps, else the live file)."""
    folder = Path(folder)
    candidates = sorted(folder.glob("setting*.yaml")) + sorted(folder.parent.glob("setting*.yaml"))
    if folder.resolve() == (ROOT / "result" / "result_txt").resolve():
        candidates.append(ROOT / "config" / "setting.yaml")
    for path in candidates:
        marker = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("marker") or {}
        right, left = marker.get("Tf_to_marker_right"), marker.get("Tf_to_marker_left")
        if right and left:
            d_pos, _, d_rot = BaseCalibrator.bracket_asymmetry(right, left, VERSION)
            return {"source": str(path), "right": [float(v) for v in right], "left": [float(v) for v in left],
                    "asym_pos_mm": float(np.max(np.abs(d_pos))), "asym_rot_deg": float(d_rot)}
    return None


def step2_summary(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    noise = data.get("measurement_noise") or {}
    weak = ((data.get("observability") or {}).get("weak_directions") or [{}])[0]
    parts = sorted((weak.get("components") or {}).items(), key=lambda kv: -abs(kv[1]))[:3]
    return {"source": str(path), "pos_noise_mm": noise.get("measurement_noise_pos_std_mm"),
            "rot_noise_deg": noise.get("measurement_noise_rot_std_deg"),
            "weakest": ", ".join(f"{name} {value:+.2f}" for name, value in parts)}


def history_range(history, arm, key):
    values = [p[key] for run in history.values() for p in run.get(arm, []) if p.get(key) is not None]
    return (min(values), max(values)) if values else None


def past_tool_limit(arm, key, value):
    """True when the value is past a warning limit the tool itself uses."""
    if value is None:
        return False
    if key == "j6_diff":
        return abs(value) > WRIST_YAW2_METHOD_DISAGREE_WARN_DEG
    if key == "r6":
        return abs(value - abs(nominal_bracket(arm)[1]) * 1000.0) > MarkerCalibrator.BRACKET_Y_WARN_MM
    return False


def spreads_outside(value, band):
    """How far the value lies outside the previous runs, in units of their spread (0 = inside).
    A few runs give a narrow band, so this is a distance to look at, not a pass/fail limit."""
    if value is None or band is None or band[1] <= band[0]:  # one earlier value has no spread to measure with
        return 0.0
    gap = max(band[0] - value, value - band[1], 0.0)
    return min(gap / (band[1] - band[0]), 99.0)


def tint(spreads):
    """White inside the previous runs, deeper yellow the further outside."""
    t = min(spreads, FULL_TINT_SPREADS) / FULL_TINT_SPREADS
    return colors.to_hex((1.0, 1.0 - 0.30 * t, 1.0 - 0.90 * t))


def build_report(folder, label, history_sets, step2_path=None):
    folder = Path(folder)
    history = {}
    for name, path in history_sets:
        path = Path(path) if Path(path).is_absolute() else ROOT / path
        if path.is_dir() and path.resolve() != folder.resolve():
            history[name] = analyze_run(path)
    report = {"label": label, "folder": str(folder), "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "arms": analyze_run(folder), "history": history, "brackets": recorded_brackets(folder),
              "step2": step2_summary(step2_path) if step2_path else None, "flags": []}
    for arm in ARMS:
        for index, item in enumerate(report["arms"][arm]):
            item["spreads_outside"] = {}
            for text, key, fmt, compare in ROWS:
                value = item.get(key)
                if past_tool_limit(arm, key, value):
                    report["flags"].append({"arm": arm, "pass": index + 1, "key": key,
                                            "text": f"{ARM_KO[arm]} 패스 {index + 1} · {text} {fmt.format(value)}"})
                if compare:
                    item["spreads_outside"][key] = spreads_outside(value, history_range(history, arm, key))
    brackets = report["brackets"]
    if brackets and (brackets["asym_pos_mm"] > BRACKET_SYMMETRY_MAX_POS_MM or brackets["asym_rot_deg"] > BRACKET_SYMMETRY_MAX_ROT_DEG):
        report["flags"].append({"arm": "both", "pass": None, "key": "bracket_asymmetry",
                                "text": f"기록된 브라켓 좌우 차 {brackets['asym_pos_mm']:.2f} mm / {brackets['asym_rot_deg']:.2f}°"})
    return report


def farthest_items(report, count=5):
    """The quantities furthest outside the previous runs: (spreads, arm, row label), largest first."""
    labels = {key: text for text, key, _, _ in ROWS}
    worst = {}
    for arm in ARMS:
        for item in report["arms"][arm]:
            for key, spreads in item["spreads_outside"].items():
                if spreads > worst.get((arm, key), 0.0):
                    worst[arm, key] = spreads
    ranked = sorted(worst.items(), key=lambda kv: -kv[1])[:count]
    return [(spreads, arm, labels[key]) for (arm, key), spreads in ranked]


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value.tolist() if isinstance(value, np.ndarray) else value


def korean_font_settings():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR") if name in installed), None)
    return {"axes.unicode_minus": False, **({"font.family": family} if family else {})}


def draw_table(ax, report, arm):
    passes = report["arms"][arm]
    ax.axis("off")
    ax.set_title(f"{ARM_KO[arm]}", fontsize=13, fontweight="bold", color=ARM_COLOR[arm], loc="left")
    if not passes:
        ax.text(0.5, 0.5, "브라켓 스윕 파일 없음", ha="center", va="center", fontsize=12)
        return
    text, colours = [], []
    for label, key, fmt, compare in ROWS:
        band = history_range(report["history"], arm, key) if compare else None
        row = [label] + ["-" if p.get(key) is None else fmt.format(p[key]) for p in passes]
        row.append("비교 안 함" if not compare else "-" if band is None else f"{fmt.format(band[0])} ~ {fmt.format(band[1])}")
        text.append(row)
        colours.append(["white"] + [RED if past_tool_limit(arm, key, p.get(key)) else tint(p["spreads_outside"].get(key, 0.0))
                                    for p in passes] + ["#f2f2f2"])
    columns = ["항목"] + [f"패스 {i + 1}" for i in range(len(passes))] + ["이전 실행 범위"]
    widths = [0.40] + [0.36 / len(passes)] * len(passes) + [0.24]
    table = ax.table(cellText=text, colLabels=columns, cellColours=colours, colWidths=widths, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.32)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if row == 0:
            cell.set_facecolor("#dddddd")
            cell.set_text_props(fontweight="bold")


def scatter_runs(ax, report, getter, mark_passes=True):
    """History in grey, this run in colour; one point per arm and pass."""
    for name, run in report["history"].items():
        for arm in ARMS:
            for p in run.get(arm, []):
                xy = getter(p)
                if xy is not None:
                    ax.plot(*xy, marker="o" if arm == "right" else "s", color="#9e9e9e", markersize=6, linestyle="none")
    for arm in ARMS:
        for index, p in enumerate(report["arms"][arm]):
            xy = getter(p)
            if xy is not None:
                ax.plot(*xy, marker="o" if arm == "right" else "s", color=ARM_COLOR[arm], markersize=10, linestyle="none")
                if mark_passes:
                    ax.annotate(str(index + 1), xy, textcoords="offset points", xytext=(7, 5), fontsize=9, color=ARM_COLOR[arm])


def draw_plots(axes, report):
    ax = axes[0]
    scatter_runs(ax, report, lambda p: (p["pitch"], p["roll"] - 90.0))
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.axvline(0, color="#cccccc", lw=0.8)
    ax.set_title("마커에서 본 J6 축의 기울기", fontsize=11)
    ax.set_xlabel("pitch (°)")
    ax.set_ylabel("roll - 90 (°)")

    ax = axes[1]
    slots = [(arm, key) for arm in ARMS for key in ("r4", "r5", "r6")]
    for x, (arm, key) in enumerate(slots):
        band = history_range(report["history"], arm, key)
        current = [p[key] for p in report["arms"][arm]]
        centre = float(np.median([p[key] for run in report["history"].values() for p in run.get(arm, [])])) if band else float(np.mean(current)) if current else 0.0
        for run in report["history"].values():
            for p in run.get(arm, []):
                ax.plot(x, p[key] - centre, "o", color="#9e9e9e", markersize=6)
        for index, value in enumerate(current):
            ax.plot(x + 0.12 * (index + 1), value - centre, "D", color=ARM_COLOR[arm], markersize=8)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels([f"{'우' if arm == 'right' else '좌'} J{key[1]}" for arm, key in slots])
    ax.set_title("원 반지름: 이전 실행 중앙값과의 차 (mm)", fontsize=11)

    ax = axes[2]
    scatter_runs(ax, report, lambda p: (p["off_x"], p["off_y"]), mark_passes=False)
    for arm in ARMS:
        for p in report["arms"][arm]:
            ax.plot(p["_track"][:, 0], p["_track"][:, 1], color=ARM_COLOR[arm], lw=1.2, alpha=0.7)
    ax.plot(0, 0, "+", color="black", markersize=14)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("J6 스윕 중 마커 위치 (+ = 화면 중심)", fontsize=11)
    ax.set_xlabel("가로 (°)")
    ax.set_ylabel("세로 (°)")

    ax = axes[3]
    names = list(report["history"]) + ["이번"]
    runs = list(report["history"].values()) + [report["arms"]]
    for x, run in enumerate(runs):
        this = x == len(runs) - 1
        for arm in ARMS:
            for index, p in enumerate(run.get(arm, [])):
                if p.get("j6_diff") is not None:
                    ax.plot(x + (-0.15 if arm == "right" else 0.15) + 0.07 * index, p["j6_diff"], marker="o" if arm == "right" else "s",
                            color=ARM_COLOR[arm] if this else "#9e9e9e", markersize=10 if this else 6, linestyle="none")
    for limit in (-WRIST_YAW2_METHOD_DISAGREE_WARN_DEG, WRIST_YAW2_METHOD_DISAGREE_WARN_DEG):
        ax.axhline(limit, color="#e57373", lw=1.0, ls="--")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_title(f"J6 두 방식의 차: 위치 방식 - 방향 방식 (°)\n점선 = 앱의 경고 기준 ±{WRIST_YAW2_METHOD_DISAGREE_WARN_DEG}", fontsize=11)
    for ax in axes:
        ax.grid(True, color="#eeeeee", lw=0.6)
        ax.tick_params(labelsize=8)


NOTES = """읽는 법
· 원 반지름: 관절 하나만 돌릴 때 마커가 그리는 원의 크기. 브라켓 위치(x·y·z)는 이 세 반지름에서 나온다. 1 mm 달라지면 브라켓 위치도 그만큼 달라진다.
· J6 축 기울기: 마커 판에서 본 J6 회전축의 방향. 브라켓 각도(pitch·roll)가 여기서 나온다. 설계대로면 pitch 0°, roll 90°.
· 축 사이 각: 손목 세 축 사이의 각은 기구로 정해져 있어 실행마다 같아야 한다. 달라졌다면 측정이 흔들렸거나 기구가 변한 것이다.
· 마커 위치: 화면 중심에서 멀수록 마커의 방향 측정이 틀어지기 쉽다 (pitch·roll과 방향 방식 J6에 영향).
· J6 두 방식: 방향 방식은 마커 방향 측정에, 위치 방식은 브라켓의 옆방향 위치에 기댄다. 차가 크면 둘 중 하나가 흔들린 것이다.
· 칸의 색: 빨강 = 앱이 쓰는 경고 기준을 넘음. 노랑 = 이전 실행 범위 밖이고, 진할수록 멀다 (범위 폭의 5배에서 가장 진함). 이전 실행이 몇 회뿐이라 옅은 노랑은 흔하다.
· 그림: 회색 점 = 이전 실행, 색 점 = 이번 실행 (빨강 ● 오른팔, 파랑 ■ 왼팔, 옆의 숫자는 패스).
· 한계: J6가 정확한지는 이 자료로 알 수 없다. J6 영점과 브라켓의 J6 축 둘레 회전은 구분되지 않는다. 여기서 보는 것은 일관성이다."""


def render(report, path):
    # No pyplot and no global backend switch: the app and its tests keep their own matplotlib state.
    with matplotlib.rc_context(korean_font_settings()):
        fig = Figure(figsize=(17, 12.6), dpi=140)
        FigureCanvasAgg(fig)
        draw_page(fig, report)
        fig.savefig(path)


def draw_page(fig, report):
    grid = fig.add_gridspec(4, 4, height_ratios=[0.9, 3.75, 2.7, 2.0], hspace=0.30, wspace=0.28,
                            left=0.035, right=0.985, top=0.975, bottom=0.03)
    head = fig.add_subplot(grid[0, :])
    head.axis("off")
    head.text(0.0, 1.0, f"보정 데이터 점검 보고서 — {report['label']}", fontsize=19, fontweight="bold", va="top")
    folder = Path(report["folder"])
    shown = folder.relative_to(ROOT).as_posix() if folder.is_relative_to(ROOT) else folder.as_posix()
    names = ", ".join(report["history"]) or "없음"
    head.text(0.0, 0.62, f"자료: {shown}   ·   비교한 이전 실행: {names}   ·   작성 {report['generated']}", fontsize=9.5, va="top", color="#444444")
    flags = report["flags"]
    head.text(0.0, 0.40, f"앱의 경고 기준 초과 {len(flags)}건" + (": " + " / ".join(f["text"] for f in flags) if flags else ""),
              fontsize=11, va="top", color="#c62828" if flags else "#2e7d32", fontweight="bold")
    far = [f"{ARM_KO[arm]} {label} ({spreads:.1f}배)" for spreads, arm, label in farthest_items(report)]
    head.text(0.0, 0.14, "이전 실행 범위에서 가장 먼 항목 (범위 폭의 몇 배): " + (" / ".join(far) if far else "없음"),
              fontsize=10, va="top", color="#6d4c00")
    draw_table(fig.add_subplot(grid[1, :2]), report, "right")
    draw_table(fig.add_subplot(grid[1, 2:]), report, "left")
    draw_plots([fig.add_subplot(grid[2, i]) for i in range(4)], report)
    notes = fig.add_subplot(grid[3, :3])
    notes.axis("off")
    notes.text(0.0, 1.0, NOTES, fontsize=9.3, va="top", linespacing=1.5, wrap=True)
    side = fig.add_subplot(grid[3, 3])
    side.axis("off")
    lines = ["기록된 값"]
    brackets = report["brackets"]
    if brackets:
        for arm in ARMS:
            v = brackets[arm]
            lines.append(f"· {ARM_KO[arm]} 브라켓: x {v[0] * 1000:+.2f}, y {v[1] * 1000:+.2f}, z {v[2] * 1000:+.2f} mm, pitch {v[4]:+.2f}°")
        same = brackets["asym_pos_mm"] < 1e-6 and brackets["asym_rot_deg"] < 1e-6
        lines.append("· 좌우 평균이 적용된 값 (팔별 원래 값은 남아 있지 않음)" if same else
                     f"· 좌우 차 {brackets['asym_pos_mm']:.2f} mm / {brackets['asym_rot_deg']:.2f}° (평균 미적용)")
    else:
        lines.append("· 브라켓 기록(setting yaml)을 찾지 못함")
    step2 = report["step2"]
    if step2:
        lines += ["", "Step 2", f"· 위치 노이즈 {step2['pos_noise_mm']:.3f} mm, 방향 노이즈 {step2['rot_noise_deg']:.2f}°",
                  f"· 가장 약한 방향: {step2['weakest']}"]
    side.text(0.0, 1.0, "\n".join(lines), fontsize=9.3, va="top", linespacing=1.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="folder with the sweep_points_* and joint_calib_debug_* files of one run")
    parser.add_argument("--label", help="name shown on the report and used in the file names")
    parser.add_argument("--out", default=str(ROOT / "archive" / "reports"), help="output folder (default: archive/reports/)")
    parser.add_argument("--step2", help="Step 2 result JSON of the same run (optional)")
    parser.add_argument("--history", nargs="*", metavar="LABEL=FOLDER", help="previous runs to compare with")
    args = parser.parse_args(argv)
    folder = Path(args.folder).resolve()
    sweeps = sorted(folder.glob("sweep_points_*_marker_axis_6.txt"))
    if not sweeps:
        raise SystemExit(f"No bracket sweep files (sweep_points_<arm>_marker_axis_6.txt) in {folder}")
    label = args.label or datetime.fromtimestamp(max(f.stat().st_mtime for f in sweeps)).strftime("%Y%m%d_%H%M")
    history = [tuple(item.split("=", 1)) for item in args.history] if args.history is not None else DEFAULT_HISTORY
    report = build_report(folder, label, history, args.step2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png, js = out / f"calib_check_{label}.png", out / f"calib_check_{label}.json"
    render(report, png)
    js.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report: {png}\nnumbers: {js}")
    for flag in report["flags"]:
        print(" - past a tool warning limit:", flag["text"])
    for spreads, arm, text in farthest_items(report):
        print(f" - {spreads:.1f} spreads outside the previous runs: {ARM_KO[arm]} {text}")
    return report


if __name__ == "__main__":
    main()
