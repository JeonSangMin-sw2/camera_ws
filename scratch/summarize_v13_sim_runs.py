"""Summarize repeated validate_full_v13_sim.py runs: Step 1 iterations to converge and Step 2 errors.

For each run folder (archive/reports/sim_v13_full_<ts>): the iteration count at which every joint stage
converged (or "no" when it fell back), read from log.txt, and the result-minus-truth errors from
summary.json. Then, over all runs: median and range of the iterations per stage with converged n/N,
and the median / worst absolute error per joint.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\summarize_v13_sim_runs.py <run folder> [<run folder> ...]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGE = re.compile(r"Moving (right|left) arm to (\w+) Ready Pose")
ITERATION = re.compile(r"\[ITERATION (\d+)/(\d+)\] offset=")

iterations = defaultdict(list)
errors = defaultdict(list)
for folder in map(Path, sys.argv[1:]):
    stage, last, found = None, 0, {}
    for line in (folder / "log.txt").read_text(encoding="utf-8").splitlines():
        m = STAGE.search(line)
        if m:
            stage, last = f"{m.group(1)} {m.group(2)}", 0
            continue
        m = ITERATION.search(line)
        if m and stage:
            last = int(m.group(1))
        elif stage and "CONVERGED" in line and stage not in found:
            found[stage] = last
        elif stage and "did not meet" in line and stage not in found:
            found[stage] = "no"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8")).get("truth_comparison", {})
    joints = {f"{side} J{i}": e for side in ("right", "left") for i, e in enumerate(summary.get(f"{side}_arm_error_deg", []))}
    for key, value in joints.items():
        errors[key].append(abs(value))
    for i, name in enumerate(("pan", "tilt")):
        if summary.get("head_error_deg"):
            errors[f"head {name}"].append(abs(summary["head_error_deg"][i]))
    stages = {k: v for k, v in found.items() if k.split()[1] != "marker"}
    for key, value in stages.items():
        iterations[key].append(value)
    worst = max(joints.values(), key=abs) if joints else float("nan")
    print(f"{folder.name}: " + ", ".join(f"{k} {v}" for k, v in stages.items()) + f" | worst joint error {worst:+.3f} deg")

print("\nStep 1 iterations to converge over the runs (median, range, converged n/N):")
for key, values in iterations.items():
    done = [v for v in values if v != "no"]
    text = f"median {np.median(done):.1f}, {min(done)}..{max(done)}" if done else "none converged"
    print(f"  {key:22s} {text}, {len(done)}/{len(values)}")
print("\nStep 2 result minus simulator truth, absolute (deg): median / worst")
print("  " + " | ".join(f"{k} {np.median(v):.2f}/{max(v):.2f}" for k, v in errors.items()))
