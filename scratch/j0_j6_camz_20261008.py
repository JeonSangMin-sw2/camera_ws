"""Three what-ifs on the 2026-10-08 09:51 run (new robot, head mode), all offline from the saved data.

 A. J0 common-mode damping: Step 2 with the damping weight (2000 in production) changed.
 B. J6: Step 2 with the Step 1 J6 anchor of one arm moved, to see whether the Step 2 poses can tell
    a J6 value apart (residual) and what else moves with it.
 C. Camera height 67 mm (design value given by the user on 2026-10-08): Step 1.5 with the height held
    there, then Step 2 on top, with the camera free as in production and with the height held too.
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\j0_j6_camz_20261008.py
"""
import sys

import step15_rerun_20261008 as step15

step2 = step15.step2_rerun
out = sys.__stdout__
DESIGN_HEIGHT_M = 0.067


def section(title):
    print(f"\n=== {title} ===", file=out)
    print(step2.COLUMNS, file=out)


section("A. J0 common-mode damping weight (production 2000)")
for weight in (0.0, 200.0, 2000.0, 20000.0, 200000.0):
    step2.row(f"w={weight:g}", step2.solve(f"j0w_{weight:g}", j0_common_weight=weight))

section("B. Step 1 J6 anchor moved (deg, sign of the result) -- one arm at a time")
step2.row("as run", step2.solve("j6_as_run"))
for side in ("right", "left"):
    for delta in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
        step2.row(f"{side[0].upper()} J6{delta:+.1f}", step2.solve(f"j6_{side}_{delta:+.1f}", j6_result_delta={side: delta}))

print(f"\n=== C. Camera height held at the design value {DESIGN_HEIGHT_M * 1000:.0f} mm ===", file=out)
print(f"{'saved':15s} | {step15.describe(step15.SAVED)}", file=out)
variants = {"free": dict(),
            "z=67": dict(lock_z_delta=DESIGN_HEIGHT_M - step15.NOMINAL[2]),
            "ez=0 z=67": dict(lock_ez=True, lock_z_delta=DESIGN_HEIGHT_M - step15.NOMINAL[2])}
solved15 = {}
for tag, options in variants.items():
    solved15[tag] = step15.solve_step15(**options)
    print(f"{tag:15s} | {step15.describe(solved15[tag])}", file=out)
print("\nStep 2 on top", file=out)
print(step2.COLUMNS, file=out)
step2.row("as run", step2.solve("camz_as_run"))
for tag in ("z=67", "ez=0 z=67"):
    camera, head = solved15[tag]["calibrated_mount_to_cam"], solved15[tag]["head_offsets_deg"]
    name = tag.replace(" ", "_").replace("=", "")
    step2.row(tag[:12], step2.solve(f"camz_{name}_free", camera=camera, head=head))
    step2.row((tag + " fix")[:12], step2.solve(f"camz_{name}_fixed", camera=camera, head=head, optimize_camera=False))
