"""Problems worth showing beside their stage in the wizard's run line (2026-11-24, user).

Judged from each stage's finished result, not from log text. Every entry is
{"key": <wizard.run_warn.<key> i18n key>, "params": {...}}; the stage is the sequence that produced it.
"""

JOINT_LABELS = {"wrist_pitch": "J5", "wrist_pitch_v13": "J5", "wrist_yaw2": "J6", "wrist_roll_v13": "J6", "elbow": "J3"}
J6_METHOD_GAP_DEG = 0.5            # same limit as the JointCalibrator wrist_yaw2 warning
HEAD_TILT_GAP_DEG = 0.3            # user, 2026-11-24 (a clean re-solve agreed to 0.01 deg)
BOUND_EPS_MM = 0.05


def _warn(key, **params):
    return {"key": key, "params": params}


def _compact_ends(ends):
    """data.mount_range_ends text ("x at the upper end (+1.0 mm)") -> "x +1.0 mm", the same in every language."""
    out = []
    for end in ends:
        axis, _, rest = str(end).partition(" at the ")
        value = rest[rest.find("(") + 1:rest.find(")")] if "(" in rest else rest
        out.append(f"{axis} {value}".strip())
    return ", ".join(out)


def step1_warnings(completed, partial):
    out = []
    j6_gap = {}         # both passes measure it; one line per arm with the larger gap
    for value in completed.values():
        if not isinstance(value, dict) or "mode" not in value or value.get("pass_idx") is None:
            continue
        arm, joint = value.get("arm_side", "?"), JOINT_LABELS.get(value["mode"], value["mode"])
        if value.get("iterations") and value.get("max_iterations") and value["iterations"] >= value["max_iterations"]:
            out.append(_warn("iter_cap", arm=arm, joint=joint, n=int(value["iterations"])))
        if value.get("clamped"):
            out.append(_warn("offset_clamped", arm=arm, joint=joint))
        orientation, position = value.get("wrist_yaw2_orientation_deg"), value.get("wrist_yaw2_position_deg")
        if orientation is not None and position is not None and abs(orientation - position) > J6_METHOD_GAP_DEG:
            j6_gap[arm] = max(j6_gap.get(arm, 0.0), round(abs(orientation - position), 2))
    out += [_warn("j6_methods", arm=arm, diff=diff) for arm, diff in j6_gap.items()]
    for value in completed.values():
        if isinstance(value, dict) and value.get("position_at_limit") and value.get("arm_side"):
            out.append(_warn("bracket_limit", arm=value["arm_side"]))
    for arm in ("right", "left"):
        change = partial.get(f"pass_change_{arm}")
        if isinstance(change, dict):
            out.append(_warn("pass_change", arm=arm, **{k: round(v, 3) for k, v in change.items()}))
    return _unique(out)


def step1_5_warnings(completed, marker_bound_mm):
    head = completed.get("head_camera")
    if not isinstance(head, dict) or head.get("skipped"):
        return []
    out = []
    if head.get("camera_at_range_end"):
        out.append(_warn("camera_range", ends=_compact_ends(head["camera_at_range_end"])))
    pinned = [side for side, shift in (head.get("marker_prior_shift_mm") or {}).items()
              if any(abs(v) >= marker_bound_mm - BOUND_EPS_MM for v in shift)]
    if pinned:
        out.append(_warn("marker_bound", sides=", ".join(sorted(pinned)), bound=round(marker_bound_mm, 1)))
    return out


def step2_warnings(completed, partial=None):
    result = completed.get("optimization")
    if not isinstance(result, dict):
        return []
    out = []
    view = ((partial or {}).get("verification") or {}).get("check_pose_view")
    if isinstance(view, dict) and not view.get("seen"):
        out.append(_warn("check_pose_unseen"))
    elif isinstance(view, dict) and view["gap_mm"] > view["limit_mm"]:
        meas = view["measured_symmetry"]
        out.append(_warn("check_pose_gap", gap=round(view["gap_mm"], 1), limit=view["limit_mm"],
                         dx=round(meas["dx_mm"], 1), dz=round(meas["dz_mm"], 1)))
    ends = (result.get("camera_mount_constraint") or {}).get("camera_at_range_end")
    if ends:
        out.append(_warn("camera_range", ends=_compact_ends(ends)))
    head, step1_5 = result.get("head_joint_offset_deg"), result.get("head_step1_5_offset_deg")
    if head and step1_5 and len(head) > 1 and len(step1_5) > 1 and abs(head[1] - step1_5[1]) > HEAD_TILT_GAP_DEG:
        out.append(_warn("head_tilt_gap", step1_5=round(step1_5[1], 2), step2=round(head[1], 2),
                         limit=HEAD_TILT_GAP_DEG))
    return out


# Step 1's pass-to-pass limits (sequences/step1.py PASS EVALUATION).
PASS_CHANGE_LIMITS = {"joint6_deg": 0.10, "joint5_deg": 0.10, "joint3_deg": 0.10, "bracket_pos_mm": 0.5,
                      "bracket_rot_deg": 0.15}


def settled_by_step2(step1_items, step2_completed):
    """Step 1 cautions left once Step 2 has solved J6 again: those about J6 alone (its two methods, or a
    pass-to-pass change in J6 only) recur run to run and Step 2 replaces that J6, so they stay in the log
    only (user, 2026-11-24). Returns (kept, settled)."""
    result = (step2_completed or {}).get("optimization")
    if not isinstance(result, dict) or not (result.get("camera_mount_constraint") or {}).get("joint6_solved"):
        return list(step1_items), []

    def j6_only(item):
        if item["key"] == "j6_methods":
            return True
        if item["key"] == "pass_change":
            return all(item["params"].get(key, 0.0) < limit
                       for key, limit in PASS_CHANGE_LIMITS.items() if key != "joint6_deg")
        return False
    return [i for i in step1_items if not j6_only(i)], [i for i in step1_items if j6_only(i)]


def stage_warnings(name, completed, partial):
    """Warnings for one finished stage ("step1", "step1_5" or "step2"); [] for anything else."""
    if name == "step1":
        return step1_warnings(completed, partial)
    if name == "step1_5":
        from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
        return step1_5_warnings(completed, HeadCameraCalibrator.MARKER_POS_BOUND_M * 1000.0)
    if name == "step2":
        return step2_warnings(completed, partial)
    return []


def _unique(items):
    seen, out = set(), []
    for item in items:
        tag = (item["key"], tuple(sorted(item["params"].items())))
        if tag not in seen:
            seen.add(tag)
            out.append(item)
    return out
