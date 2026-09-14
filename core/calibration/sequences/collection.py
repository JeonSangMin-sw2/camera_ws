from core.robot.motion import build_incremental_motion_plan, execute_auto_motion_step, move_to_auto_ready_pose, verify_and_align_head_at_ready_pose
from ..data import capture_one_sample, get_both_arm_config, get_head_config


def run_collection(core, context, prepare=False, plan=None, start_index=0, max_samples=None):
    context.check_cancelled()
    if core.robot is None:
        raise RuntimeError("Robot is not connected")
    if start_index < 0 or (max_samples is not None and max_samples < 1):
        raise ValueError("start_index must be nonnegative and max_samples must be positive")
    if prepare:
        move_to_auto_ready_pose(core.robot, ["right", "left"], include_head_motion=core.include_head_motion,
                                robot_version=core.get_robot_version())
        context.check_cancelled()
        verify_and_align_head_at_ready_pose(
            core.robot, core.observer, core.model, ["right", "left"], 10,
            include_head_motion=core.include_head_motion,
            prompt_teaching_cb=core.prompt_teaching, log_cb=core.log_msg,
        )
    context.check_cancelled()
    plan = plan if plan is not None else build_incremental_motion_plan(
        core.robot, core.robot.get_dynamics(), core.auto_config, ["right", "left"],
        include_head_motion=core.include_head_motion,
    )
    arm = get_both_arm_config(core.model, version=core.get_robot_version())
    head = get_head_config(core.model)["head_idx"] if core.include_head_motion else None
    samples = []
    context.result.partial["samples"] = samples
    failures = 0
    for index, motion in enumerate(plan[start_index:], start=start_index):
        context.check_cancelled()
        execute_auto_motion_step(core.robot, core.auto_config, motion, ["right", "left"],
                                 include_head_motion=core.include_head_motion)
        context.check_cancelled()
        q_arm, q_head, transforms = capture_one_sample(
            core.robot, arm["arm_idx"], core.observer,
            sampling_time=0 if core.observer.sim else 1, side="all", head_idx=head,
        )
        # Finish recording an acquired sample before acknowledging cancellation.
        if transforms is not None:
            samples.append({"q_arm": q_arm, "q_head": q_head, "marker": transforms,
                            "motion_index": index, "frame": core.observer.last_frame.copy()})
            failures = 0
        else:
            failures += 1
        context.checkpoint("next_motion_index", index + 1)
        context.check_cancelled()
        if failures >= 3:
            raise RuntimeError("Marker not detected at three consecutive poses")
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError("No valid calibration samples collected")
    context.complete("samples", samples)
    return samples
