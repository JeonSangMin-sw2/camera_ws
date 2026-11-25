"""Full execution reuses the exact same public step implementations."""


def run_full(core, context, step1=None, step1_5=None, step2=None):
    step2 = dict(step2 or {})
    for name, options in (("step1", step1), ("step1_5", step1_5), ("step2", step2)):
        context.check_cancelled()
        child = core._execute(name, dict(options or {}))
        if child.success:
            if name == "step2":
                _settle_step1_j6(core, context, child)
            context.complete(name, child)
            head_done = (not child.completed.get("head_camera", {}).get("skipped", False)
                         and dict(step1_5 or {}).get("prepare", True))
            if name == "step1_5" and head_done:
                # Step 1.5 finished at the same ready pose Step 2 starts from, and only the head
                # moved after that. Skip Step 2's joint ready pose so the arms stop travelling
                # back to it and straight out again.
                collection = dict(step2.get("collection", {}))
                collection["skip_joint_pose"] = True
                step2["collection"] = collection
        else:
            context.checkpoint(name, child)
            context.result.finish(child.status, child.error)
            return


def _settle_step1_j6(core, context, step2):
    """Step 1's J6-only cautions go back to green once Step 2 has solved J6 again; their log lines stay."""
    from .run_warnings import settled_by_step2
    step1 = context.result.completed.get("step1")
    items = getattr(step1, "completed", {}).get("warnings") if step1 is not None else None
    if not items:
        return
    kept, settled = settled_by_step2(items, step2.completed)
    if not settled:
        return
    if kept:
        step1.completed["warnings"] = kept
    else:
        step1.completed.pop("warnings", None)
    core.log_msg("[INFO] Step 1 J6 cautions settled by Step 2 (it solved J6 again): "
                 + "; ".join(f"{i['key']} {i['params']}" for i in settled))
    children = [context.result.completed.get("step1_5"), step2, step1]
    core.run_warned = any(getattr(c, "completed", {}).get("warnings") for c in children if c is not None)
