"""Full execution reuses the exact same public step implementations."""


def run_full(core, context, step1=None, step1_5=None, step2=None):
    for name, options in (("step1", step1), ("step1_5", step1_5), ("step2", step2)):
        context.check_cancelled()
        child = core._execute(name, dict(options or {}))
        if child.success:
            context.complete(name, child)
        else:
            context.checkpoint(name, child)
            context.result.finish(child.status, child.error)
            return
