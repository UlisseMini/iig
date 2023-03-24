import circrl.module_hook as cmh
from captum.attr import IntegratedGradients


# will use later, when doing intermediate things
def _hook_if_needed(model):
    if not isinstance(model, cmh.ModuleHook):
        model = cmh.ModuleHook(model)
    return model


def trace(model, input, baseline, target):
    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(
        (input,), target=target, return_convergence_delta=True, baselines=baseline
    )

    return attr, delta
