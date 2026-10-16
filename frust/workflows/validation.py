"""Role-based validation graphs shared by chemistry-specific workflows."""

from frust.workflows.core import StageDef


def _validation_stages(method, *, profile, preopt=False, selected_candidates=False):
    """Build the stationary-point graph from resolved calculator roles."""
    roles = method.resolved_validation_roles()
    required = {"frequencies"}
    if profile == "transition_state":
        required.update({"hessian_seed", "ts_search"})
    else:
        required.add("minimum_optimization")
    if preopt:
        required.add("constrained_preoptimization")
    if method.include_terminal_solv_sp:
        required.add("analysis_single_point")
    missing = required.difference(roles)
    if missing:
        raise ValueError(
            f"Method plan lacks required validation operations: {sorted(missing)}"
        )
    ts = profile == "transition_state"
    stages = []
    if preopt:
        key = roles["constrained_preoptimization"]
        stage_id = roles.get("preoptimization_id", key)
        stages.append(
            StageDef(
                stage_id,
                "constrained preoptimization",
                method_stage=key,
                constraint=True,
                lowest=None if selected_candidates or ts else 1,
                rank_by=stage_id,
                operation="constrained_preoptimization",
                phase="validation",
            )
        )
    if ts:
        key = roles["hessian_seed"]
        stages.append(
            StageDef(
                key,
                "Hessian seed",
                read_files=["input.hess"],
                operation="hessian_seed",
                phase="validation",
            )
        )
        key = roles["ts_search"]
        stages.append(
            StageDef(
                key,
                f"{method.result_family.upper()} transition-state optimization",
                use_last_hess=True,
                operation="ts_search",
                phase="validation",
            )
        )
    else:
        key = roles["minimum_optimization"]
        stages.append(
            StageDef(
                key,
                "minimum optimization",
                lowest=None if selected_candidates else 1,
                rank_by=key,
                operation="minimum_optimization",
                phase="validation",
            )
        )
    key = roles["frequencies"]
    stages.append(
        StageDef(key, "final frequencies", operation="frequencies", phase="validation")
    )
    if method.include_terminal_solv_sp:
        key = roles["analysis_single_point"]
        stages.append(
            StageDef(
                key,
                "analysis single point",
                operation="analysis_single_point",
                phase="validation",
            )
        )
    if len({stage.id for stage in stages}) != len(stages):
        raise ValueError(
            "Validation operations require distinct stored stage IDs; "
            "set preoptimization_id when reusing the minimum calculator"
        )
    for stage in stages:
        spec = method.for_stage(stage.method_stage or stage.id)
        if not spec.supports_operation(stage.operation):
            raise ValueError(
                f"Calculator for {stage.id!r} does not support {stage.operation}"
            )
    potentials = {
        (spec.kwargs.get("uma"), spec.kwargs.get("uma_xtb_alpb"))
        for stage in stages
        if stage.operation != "analysis_single_point"
        for spec in [method.for_stage(stage.method_stage or stage.id)]
        if spec.kwargs.get("uma") is not None
    }
    if len(potentials) > 1:
        raise ValueError("Full UMA stages must use one model and environment")
    return stages
