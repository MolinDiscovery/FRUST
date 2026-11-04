"""Native operation dispatch for reaction-aware calculation plans."""


def _execution_children(
    plan,
    *,
    method,
    screening,
    candidate_specs=None,
    spec_profile="auto",
    random_seed=None,
    n_confs=None,
    validation_top_n=1,
    top_n=20,
    prune_initial=True,
    calculation_level="full",
    ranking_top_n=None,
    include_dft_rank_sp=False,
    spec_match="prefer-exact",
    ts_refine_n=None,
):
    """Bind a private reviewed plan to existing homogeneous execution graphs.

    Children retain their ordinary local ``run`` and cluster ``submit`` paths.
    The composed owner supplies manifests, reference reuse and result-bound
    pathway decisions without changing these native execution graphs.
    """
    from frust.workflows.factories import (
        ScreenTSWorkflow,
        IntermediateWorkflow,
        MolsWorkflow,
    )
    from frust.workflows.methods import preset

    method = preset(method) if isinstance(method, str) else method
    if method.fingerprint() != plan["method_fingerprint"]:
        raise ValueError("Execution method does not match the reviewed target plan")
    from frust.structures.specs import get_state_spec

    default_groups = {
        "transition_state": "transition_states",
        "constrained_minimum": "intermediates",
        "minimum": "references",
    }
    grouped = {"references": [], "transition_states": []}
    for target in plan["targets"]:
        spec = get_state_spec(target.state_id)
        branch = plan.get("execution_groups", {}).get(
            target.target_id, default_groups[target.state_kind]
        )
        # A reference explicitly required by a local expression is an ordinary
        # minimum operation even if its historical full-cycle group differs.
        grouped.setdefault(branch, []).append(target)
    children = {}
    for branch, targets in grouped.items():
        if not targets and branch not in {"references", "transition_states"}:
            continue
        import pandas as pd

        systems = pd.DataFrame(
            [{**t.system.as_dict(), "rpos": t.rpos} for t in targets]
        )
        states = tuple(dict.fromkeys(t.state_id for t in targets))
        common = dict(
            dataframe=systems,
            method=method,
            screening=screening,
            calculation_level=calculation_level,
            n_confs=n_confs,
            top_n=top_n,
            validation_top_n=validation_top_n,
            prune_initial=prune_initial,
            ranking_top_n=ranking_top_n,
            include_dft_rank_sp=include_dft_rank_sp,
        )
        kind = (
            targets[0].state_kind
            if targets
            else ("transition_state" if branch == "transition_states" else "minimum")
        )
        if kind == "minimum":
            child = MolsWorkflow(select_mols=states or ("HH",), **common)
        else:
            candidates = {
                s: spec for s, spec in (candidate_specs or {}).items() if s in states
            }
            connected = dict(
                candidate_specs=candidates,
                spec_profile=spec_profile,
                spec_match=spec_match,
                random_seed=random_seed,
            )
            child = (
                ScreenTSWorkflow(
                    ts_types=states,
                    ts_refine_n=ts_refine_n if validation_top_n is None else None,
                    **common,
                    **connected,
                )
                if kind == "transition_state"
                else IntermediateWorkflow(states=states, **common, **connected)
            )
            if targets:
                child._stage_defs()  # Validate capabilities before profile selection.
                child.targets()  # Resolve applicable profiles without embedding.
        child._target_cache = list(targets)
        children[branch] = child
    return children
