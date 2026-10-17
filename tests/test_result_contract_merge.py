"""Compatibility of legacy and modular result metadata during collection."""

from copy import deepcopy

import pandas as pd
import pytest

from frust.results import attach_result_contract
from frust.utils.dataframes import merge_dataframe_attrs
from frust.workflows.screening import _concat_reference_results


def _frames(family, profile="minimum"):
    legacy = pd.DataFrame({"value": [1.0]})
    attach_result_contract(
        legacy, profile, dft=family == "dft", calculation_level="full",
        full_method=family, include_terminal_solv_sp=True,
    )
    current = legacy.copy(deep=True)
    geometry = "ts_search" if profile == "transition_state" else "minimum_optimization"
    optimized = current.attrs["frust_results"]["columns"]["optimized"]["coords"].removesuffix("-oc")
    current.attrs["frust_results"]["stage_roles"] = {
        geometry: optimized, "frequencies": f"{family}_freq",
        "analysis_single_point": f"{family}_solv_sp",
    }
    legacy.attrs["frust_results"]["artifact_policy"] = "screening"
    return legacy, current


@pytest.mark.parametrize("family", ["uma", "dft"])
@pytest.mark.parametrize("profile", ["minimum", "transition_state", "constrained_minimum"])
def test_legacy_roles_and_mixed_retention_preserve_semantic_contract(family, profile):
    legacy, current = _frames(family, profile)
    originals = [deepcopy(frame.attrs) for frame in (legacy, current)]
    attrs = merge_dataframe_attrs([legacy, current], source_files=["legacy", "current"])

    assert attrs["frust_results"] == current.attrs["frust_results"]
    assert "frust_results" not in attrs["frust_merge"]["attr_conflicts"]
    assert [frame.attrs for frame in (legacy, current)] == originals


@pytest.mark.parametrize("family", ["uma", "dft"])
def test_reference_collection_accepts_legacy_roles(family):
    legacy, current = _frames(family)
    merged = _concat_reference_results([legacy, current], source_files=["legacy", "current"])

    assert merged["value"].tolist() == [1.0, 1.0]
    assert merged.attrs["frust_results"] == current.attrs["frust_results"]


@pytest.mark.parametrize("difference", ["stage_roles", "columns", "thermochemistry", "energy_protocol"])
def test_conflicting_scientific_definitions_remain_conflicts(difference):
    _, first = _frames("uma")
    second = first.copy(deep=True)
    if difference == "stage_roles":
        second.attrs["frust_results"][difference]["frequencies"] = "other_freq"
    elif difference == "columns":
        second.attrs["frust_results"][difference]["analysis"]["electronic_energy"] = "other_sp-EE"
    else:
        second.attrs["frust_results"][difference] = {"different": True}
    attrs = merge_dataframe_attrs([first, second])

    assert "frust_results" not in attrs
    assert "frust_results" in attrs["frust_merge"]["attr_conflicts"]
    with pytest.raises(ValueError, match="incompatible canonical"):
        _concat_reference_results([first, second], source_files=["first", "second"])


def test_matching_compact_policy_is_retained():
    legacy, current = _frames("uma")
    current.attrs["frust_results"]["artifact_policy"] = "screening"
    attrs = merge_dataframe_attrs([legacy, current])

    assert attrs["frust_results"]["artifact_policy"] == "screening"
