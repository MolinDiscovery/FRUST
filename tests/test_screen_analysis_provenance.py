"""Analysis retains results without copying large source audit trails."""

from copy import deepcopy

import pandas as pd
import pytest

from frust.results import attach_result_contract
from frust.screen.runs import _state_rows
from frust.workflows.methods import ThermochemistrySpec


class _AuditCopyGuard:
    def __deepcopy__(self, memo):
        raise AssertionError("Analysis must not copy the source audit trail")


@pytest.mark.parametrize("family", ["uma", "dft"])
@pytest.mark.parametrize("profile", ["minimum", "transition_state", "constrained_minimum"])
def test_state_analysis_preserves_results_and_source_provenance(family, profile):
    transition_state = profile == "transition_state"
    geometry = f"{family}_ts_opt" if transition_state else "uma_min_opt" if family == "uma" else "dft_opt"
    frequencies = [-250.0, 30.0] if transition_state else [20.0, 50.0]
    frame = pd.DataFrame([{
        "atoms": ["H", "H", "H"], "structure_id": "sample", "cid": 0,
        "state_id": "TS1" if transition_state else "int1",
        "state_kind": "transition_state" if transition_state else "minimum",
        "system_name": "sample", "substrate_name": "substrate",
        "catalyst_name": "catalyst", "rpos": 1,
        f"{geometry}-oc": [[0.0, 0.0, 0.0]] * 3,
        f"{geometry}-NT": True, f"{family}_freq-NT": True,
        f"{family}_solv_sp-NT": True, f"{family}_solv_sp-EE": -1.3,
        f"{family}_freq-EE": -1.2, f"{family}_freq-GE": -1.1,
        f"{family}_freq-vibs": [
            {"frequency": value, "mode": [[0.1, 0.2, 0.3]] * 3}
            for value in frequencies
        ],
    }])
    recipe = ThermochemistrySpec("electronic_plus_thermal").to_dict()
    attach_result_contract(
        frame, profile, dft=family == "dft", full_method=family,
        calculation_level="full", include_terminal_solv_sp=True,
        thermochemistry=ThermochemistrySpec("electronic_plus_thermal"),
    )
    kwargs = dict(source="sample", recipe=recipe, reviews={}, calculation_level="full",
                  method={"result_family": family, "stages": {}})
    expected = _state_rows(frame, **kwargs)
    contract = deepcopy(frame.attrs["frust_results"])
    audit = _AuditCopyGuard()
    frame.attrs["frust_large_audit"] = audit

    actual = _state_rows(frame, **kwargs)

    pd.testing.assert_frame_equal(actual, expected)
    assert frame.attrs["frust_large_audit"] is audit
    assert frame.attrs["frust_results"] == contract
    assert "frust_large_audit" not in actual.attrs
