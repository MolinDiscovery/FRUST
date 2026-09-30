"""Executable contracts for full UMA transition-state refinement."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.artifacts import compact_result_dataframe
from frust.screen.runs import _vibration_status
from frust.stepper import Stepper
from frust.workflows.factories import ScreenTSWorkflow


def _systems() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system_name": ["sample"],
            "substrate_name": ["pyrrole"],
            "catalyst_name": ["NMe"],
            "substrate_smiles": ["CN1C=CC=C1"],
            "catalyst_smiles": ["BC1=C(N(C)C)C=CC=C1"],
            "rpos": [2],
        }
    )


@pytest.mark.parametrize("environment,solvent", [("uma-gas", None), ("uma-alpb-chloroform", "chloroform")])
def test_full_uma_ts_plan_releases_constraints_and_keeps_one_potential(environment, solvent):
    workflow = ft.workflows.screen_ts(
        dataframe=_systems(), method=environment, calculation_level="full",
        ts_refine_n=3, ts_types=["TS1"],
    )
    stages = workflow.show_stages(detail="full")
    tail = stages["stage"].tolist()[-5:]

    assert tail == ["uma_refine_prune", "uma_refine_filter", "uma_hessian", "uma_ts_opt", "uma_freq"]
    assert stages.set_index("stage").loc["uma_opt", "constraint"]
    assert not stages.set_index("stage").loc["uma_ts_opt", "constraint"]
    assert stages.set_index("stage").loc["uma_ts_opt", "use_last_hess"]
    assert workflow._stage_defs()[-3].read_files == ["input.hess"]
    assert workflow.resolved_spec_profile == "omol-uma-s-1p2p1/gas"
    assert len(workflow._stage_groups("dft_staged")) == 1
    for stage in ("uma_sp", "uma_opt", "uma_hessian", "uma_ts_opt", "uma_freq"):
        spec = workflow.method.for_stage(stage)
        assert spec.kwargs["uma"] == "omol@uma-s-1p2p1"
        assert spec.kwargs.get("uma_xtb_alpb") == solvent
    assert "NumFreq" in workflow.method.for_stage("uma_hessian").options
    assert "NumFreq" in workflow.method.for_stage("uma_freq").options


def test_catalyst_screen_passes_ts_candidate_limit_to_child():
    components = pd.DataFrame({
        "role": ["substrate", "catalyst"],
        "smiles": ["CN1C=CC=C1", "BC1=C(N(C)C)C=CC=C1"],
        "compound_name": ["pyrrole", "NMe"],
        "rpos": ["2", None],
    })
    workflow = ft.workflows.catalyst_screen(
        dataframe=components, screening="uma-gas", method="uma-gas",
        level="full", ts_refine_n=2, ts_types=["TS1"],
    )
    child = workflow.children()["transition_states"]
    assert child.ts_refine_n == 2
    assert child.show_stages()["stage"].tolist()[-1] == "uma_freq"


def test_full_uma_rejects_mixed_screen_and_refinement_environment():
    workflow = ft.workflows.screen_ts(
        dataframe=_systems(), method="uma-alpb-chloroform",
        screening="uma-gas", calculation_level="full", ts_types=["TS1"],
    )
    with pytest.raises(ValueError, match="one model and environment"):
        workflow.show_stages()


def test_uma_ts_refines_multiple_source_guesses_and_keeps_modes():
    calls: list[tuple[str, list[int], dict]] = []

    def prepare(self, target, *, save_dir, options):
        return pd.DataFrame(
            {
                "structure_id": ["sample:TS1"] * 3,
                "state_id": ["TS1"] * 3,
                "state_kind": ["transition_state"] * 3,
                "substrate_name": ["pyrrole"] * 3,
                "cid": [0, 1, 2],
                "atoms": [["H", "H"]] * 3,
                "coords_embedded": [[[0, 0, 0], [0, 0, 1]]] * 3,
                "constraint_roles": [{"a": 0, "b": 1}] * 3,
                "constraint_spec": [{"distances": []}] * 3,
            }
        )

    class FakeStepper:
        def __init__(self, **kwargs):
            pass

        def prune_conformers(self, df, *, name, **kwargs):
            assert name == "UMA geometry diversity selection"
            return df[df["cid"] != 2].copy()

        def xtb(self, df, *, name, **kwargs):
            return self._calc(df, name, kwargs)

        def orca(self, df, *, name, **kwargs):
            return self._calc(df, name, kwargs)

        @staticmethod
        def _calc(df, name, kwargs):
            calls.append((name, df["cid"].tolist(), kwargs))
            out = df.copy()
            out[f"{name}-EE"] = [-10.0 - cid for cid in out["cid"]]
            out[f"{name}-oc"] = out.iloc[:, out.columns.get_loc("coords_embedded")]
            out[f"{name}-NT"] = True
            if name == "uma_hessian":
                out["uma_hessian-input.hess"] = ["seed"] * len(out)
            if name == "uma_freq":
                out["uma_freq-GE"] = [-9.0 - cid for cid in out["cid"]]
                out["uma_freq-vibs"] = [
                    [{"frequency": -350.0, "mode": [[1, 0, 0], [-1, 0, 0]]},
                     {"frequency": 25.0, "mode": [[0, 1, 0], [0, -1, 0]]}]
                    for _ in range(len(out))
                ]
            return out

    workflow = ft.workflows.screen_ts(
        dataframe=_systems(), method="uma-gas", calculation_level="full",
        ts_types=["TS1"], top_n=3, ts_refine_n=2, prune_initial=False,
    )
    with (
        patch.object(ScreenTSWorkflow, "_prepare_initial_df", prepare),
        patch("frust.workflows.core.Stepper", FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        result = workflow.run(targets=[0])

    assert set(result["cid"]) == {0, 1}
    assert [name for name, _, _ in calls][-3:] == ["uma_hessian", "uma_ts_opt", "uma_freq"]
    assert all(set(cids) == {0, 1} for name, cids, _ in calls if name in {"uma_hessian", "uma_ts_opt", "uma_freq"})
    assert next(kwargs for name, _, kwargs in calls if name == "uma_opt")["constraint"]
    ts_kwargs = next(kwargs for name, _, kwargs in calls if name == "uma_ts_opt")
    assert ts_kwargs["constraint"] is False and ts_kwargs["use_last_hess"] is True
    assert result.attrs["frust_results"]["dft"] is False
    assert result.attrs["frust_results"]["columns"]["optimized"]["coords"] == "uma_ts_opt-oc"
    assert result.attrs["frust_workflow"]["ts_refine_n"] == 2
    assert result.attrs["frust_workflow"]["guess_profile"] == "omol-uma-s-1p2p1/gas"
    compact = compact_result_dataframe(result)
    assert compact["uma_freq-frequencies_cm1"].iloc[0] == [-350.0, 25.0]
    assert "uma_freq-vibs" in compact
    assert compact.attrs["frust_artifacts"]["vibration_displacements"] is True
    quality = _vibration_status(compact.iloc[0], "transition_state")
    assert quality["valid"] is True and quality["n_imag"] == 1
    assert quality["flags"] == []
    assert _vibration_status(
        pd.Series({"uma_freq-frequencies_cm1": [-350.0, -70.0, 25.0]}),
        "transition_state",
    )["valid"] is False


def test_uma_orca_reuses_seed_hessian_and_includes_alpb_flag(tmp_path, monkeypatch):
    oet = tmp_path / "oet"
    (oet / "bin").mkdir(parents=True)
    (oet / "bin" / "oet_uma").write_text("#!/bin/sh\n")
    monkeypatch.setenv("OET_TOOLS", str(oet))
    seen = []

    def fake_orca(atoms, coords, n_cores, scr, data2file, options, xtra_inp_str, memory, read_files):
        seen.append((data2file, options, xtra_inp_str))
        return {"normal_termination": True, "electronic_energy": -1.0, "opt_coords": coords}

    step = Stepper(n_cores=1, memory_gb=2, output_base=tmp_path, save_output_dir=False)
    step.orca_fn = fake_orca
    frame = pd.DataFrame({
        "substrate_name": ["test"], "atoms": [["H", "H"]],
        "coords_embedded": [[[0, 0, 0], [0, 0, 1]]],
        "uma_hessian-input.hess": ["seed-data"],
    })
    step.orca(
        frame, name="uma_ts_opt", options={"ExtOpt": None, "OptTS": None},
        uma="omol@uma-s-1p2p1", uma_server=False,
        uma_xtb_alpb="chloroform", use_last_hess=True,
    )
    assert seen[0][0] == {"private_input.hess": "seed-data"}
    assert "--xtb-alpb chloroform" in seen[0][2]
    assert "inhess Read" in seen[0][2]
