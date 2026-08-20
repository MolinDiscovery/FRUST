from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

import frust as ft
from frust.results import attach_result_contract, free_energy_components
from frust.screen.references import ReferenceLibrary
from frust.structures import ChemicalSystem, StructureTarget
from frust.transformers import transformer_mols
from frust.workflows.screening import _finalize_run


CATALYST = "BC1=C(N(C)C)C=CC=C1"


class _FakeJob:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id


class _FakeExecutor:
    def __init__(self) -> None:
        self.parameters: list[dict] = []
        self.submissions: list[tuple] = []

    def update_parameters(self, **kwargs):
        self.parameters.append(dict(kwargs))

    def submit(self, function, *args, **kwargs):
        self.submissions.append((function, args, kwargs))
        return _FakeJob(f"job-{len(self.submissions)}")


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", CATALYST],
            "compound_name": ["pyrrole", "NMe"],
            "rpos": ["2", None],
        }
    )


def _atoms(formula: dict[str, int]) -> list[str]:
    return [element for element, count in formula.items() for _ in range(count)]


def _row(
    state_id: str,
    energy: float,
    formula: dict[str, int],
    *,
    state_kind: str = "minimum",
    system_name: str = "pyrrole__NMe",
    substrate_name: str = "pyrrole",
    catalyst_name: str = "NMe",
    rpos: int | None = None,
) -> dict:
    atoms = _atoms(formula)
    vibrations = (
        [{"frequency": -250.0}, {"frequency": 30.0}]
        if state_kind == "transition_state"
        else [{"frequency": 20.0}, {"frequency": 50.0}]
    )
    return {
        "structure_id": f"{state_id}:{system_name}:r{rpos}",
        "state_id": state_id,
        "state_kind": state_kind,
        "system_name": system_name,
        "substrate_name": substrate_name,
        "catalyst_name": catalyst_name,
        "rpos": rpos,
        "cid": 0,
        "atoms": atoms,
        "dft_opt-oc": np.zeros((len(atoms), 3)).tolist(),
        "dft_ts_opt-oc": np.zeros((len(atoms), 3)).tolist(),
        "dft_freq-EE": energy - 0.1,
        "dft_freq-GE": energy,
        "dft_freq-vibs": vibrations,
        "dft_freq-NT": True,
    }


def _write_result(path: Path, rows: list[dict], profile: str) -> None:
    df = pd.DataFrame(rows)
    attach_result_contract(
        df,
        profile,
        dft=True,
        include_terminal_solv_sp=False,
        thermochemistry=ft.workflows.ThermochemistrySpec("frequency_gibbs"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _manifest(scope: str = "barriers") -> dict:
    method = ft.workflows.methods.preset("r2scan-3c-solv")
    return {
        "schema_version": 1,
        "run_type": "catalyst_screen",
        "scope": scope,
        "method": method.to_dict(),
        "method_fingerprint": method.fingerprint(),
        "corrections_kcal_mol": {"TS1": -1.89, "TS3": -1.89},
        "analysis_targets": [
            {
                "state_id": state,
                "system_name": "pyrrole__NMe",
                "substrate_name": "pyrrole",
                "catalyst_name": "NMe",
                "rpos": 2,
            }
            for state in ("TS1", "TS2", "TS3", "TS4")
        ],
        "calculation_results": {
            "transition_states": "calculations/transition_states/merged.parquet",
            "references": "calculations/references/merged.parquet",
            "cycle_molecules": (
                "calculations/full_cycle/molecular_states/merged.parquet"
                if scope == "full_cycle"
                else None
            ),
            "int3": (
                "calculations/full_cycle/int3/merged.parquet"
                if scope == "full_cycle"
                else None
            ),
        },
    }


def _write_barrier_bundle(root: Path, *, scope: str = "barriers") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(_manifest(scope), indent=2))
    formulas = {
        "ligand": {"C": 5, "H": 7, "N": 1},
        "dimer": {"C": 16, "H": 24, "B": 2, "N": 2},
        "HBpin-mol": {"C": 6, "H": 13, "B": 1, "O": 2},
        "HH": {"H": 2},
    }
    refs = [
        _row("ligand", -10.0, formulas["ligand"]),
        _row("dimer", -40.0, formulas["dimer"]),
        _row("HBpin-mol", -5.0, formulas["HBpin-mol"]),
        _row("HH", -1.0, formulas["HH"]),
    ]
    ts_formulas = {
        "TS1": {"C": 13, "H": 19, "B": 1, "N": 2},
        "TS2": {"C": 13, "H": 19, "B": 1, "N": 2},
        "TS3": {"C": 19, "H": 30, "B": 2, "N": 2, "O": 2},
        "TS4": {"C": 19, "H": 30, "B": 2, "N": 2, "O": 2},
    }
    energies = {"TS1": -29.9, "TS2": -29.8, "TS3": -33.7, "TS4": -33.6}
    ts_rows = [
        _row(
            state,
            energies[state],
            ts_formulas[state],
            state_kind="transition_state",
            rpos=2,
        )
        for state in energies
    ]
    _write_result(
        root / "calculations/transition_states/merged.parquet",
        ts_rows,
        "transition_state",
    )
    _write_result(
        root / "calculations/references/merged.parquet",
        refs,
        "minimum",
    )

    if scope == "full_cycle":
        cycle = [
            _row("catalyst", -20.0, {"C": 8, "H": 12, "B": 1, "N": 1}),
            _row("int1", -29.7, ts_formulas["TS1"], rpos=2),
            _row("int2", -29.6, {"C": 13, "H": 17, "B": 1, "N": 2}, rpos=2),
            _row("HBpin-ligand", -14.5, {"C": 11, "H": 18, "B": 1, "N": 1, "O": 2}, rpos=2),
        ]
        _write_result(
            root / "calculations/full_cycle/molecular_states/merged.parquet",
            cycle,
            "minimum",
        )
        _write_result(
            root / "calculations/full_cycle/int3/merged.parquet",
            [_row("INT3", -33.5, ts_formulas["TS3"], state_kind="constrained_minimum", rpos=2)],
            "constrained_minimum",
        )


def test_thermochemistry_recipes_and_method_fingerprints():
    frame = pd.DataFrame(
        {
            "dft_freq-EE": [-10.0],
            "dft_freq-GE": [-9.8],
            "dft_solv_sp-EE": [-10.1],
        }
    )
    attach_result_contract(
        frame,
        "minimum",
        dft=True,
        include_terminal_solv_sp=True,
        thermochemistry=ft.workflows.ThermochemistrySpec("electronic_plus_thermal"),
    )
    assert free_energy_components(frame).iloc[0]["free_energy_hartree"] == pytest.approx(-9.9)

    gas = ft.workflows.methods.preset("r2scan-3c")
    renamed = replace(gas, name="same-science-new-name")
    solvent = ft.workflows.methods.preset("r2scan-3c-solv")
    assert gas.fingerprint() == renamed.fingerprint()
    assert gas.fingerprint() != solvent.fingerprint()


def test_reference_library_keeps_inspectable_structure_and_review(tmp_path):
    method = ft.workflows.methods.preset("r2scan-3c-solv")
    system = ChemicalSystem("pyrrole__NMe", "pyrrole", "NMe", "CN1C=CC=C1", CATALYST)
    target = StructureTarget(
        target_id="dimer:NMe",
        tag="dimer__NMe",
        system=system,
        state_id="dimer",
        state_kind="minimum",
        builder_spec="cycle::dimer::v2",
        scope="catalyst",
    )
    df = pd.DataFrame([_row("dimer", -40.0, {"H": 2})])
    attach_result_contract(
        df,
        "minimum",
        dft=True,
        include_terminal_solv_sp=False,
        thermochemistry=method.thermochemistry,
    )
    library = ReferenceLibrary(tmp_path / "library").initialize()
    record = library.publish(df, target, method, protocol={"n_confs": 10})

    assert record.xyz_path().exists()
    assert record.dataframe().iloc[0]["state_id"] == "dimer"
    assert record.metadata["formula"] == "H2"
    assert library.review_queue()["reference_id"].tolist() == [record.reference_id]
    assert library.find(target, method, protocol={"n_confs": 10}) is None

    record.approve(note="Inspected minimum")
    found = library.find(target, method, protocol={"n_confs": 10})
    assert found.reference_id == record.reference_id
    assert library.search(review="approved")["reference_id"].tolist() == [record.reference_id]

    recalculated = df.copy()
    recalculated["dft_freq-EE"] -= 0.01
    recalculated["dft_freq-GE"] -= 0.01
    changed = library.publish(recalculated, target, method, protocol={"n_confs": 10})
    assert changed.reference_id != record.reference_id
    assert changed.metadata["cache_key"] == record.metadata["cache_key"]
    assert len(library.index()) == 2
    found = library.find(target, method, protocol={"n_confs": 10})
    assert found.reference_id == record.reference_id
    changed.approve(note="Inspected recalculation")
    found = library.find(target, method, protocol={"n_confs": 10})
    assert found.reference_id == changed.reference_id

    changed.xyz_path().write_text("tampered")
    found = library.find(target, method, protocol={"n_confs": 10})
    assert found.reference_id == record.reference_id
    metadata_path = record.path / "metadata.json"
    metadata_path.write_text(
        metadata_path.read_text().replace('"formula": "H2"', '"formula": "H3"')
    )
    assert library.find(target, method, protocol={"n_confs": 10}) is None


def test_barrier_analysis_matches_supplied_formulas_and_survives_relocation(tmp_path):
    original = tmp_path / "original"
    _write_barrier_bundle(original)
    run = ft.screen.open_run(original).refresh_analysis()
    barriers = run.barriers().set_index("ts_type")

    assert barriers.loc["TS1", "barrier_kcal_mol"] == pytest.approx(0.1 * 627.5094740631 - 1.89)
    assert barriers.loc["TS2", "barrier_kcal_mol"] == pytest.approx(0.2 * 627.5094740631)
    assert barriers.loc["TS3", "barrier_kcal_mol"] == pytest.approx(0.3 * 627.5094740631 - 1.89)
    assert barriers.loc["TS4", "barrier_kcal_mol"] == pytest.approx(0.4 * 627.5094740631)
    assert set(barriers["quality_status"]) == {"review"}

    relocated = tmp_path / "downloaded" / "run"
    relocated.parent.mkdir()
    shutil.copytree(original, relocated)
    moved = ft.screen.open_run(relocated)
    pd.testing.assert_frame_equal(run.barriers(), moved.barriers())


def test_full_cycle_is_balanced_and_review_persists(tmp_path):
    root = tmp_path / "full"
    _write_barrier_bundle(root, scope="full_cycle")
    run = ft.screen.open_run(root).refresh_analysis()
    profile = run.profile(system_name="pyrrole__NMe", rpos=2, include_invalid=True)

    assert profile["profile_state"].tolist() == [
        "Dimer", "Cat", "TS1", "int1", "TS2", "int2", "TS3", "INT3", "TS4", "Product"
    ]
    assert "unbalanced_composition" not in ";".join(profile["quality_issues"])
    assert profile.iloc[0]["relative_g_kcal_mol"] == pytest.approx(0.0)

    ts1 = run.states().query("state_id == 'TS1'").iloc[0]
    run.set_review(ts1["result_id"], "approved", note="Correct reactive mode")
    reviewed = run.states().query("state_id == 'TS1'").iloc[0]
    assert reviewed["review_status"] == "approved"
    assert reviewed["quality_status"] == "ready"

    transition_path = root / "calculations/transition_states/merged.parquet"
    transition_states = pd.read_parquet(transition_path)
    modes = deepcopy(list(transition_states.at[0, "dft_freq-vibs"]))
    for index, mode in enumerate(modes):
        mode["mode_fingerprint_test"] = float(index)
    transition_states.at[0, "dft_freq-vibs"] = modes
    transition_states.to_parquet(transition_path, index=False)
    run.refresh_analysis()
    changed = run.states().query("state_id == 'TS1'").iloc[0]
    assert changed["result_id"] != ts1["result_id"]
    assert changed["review_status"] == "unreviewed"
    assert changed["quality_status"] == "review"


def test_balanced_profile_compositions_match_standard_molecule_builders():
    molecules = transformer_mols(
        "CN1C=CC=C1",
        CATALYST,
        show_IUPAC=False,
        rpos_list=[2],
    )
    formulas = {
        key.split("_")[-1].split("_rpos")[0]: CalcMolFormula(mol)
        for key, mol in molecules.items()
    }

    assert formulas["dimer"] == "C16H24B2N2"
    assert formulas["ligand"] == "C5H7N"
    assert formulas["catalyst"] == "C8H12BN"
    assert formulas["HBpin-mol"] == "C6H13BO2"
    assert formulas["HH"] == "H2"
    assert any(
        CalcMolFormula(mol) == "C13H19BN2"
        for key, mol in molecules.items()
        if "_int1_" in key
    )
    assert any(
        CalcMolFormula(mol) == "C13H17BN2"
        for key, mol in molecules.items()
        if "_int2_" in key
    )
    assert any(
        CalcMolFormula(mol) == "C11H18BNO2"
        for key, mol in molecules.items()
        if "_HBpin-ligand_" in key
    )


def test_composed_workflow_plans_required_states_without_calculation(tmp_path):
    workflow = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c",
        reference_store=tmp_path / "missing_library",
    )
    plan = workflow.plan()

    assert set(plan.query("branch == 'transition_states'")["state_id"]) == {
        "TS1", "TS2", "TS3", "TS4",
    }
    assert set(plan.query("branch == 'references'")["state_id"]) == {
        "ligand", "dimer", "HBpin-mol", "HH",
    }
    assert set(plan["action"]) == {"calculate"}
    assert not (tmp_path / "missing_library").exists()

    run_dir = tmp_path / "signature"
    run_dir.mkdir()
    workflow._write_manifest(run_dir)
    workflow._write_manifest(run_dir)
    incompatible = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
    )
    with pytest.raises(FileExistsError, match="different catalyst-screen manifest"):
        incompatible._write_manifest(run_dir)


def test_finalizer_publishes_and_future_workflow_reuses_approved_references(tmp_path):
    store = tmp_path / "shared_references"
    root = tmp_path / "run"
    workflow = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
        reference_store=store,
    )
    root.mkdir()
    workflow._write_manifest(root)
    reference_dir = root / "calculations" / "references"
    formulas = {
        "ligand": {"C": 5, "H": 7, "N": 1},
        "dimer": {"C": 16, "H": 24, "B": 2, "N": 2},
        "HBpin-mol": {"C": 6, "H": 13, "B": 1, "O": 2},
        "HH": {"H": 2},
    }
    for index, target in enumerate(workflow.children()["references"].targets()):
        frame = pd.DataFrame([_row(target.state_id, -10.0 - index, formulas[target.state_id])])
        attach_result_contract(
            frame,
            "minimum",
            dft=True,
            include_terminal_solv_sp=False,
            thermochemistry=workflow.method.thermochemistry,
        )
        target_dir = reference_dir / target.tag
        target_dir.mkdir(parents=True)
        frame.to_parquet(target_dir / "final.parquet", index=False)
    _write_result(
        root / "calculations/transition_states/merged.parquet",
        [
            _row(
                state,
                -30.0 - index,
                {"C": 13, "H": 19, "B": 1, "N": 2}
                if state in {"TS1", "TS2"}
                else {"C": 19, "H": 30, "B": 2, "N": 2, "O": 2},
                state_kind="transition_state",
                rpos=2,
            )
            for index, state in enumerate(("TS1", "TS2", "TS3", "TS4"))
        ],
        "transition_state",
    )

    report = _finalize_run(workflow, root)
    library = ft.screen.open_reference_library(store)

    assert report["n_references_calculated"] == 4
    assert len(library.index()) == 4
    assert (root / "calculations/references/merged.parquet").exists()
    assert len(list((root / "calculations/references/entries").rglob("optimized.xyz"))) == 4

    for reference_id in library.index()["reference_id"]:
        library.get(reference_id).approve(note="One-time scientific structure review")
    repeated = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
        reference_store=store,
    )
    reference_plan = repeated.plan().query("branch == 'references'")
    assert set(reference_plan["action"]) == {"reuse"}


def test_composed_slurm_submission_finishes_with_afterany_finalizer(tmp_path):
    workflow = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
    )
    fake = _FakeExecutor()
    cluster = ft.ClusterConfig(backend="slurm", partition="kemi1", log_dir=tmp_path / "logs")

    with (
        patch("frust.workflows.core.create_executor", return_value=fake),
        patch("frust.workflows.screening.create_executor", return_value=fake),
    ):
        submission = workflow.submit(out_dir=tmp_path / "results", cluster=cluster)

    assert set(submission.child_submissions) == {"transition_states", "references"}
    assert submission.finalization_job_id == f"job-{len(fake.submissions)}"
    finalizer, finalizer_args, _ = fake.submissions[-1]
    assert finalizer.__name__ == "_finalize_submitted_run"
    assert finalizer_args[1] == tmp_path / "results"
    dependency = fake.parameters[-1]["slurm_additional_parameters"]["dependency"]
    assert dependency.startswith("afterany:")
    assert (tmp_path / "results" / "manifest.json").exists()
