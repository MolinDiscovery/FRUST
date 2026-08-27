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
import frust.screen.runs as screen_runs
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
    mode = np.zeros((len(atoms), 3)).tolist()
    vibrations = (
        [
            {"frequency": -250.0, "mode": mode},
            {"frequency": 30.0, "mode": mode},
        ]
        if state_kind == "transition_state"
        else [
            {"frequency": 20.0, "mode": mode},
            {"frequency": 50.0, "mode": mode},
        ]
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
        "schema_version": 2,
        "run_type": "catalyst_screen",
        "scope": scope,
        "calculation_level": "full",
        "screening": ft.workflows.methods.screening_preset().to_dict(),
        "screening_fingerprint": ft.workflows.methods.screening_preset().fingerprint(),
        "method": method.to_dict(),
        "method_fingerprint": method.fingerprint(),
        "ranking_solvation": {
            "requested": "method",
            "model": "smd",
            "solvent": "chloroform",
            "applied": True,
        },
        "g_corrections_kcal_mol": {"TS1": -1.89, "TS3": -1.89},
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
            _row("catalyst", -19.8, {"C": 8, "H": 12, "B": 1, "N": 1}),
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


def test_result_hash_normalizes_nested_numpy_arrays():
    mode = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    list_payload = {
        "vibrations": [{"frequency": -123.4, "mode": mode}],
    }
    array_vibrations = np.empty(1, dtype=object)
    array_vibrations[0] = {
        "frequency": np.float64(-123.4),
        "mode": np.asarray(mode),
    }
    array_payload = {"vibrations": array_vibrations}

    assert screen_runs._json_hash(list_payload) == screen_runs._json_hash(array_payload)

    changed = np.empty(1, dtype=object)
    changed[0] = {
        "frequency": np.float64(-123.4),
        "mode": np.asarray([[0.1, 0.2, 0.4], [-0.1, -0.2, -0.3]]),
    }
    assert screen_runs._json_hash(array_payload) != screen_runs._json_hash(
        {"vibrations": changed}
    )


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


def test_weak_minimum_reference_requires_approval_before_reuse(tmp_path):
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
    vibrations = deepcopy(list(df.at[0, "dft_freq-vibs"]))
    vibrations[0]["frequency"] = -8.14
    df.at[0, "dft_freq-vibs"] = vibrations
    attach_result_contract(
        df,
        "minimum",
        dft=True,
        include_terminal_solv_sp=False,
        thermochemistry=method.thermochemistry,
    )
    library = ReferenceLibrary(tmp_path / "library").initialize()

    record = library.publish(df, target, method)

    assert record.metadata["validation_status"] == "review"
    assert record.metadata["auto_validation"]["flags"] == ["weak_minimum_imag"]
    assert library.index().iloc[0]["validation_status"] == "review"
    assert library.find(target, method, reuse_policy="auto_valid") is None
    assert library.find(target, method, reuse_policy="approved") is None

    record.approve(note="Inspected the weak, peripheral mode")

    assert library.find(
        target,
        method,
        reuse_policy="approved",
    ).reference_id == record.reference_id
    assert library.find(target, method, reuse_policy="auto_valid") is None


def test_reference_identity_uses_level_and_only_active_reference_stages(tmp_path):
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
    changed_ts_only = method.replace(
        dft_ts_opt=ft.workflows.methods.orca_composite(
            "r2SCAN-3c",
            job="optts",
            solvent="chloroform",
            uma="omol",
        )
    )
    from frust.screen.references import reference_identity

    full_key, _ = reference_identity(target, method, calculation_level="full")
    changed_key, _ = reference_identity(
        target,
        changed_ts_only,
        calculation_level="full",
    )
    ranked_key, _ = reference_identity(
        target,
        method,
        calculation_level="dft_ranked",
    )
    gas_method, _ = ft.workflows.methods.with_ranking_solvation(method, "gas")
    ranked_gas_key, _ = reference_identity(
        target,
        gas_method,
        calculation_level="dft_ranked",
    )
    assert changed_key == full_key
    assert ranked_key != full_key
    assert ranked_gas_key != ranked_key

    atoms = ["H", "H"]
    frame = pd.DataFrame(
        {
            "state_id": ["dimer"],
            "atoms": [atoms],
            "xtb_opt-oc": [np.zeros((2, 3)).tolist()],
            "dft_rank_sp-EE": [-1.0],
            "dft_rank_sp-NT": [True],
        }
    )
    attach_result_contract(
        frame,
        "minimum",
        dft=False,
        calculation_level="dft_ranked",
    )
    library = ReferenceLibrary(tmp_path / "tiered").initialize()
    record = library.publish(
        frame,
        target,
        method,
        calculation_level="dft_ranked",
    )
    assert record.metadata["calculation_level"] == "dft_ranked"
    assert record.metadata["free_energy_hartree"] is None
    assert "entries/screening/dft_ranked" in str(record.path)
    assert library.find(
        target,
        method,
        calculation_level="dft_ranked",
    ).reference_id == record.reference_id
    assert library.review_queue().empty


def test_barrier_analysis_matches_supplied_formulas_and_survives_relocation(tmp_path):
    original = tmp_path / "original"
    _write_barrier_bundle(original)
    run = ft.screen.open_run(original).refresh_analysis()
    barriers = run.barriers().set_index("ts_type")

    assert barriers.loc["TS1", "delta_e_kcal_mol"] == pytest.approx(0.15 * 627.5094740631)
    assert barriers.loc["TS1", "delta_g_kcal_mol"] == pytest.approx(0.1 * 627.5094740631)
    assert barriers.loc["TS1", "delta_g_corrected_kcal_mol"] == pytest.approx(
        0.1 * 627.5094740631 - 1.89
    )
    assert barriers.loc["TS2", "delta_g_corrected_kcal_mol"] == pytest.approx(
        0.2 * 627.5094740631
    )
    assert barriers.loc["TS3", "delta_g_corrected_kcal_mol"] == pytest.approx(
        0.3 * 627.5094740631 - 1.89
    )
    assert barriers.loc["TS4", "delta_g_corrected_kcal_mol"] == pytest.approx(
        0.4 * 627.5094740631
    )
    assert set(barriers["quality_status"]) == {"review"}

    relocated = tmp_path / "downloaded" / "run"
    relocated.parent.mkdir()
    shutil.copytree(original, relocated)
    moved = ft.screen.open_run(relocated)
    pd.testing.assert_frame_equal(run.barriers(), moved.barriers())


@pytest.mark.parametrize(
    ("level", "energy_stage"),
    [("low_cost", "xtb_opt"), ("dft_ranked", "dft_rank_sp")],
)
def test_electronic_levels_build_delta_e_without_frequencies(
    tmp_path,
    level,
    energy_stage,
):
    root = tmp_path / level
    root.mkdir()
    manifest = _manifest()
    manifest["calculation_level"] = level
    manifest["ranking_solvation"]["applied"] = level == "dft_ranked"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    formulas = {
        "ligand": {"C": 5, "H": 7, "N": 1},
        "dimer": {"C": 16, "H": 24, "B": 2, "N": 2},
        "HBpin-mol": {"C": 6, "H": 13, "B": 1, "O": 2},
        "HH": {"H": 2},
    }

    def electronic_row(state_id, energy, formula, *, state_kind="minimum"):
        atoms = _atoms(formula)
        return {
            "state_id": state_id,
            "state_kind": state_kind,
            "system_name": "pyrrole__NMe",
            "substrate_name": "pyrrole",
            "catalyst_name": "NMe",
            "rpos": 2 if state_kind == "transition_state" else None,
            "atoms": atoms,
            "xtb_opt-oc": np.zeros((len(atoms), 3)).tolist(),
            f"{energy_stage}-EE": energy,
            f"{energy_stage}-NT": True,
        }

    references = pd.DataFrame(
        [
            electronic_row("ligand", -10.0, formulas["ligand"]),
            electronic_row("dimer", -40.0, formulas["dimer"]),
            electronic_row("HBpin-mol", -5.0, formulas["HBpin-mol"]),
            electronic_row("HH", -1.0, formulas["HH"]),
        ]
    )
    transition_states = pd.DataFrame(
        [
            electronic_row(
                state,
                energy,
                {"C": 13, "H": 19, "B": 1, "N": 2}
                if state in {"TS1", "TS2"}
                else {"C": 19, "H": 30, "B": 2, "N": 2, "O": 2},
                state_kind="transition_state",
            )
            for state, energy in {
                "TS1": -29.9,
                "TS2": -29.8,
                "TS3": -33.7,
                "TS4": -33.6,
            }.items()
        ]
    )
    for frame, profile, path in (
        (
            references,
            "minimum",
            root / "calculations/references/merged.parquet",
        ),
        (
            transition_states,
            "transition_state",
            root / "calculations/transition_states/merged.parquet",
        ),
    ):
        attach_result_contract(
            frame,
            profile,
            dft=False,
            calculation_level=level,
            thermochemistry=None,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    run = ft.screen.open_run(root).refresh_analysis()
    barriers = run.barriers().set_index("ts_type")
    assert barriers.loc["TS1", "delta_e_kcal_mol"] == pytest.approx(
        0.1 * 627.5094740631
    )
    assert barriers["delta_g_kcal_mol"].isna().all()
    assert barriers["delta_g_corrected_kcal_mol"].isna().all()
    assert set(barriers["quality_status"]) == {"ready"}
    assert run.review_queue().empty


def test_end_to_end_levels_use_consistent_stages_and_solvent(tmp_path):
    expected = {
        "low_cost": {"xtb_preopt", "xtb_sp", "xtb_opt"},
        "dft_ranked": {"xtb_preopt", "xtb_sp", "xtb_opt", "dft_rank_sp"},
        "full": {
            "xtb_preopt",
            "xtb_sp",
            "xtb_opt",
            "dft_rank_sp",
            "dft_freq",
        },
    }
    for level, required in expected.items():
        workflow = ft.workflows.catalyst_screen(
            dataframe=_components(),
            screening="gxtb-default",
            level=level,
            method="r2scan-3c",
        )
        stages = workflow.show_stages()
        for branch in ("transition_states", "references"):
            branch_stages = set(stages.query("branch == @branch")["stage"])
            assert required <= branch_stages
            assert ("dft_rank_sp" in branch_stages) == (level != "low_cost")
        rank_rows = stages.query("stage == 'dft_rank_sp'")
        if level == "low_cost":
            assert rank_rows.empty
        else:
            assert set(rank_rows["solvent"]) == {"SMD(chloroform)"}

    gas = ft.workflows.catalyst_screen(
        dataframe=_components(),
        level="dft_ranked",
        method="r2scan-3c",
        ranking_solvation="gas",
    )
    assert gas.show_stages().query("stage == 'dft_rank_sp'")["solvent"].isna().all()


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
    indexed_profile = profile.set_index("profile_state")
    assert indexed_profile.loc["Cat", "relative_g_kcal_mol"] == pytest.approx(
        0.2 * 627.5094740631
    )
    assert indexed_profile.loc["Product", "relative_e_kcal_mol"] == pytest.approx(
        -0.5 * 627.5094740631
    )
    assert indexed_profile.loc["Product", "relative_g_kcal_mol"] == pytest.approx(
        -0.5 * 627.5094740631
    )
    assert indexed_profile.loc["Product", "mechanism_id"] == "frust_balanced_cycle::v2"

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


def test_weak_dimer_quality_propagates_to_full_profile(tmp_path):
    root = tmp_path / "full"
    _write_barrier_bundle(root, scope="full_cycle")
    references_path = root / "calculations/references/merged.parquet"
    references = pd.read_parquet(references_path)
    dimer_index = references.index[references["state_id"].eq("dimer")][0]
    vibrations = deepcopy(list(references.at[dimer_index, "dft_freq-vibs"]))
    vibrations[0]["frequency"] = -8.14
    references.at[dimer_index, "dft_freq-vibs"] = vibrations
    references.to_parquet(references_path, index=False)

    run = ft.screen.open_run(root).refresh_analysis()
    profile = run.profile(system_name="pyrrole__NMe", rpos=2)

    assert profile["relative_e_kcal_mol"].notna().all()
    assert profile["relative_g_kcal_mol"].notna().all()
    assert set(profile["quality_status"]) == {"review"}
    assert profile["quality_issues"].str.contains(
        "dependency_review:dimer",
        regex=False,
    ).all()


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

    for filename in ("computed.parquet", "merged.parquet"):
        frame = pd.read_parquet(reference_dir / filename)
        assert isinstance(frame.attrs.get("frust_results"), dict)
    first_run = ft.screen.open_run(root)
    assert first_run.summary().query(
        "artifact == 'barriers' and quality_status == 'review'"
    ).iloc[0]["count"] == 4

    first_two = library.index()["reference_id"].iloc[:2]
    for reference_id in first_two:
        library.get(reference_id).approve(note="One-time scientific structure review")

    mixed = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
        reference_store=store,
    )
    mixed_plan = mixed.plan().query("branch == 'references'")
    assert set(mixed_plan["action"]) == {"calculate", "reuse"}
    mixed_root = tmp_path / "mixed_run"
    mixed_root.mkdir()
    mixed._write_manifest(mixed_root)
    mixed_items = [item for item in mixed.targets() if item.branch == "references"]
    mixed._snapshot_reused_references(mixed_root, mixed_items)
    for index, item in enumerate(mixed_items):
        if item.action != "calculate":
            continue
        frame = pd.DataFrame(
            [_row(item.target.state_id, -10.0 - index, formulas[item.target.state_id])]
        )
        attach_result_contract(
            frame,
            "minimum",
            dft=True,
            include_terminal_solv_sp=False,
            thermochemistry=mixed.method.thermochemistry,
        )
        target_dir = mixed_root / "calculations" / "references" / item.target.tag
        target_dir.mkdir(parents=True)
        frame.to_parquet(target_dir / "final.parquet", index=False)
    _write_result(
        mixed_root / "calculations/transition_states/merged.parquet",
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
    mixed_report = _finalize_run(mixed, mixed_root)
    mixed_references = mixed_root / "calculations" / "references"
    assert mixed_report["n_references_calculated"] == 2
    assert mixed_report["n_references_reused"] == 2
    assert isinstance(
        pd.read_parquet(mixed_references / "reused.parquet").attrs.get("frust_results"),
        dict,
    )
    assert isinstance(
        pd.read_parquet(mixed_references / "merged.parquet").attrs.get("frust_results"),
        dict,
    )
    assert len(ft.screen.open_run(mixed_root).states()) == 8

    for reference_id in library.index()["reference_id"]:
        library.get(reference_id).approve(note="One-time scientific structure review")
    repeated = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
        reference_store=store,
    )
    reference_plan = repeated.plan().query("branch == 'references'")
    assert set(reference_plan["action"]) == {"reuse"}

    reused_root = tmp_path / "reused_run"
    reused_root.mkdir()
    repeated._write_manifest(reused_root)
    reused_items = [item for item in repeated.targets() if item.branch == "references"]
    repeated._snapshot_reused_references(reused_root, reused_items)
    _write_result(
        reused_root / "calculations/transition_states/merged.parquet",
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
    reused_report = _finalize_run(repeated, reused_root)
    reused_references = reused_root / "calculations" / "references"
    assert reused_report["n_references_calculated"] == 0
    assert reused_report["n_references_reused"] == 4
    assert isinstance(
        pd.read_parquet(reused_references / "merged.parquet").attrs.get("frust_results"),
        dict,
    )
    reuse_publication = json.loads(
        (reused_references / "publication_report.json").read_text()
    )
    assert reuse_publication["n_reused"] == 4
    assert reuse_publication["n_missing_results"] == 0
    assert len(ft.screen.open_run(reused_root).states()) == 8


@pytest.mark.parametrize(
    (
        "imaginary_frequency",
        "expected_state_quality",
        "expected_publication_status",
        "expected_published",
    ),
    [
        (-8.14, "review", "published", 4),
        (-80.0, "invalid", "not_published", 3),
    ],
)
def test_finalizer_retains_flagged_reference_results(
    tmp_path,
    imaginary_frequency,
    expected_state_quality,
    expected_publication_status,
    expected_published,
):
    root = tmp_path / "run"
    workflow = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
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
        frame = pd.DataFrame(
            [_row(target.state_id, -10.0 - index, formulas[target.state_id])]
        )
        if target.state_id == "dimer":
            vibrations = deepcopy(list(frame.at[0, "dft_freq-vibs"]))
            vibrations[0]["frequency"] = imaginary_frequency
            frame.at[0, "dft_freq-vibs"] = vibrations
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
    computed = pd.read_parquet(reference_dir / "computed.parquet")
    merged = pd.read_parquet(reference_dir / "merged.parquet")
    run = ft.screen.open_run(root)
    dimer = run.states().query("state_id == 'dimer'").iloc[0]
    barriers = run.barriers()
    publication = json.loads(
        (reference_dir / "publication_report.json").read_text()
    )
    dimer_publication = next(
        entry for entry in publication["entries"] if entry["state_id"] == "dimer"
    )

    assert len(computed) == 4
    assert len(merged) == 4
    assert set(computed["state_id"]) == {"ligand", "dimer", "HBpin-mol", "HH"}
    assert dimer["quality_status"] == expected_state_quality
    assert dimer["n_imag"] == 1
    assert barriers["delta_e_kcal_mol"].notna().all()
    assert barriers["delta_g_kcal_mol"].notna().all()
    assert set(barriers["quality_status"]) == {expected_state_quality}
    assert barriers["quality_issues"].str.contains(
        f"dependency_{expected_state_quality}:dimer",
        regex=False,
    ).all()
    assert dimer_publication["status"] == expected_publication_status
    assert dimer_publication["validation_status"] == expected_state_quality
    assert publication["n_published"] == expected_published
    assert report["n_references_calculated"] == 4
    assert report["n_references_published"] == expected_published


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
