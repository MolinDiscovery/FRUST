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
from frust.results import attach_result_contract, free_energy_components, result_contract
from frust.screen.references import ReferenceLibrary
from frust.structures import ChemicalSystem, StructureTarget
from frust.transformers import transformer_mols
from frust.workflows.screening import _concat_reference_results, _finalize_run


CATALYST = "BC1=C(N(C)C)C=CC=C1"


def test_lower_tier_contract_omits_inert_thermochemistry():
    contract = result_contract(
        "minimum",
        dft=False,
        calculation_level="dft_ranked",
        thermochemistry=ft.workflows.ThermochemistrySpec(
            "electronic_plus_thermal"
        ),
    )

    assert "thermochemistry" not in contract


def test_reference_concat_accepts_legacy_lower_tier_thermochemistry(tmp_path):
    current = pd.DataFrame({"dft_rank_sp-EE": [-1.0]})
    attach_result_contract(
        current,
        "minimum",
        dft=False,
        calculation_level="dft_ranked",
    )
    legacy = current.copy()
    legacy.attrs["frust_results"]["thermochemistry"] = {
        "schema_version": 1,
        "mode": "electronic_plus_thermal",
        "temperature_k": 298.15,
        "energy_unit": "hartree",
    }

    merged = _concat_reference_results(
        [current, legacy],
        source_files=[tmp_path / "current.parquet", tmp_path / "legacy.parquet"],
    )

    assert len(merged) == 2
    assert "thermochemistry" not in merged.attrs["frust_results"]


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


def _write_reference_tier_snapshots(
    frame: pd.DataFrame,
    target_dir: Path,
) -> None:
    """Write compact synthetic nested-tier winners for cache tests."""
    low_cost = frame.copy()
    low_cost["xtb_opt-oc"] = low_cost["dft_opt-oc"]
    low_cost["xtb_opt-EE"] = low_cost["dft_freq-EE"]
    low_cost["xtb_opt-NT"] = True
    attach_result_contract(
        low_cost,
        "minimum",
        dft=False,
        calculation_level="low_cost",
        thermochemistry=None,
    )
    low_cost.to_parquet(target_dir / "tier_low_cost.parquet", index=False)

    ranked = low_cost.copy()
    ranked["dft_rank_sp-EE"] = ranked["dft_freq-EE"]
    ranked["dft_rank_sp-NT"] = True
    attach_result_contract(
        ranked,
        "minimum",
        dft=False,
        calculation_level="dft_ranked",
        thermochemistry=None,
    )
    ranked.to_parquet(target_dir / "tier_dft_ranked.parquet", index=False)


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


def _add_nested_barrier_tiers(root: Path) -> None:
    """Add synthetic low-cost and DFT-ranked winners to a full bundle."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tier_results: dict[str, dict[str, str | None]] = {}
    for level, stage, shift in (
        ("low_cost", "xtb_opt", 0.01),
        ("dft_ranked", "dft_rank_sp", 0.02),
    ):
        paths: dict[str, str | None] = {
            "transition_states": None,
            "references": None,
            "cycle_molecules": None,
            "int3": None,
        }
        for source, profile in (
            ("transition_states", "transition_state"),
            ("references", "minimum"),
        ):
            terminal_path = root / manifest["calculation_results"][source]
            frame = pd.read_parquet(terminal_path)
            frame["xtb_opt-oc"] = frame[
                "dft_ts_opt-oc" if profile == "transition_state" else "dft_opt-oc"
            ]
            frame[f"{stage}-EE"] = frame["dft_freq-EE"]
            if source == "transition_states":
                frame[f"{stage}-EE"] += shift
            frame[f"{stage}-NT"] = True
            attach_result_contract(
                frame,
                profile,
                dft=False,
                calculation_level=level,
                thermochemistry=None,
            )
            relative = f"calculations/{source}/tiers/{level}/merged.parquet"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            paths[source] = relative
        tier_results[level] = paths
    tier_results["full"] = manifest["calculation_results"]
    manifest["analysis_levels"] = ["low_cost", "dft_ranked", "full"]
    manifest["tier_calculation_results"] = tier_results
    manifest_path.write_text(json.dumps(manifest, indent=2))


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

    renamed_system = replace(
        system,
        system_name="descriptive_pyrrole__renamed_NMe",
        substrate_name="descriptive_pyrrole",
        catalyst_name="renamed_NMe",
    )
    renamed_target = replace(
        target,
        target_id="dimer:renamed_NMe",
        tag="dimer__renamed_NMe",
        system=renamed_system,
    )
    rebound = record.materialize(renamed_target)
    assert rebound.loc[0, "structure_id"] == "dimer:renamed_NMe"
    assert rebound.loc[0, "system_name"] == "descriptive_pyrrole__renamed_NMe"
    assert rebound.loc[0, "substrate_name"] == "descriptive_pyrrole"
    assert rebound.loc[0, "catalyst_name"] == "renamed_NMe"
    binding = rebound.attrs["frust_reference_bindings"]["bindings"][0]
    assert binding["source_labels"]["catalyst_name"] == "NMe"
    assert binding["target_labels"]["catalyst_name"] == "renamed_NMe"
    assert record.dataframe().loc[0, "catalyst_name"] == "NMe"

    incompatible = replace(
        renamed_target,
        system=replace(renamed_system, catalyst_smiles="CC"),
    )
    with pytest.raises(ValueError, match="chemical_identity"):
        record.materialize(incompatible)

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


def test_full_run_compares_independently_selected_screening_and_full_barriers(
    tmp_path,
):
    root = tmp_path / "nested-tiers"
    _write_barrier_bundle(root, scope="full_cycle")
    _add_nested_barrier_tiers(root)

    run = ft.screen.open_run(root).refresh_analysis()

    assert run.available_analysis_levels() == (
        "low_cost",
        "dft_ranked",
        "full",
    )
    low_cost = run.barriers(level="low_cost").set_index("ts_type")
    ranked = run.barriers(level="dft_ranked").set_index("ts_type")
    full = run.barriers(level="full").set_index("ts_type")
    comparison = run.compare_barriers().set_index("ts_type")

    assert low_cost.loc["TS1", "delta_e_kcal_mol"] == pytest.approx(
        0.16 * screen_runs.HARTREE_TO_KCAL_MOL
    )
    assert ranked.loc["TS1", "delta_e_kcal_mol"] == pytest.approx(
        0.17 * screen_runs.HARTREE_TO_KCAL_MOL
    )
    assert full.loc["TS1", "delta_e_kcal_mol"] == pytest.approx(
        0.15 * screen_runs.HARTREE_TO_KCAL_MOL
    )
    assert ranked["delta_g_kcal_mol"].isna().all()
    assert comparison.loc["TS1", "delta_e_dft_ranked_kcal_mol"] == pytest.approx(
        ranked.loc["TS1", "delta_e_kcal_mol"]
    )
    assert comparison.loc["TS1", "delta_e_full_kcal_mol"] == pytest.approx(
        full.loc["TS1", "delta_e_kcal_mol"]
    )
    assert comparison.loc["TS1", "delta_g_full_kcal_mol"] == pytest.approx(
        full.loc["TS1", "delta_g_kcal_mol"]
    )
    assert pd.isna(comparison.loc["TS1", "delta_g_dft_ranked_kcal_mol"])
    pd.testing.assert_frame_equal(run.barriers(), run.barriers(level="full"))

    with pytest.raises(ValueError, match="Gibbs profiles require"):
        run.profile(level="dft_ranked", quantity="gibbs")


def test_lowest_dimer_reference_uses_gibbs_ranking_and_one_exact_result(tmp_path):
    root = tmp_path / "lowest-dimer"
    _write_barrier_bundle(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["dimer_reference"] = "lowest"
    manifest["dimer_candidates"] = list(screen_runs.DIMER_STATES)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    formula = {"C": 16, "H": 24, "B": 2, "N": 2}
    references = [
        _row("ligand", -10.0, {"C": 5, "H": 7, "N": 1}),
        _row("dimer", -40.0, formula),
        _row("dimer_bh_bridged", -40.2, formula),
        _row("dimer_eight_membered", -40.1, formula),
        _row("HBpin-mol", -5.0, {"C": 6, "H": 13, "B": 1, "O": 2}),
        _row("HH", -1.0, {"H": 2}),
    ]
    references[1]["dft_freq-EE"] = -40.2
    references[2]["dft_freq-EE"] = -39.4
    references[3]["dft_freq-EE"] = -41.0
    _write_result(
        root / "calculations/references/merged.parquet",
        references,
        "minimum",
    )

    run = ft.screen.open_run(root).refresh_analysis()
    choices = run.dimer_references()
    selected = choices.loc[choices["selected"]].iloc[0]
    barrier = run.barriers().query("ts_type == 'TS1'").iloc[0]

    assert choices["selection_quantity"].unique().tolist() == ["gibbs"]
    assert selected["state_id"] == "dimer_bh_bridged"
    assert selected["selection_quality_status"] == "ready"
    assert barrier["dimer_state_id"] == "dimer_bh_bridged"
    assert barrier["dimer_result_id"] == selected["result_id"]
    assert barrier["delta_e_kcal_mol"] == pytest.approx(
        (-30.0 + 10.1 + 0.5 * 39.4) * screen_runs.HARTREE_TO_KCAL_MOL
    )


def test_lowest_dimer_reference_is_strict_when_a_topology_is_missing(tmp_path):
    root = tmp_path / "missing-dimer"
    _write_barrier_bundle(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["dimer_reference"] = "lowest"
    manifest["dimer_candidates"] = list(screen_runs.DIMER_STATES)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    references_path = root / "calculations/references/merged.parquet"
    references = pd.read_parquet(references_path)
    bridged = references.query("state_id == 'dimer'").copy()
    bridged["state_id"] = "dimer_bh_bridged"
    references = pd.concat([references, bridged], ignore_index=True)
    references.to_parquet(references_path, index=False)

    run = ft.screen.open_run(root).refresh_analysis()
    choices = run.dimer_references()

    assert not choices["selected"].any()
    assert set(choices["selection_quality_status"]) == {"incomplete"}
    assert choices.iloc[0]["selection_issues"] == "missing:dimer_eight_membered"
    assert set(run.barriers()["quality_status"]) == {"incomplete"}
    assert run.barriers()["dimer_state_id"].isna().all()


def test_low_cost_dimer_reference_ranks_by_electronic_energy():
    states = pd.DataFrame(
        {
            "catalyst_name": ["cat", "cat", "cat"],
            "state_id": list(screen_runs.DIMER_STATES),
            "result_id": ["a", "b", "c"],
            "cid": [0, 0, 0],
            "quality_status": ["ready", "ready", "ready"],
            "electronic_energy_hartree": [-10.0, -10.2, -10.1],
            "free_energy_hartree": [-9.0, -8.0, -10.0],
        }
    )
    manifest = {
        "calculation_level": "low_cost",
        "dimer_reference": "lowest",
        "dimer_candidates": list(screen_runs.DIMER_STATES),
        "analysis_targets": [{"catalyst_name": "cat"}],
    }

    choices = screen_runs._build_dimer_references(states, manifest)

    selected = choices.loc[choices["selected"]].iloc[0]
    assert selected["state_id"] == "dimer_bh_bridged"
    assert selected["selection_quantity"] == "electronic"

    invalid_states = states.copy()
    invalid_states.loc[
        invalid_states["state_id"].eq("dimer_eight_membered"),
        "quality_status",
    ] = "invalid"
    invalid_choices = screen_runs._build_dimer_references(
        invalid_states,
        manifest,
    )
    assert not invalid_choices["selected"].any()
    assert set(invalid_choices["selection_quality_status"]) == {"invalid"}
    assert invalid_choices.iloc[0]["selection_issues"] == (
        "invalid:dimer_eight_membered"
    )


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
    assert indexed_profile.loc["Product", "mechanism_id"] == "frust_balanced_cycle::v3"

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


def test_profile_and_plot_profile_can_select_an_available_dimer(tmp_path):
    root = tmp_path / "dimer-profile"
    _write_barrier_bundle(root, scope="full_cycle")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dimer_reference"] = "lowest"
    manifest["dimer_candidates"] = list(screen_runs.DIMER_STATES)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    references_path = root / "calculations/references/merged.parquet"
    references = pd.read_parquet(references_path)
    dimer = references.loc[references["state_id"].eq("dimer")].copy()
    bridged = dimer.copy()
    bridged["state_id"] = "dimer_bh_bridged"
    bridged["structure_id"] = "dimer_bh_bridged:pyrrole__NMe:rNone"
    bridged["dft_freq-EE"] -= 0.2
    bridged["dft_freq-GE"] -= 0.2
    eight_membered = dimer.copy()
    eight_membered["state_id"] = "dimer_eight_membered"
    eight_membered["structure_id"] = "dimer_eight_membered:pyrrole__NMe:rNone"
    eight_membered["dft_freq-EE"] -= 0.1
    eight_membered["dft_freq-GE"] -= 0.1
    references = pd.concat(
        [references, bridged, eight_membered],
        ignore_index=True,
    )
    references.to_parquet(references_path, index=False)

    run = ft.screen.open_run(root).refresh_analysis()
    recorded = run.profile(system_name="pyrrole__NMe", rpos=2)
    ordinary = run.profile(
        system_name="pyrrole__NMe",
        rpos=2,
        dimer_reference="dimer",
    )

    assert set(recorded["dimer_state_id"]) == {"dimer_bh_bridged"}
    assert set(ordinary["dimer_state_id"]) == {"dimer"}
    assert not ordinary["relative_g_kcal_mol"].equals(
        recorded["relative_g_kcal_mol"]
    )

    with patch("frust.vis.plot_energy_profile", return_value="profile-plot") as plot:
        result = run.plot_profile(
            system_name="pyrrole__NMe",
            rpos=2,
            dimer_reference="dimer_eight_membered",
            ylabel="dG",
        )

    assert result == "profile-plot"
    assert plot.call_args.kwargs["ylabel"] == "dG"
    plotted_states = dict(plot.call_args.args[0])
    assert plotted_states["Dimer"] == pytest.approx(0.0)


def test_profile_rejects_a_dimer_that_was_not_calculated(tmp_path):
    root = tmp_path / "single-dimer-profile"
    _write_barrier_bundle(root, scope="full_cycle")
    run = ft.screen.open_run(root).refresh_analysis()

    with pytest.raises(ValueError, match=r"available dimers are \['dimer'\]"):
        run.profile(dimer_reference="dimer_bh_bridged")


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
        "ligand",
        "dimer",
        "dimer_bh_bridged",
        "dimer_eight_membered",
        "HBpin-mol",
        "HH",
    }
    assert plan.attrs["dimer_reference"] == "lowest"
    assert set(plan["action"]) == {"calculate"}
    assert not (tmp_path / "missing_library").exists()

    fixed = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c",
        dimer_reference="dimer_eight_membered",
    )
    fixed_references = set(
        fixed.plan().query("branch == 'references'")["state_id"]
    )
    assert fixed_references == {
        "ligand",
        "dimer_eight_membered",
        "HBpin-mol",
        "HH",
    }
    assert fixed._reference_protocol() == workflow._reference_protocol()

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
        dimer_reference="dimer",
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
        _write_reference_tier_snapshots(frame, target_dir)
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
    assert len(library.index()) == 12
    assert (root / "calculations/references/merged.parquet").exists()
    assert len(list((root / "calculations/references/entries").rglob("optimized.xyz"))) == 12

    for filename in ("computed.parquet", "merged.parquet"):
        frame = pd.read_parquet(reference_dir / filename)
        assert isinstance(frame.attrs.get("frust_results"), dict)
    first_run = ft.screen.open_run(root)
    assert first_run.summary().query(
        "artifact == 'barriers' and quality_status == 'review'"
    ).iloc[0]["count"] == 4

    first_two = library.search(calculation_level="full")["reference_id"].iloc[:2]
    for reference_id in first_two:
        library.get(reference_id).approve(note="One-time scientific structure review")

    mixed = ft.workflows.catalyst_screen(
        dataframe=_components(),
        method="r2scan-3c-solv",
        reference_store=store,
        dimer_reference="dimer",
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
        _write_reference_tier_snapshots(frame, target_dir)
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
    renamed_components = _components().copy()
    renamed_components["compound_name"] = [
        "descriptive_pyrrole",
        "renamed_NMe",
    ]
    repeated = ft.workflows.catalyst_screen(
        dataframe=renamed_components,
        method="r2scan-3c-solv",
        reference_store=store,
        dimer_reference="dimer",
    )
    reference_plan = repeated.plan().query("branch == 'references'")
    assert set(reference_plan["action"]) == {"reuse"}

    repeated_items = repeated.targets()
    ligand_index = next(
        index
        for index, item in enumerate(repeated_items)
        if item.branch == "references" and item.target.state_id == "ligand"
    )
    cached_preview = repeated.preview(targets=[ligand_index])
    assert cached_preview.loc[0, "substrate_name"] == "descriptive_pyrrole"
    assert cached_preview.loc[0, "catalyst_name"] == "renamed_NMe"

    reused_root = tmp_path / "reused_run"
    reused_root.mkdir()
    repeated._write_manifest(reused_root)
    reused_items = [item for item in repeated_items if item.branch == "references"]
    repeated._snapshot_reused_references(reused_root, reused_items)
    rebound_references = pd.read_parquet(
        reused_root / "calculations/references/reused.parquet"
    )
    assert set(rebound_references["system_name"]) == {
        "descriptive_pyrrole__renamed_NMe"
    }
    assert set(rebound_references["substrate_name"]) == {"descriptive_pyrrole"}
    assert set(rebound_references["catalyst_name"]) == {"renamed_NMe"}
    assert len(
        rebound_references.attrs["frust_reference_bindings"]["bindings"]
    ) == 4
    for level in ("low_cost", "dft_ranked"):
        tier_reused = pd.read_parquet(
            reused_root
            / "calculations/references/tiers"
            / level
            / "reused.parquet"
        )
        assert set(tier_reused["substrate_name"]) == {"descriptive_pyrrole"}
        assert set(tier_reused["catalyst_name"]) == {"renamed_NMe"}
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
                system_name="descriptive_pyrrole__renamed_NMe",
                substrate_name="descriptive_pyrrole",
                catalyst_name="renamed_NMe",
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
    reused_screen = ft.screen.open_run(reused_root)
    assert len(reused_screen.states()) == 8
    assert not reused_screen.barriers()["quality_issues"].str.contains(
        "missing:ligand"
    ).any()

    repair_paths = [
        reused_references / "reused.parquet",
        reused_references / "merged.parquet",
        reused_references / "tiers/low_cost/reused.parquet",
        reused_references / "tiers/low_cost/merged.parquet",
        reused_references / "tiers/dft_ranked/reused.parquet",
        reused_references / "tiers/dft_ranked/merged.parquet",
    ]
    for path in repair_paths:
        stale = pd.read_parquet(path)
        ligand = stale["state_id"].eq("ligand")
        stale.loc[ligand, "structure_id"] = "ligand:pyrrole"
        stale.loc[ligand, "system_name"] = "pyrrole__NMe"
        stale.loc[ligand, "substrate_name"] = "pyrrole"
        stale.loc[ligand, "catalyst_name"] = "NMe"
        stale.to_parquet(path, index=False)

    proposed = ft.screen.repair_reference_bindings(reused_root)
    assert len(proposed) == len(repair_paths)
    assert set(proposed["old_substrate_name"]) == {"pyrrole"}
    assert set(proposed["substrate_name"]) == {"descriptive_pyrrole"}
    assert not proposed.attrs["applied"]
    assert set(pd.read_parquet(repair_paths[0])["substrate_name"]) == {
        "pyrrole",
        "descriptive_pyrrole",
    }

    backup_dir = tmp_path / "reference_binding_backup"
    repair = ft.screen.repair_reference_bindings(
        reused_root,
        apply=True,
        backup_dir=backup_dir,
    )
    assert repair.attrs["applied"]
    assert repair.attrs["backup_dir"] == str(backup_dir)
    assert (backup_dir / repair_paths[0].relative_to(reused_root)).exists()
    assert (reused_root / "reference_binding_repair.json").exists()
    for path in repair_paths:
        repaired = pd.read_parquet(path)
        ligand = repaired["state_id"].eq("ligand")
        assert set(repaired.loc[ligand, "substrate_name"]) == {
            "descriptive_pyrrole"
        }
    assert not ft.screen.open_run(reused_root).barriers()[
        "quality_issues"
    ].str.contains("missing:ligand").any()


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
        dimer_reference="dimer",
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
