from __future__ import annotations

import json

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

import frust as ft
from frust.results import attach_result_contract
from frust.structures.planner import molecule_states
from frust.transformers import transformer_mols

CATALYST = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"
MOLECULE_STATES = (
    "dimer",
    "HH",
    "ligand",
    "catalyst",
    "int1",
    "int2",
    "HBpin-ligand",
    "HBpin-mol",
)
DIMER_STATES = (
    "dimer",
    "dimer_bh_bridged",
    "dimer_eight_membered",
)


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", CATALYST],
            "compound_name": ["pyrrole", "cat"],
            "rpos": ["2", None],
        }
    )


def test_mols_targets_are_lightweight_json_structure_plans(monkeypatch):
    substrate = _components().iloc[[0]][["smiles", "compound_name", "rpos"]]

    def unexpected_builder(*args, **kwargs):
        raise AssertionError("target inspection must not construct molecules")

    monkeypatch.setattr(
        "frust.workflows.factories.create_mol_per_rpos", unexpected_builder
    )
    wf = ft.workflows.mols(dataframe=substrate, select_mols=["int1"])
    target = wf.targets()[0]

    assert target.state_id == "int1"
    assert target.state_kind == "minimum"
    assert target.builder_spec == "cycle::int1::v3"
    assert json.loads(json.dumps(target.payload))["rpos"] == 2


def test_mols_docstring_and_shortcuts_define_the_public_state_vocabulary():
    docstring = ft.workflows.mols.__doc__ or ""

    assert all(
        f'``"{state}"``' in docstring
        for state in (*MOLECULE_STATES, *DIMER_STATES)
    )
    assert molecule_states("all") == MOLECULE_STATES
    assert molecule_states("dimers") == DIMER_STATES
    assert molecule_states("uniques") == ("ligand", "int1", "int2", "HBpin-ligand")
    assert molecule_states("generics") == ("dimer", "HH", "catalyst", "HBpin-mol")


def test_dimer_variants_have_expected_formula_charge_and_interaction_ring():
    dimers = transformer_mols(
        ligand_smiles="CN1C=CC=C1",
        catalyst_smiles=CATALYST,
        select="dimers",
    )
    expected_largest_charged_ring = {
        "dimer": 4,
        "dimer_bh_bridged": 6,
        "dimer_eight_membered": 8,
    }

    for state_id, expected_ring_size in expected_largest_charged_ring.items():
        mol = next(value for key, value in dimers.items() if key.endswith(state_id))
        charged_ring_sizes = [
            len(ring)
            for ring in Chem.GetSymmSSSR(mol)
            if any(mol.GetAtomWithIdx(index).GetFormalCharge() for index in ring)
        ]
        assert CalcMolFormula(mol) == "C30H48B2N2"
        assert Chem.GetFormalCharge(mol) == 0
        assert max(charged_ring_sizes) == expected_ring_size


def test_workflow_stage_tables_stay_separate_and_homogeneous():
    substrate = _components().iloc[[0]][["smiles", "compound_name", "rpos"]]
    workflows = {
        "mols": ft.workflows.mols(dataframe=substrate, select_mols=["int2"], dft=True),
        "screen_ts": ft.workflows.screen_ts(
            dataframe=_components(), ts_types=["TS1"], dft=True, prune_initial=False
        ),
        "int3": ft.workflows.int3(dataframe=_components(), dft=True),
    }

    assert "dft_ts_opt" not in workflows["mols"].show_stages()["stage"].tolist()
    assert "dft_ts_opt" in workflows["screen_ts"].show_stages()["stage"].tolist()
    assert "dft_ts_opt" not in workflows["int3"].show_stages()["stage"].tolist()
    assert "dft_opt" in workflows["int3"].show_stages()["stage"].tolist()
    for workflow in workflows.values():
        stages = workflow.show_stages(execution="dft_staged")
        assert stages["stage"].is_unique
        assert "rank_by" in stages


def test_calculation_free_structure_apis_use_workflow_targets(monkeypatch):
    systems = ft.screen.expand(ft.screen.read(_components()))
    calls = []

    def fake_build(target, **kwargs):
        calls.append((target, kwargs))
        return pd.DataFrame(
            {
                "system_name": [target.system.system_name],
                "state_id": [target.state_id],
                "state_kind": [target.state_kind],
                "rpos": [target.rpos if target.rpos is not None else pd.NA],
                "cid": [0],
                "atoms": [["H"]],
                "coords_embedded": [[(0.0, 0.0, 0.0)]],
            }
        )

    monkeypatch.setattr("frust.structures.api.build", fake_build)

    mols = ft.structures.create_mols(
        systems,
        states=["HH", "int1", "int2"],
        n_confs=2,
        n_cores=3,
    )
    mol_targets = ft.workflows.mols(
        dataframe=systems,
        select_mols=["HH", "int1", "int2"],
    ).targets()

    assert [target.target_id for target, _ in calls] == [
        target.target_id for target in mol_targets
    ]
    assert set(mols["state_id"]) == {"HH", "int1", "int2"}
    assert {
        "system_name",
        "state_id",
        "state_kind",
        "rpos",
        "atoms",
        "coords_embedded",
    } <= set(mols.columns)
    assert all(kwargs["n_confs"] == 2 for _, kwargs in calls)
    assert all(kwargs["n_cores"] == 3 for _, kwargs in calls)
    assert mols.attrs["frust_structure_generation"] == {
        "schema_version": 1,
        "source": "frust.structures.create_mols",
        "calculation_free": True,
        "requested_n_confs": 2,
        "n_cores": 3,
        "n_targets": 3,
        "states": ["HH", "int1", "int2"],
    }

    calls.clear()
    int3 = ft.structures.create_int3_guesses(systems, n_confs=1)
    int3_targets = ft.workflows.int3(dataframe=systems).targets()

    assert [target.target_id for target, _ in calls] == [
        target.target_id for target in int3_targets
    ]
    assert set(int3["state_id"]) == {"INT3"}
    assert set(int3["state_kind"]) == {"constrained_minimum"}
    assert set(mols["state_id"]).isdisjoint(set(int3["state_id"]))


def test_workflow_preview_uses_selected_cached_typed_targets(monkeypatch):
    wf = ft.workflows.mols(
        dataframe=_components(),
        select_mols=["HH", "int1", "int2"],
        dft=True,
    )
    planned = wf.targets()
    calls = []

    def fake_build(target, **kwargs):
        calls.append((target, kwargs))
        return pd.DataFrame(
            {
                "system_name": [target.system.system_name],
                "state_id": [target.state_id],
                "state_kind": [target.state_kind],
                "rpos": [target.rpos if target.rpos is not None else pd.NA],
                "atoms": [["H"]],
                "coords_embedded": [[(0.0, 0.0, 0.0)]],
            }
        )

    monkeypatch.setattr("frust.structures.api.build", fake_build)

    preview = wf.preview(n_confs=2, targets=[1], n_cores=3)

    assert calls == [
        (
            planned[1],
            {
                "n_confs": 2,
                "n_cores": 3,
                "memory_gb": 4,
                "debug": False,
            },
        )
    ]
    assert preview["state_id"].tolist() == [planned[1].state_id]
    assert preview.attrs["frust_structure_generation"]["source"] == (
        "frust.workflows.mols.preview"
    )
    assert not any(column.endswith(("-EE", "-NT", "-oc")) for column in preview)


def test_preview_rejects_non_typed_legacy_targets():
    wf = ft.workflows.mols(smiles=["c1ccccc1"], split="per_input")

    with pytest.raises(TypeError, match="requires typed StructureTarget"):
        wf.preview()


def test_mols_component_input_preserves_variable_catalyst_smiles():
    components = pd.DataFrame(
        {
            "compound_name": ["furan", "NMe"],
            "role": ["substrate", "catalyst"],
            "smiles": ["C1=CC=CO1", "BC1=C(N(C)C)C=CC=C1"],
        }
    )

    wf = ft.workflows.mols(
        dataframe=components,
        select_mols=["catalyst", "int1", "int2"],
    )
    targets = wf.targets()

    assert targets[0].tag == "catalyst__NMe"
    assert {target.system.catalyst_smiles for target in targets} == {
        "BC1=C(N(C)C)C=CC=C1"
    }
    assert all("NMe" in target.tag for target in targets)

    built = ft.create_mol_per_rpos(
        pd.DataFrame(
            {
                "smiles": ["C1=CC=CO1"],
                "catalyst_smiles": ["BC1=C(N(C)C)C=CC=C1"],
                "system_name": ["furan__NMe"],
                "rpos": [0],
            }
        ),
        select_mols=["int1", "int2"],
        show_iupac=False,
    )
    by_role = {
        metadata["molecule_role"]: molecule for molecule, metadata in built.values()
    }
    assert set(by_role) == {"int1", "int2"}
    assert all(molecule.GetNumAtoms() == 15 for molecule in by_role.values())
    assert sorted(atom.GetFormalCharge() for atom in by_role["int1"].GetAtoms()) == [
        -1,
        *([0] * 13),
        1,
    ]
    assert all(atom.GetFormalCharge() == 0 for atom in by_role["int2"].GetAtoms())


def test_mols_rejects_removed_mol2_name_with_migration_guidance():
    substrate = _components().iloc[[0]][["smiles", "compound_name", "rpos"]]

    with pytest.raises(ValueError, match="mol2.*renamed to 'int2'"):
        ft.workflows.mols(dataframe=substrate, select_mols=["mol2"]).targets()


def test_legacy_result_upgrade_adds_canonical_names_and_semantics():
    old = pd.DataFrame(
        {
            "structure_type": ["MOL", "MOL"],
            "molecule_role": ["int2", "mol2"],
            "structure_id": ["MOL:pyrrole:int2:r2", "MOL:pyrrole:mol2:r2"],
            "custom_name": ["pyrrole_int2_rpos(2)", "pyrrole_mol2_rpos(2)"],
            "DFT-Opt-oc": [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]],
            "DFT-SP-EE": [-100.0, -99.0],
        }
    )
    old.attrs["frust_workflow"] = {"workflow": "mols"}

    upgraded = ft.upgrade_dataframe(old)

    assert upgraded["state_id"].tolist() == ["int1", "int2"]
    assert upgraded["molecule_role"].tolist() == ["int1", "int2"]
    assert upgraded["structure_id"].tolist() == [
        "MOL:pyrrole:int1:r2",
        "MOL:pyrrole:int2:r2",
    ]
    assert upgraded["custom_name"].tolist() == [
        "pyrrole_int1_rpos(2)",
        "pyrrole_int2_rpos(2)",
    ]
    assert upgraded.loc[0, "state_kind"] == "minimum"
    assert upgraded.attrs["frust_schema"]["version"] == 3
    assert ft.result_column(upgraded) == "dft_solv_sp-EE"
    assert ft.get_result(upgraded).iloc[0] == -100.0
    assert "dft_opt-oc" in upgraded


def test_legacy_result_upgrade_refuses_ambiguous_energy_names():
    old = pd.DataFrame(
        {
            "structure_type": ["MOL"],
            "molecule_role": ["int2"],
            "DFT-SP-EE": [-100.0],
        }
    )
    with pytest.raises(ValueError, match="Cannot safely map"):
        ft.upgrade_dataframe(old)


def test_current_schema_int2_is_not_remapped_during_upgrade():
    current = pd.DataFrame(
        {
            "structure_type": ["MOL"],
            "molecule_role": ["int2"],
            "state_id": ["int2"],
            "structure_id": ["MOL:pyrrole:int2:r2"],
        }
    )
    current.attrs["frust_schema"] = {"name": "frust-results", "version": 3}

    upgraded = ft.upgrade_dataframe(current)

    assert upgraded.loc[0, "state_id"] == "int2"
    assert upgraded.loc[0, "molecule_role"] == "int2"
    assert upgraded.loc[0, "structure_id"] == "MOL:pyrrole:int2:r2"


def test_lowest_energy_rows_uses_semantic_analysis_energy():
    df = pd.DataFrame(
        {
            "structure_id": ["a", "a"],
            "xtb_opt-EE": [-10.0, -9.0],
            "dft_solv_sp-EE": [-98.0, -99.0],
        }
    )
    attach_result_contract(df, "minimum", dft=True)

    selected = ft.lowest_energy_rows(df)

    assert selected.index.tolist() == [1]


@pytest.mark.slow
def test_native_int3_builder_emits_role_based_constrained_minimum():
    systems = ft.screen.expand(ft.screen.read(_components()))
    df = ft.structures.create_int3_guesses(systems, n_confs=1)

    assert set(df["state_id"]) == {"INT3"}
    assert set(df["state_kind"]) == {"constrained_minimum"}
    roles = df.iloc[0]["constraint_roles"]
    assert {"cat_B", "transfer_H", "pin_B", "substrate_C"} <= set(roles)
    assert (
        df.iloc[0]["ts_spec_id"]
        == "INT3::tsguess2-v2::wb97xd3-631g::gas::r1"
    )
    assert len(df.iloc[0]["constraint_spec"]) == 6
    assert not any(column.endswith(("-EE", "-NT", "-oc")) for column in df)


@pytest.mark.slow
def test_create_mols_returns_embedded_canonical_structures_only():
    systems = ft.screen.expand(ft.screen.read(_components()))

    df = ft.structures.create_mols(
        systems,
        states=["HH", "int1", "int2"],
        n_confs=1,
    )

    assert set(df["state_id"]) == {"HH", "int1", "int2"}
    assert set(df["state_kind"]) == {"minimum"}
    assert df["rpos"].isna().sum() == 1
    assert all(
        len(atoms) == len(coords)
        for atoms, coords in zip(df["atoms"], df["coords_embedded"])
    )
    assert not any(column.endswith(("-EE", "-NT", "-oc")) for column in df)
