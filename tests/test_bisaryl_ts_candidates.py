"""Bisaryl graph, lineage and explicit-candidate workflow contracts."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import frust as ft
from frust.tsguess2.api import _role_mapping_from_mol
from frust.tsguess2.builders import build_bisaryl_h2_connected_smiles
from frust.tsguess2.models import TSGuess2Spec
from frust.workflows.core import ExecutionOptions


TMP = "CC1(C)CCCC(C)(C)N1c1ccccc1B"
PYRROLE = "CN1C=CC=C1"


def candidate(state="TS6"):
    builder = {"TS5": "bisaryl_adduct", "TS6": "bisaryl_h2",
               "INT4": "bisaryl_adduct", "INT5": "bisaryl_diaryl",
               "TS7": "bisaryl_exchange", "INT6": "bisaryl_exchange", "TS8": "bisaryl_exchange"}[state]
    return TSGuess2Spec(
        name=state, spec_id=f"{state}::test-candidate", profile_id="omol-uma-s-1p2p1/gas",
        builder_key=builder,
        core_smarts="[#5]", role_coordinates={}, constraints=(),
    )


def systems():
    return pd.DataFrame([dict(system_name="pyrrole__TMP", substrate_name="pyrrole",
                              catalyst_name="TMP", substrate_smiles=PYRROLE,
                              catalyst_smiles=TMP, rpos=2)])


@pytest.mark.parametrize("catalyst,formula,atoms", [
    (TMP, "C25H36BN3", 65),
    ("CN(C)c1ccccc1B", "C18H24BN3", 46),
    ("B(c1ccccc1N1CCCCC1)", "C21H28BN3", 53),
])
def test_bisaryl_inventory_and_eliminating_hydrogens(catalyst, formula, atoms):
    smiles = build_bisaryl_h2_connected_smiles(catalyst, PYRROLE, [2])[2]
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdMolDescriptors.CalcMolFormula(mol) == formula
    assert mol.GetNumAtoms() == atoms
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
    roles = _role_mapping_from_mol(mol, candidate())
    assert roles["bound_C"] != roles["incoming_C"]
    assert roles["N_transfer_H"] != roles["B_transfer_H"]
    assert mol.GetAtomWithIdx(roles["cat_B"]).GetFormalCharge() == -1
    assert mol.GetAtomWithIdx(roles["cat_N"]).GetFormalCharge() == 1
    assert len(Chem.GetMolFrags(mol)) == 1
    for h, parent in [("N_transfer_H", "cat_N"), ("B_transfer_H", "cat_B")]:
        atom = mol.GetAtomWithIdx(roles[h])
        assert [a.GetIdx() for a in atom.GetNeighbors()] == [roles[parent]]


def test_bound_and_incoming_chemical_identities_are_independent():
    smiles = build_bisaryl_h2_connected_smiles(
        TMP, PYRROLE, [2], bound_substrate_smiles="c1ccoc1", bound_rpos=4,
    )[2]
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdMolDescriptors.CalcMolFormula(mol) == "C24H33BN2O"
    roles = _role_mapping_from_mol(mol, candidate())
    assert mol.GetAtomWithIdx(roles["bound_frame"]).GetAtomicNum() == 8
    assert mol.GetAtomWithIdx(roles["incoming_frame"]).GetAtomicNum() == 7
    with pytest.raises(ValueError, match="bound_rpos is required"):
        build_bisaryl_h2_connected_smiles(TMP, PYRROLE, [2], bound_substrate_smiles="c1ccoc1")


def test_bisaryl_roles_survive_atom_reordering():
    smiles = build_bisaryl_h2_connected_smiles(TMP, PYRROLE, [2])[2]
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    roles = _role_mapping_from_mol(mol, candidate())
    permutation = np.random.default_rng(19).permutation(mol.GetNumAtoms()).tolist()
    reordered = Chem.RenumberAtoms(mol, permutation)
    remapped = _role_mapping_from_mol(reordered, candidate())
    assert {role: permutation[index] for role, index in remapped.items()} == roles


def test_input_catalyst_stereochemistry_survives_bisaryl_assembly():
    catalyst = "C[C@H]1CCC[C@@H](C)N1c1ccccc1B"
    before = sorted(label for _, label in Chem.FindMolChiralCenters(Chem.MolFromSmiles(catalyst)))
    smiles = build_bisaryl_h2_connected_smiles(catalyst, PYRROLE, [2])[2]
    after = sorted(label for _, label in Chem.FindMolChiralCenters(Chem.MolFromSmiles(smiles)))
    assert before and before == after


def test_invalid_sites_and_non_bh2_parents_fail_before_embedding():
    with pytest.raises(ValueError, match="Invalid rpos"):
        build_bisaryl_h2_connected_smiles(TMP, PYRROLE, [1])
    with pytest.raises(ValueError, match="BH2"):
        build_bisaryl_h2_connected_smiles("CN(C)c1ccccc1B(C)", PYRROLE, [2])


def fake_embedding(mol, *, numConfs, **kwargs):
    for cid in range(numConfs):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for index in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(index, (float(index), float(cid), 0.0))
        mol.AddConformer(conf, assignId=True)
    return list(range(numConfs))


def test_explicit_candidate_is_call_scoped_and_records_seed(monkeypatch):
    before = ft.show_spec_profiles().copy()
    monkeypatch.setattr("frust.tsguess2.api.AllChem.EmbedMultipleConfs", fake_embedding)
    inputs = systems().assign(bound_substrate_smiles=PYRROLE, bound_rpos=2)
    frame = ft.tsguess2.create_ts_guess_dataframes(
        inputs, ts_types=["TS6"], n_confs=2, spec_profile="uma-gas",
        candidate_specs={"TS6": candidate()}, random_seed=190201,
    )["TS6"]
    assert len(frame) == 2
    assert len(frame.iloc[0].atoms) == 65
    assert frame.attrs["frust_tsguess2"]["candidate_spec"] is True
    assert frame.attrs["frust_tsguess2"]["scientific_status"] == "unvalidated_candidate"
    assert frame.attrs["frust_tsguess2"]["embedding"]["randomSeed"] == 190201
    assert "bound_rpos" not in frame and "bound_substrate_smiles" not in frame
    lineage = frame.attrs["frust_tsguess2"]["generated_smiles"][0]["builder_inputs"]
    assert lineage["bound_rpos"] == lineage["incoming_rpos"] == 2
    assert lineage["bound_substrate_smiles"] == lineage["incoming_substrate_smiles"] == PYRROLE
    pd.testing.assert_frame_equal(before, ft.show_spec_profiles())
    with pytest.raises(ValueError, match="No selectable TS6"):
        ft.tsguess2.resolve_profile_spec("TS6", "uma-gas", match="exact")
    with pytest.raises(ValueError, match="state and method/profile"):
        ft.tsguess2.create_ts_guess_dataframes(
            systems(), ts_types=["TS6"], candidate_specs={"TS6": candidate()},
        )


@pytest.mark.parametrize("method,screening", [("uma-gas", "uma-gas"), ("r2scan-3c", "gxtb-default")])
@pytest.mark.parametrize("state", ["TS5", "TS6", "TS7", "TS8"])
def test_candidate_target_and_preparation_are_lazy_and_method_independent(monkeypatch, method, screening, state):
    calls = []
    def embed(mol, **kwargs):
        calls.append(kwargs["numThreads"])
        return fake_embedding(mol, **kwargs)
    monkeypatch.setattr("frust.tsguess2.api.AllChem.EmbedMultipleConfs", embed)
    workflow = ft.workflows.screen_ts(
        dataframe=systems(), ts_types=[state], method=method, screening=screening,
        spec_profile="uma-gas", candidate_specs={state: candidate(state)}, random_seed=190201,
        n_confs=3, validation_top_n=3, prune_initial=False, calculation_level="full",
    )
    targets = workflow.targets()
    assert len(targets) == 1 and calls == []
    local = workflow._prepare_initial_df(targets[0], save_dir=None, options=ExecutionOptions(n_cores=2))
    cluster = workflow._prepare_initial_df(targets[0], save_dir=None, options=ExecutionOptions(n_cores=8))
    assert calls == [2, 8]
    assert local.smiles.tolist() == cluster.smiles.tolist()
    assert local.constraint_roles.tolist() == cluster.constraint_roles.tolist()
    assert len(local) == len(cluster) == 3
    stages = workflow._stage_defs()
    assert all(not s.constraint for s in stages if s.operation in {"hessian_seed", "ts_search", "frequencies"})
    assert next(s for s in stages if s.operation == "ts_search").use_last_hess
    boundary = [s for s in stages if s.kind == "filter" and s.lowest == 3]
    assert boundary


def test_experimental_templates_do_not_enter_legacy_backend():
    with pytest.raises(ValueError, match="require"):
        ft.screen.create_ts_guesses(systems(), backend="tsguess", candidate_specs={"TS6": candidate()})


def test_second_activation_and_h2_release_have_distinct_reactive_contracts():
    from frust.tsguess2.topologies import BISARYL_CH_TOPOLOGY, BISARYL_H2_TOPOLOGY, CORE_TOPOLOGIES

    activation = {frozenset(c.roles) for c in BISARYL_CH_TOPOLOGY.constraints}
    release = {frozenset(c.roles) for c in BISARYL_H2_TOPOLOGY.constraints}
    assert frozenset(("incoming_C", "N_transfer_H")) in activation
    assert frozenset(("N_transfer_H", "B_transfer_H")) not in activation
    assert frozenset(("N_transfer_H", "B_transfer_H")) in release
    assert frozenset(("incoming_C", "N_transfer_H")) not in release
    assert "TS5" not in CORE_TOPOLOGIES and "TS6" not in CORE_TOPOLOGIES


@pytest.mark.parametrize("catalyst,formula,count", [
    (TMP, "C31H47B2N3O2", 85),
    ("CN(C)c1ccccc1B", "C24H35B2N3O2", 66),
    ("B(c1ccccc1N1CCCCC1)", "C27H39B2N3O2", 73),
])
def test_diaryl_hbpin_bridge_preserves_inventory_and_roles(catalyst, formula, count):
    from frust.tsguess2.builders import build_bisaryl_exchange_connected_smiles

    text = build_bisaryl_exchange_connected_smiles(catalyst, PYRROLE, [2])[2]
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mol = Chem.AddHs(Chem.MolFromSmiles(text, parser))
    assert rdMolDescriptors.CalcMolFormula(mol) == formula
    assert mol.GetNumAtoms() == count and len(Chem.GetMolFrags(mol)) == 1
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
    roles = _role_mapping_from_mol(mol, candidate("TS7"))
    assert roles["cat_B"] != roles["pin_B"]
    assert roles["exchanging_C"] != roles["retained_C"]
    h = mol.GetAtomWithIdx(roles["transfer_H"])
    assert {a.GetIdx() for a in h.GetNeighbors()} == {roles["cat_B"], roles["pin_B"]}
    assert mol.GetAtomWithIdx(roles["cat_N"]).GetFormalCharge() == 0
    permutation = np.random.default_rng(78).permutation(count).tolist()
    remapped = _role_mapping_from_mol(Chem.RenumberAtoms(mol, permutation), candidate("TS8"))
    assert {role: permutation[index] for role, index in remapped.items()} == roles


def test_exchange_graph_keeps_independent_first_aryl_identity():
    from frust.tsguess2.builders import build_bisaryl_exchange_connected_smiles

    text = build_bisaryl_exchange_connected_smiles(
        TMP, PYRROLE, [2], bound_substrate_smiles="c1ccoc1", bound_rpos=4,
    )[2]
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mol = Chem.AddHs(Chem.MolFromSmiles(text, parser))
    roles = _role_mapping_from_mol(mol, candidate("INT6"))
    assert rdMolDescriptors.CalcMolFormula(mol) == "C30H44B2N2O3"
    assert mol.GetAtomWithIdx(roles["exchanging_frame"]).GetAtomicNum() == 8
    assert mol.GetAtomWithIdx(roles["retained_frame"]).GetAtomicNum() == 7


def test_screen_ts_rejects_minimum_candidate_before_calculation():
    with pytest.raises(ValueError, match="intermediate candidates require a minimum workflow"):
        ft.workflows.screen_ts(
            dataframe=systems(), ts_types=["INT6"], method="uma-gas",
            candidate_specs={"INT6": candidate("INT6")}, spec_profile="uma-gas",
        )


@pytest.mark.parametrize("catalyst,formula,atoms", [
    (TMP, "C25H34BN3", 63),
    ("CN(C)c1ccccc1B", "C18H22BN3", 44),
    ("B(c1ccccc1N1CCCCC1)", "C21H26BN3", 51),
])
def test_diaryl_minimum_retains_both_aryls_without_eliminated_hydrogens(catalyst, formula, atoms):
    from frust.tsguess2.builders import build_bisaryl_diaryl_connected_smiles

    text = build_bisaryl_diaryl_connected_smiles(catalyst, PYRROLE, [2])[2]
    mol = Chem.AddHs(Chem.MolFromSmiles(text))
    roles = _role_mapping_from_mol(mol, candidate("INT5"))
    assert rdMolDescriptors.CalcMolFormula(mol) == formula
    assert mol.GetNumAtoms() == atoms and len(Chem.GetMolFrags(mol)) == 1
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
    assert "N_transfer_H" not in roles and "B_transfer_H" not in roles
    for parent in ("cat_B", "cat_N"):
        atom = mol.GetAtomWithIdx(roles[parent])
        assert atom.GetFormalCharge() == 0 and atom.GetTotalNumHs(includeNeighbors=True) == 0
    for role in ("bound_C", "incoming_C", "scaffold_B_C"):
        assert mol.GetBondBetweenAtoms(roles["cat_B"], roles[role]) is not None
    permutation = np.random.default_rng(25).permutation(mol.GetNumAtoms()).tolist()
    reordered = Chem.RenumberAtoms(mol, permutation)
    remapped = _role_mapping_from_mol(reordered, candidate("INT5"))
    assert {role: permutation[index] for role, index in remapped.items()} == roles


def test_diaryl_keeps_independent_first_substrate_identity():
    from frust.tsguess2.builders import build_bisaryl_diaryl_connected_smiles

    text = build_bisaryl_diaryl_connected_smiles(
        TMP, PYRROLE, [2], bound_substrate_smiles="c1ccoc1", bound_rpos=4,
    )[2]
    mol = Chem.AddHs(Chem.MolFromSmiles(text))
    assert rdMolDescriptors.CalcMolFormula(mol) == "C24H31BN2O"
    roles = _role_mapping_from_mol(mol, candidate("INT5"))
    assert mol.GetAtomWithIdx(roles["bound_frame"]).GetAtomicNum() == 8
    assert mol.GetAtomWithIdx(roles["incoming_frame"]).GetAtomicNum() == 7


@pytest.mark.parametrize("state,atoms", [("INT4", 65), ("INT5", 63), ("INT6", 85)])
def test_explicit_bisaryl_minimum_construction_is_separate_from_ts_execution(monkeypatch, state, atoms):
    from frust.tsguess2.topologies import CORE_TOPOLOGIES

    monkeypatch.setattr("frust.tsguess2.api.AllChem.EmbedMultipleConfs", fake_embedding)
    before = ft.show_spec_profiles().copy()
    frame = ft.screen.create_ts_guesses(
        systems(), ts_types=[state], n_confs=2, spec_profile="uma-gas",
        candidate_specs={state: candidate(state)}, random_seed=190401,
    )[state]
    assert len(frame) == 2 and len(frame.iloc[0].atoms) == atoms
    assert set(frame.state_kind) == {"constrained_minimum"}
    assert state not in CORE_TOPOLOGIES
    pd.testing.assert_frame_equal(before, ft.show_spec_profiles())
    with pytest.raises(ValueError, match="intermediate candidates require a minimum workflow"):
        ft.workflows.screen_ts(
            dataframe=systems(), ts_types=[state], method="uma-gas",
            candidate_specs={state: candidate(state)}, spec_profile="uma-gas",
        )
