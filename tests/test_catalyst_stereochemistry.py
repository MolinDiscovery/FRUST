"""Explicit carbon configurations, ordered dimers and portable reference coverage."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import frust as ft
from frust.screen.stereo import resolve_stereoisomers
from frust.structures.planner import plan_targets
from frust.structures.specs import DIMER_STATES
from frust.structures.stereo import _stereo_spec, check_geometry
from frust.transformers import _build_dimer_pair, transformer_mols
from frust.utils.dataframes import merge_dataframe_attrs

PIP = 'Bc1ccccc1N1CCCCC1F'


def systems(smiles=PIP):
    return pd.DataFrame([dict(system_name='pyrrole__cat', substrate_name='pyrrole',
                             substrate_smiles='CN1C=CC=C1', smiles='CN1C=CC=C1',
                             catalyst_name='cat', catalyst_smiles=smiles, rpos=2)])


def test_resolution_is_explicit_idempotent_and_keeps_selection():
    source = systems().assign(screening_score_kcal_mol=12.3)
    resolved = resolve_stereoisomers(source, catalyst_stereoisomers='all')
    assert set(resolved.catalyst_name) == {'cat__R', 'cat__S'}
    assert resolved.screening_score_kcal_mol.eq(12.3).all()
    assert resolved.catalyst_parent_name.eq('cat').all()
    pd.testing.assert_frame_equal(resolved, resolve_stereoisomers(resolved, catalyst_stereoisomers='all'))
    pd.testing.assert_frame_equal(source, resolve_stereoisomers(source))
    assert len(resolve_stereoisomers(source, catalyst_stereoisomers='representative')) == 1


@pytest.mark.parametrize('smiles', [PIP, 'Bc1ccccc1N(CC)C(C)F', 'Bc1ccccc1N1C(C)(C)CCCC1(C)CF'])
def test_carbon_configurations_and_intermediate_graphs(smiles):
    resolved = resolve_stereoisomers(systems(smiles), catalyst_stereoisomers='all')
    assert len(resolved) == 2
    for row in resolved.itertuples():
        original = Chem.MolFromSmiles(row.catalyst_smiles)
        expected = [tag for _, tag in Chem.FindMolChiralCenters(original)]
        products = transformer_mols('CN1C=CC=C1', row.catalyst_smiles,
                                    select=['catalyst', 'int1', 'int2'],
                                    show_IUPAC=False, rpos_list=[2])
        for mol in products.values():
            observed = [tag for _, tag in Chem.FindMolChiralCenters(mol)
                        if tag in {'R', 'S'}]
            assert observed == expected


def test_pair_order_deduplicated_only_for_exact_graph_equivalence():
    resolved = resolve_stereoisomers(systems(), catalyst_stereoisomers='all')
    targets = plan_targets(resolved, states=DIMER_STATES,
                           builder_options={'dimer_pairing': 'racemic'})
    assert pd.Series([t.state_id for t in targets]).value_counts().to_dict() == {
        'dimer': 3, 'dimer_bh_bridged': 4, 'dimer_eight_membered': 3}
    for target in targets:
        metadata = target.system.metadata
        for a, b in metadata['dimer_equivalent_pairings']:
            lookup = resolved.set_index('catalyst_name').catalyst_smiles
            graph = _build_dimer_pair(target.state_id, lookup[a], lookup[b])
            assert Chem.MolToSmiles(graph) == metadata['dimer_graph_smiles']
        assert pickle.loads(pickle.dumps(target)) == target
        json.dumps(target.payload)
    assert len({t.target_id for t in targets}) == 10


def test_achiral_racemate_has_three_dimers_and_representative_has_full_pool():
    achiral = systems('Bc1ccccc1N(C)C')
    assert len(plan_targets(achiral, states=DIMER_STATES,
                           builder_options={'dimer_pairing': 'racemic'})) == 3
    rep = resolve_stereoisomers(systems(), catalyst_stereoisomers='representative')
    assert len(plan_targets(rep, states=DIMER_STATES,
                           builder_options={'dimer_pairing': 'racemic'})) == 10


def test_geometry_checks_orientation_missing_and_planar_evidence():
    mol = Chem.AddHs(Chem.MolFromSmiles('C[C@H](F)Cl'))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    spec = _stereo_spec(mol)
    xyz = mol.GetConformer().GetPositions()
    assert check_geometry(spec, xyz)['status'] == 'preserved'
    mirrored = xyz.copy()
    mirrored[:, 0] *= -1
    assert check_geometry(spec, mirrored)['status'] == 'inverted'
    planar = xyz.copy()
    planar[:, 2] = 0
    assert check_geometry(spec, planar)['status'] == 'unresolved'
    assert check_geometry(spec, None)['status'] == 'unavailable'
    assert check_geometry(spec, xyz, ['H'] * len(xyz))['status'] == 'unavailable'


def test_geometry_specs_survive_merge():
    a, b = pd.DataFrame({'structure_id': ['a']}), pd.DataFrame({'structure_id': ['b']})
    a.attrs['frust_stereo'] = {'targets': {'a': {'centres': [1]}}}
    b.attrs['frust_stereo'] = {'targets': {'b': {'centres': [2]}}}
    assert set(merge_dataframe_attrs([a, b])['frust_stereo']['targets']) == {'a', 'b'}


@pytest.mark.parametrize('method', ['wb97xd3-631g', 'uma-gas-opt-alpb-chloroform'])
def test_native_plan_method_independent_and_manifest_isolated(method, tmp_path):
    workflow = ft.workflows.catalyst_screen(systems=systems(), method=method,
        catalyst_stereoisomers='all', dimer_pairing='racemic', n_confs=1,
        validation_top_n=3)
    plan = workflow.plan()
    assert plan.groupby('branch').size().to_dict() == {'references': 13, 'transition_states': 8}
    workflow._write_manifest(tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['dimer_pairing'] == 'racemic'
    assert len(manifest['reference_plan']) == 13
    assert len({t['catalyst_name'] for t in manifest['analysis_targets']}) == 2
    old = ft.workflows.catalyst_screen(systems=systems(), method=method, n_confs=1)
    assert len(old.children()['transition_states'].targets()) == 4
    assert workflow._reference_protocol() != old._reference_protocol()


def test_racemic_reference_requires_each_pair_and_shares_exact_result():
    from frust.screen.runs import _build_dimer_references, _resolve_terms

    resolved = resolve_stereoisomers(systems(), catalyst_stereoisomers='all')
    targets = plan_targets(resolved, states=DIMER_STATES,
                           builder_options={'dimer_pairing': 'racemic'})
    manifest = {'dimer_pairing': 'racemic', 'dimer_reference': 'lowest',
                'calculation_level': 'low_cost', 'dimer_candidates': list(DIMER_STATES),
                'analysis_targets': [{'catalyst_name': row.catalyst_name, 'catalyst_parent_name': 'cat'}
                                     for row in resolved.itertuples()],
                'reference_plan': [dict(target_id=t.target_id, state_id=t.state_id,
                                        catalyst_parent_name='cat') for t in targets]}
    states = pd.DataFrame([dict(state_id=t.state_id, structure_id=t.target_id,
                               catalyst_name=t.system.catalyst_name, result_id=f'result_{i}',
                               electronic_energy_hartree=-100-i, quality_status='ready', rpos=None)
                           for i, t in enumerate(targets)])
    refs = _build_dimer_references(states, manifest)
    assert len(refs) == 20
    selected = refs[refs.selected]
    assert len(selected) == 2 and selected.result_id.nunique() == 1
    choice = selected.iloc[0]
    selected_terms, issues = _resolve_terms(states, {choice.state_id: -.5},
        system_name='pyrrole__cat__R', substrate_name='pyrrole', catalyst_name='cat__R',
        rpos=2, result_ids={choice.state_id: choice.result_id})
    assert not issues and selected_terms[choice.state_id].result_id == choice.result_id
    incomplete = _build_dimer_references(states.iloc[:-1], manifest)
    assert not incomplete.selected.any()
    assert incomplete.selection_quality_status.eq('incomplete').all()
