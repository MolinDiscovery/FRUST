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


@pytest.mark.slow
def test_actual_builder_geometry_and_parquet_stereo(tmp_path):
    from frust.structures.builders import build
    resolved = resolve_stereoisomers(systems(), catalyst_stereoisomers='all')
    targets = plan_targets(resolved, states=['TS1', 'TS2', 'TS3', 'TS4', 'INT3', 'catalyst', 'int1', 'int2', *DIMER_STATES],
                           builder_options={'dimer_pairing': 'racemic'})
    for target in targets:
        frame = build(target, n_confs=1, n_cores=1, memory_gb=2, debug=False)
        path = tmp_path / f'{target.tag}.parquet'
        frame.to_parquet(path)
        restored = pd.read_parquet(path)
        spec = restored.attrs['frust_stereo']['targets'][target.target_id]
        assert spec['centres']
        for row in restored.itertuples():
            assert check_geometry(spec, row.coords_embedded, row.atoms)['status'] == 'preserved'
        from frust.results import attach_result_contract
        from frust.screen.runs import _state_rows
        restored['xtb_opt-oc'] = restored.coords_embedded
        restored['xtb_opt-EE'] = -100.0
        restored['xtb_opt-NT'] = True
        attach_result_contract(restored, target.state_kind, dft=False)
        analysis = _state_rows(restored, source='test', recipe=None, reviews={},
            calculation_level='low_cost', method=ft.workflows.methods.preset('wb97xd3-631g').to_dict())
        assert analysis.stereochemistry_status.eq('preserved').all()


@pytest.mark.parametrize('method', ['wb97xd3-631g', 'uma-gas-opt-alpb-chloroform'])
def test_native_mocked_racemic_execution_and_collection(method, tmp_path):
    from test_modular_workflow_validation import run_mocked

    wf = ft.workflows.catalyst_screen(systems=systems(), ts_types=['TS1'], method=method,
        catalyst_stereoisomers='all', dimer_pairing='racemic',
        ranking='uma-alpb-chloroform', top_n=4, ranking_top_n=3, validation_top_n=3,
        prune_initial=False)
    run = run_mocked(wf, tmp_path, artifact_policy='screening')
    refs = run.dimer_references()
    assert len(refs) == 20 and refs.selected.sum() == 2
    assert refs.loc[refs.selected, 'result_id'].nunique() == 1
    barriers = run.barriers()
    assert len(barriers) == 2
    assert barriers.dimer_result_id.nunique() == 1
    ts = pd.read_parquet(tmp_path / 'run/calculations/transition_states/merged.parquet')
    assert ts.groupby('catalyst_name').size().eq(3).all()
    assert len(run.candidate_barriers()) == 6
    if method == 'uma-gas-opt-alpb-chloroform':
        comparison = ft.workflows.wb97_comparison(run)
        assert comparison.dimer_pairing == 'racemic'
        assert len(comparison.children()['references'].targets()) == 13


def test_explicit_assigned_identity_and_unresolved_other_stereo_are_preserved():
    source = systems('Bc1ccccc1N1CCCC[C@H]1F')
    resolved = resolve_stereoisomers(source, catalyst_stereoisomers='all')
    assert resolved.catalyst_name.tolist() == ['cat']
    pool = resolve_stereoisomers(source, catalyst_stereoisomers='all', racemic=True)
    assert len(pool) == 2 and 'cat' in set(pool.catalyst_name)
    imine = resolve_stereoisomers(systems('Bc1ccccc1N1CCCCC1C=N'), catalyst_stereoisomers='all')
    assert len(imine) == 2
    assert imine.attrs['frust_stereoisomers']['unresolved_other_stereo']


def test_separately_supplied_enantiomers_share_parent_without_renaming():
    resolved = resolve_stereoisomers(systems(), catalyst_stereoisomers='all')
    explicit = resolved.drop(columns=['catalyst_parent_name', 'catalyst_parent_smiles'])
    explicit.catalyst_name = ['original_S', 'original_R']
    explicit.system_name = ['pyrrole__original_S', 'pyrrole__original_R']
    out = resolve_stereoisomers(explicit, catalyst_stereoisomers='all')
    assert out.catalyst_name.tolist() == explicit.catalyst_name.tolist()
    assert out.catalyst_parent_name.nunique() == 1
    assert len(plan_targets(out, states=DIMER_STATES,
                            builder_options={'dimer_pairing': 'racemic'})) == 10


def test_explicit_stereo_without_atom_mapping_is_unavailable():
    from frust.structures.stereo import _check_row_stereo
    assert _check_row_stereo({'smiles': 'C[C@H](F)Cl'}, None, None)['status'] == 'unavailable'


def test_cache_keys_distinguish_racemic_pairs_and_historical_homodimers():
    from frust.screen.references import reference_identity
    method = ft.workflows.methods.preset('wb97xd3-631g')
    resolved = resolve_stereoisomers(systems(), catalyst_stereoisomers='all')
    old = plan_targets(resolved, states=DIMER_STATES)
    pairs = plan_targets(resolved, states=DIMER_STATES, builder_options={'dimer_pairing': 'racemic'})
    old_keys = {reference_identity(t, method)[0] for t in old}
    new_keys = {reference_identity(t, method)[0] for t in pairs}
    assert len(old_keys) == 6 and len(new_keys) == 10
    assert old_keys.isdisjoint(new_keys)
