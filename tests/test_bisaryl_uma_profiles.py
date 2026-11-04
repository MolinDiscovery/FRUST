"""Immutable candidates, reviewed publication and method-resolution boundaries."""
import numpy as np
import pytest

import frust as ft
from frust.tsguess2.profiles.uma_bisaryl_gas import GEOMETRIES
from frust.tsguess2.topologies import CORE_TOPOLOGIES
from frust.tsguess2.specs import BUILTIN_TS_SPECS_V2


@pytest.mark.parametrize('state',['TS5','TS6','TS7','TS8','INT4','INT5','INT6'])
def test_reviewed_candidate_constraints_match_exact_exported_coordinates(state):
    geometry=GEOMETRIES[state]
    assert geometry.status=='candidate'
    assert geometry.reference.mode_reviewed
    assert len(geometry.reference.negative_frequencies)==(0 if state.startswith('INT') else 1)
    assert len(geometry.reference.source_sha256)==64
    assert 'cid' in geometry.reference.notes
    assert 'Source row' in geometry.reference.notes
    for constraint in CORE_TOPOLOGIES[state].constraints:
        a,b=[np.asarray(geometry.role_coordinates[r]) for r in constraint.roles]
        assert geometry.constraint_values[constraint.name]==pytest.approx(np.linalg.norm(a-b),abs=1e-10)


@pytest.mark.parametrize('state',['TS5','TS6','TS7','TS8','INT4','INT5','INT6'])
def test_reviewed_publication_preserves_candidate_and_method_boundaries(state, monkeypatch):
    from frust.tsguess2 import profiles
    from frust.tsguess2.profiles.uma_bisaryl_gas_reviewed import GEOMETRIES as published

    original=GEOMETRIES[state];active=published[state]
    assert original.status=='candidate' and original.revision==1
    assert active.status=='active' and active.revision==2
    assert active.role_coordinates==original.role_coordinates
    assert active.constraint_values==original.constraint_values
    assert active.reference.source_sha256==original.reference.source_sha256
    selection=ft.tsguess2.resolve_profile_spec(state,'uma-gas',match='exact')
    assert selection.match=='exact'
    assert selection.spec.spec_id.endswith('::r2')
    assert selection.spec.role_coordinates==active.role_coordinates
    for profile in ['r2scan-3c','wb97xd3-631g']:
        with pytest.raises(ValueError,match=f'No selectable {state}'):
            ft.tsguess2.resolve_profile_spec(state,profile)
    summary=ft.show_spec_profiles()
    rows=summary.loc[summary.state.eq(state)&summary.profile.eq(active.geometry_key.profile_id)]
    assert rows.status.eq('active').all() and rows.selectable_prefer_exact.all()
    monkeypatch.setitem(profiles._GEOMETRIES[active.geometry_key.profile_id],state,original)
    with pytest.raises(ValueError,match=f'No selectable {state}'):
        ft.tsguess2.resolve_profile_spec(state,'uma-gas')


def test_public_native_workflow_needs_no_override_and_does_not_assign_new_system(monkeypatch):
    from test_bisaryl_ts_candidates import systems
    from frust.workflows.factories import ScreenTSWorkflow, IntermediateWorkflow, MolsWorkflow

    for cls in (ScreenTSWorkflow,IntermediateWorkflow,MolsWorkflow):
        monkeypatch.setattr(cls,'_prepare_initial_df',
                            lambda *args,**kwargs:pytest.fail('Planning must not embed'))
    workflow=ft.workflows.bisarylation_screen(systems=systems(),method='uma-gas',screening='uma-gas')
    assert not workflow.candidate_specs
    targets=workflow.targets()
    assert {'TS5','TS6','TS7','TS8','INT4','INT5','INT6'}.issubset({t.target.state_id for t in targets})
    assert workflow._reaction_plan["cases"][0]["variant"] == "unresolved"


def test_default_compatibility_specs_keep_only_existing_dft_references():
    assert set(BUILTIN_TS_SPECS_V2)=={'TS1','TS2','TS3','TS4','INT3'}
    assert ft.tsguess2.resolve_profile_spec('TS3','uma-gas').spec.spec_id.endswith('::r2')


def test_ts8_candidate_documents_short_irc_and_separate_relaxation():
    geometry=GEOMETRIES['TS8']
    assert geometry.reference.negative_frequencies==(-105.01,)
    assert '117-cycle' in geometry.reference.notes
    assert 'not IRC' in geometry.reference.notes
    assert geometry.reference.coordinates_column=='uma_ts_opt-oc'
    assert 'conformer' in GEOMETRIES['INT6'].reference.notes


def test_single_imaginary_mode_challenge_is_methyl_rotation_not_exchange():
    import json
    from pathlib import Path

    fixture=json.loads((Path(__file__).parent/'data/bisaryl_ts8_wrong_mode.json').read_text())
    assert len(fixture['negative_frequencies_cm1'])==1
    xyz=np.asarray(fixture['coords']);mode=np.asarray(fixture['mode'])
    scaled=mode/np.linalg.norm(mode,axis=1).max()*0.2
    roles=fixture['roles']
    for a,b in [('cat_B','transfer_H'),('pin_B','transfer_H'),('cat_B','exchanging_C'),('pin_B','exchanging_C')]:
        distances=[np.linalg.norm(q[roles[a]]-q[roles[b]]) for q in [xyz-scaled,xyz+scaled]]
        assert abs(distances[1]-distances[0])<0.01
    dominant=np.argsort(np.linalg.norm(mode,axis=1))[-3:]
    assert all(fixture['atoms'][int(i)]=='H' for i in dominant)
    neighbours=[{b if a==i else a for a,b in fixture['connectivity_bonds'] if i in (a,b)} for i in dominant]
    assert len(set.intersection(*neighbours))==1
    carbon=next(iter(set.intersection(*neighbours)))
    assert fixture['atoms'][carbon]=='C'
    assert not fixture['native_optimization_converged']
    selected=GEOMETRIES['TS8'].reference
    assert 'Source row cid5' in selected.notes
    assert selected.negative_frequencies!=tuple(fixture['negative_frequencies_cm1'])
