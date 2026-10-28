"""Inactive reference geometry bindings and method-resolution boundaries."""
import numpy as np
import pytest

import frust as ft
from frust.tsguess2.profiles.uma_bisaryl_gas import GEOMETRIES
from frust.tsguess2.topologies import CORE_TOPOLOGIES
from frust.tsguess2.specs import BUILTIN_TS_SPECS_V2


@pytest.mark.parametrize('state',['TS5','TS6','TS7','TS8','INT6'])
def test_reviewed_candidate_constraints_match_exact_exported_coordinates(state):
    geometry=GEOMETRIES[state]
    assert geometry.status=='candidate'
    assert geometry.reference.mode_reviewed
    assert len(geometry.reference.negative_frequencies)==(0 if state=='INT6' else 1)
    assert len(geometry.reference.source_sha256)==64
    assert 'cid' in geometry.reference.notes
    assert 'Source row' in geometry.reference.notes
    for constraint in CORE_TOPOLOGIES[state].constraints:
        a,b=[np.asarray(geometry.role_coordinates[r]) for r in constraint.roles]
        assert geometry.constraint_values[constraint.name]==pytest.approx(np.linalg.norm(a-b),abs=1e-10)


@pytest.mark.parametrize('state',['TS5','TS6','TS7','TS8','INT4','INT5','INT6'])
def test_construction_support_does_not_silently_activate_a_profile(state):
    for profile in ['uma-gas','r2scan-3c','wb97xd3-631g']:
        with pytest.raises(ValueError,match=f'No selectable {state}'):
            ft.tsguess2.resolve_profile_spec(state,profile)
    summary=ft.show_spec_profiles()
    rows=summary.loc[summary.state.eq(state)]
    assert not rows.selectable_prefer_exact.any()
    assert set(rows.status).issubset({'candidate','missing'})


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
