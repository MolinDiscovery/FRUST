"""Local balanced barriers, review bindings and independent missing dependencies."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from frust.workflows.bisarylation import _state_applicability, _profile_terms
from frust.workflows.bisarylation_analysis import _barrier_rows, _review_roles, _review_status
from frust.screen.runs import HARTREE_TO_KCAL_MOL


def fixture(variant='two_saddle'):
    formulas={'dimer':dict(C=30,H=48,B=2,N=2),'catalyst':dict(C=15,H=24,B=1,N=1),
        'incoming_ligand':dict(C=5,H=7,N=1),'bound_ligand':dict(C=5,H=7,N=1),
        'HBpin-mol':dict(C=6,H=13,B=1,O=2),'HH':dict(H=2),
        'int2':dict(C=20,H=29,B=1,N=2),'bound_int2':dict(C=20,H=29,B=1,N=2),
        'bound_product':dict(C=11,H=18,B=1,N=1,O=2)}
    for state in ['TS5','INT4','TS6']:formulas[state]=dict(C=25,H=36,B=1,N=3)
    formulas['INT5']=dict(C=25,H=34,B=1,N=3)
    for state in ['TS7','INT6','TS8']:formulas[state]=dict(C=31,H=47,B=2,N=3,O=2)
    energies={'HH':-1.,'HBpin-mol':-5.,'int2':-20.,'bound_int2':-20.,'incoming_ligand':-2.,
        'bound_ligand':-2.,'bound_product':-8.,'catalyst':-10.,'dimer':-21.,'TS5':-21.9,
        'INT4':-22.1,'TS6':-22.,'INT5':-21.,'TS7':-25.9,'INT6':-26.2,'TS8':-26.0}
    roles=set().union(*(set(t) for t in _profile_terms(variant).values()))
    states=pd.DataFrame([dict(structure_id=r,result_id=r+'-result',state_id=r,cid=0,
        formula=formulas[r],electronic_energy_hartree=energies[r],free_energy_hartree=energies[r]+.01,
        energy_protocol_fingerprint='protocol',quality_status='ready',review_status='approved',
        quality_issues='',n_imag=int(r.startswith('TS')),method_family='uma') for r in sorted(roles)])
    case=dict(system_name='system',substrate_name='substrate',catalyst_name='catalyst',rpos=2,
        system_signature='chemical-system',method_fingerprint='method',variant=variant,
        applicability=_state_applicability(variant),bindings={r:r for r in roles},
        pathway_review_status='approved')
    return states,case


@pytest.mark.parametrize('variant',['two_saddle','ts7_only','ts8_only'])
def test_local_barriers_use_correct_precursor_in_each_variant(variant):
    states,case=fixture(variant)
    rows=_barrier_rows(states,case,quantity='electronic').set_index('ts_type')
    assert rows.loc['TS5','reference_state']=='M'
    assert rows.loc['TS6','reference_state']=='INT4'
    assert rows.loc['TS5','delta_e_kcal_mol']==pytest.approx(.1*HARTREE_TO_KCAL_MOL)
    assert rows.loc['TS6','delta_e_kcal_mol']==pytest.approx(.1*HARTREE_TO_KCAL_MOL)
    if variant=='two_saddle':
        assert rows.loc['TS8','reference_state']=='INT6'
        assert rows.loc['TS8','delta_e_kcal_mol']==pytest.approx(.2*HARTREE_TO_KCAL_MOL)
    elif variant=='ts8_only':
        assert rows.loc['TS8','reference_state']=='INT5'
        assert rows.loc['TS8','delta_e_kcal_mol']==pytest.approx(0.)
        assert 'TS7' not in rows.index
    else:
        assert rows.loc['TS7','product_state']=='Product'
        assert 'TS8' not in rows.index


def test_local_barrier_survives_missing_dimer_and_reverse_minimum():
    states,case=fixture()
    states=states.loc[~states.structure_id.isin(['dimer','bound_product'])]
    rows=_barrier_rows(states,case,quantity='electronic').set_index('ts_type')
    assert rows.quality_status.eq('ready').all()
    assert rows.dimer_relative_e_kcal_mol.isna().all()
    assert rows.loc['TS8','reverse_quality_status']=='incomplete'
    assert pd.isna(rows.loc['TS8','reverse_delta_e_kcal_mol'])


def test_cancelled_spectators_are_not_required_for_local_barriers():
    states,case=fixture()
    states=states.loc[~states.structure_id.isin(['HH','HBpin-mol'])]
    rows=_barrier_rows(states,case,quantity='electronic').set_index('ts_type')
    assert rows.loc[['TS5','TS6','TS8'],'quality_status'].eq('ready').all()
    assert rows.loc['TS7','quality_status']=='incomplete'  # HBpin actually enters TS7.
    assert rows.loc['TS6','reverse_quality_status']=='incomplete'  # H2 is formed.


def test_planning_variant_does_not_approve_new_results_or_other_candidates():
    states,case=fixture()
    case['pathway_review_status']='unreviewed'
    rows=_barrier_rows(states,case,quantity='electronic').set_index('ts_type')
    assert rows.loc[['TS7','TS8'],'quality_status'].eq('review').all()
    assert rows.loc[['TS7','TS8'],'delta_e_kcal_mol'].isna().all()
    assert rows.loc['TS5','quality_status']=='ready'


def test_distinct_neighbor_conformers_are_not_spliced_into_one_local_barrier():
    states,case=fixture()
    alternate=states.loc[states.state_id.eq('INT6')].iloc[0].copy()
    alternate['result_id']='INT6-other-conformer'
    alternate['electronic_energy_hartree']+=.03
    alternate['free_energy_hartree']+=.03
    states=pd.concat([states,pd.DataFrame([alternate])],ignore_index=True)
    case['reviewed_result_ids']={r:r+'-result' for r in _review_roles('two_saddle')}
    case['reviewed_minimum_result_ids']={'TS8':{'INT6':alternate.result_id}}
    rows=_barrier_rows(states,case,quantity='electronic').set_index('ts_type')
    assert rows.loc['TS8','delta_e_kcal_mol']==pytest.approx(.17*HARTREE_TO_KCAL_MOL)
    assert json.loads(rows.loc['TS8','reference_result_ids'])['INT6']==alternate.result_id
    assert json.loads(rows.loc['TS7','product_result_ids'])['INT6']=='INT6-result'
    assert rows.loc['TS8','uses_alternate_minimum']
    assert not rows.loc['TS7','uses_alternate_minimum']
    case['reviewed_minimum_result_ids']={'TS8':{'INT6':'INT6-result'}}
    matching=_barrier_rows(states,case,quantity='electronic').set_index('ts_type')
    assert not matching.loc['TS8','uses_alternate_minimum']


def test_pathway_review_invalidated_by_geometry_result_method_or_file(tmp_path):
    import hashlib

    states,case=fixture()
    path=tmp_path/'analysis/pathway_evidence/proof.json';path.parent.mkdir(parents=True);path.write_text('{}')
    record=dict(system_signature=case['system_signature'],method_fingerprint=case['method_fingerprint'],
        variant='two_saddle',result_ids={r:r+'-result' for r in _review_roles('two_saddle')},
        evidence=[dict(path=str(path.relative_to(tmp_path)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    assert _review_status(states,case,record,tmp_path)=='approved'
    changed=deepcopy(record);changed['method_fingerprint']='different'
    assert _review_status(states,case,changed,tmp_path)=='stale'
    modified=states.copy();modified.loc[modified.state_id.eq('INT6'),'result_id']='recalculated-geometry'
    assert _review_status(modified,case,record,tmp_path)=='stale'
    modified=states.copy();modified.loc[modified.state_id.eq('INT6'),'quality_status']='invalid'
    assert _review_status(modified,case,record,tmp_path)=='stale'
    path.write_text('{"changed":true}')
    assert _review_status(states,case,record,tmp_path)=='changed_evidence'
    path.unlink()
    assert _review_status(states,case,record,tmp_path)=='missing_evidence'


def test_pathway_review_rejects_wrong_target_or_invalid_result():
    from frust.workflows.bisarylation_analysis import _check_result_bindings

    states,case=fixture()
    ids={r:r+'-result' for r in _review_roles('two_saddle')}
    with pytest.raises(ValueError,match='exact current target'):
        _check_result_bindings(states,case,'two_saddle',{**ids,'TS7':ids['TS8']})
    states.loc[states.state_id.eq('INT6'),'quality_status']='invalid'
    with pytest.raises(ValueError,match='numerically valid'):
        _check_result_bindings(states,case,'two_saddle',ids)


def test_protocol_mismatch_between_nodes_cannot_produce_a_local_barrier():
    from frust.workflows.bisarylation import _balanced_profile_rows

    states,case=fixture()
    # A cached, internally consistent node can still disagree with its reference.
    from frust.workflows.bisarylation import _resolved_profile_nodes
    nodes=_resolved_profile_nodes(states,case,quantity='electronic')
    nodes['TS7']['energy_protocol_fingerprint']='other-surface'
    profile=_balanced_profile_rows(states,case,reference='INT5',_nodes=nodes).set_index('profile_state')
    assert profile.loc['TS7','quality_status']=='invalid'
    assert pd.isna(profile.loc['TS7','relative_e_kcal_mol'])
