"""Balanced graphs and evidence-scoped applicability across methods."""
from collections import Counter
import json
import pandas as pd
import pytest
import frust as ft
from frust.workflows.bisarylation import _state_applicability,_profile_terms,_system_signature,_validated_review,_reference_roles
from frust.workflows.bisarylation import _plan_bisarylation,_balanced_profile_rows
from frust.workflows.bisarylation import _execution_children


SYSTEM=dict(catalyst_smiles='CN(C)c1ccccc1B',substrate_smiles='CN1C=CC=C1')


@pytest.mark.parametrize('variant,absent', [('two_saddle',set()),('ts7_only',{'TS8','INT6'}),('ts8_only',{'TS7','INT6'})])
def test_variants_do_not_create_dummy_saddles_or_intermediates(variant,absent):
    applicability=_state_applicability(variant)
    assert {s for s,a in applicability.items() if a=='not_applicable'}==absent
    assert not(set(_profile_terms(variant))&absent)


def test_unresolved_cases_remain_discovery_instead_of_inapplicable():
    values=_state_applicability('unresolved')
    assert {s for s,a in values.items() if a=='discovery'}=={'TS7','INT6','TS8'}
    assert 'not_applicable' not in values.values()


@pytest.mark.parametrize('variant',['two_saddle','ts7_only','ts8_only','unresolved'])
def test_every_profile_node_has_the_same_89_atom_tmp_ledger(variant):
    formula={
        'dimer':dict(C=30,H=48,B=2,N=2),'ligand':dict(C=5,H=7,N=1),
        'HBpin-mol':dict(C=6,H=13,B=1,O=2),'catalyst':dict(C=15,H=24,B=1,N=1),
        'int2':dict(C=20,H=29,B=1,N=2),'HH':dict(H=2),
        'TS5':dict(C=25,H=36,B=1,N=3),'INT4':dict(C=25,H=36,B=1,N=3),
        'TS6':dict(C=25,H=36,B=1,N=3),'INT5':dict(C=25,H=34,B=1,N=3),
        'TS7':dict(C=31,H=47,B=2,N=3,O=2),'INT6':dict(C=31,H=47,B=2,N=3,O=2),
        'TS8':dict(C=31,H=47,B=2,N=3,O=2),'bound_product':dict(C=11,H=18,B=1,N=1,O=2)}
    formula.update(incoming_ligand=formula['ligand'],bound_ligand=formula['ligand'],bound_int2=formula['int2'])
    for terms in _profile_terms(variant).values():
        total=Counter()
        for state,weight in terms.items():
            for atom,count in formula[state].items():total[atom]+=weight*count
        assert total==dict(C=31,H=51,B=2,N=3,O=2)
        assert sum(total.values())==89


@pytest.mark.parametrize('method',['uma-gas','r2scan-3c'])
def test_pathway_assignment_is_method_scoped_and_requires_actual_evidence(method):
    fingerprint=ft.workflows.methods.preset(method).fingerprint()
    review=dict(variant='ts7_only',system_signature=_system_signature(SYSTEM,2),method_fingerprint=fingerprint,evidence_sha256=['a'*64])
    assert _validated_review(review,system=SYSTEM,rpos=2,method_fingerprint=fingerprint)=='ts7_only'
    with pytest.raises(ValueError,match='method/environment'):
        _validated_review(review,system=SYSTEM,rpos=2,method_fingerprint='wrong')
    with pytest.raises(ValueError,match='evidence hashes'):
        _validated_review({**review,'evidence_sha256':[]},system=SYSTEM,rpos=2,method_fingerprint=fingerprint)
    changed={**SYSTEM,'bound_substrate_smiles':'c1ccoc1','bound_rpos':4}
    with pytest.raises(ValueError,match='chemical system'):
        _validated_review(review,system=changed,rpos=2,method_fingerprint=fingerprint)
    assert _validated_review(None,system=changed,rpos=2,method_fingerprint=fingerprint)=='unresolved'


def test_bound_and_incoming_reference_chemistry_are_not_collapsed_into_one_ligand():
    system={**SYSTEM,'bound_substrate_smiles':'c1ccoc1','bound_rpos':4}
    roles=_reference_roles(system,2)
    assert roles['bound_int2']['substrate_smiles']=='c1ccoc1'
    assert roles['bound_product']['rpos']==4
    assert roles['bound_product']['state_id']=='HBpin-ligand'
    assert roles['int2']['substrate_smiles']==SYSTEM['substrate_smiles']
    assert roles['int2']['rpos']==2
    assert roles['incoming_ligand']!=roles['bound_ligand']


@pytest.mark.parametrize('method',['uma-gas','r2scan-3c'])
def test_native_plan_contains_different_reviewed_routes_and_unresolved_discovery(method,monkeypatch):
    import frust.structures.builders
    monkeypatch.setattr(frust.structures.builders,'build',lambda *a,**k:pytest.fail('Planning must not embed'))
    fingerprint=ft.workflows.methods.preset(method).fingerprint()
    rows=[];reviews={}
    for i,variant in enumerate(['two_saddle','ts7_only','ts8_only','unresolved']):
        row={**SYSTEM,'system_name':f'system{i}','substrate_name':'pyrrole','catalyst_name':f'cat{i}',
             'catalyst_smiles':f'B'+'C'*(i+1),'rpos':2}
        rows.append(row)
        if variant!='unresolved':
            signature=_system_signature(row,2)
            reviews[signature]=dict(variant=variant,system_signature=signature,
                                   method_fingerprint=fingerprint,evidence_sha256=['a'*64])
    plan=_plan_bisarylation(pd.DataFrame(rows),method_fingerprint=fingerprint,pathway_reviews=reviews)
    assert [case['variant'] for case in plan['cases']]==['two_saddle','ts7_only','ts8_only','unresolved']
    for case in plan['cases']:
        absent={s for s,a in case['applicability'].items() if a=='not_applicable'}
        assert not absent.intersection(case['bindings'])
        assert all(t.target_id in case['bindings'].values() or t.system.system_name!=case['system_name'] or not t.state_id.startswith('TS') for t in plan['targets'])
    json.dumps({**plan,'targets':[t.as_dict() for t in plan['targets']]})


def test_mixed_aryl_plan_binds_distinct_reference_molecules_and_missing_values():
    system={**SYSTEM,'system_name':'mixed','substrate_name':'pyrrole','catalyst_name':'cat','rpos':2,
            'bound_substrate_smiles':'c1ccoc1','bound_rpos':4}
    plan=_plan_bisarylation(pd.DataFrame([system]),method_fingerprint='method')
    case=plan['cases'][0];targets={t.target_id:t for t in plan['targets']}
    assert case['bindings']['bound_int2']!=case['bindings']['int2']
    assert targets[case['bindings']['bound_product']].system.substrate_smiles=='c1ccoc1'
    assert targets[case['bindings']['bound_product']].rpos==4
    missing={**system,'bound_substrate_smiles':pd.NA,'bound_rpos':pd.NA}
    other=_plan_bisarylation(pd.DataFrame([missing]),method_fingerprint='method')['cases'][0]
    assert other['bindings']['bound_int2']==other['bindings']['int2']


@pytest.mark.parametrize('method,screening',[('uma-gas','uma-gas'),('r2scan-3c','gxtb-default')])
@pytest.mark.parametrize('variant',['two_saddle','ts7_only','ts8_only','unresolved'])
def test_planned_children_keep_native_execution_graphs_and_requested_candidates(method,screening,variant):
    import pickle
    from test_bisaryl_ts_candidates import candidate

    system={**SYSTEM,'system_name':'native','substrate_name':'pyrrole','catalyst_name':'cat','rpos':2}
    final=ft.workflows.methods.preset(method);fingerprint=final.fingerprint();signature=_system_signature(system,2)
    reviews={} if variant=='unresolved' else {signature:dict(variant=variant,system_signature=signature,
             method_fingerprint=fingerprint,evidence_sha256=['a'*64])}
    plan=_plan_bisarylation(pd.DataFrame([system]),method_fingerprint=fingerprint,pathway_reviews=reviews)
    required={t.state_id for t in plan['targets'] if t.state_kind!='minimum'}
    children=_execution_children(plan,method=final,screening=screening,
                                candidate_specs={s:candidate(s) for s in required},
                                spec_profile='uma-gas',n_confs=4,validation_top_n=3)
    assert {t.target_id for c in children.values() for t in c.targets()}=={t.target_id for t in plan['targets']}
    for branch,child in children.items():
        restored=pickle.loads(pickle.dumps(child))
        assert [t.as_dict() for t in restored.targets()]==[t.as_dict() for t in child.targets()]
        assert child.validation_top_n==3
        stages=child._stage_defs()
        names=[s.method_stage or s.id for s in stages if s.kind=='calc']
        roles=child.method.resolved_validation_roles()
        assert roles['frequencies'] in names
        expected=roles['ts_search'] if branch=='transition_states' else roles['minimum_optimization']
        assert expected in names
    with pytest.raises(ValueError,match='Execution method'):
        _execution_children(plan,method=ft.workflows.methods.preset('wb97xd3-631g'),screening=screening)


@pytest.mark.parametrize('variant',['two_saddle','ts7_only','ts8_only','unresolved'])
def test_profile_uses_exact_roles_quality_and_missing_evidence(variant):
    applicability=_state_applicability(variant)
    terms=_profile_terms(variant)
    roles=set().union(*(set(value) for value in terms.values()))
    case=dict(system_name='mixed',rpos=2,variant=variant,applicability=applicability,
              bindings={role:role for role in roles})
    # Use the same atom-balanced ledger, represented as independently derived
    # formula counts rather than rounded numerical energy comparisons.
    atoms={'dimer':dict(C=30,H=48,B=2,N=2),'catalyst':dict(C=15,H=24,B=1,N=1),
           'incoming_ligand':dict(C=5,H=7,N=1),'bound_ligand':dict(C=5,H=7,N=1),
           'HBpin-mol':dict(C=6,H=13,B=1,O=2),'HH':dict(H=2),
           'bound_int2':dict(C=20,H=29,B=1,N=2),'int2':dict(C=20,H=29,B=1,N=2),
           'bound_product':dict(C=11,H=18,B=1,N=1,O=2)}
    for state in ['TS5','INT4','TS6']:atoms[state]=dict(C=25,H=36,B=1,N=3)
    atoms['INT5']=dict(C=25,H=34,B=1,N=3)
    for state in ['TS7','INT6','TS8']:atoms[state]=dict(C=31,H=47,B=2,N=3,O=2)
    rows=[dict(structure_id=role,result_id=role+'-qualified',electronic_energy_hartree=-10.,
               quality_status='ready',energy_protocol_fingerprint='method',formula=atoms[role]) for role in roles]
    rows.append({**next(r for r in rows if r['structure_id']=='TS5'),
                 'quality_status':'invalid','electronic_energy_hartree':-20.,'result_id':'wrong-lower'})
    frame=pd.DataFrame(rows);result=_balanced_profile_rows(frame,case)
    required=result[~result.profile_state.isin(s for s,a in applicability.items() if a=='discovery')]
    assert required.quality_status.eq('ready').all()
    if variant=='unresolved':
        assert result[result.profile_state.isin(['TS7','INT6','TS8'])].quality_status.eq('review').all()
    assert result.set_index('profile_state').loc['TS5','result_ids']['TS5']=='TS5-qualified'
    assert not set(result.profile_state).intersection(s for s,a in applicability.items() if a=='not_applicable')
    missing=frame.loc[~frame.structure_id.eq('INT4')]
    assert _balanced_profile_rows(missing,case).set_index('profile_state').loc['INT4','quality_status']=='incomplete'
    frame.loc[frame.structure_id.eq('INT4'),'formula']=pd.Series([{'C':1}],index=frame.index[frame.structure_id.eq('INT4')])
    assert _balanced_profile_rows(frame,case).set_index('profile_state').loc['INT4','quality_status']=='invalid'


def test_full_ledger_selection_uses_final_free_energy_and_relative_corrections():
    case=dict(system_name='single',rpos=2,variant='two_saddle',
              applicability=_state_applicability('two_saddle'),
              bindings={'INT4':'adduct','HBpin-mol':'HBpin','HH':'HH'})
    frame=pd.DataFrame([
        dict(structure_id='adduct',result_id='lowest-electronic',quality_status='ready',
             electronic_energy_hartree=-10.,free_energy_hartree=-9.,formula='BH3',energy_protocol_fingerprint='method'),
        dict(structure_id='adduct',result_id='lowest-free',quality_status='ready',
             electronic_energy_hartree=-9.8,free_energy_hartree=-9.5,formula='BH3',energy_protocol_fingerprint='method'),
        *[dict(structure_id=identity,result_id=identity,quality_status='ready',electronic_energy_hartree=0.,
               free_energy_hartree=0.,formula=formula,energy_protocol_fingerprint='method')
          for identity,formula in [('HBpin','C6H13BO2'),('HH','H2')]]])
    result=_balanced_profile_rows(frame,case,reference='INT4',quantity='gibbs',corrections={'INT4':1.2})
    row=result.set_index('profile_state').loc['INT4']
    assert row.result_ids['INT4']=='lowest-free'
    assert row.relative_g_corrected_kcal_mol==0.
    frame['free_energy_hartree']=float('nan')
    result=_balanced_profile_rows(frame,case,reference='INT4',quantity='gibbs')
    assert result.set_index('profile_state').loc['INT4','quality_status']=='incomplete'
