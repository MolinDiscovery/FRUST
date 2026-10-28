"""Private composition uses shared local execution, collection and review."""
import json
from unittest.mock import patch

import pandas as pd
import pytest

from frust.workflows.bisarylation import _system_signature
from frust.workflows.bisarylation_workflow import _BisarylationWorkflow
from frust.workflows.factories import IntermediateWorkflow


def workflow(method='uma-gas',variant='unresolved'):
    from test_bisaryl_ts_candidates import candidate
    system=dict(system_name='pyrrole__TMP',substrate_name='pyrrole',catalyst_name='TMP',
                substrate_smiles='CN1C=CC=C1',catalyst_smiles='CC1(C)CCCC(C)(C)N1c1ccccc1B',rpos=2)
    wf=_BisarylationWorkflow(systems=pd.DataFrame([system]),method=method,
                            screening='uma-gas' if method=='uma-gas' else 'gxtb-default',
                            candidate_specs={s:candidate(s) for s in ['TS5','TS6','TS7','TS8','INT4','INT5','INT6']},
                            spec_profile='uma-gas',n_confs=4,validation_top_n=2,prune_initial=False)
    if variant!='unresolved':
        signature=_system_signature(system,2)
        wf.pathway_reviews={signature:dict(variant=variant,system_signature=signature,
                            method_fingerprint=wf.method.fingerprint(),evidence_sha256=['a'*64])}
    return wf


@pytest.mark.parametrize('variant',['two_saddle','ts7_only','ts8_only','unresolved'])
def test_composed_manifest_binds_pathway_profiles_and_retry_identity(tmp_path,variant):
    wf=workflow(variant=variant);wf._write_manifest(tmp_path)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['scope']=='bisarylation'
    assert manifest['bisarylation']['cases'][0]['variant']==variant
    assert 'intermediates' in manifest['calculation_results']
    wf._write_manifest(tmp_path)
    wf.random_seed=32
    with pytest.raises(FileExistsError,match='different bisarylation'):
        wf._write_manifest(tmp_path)


def test_missing_calculation_branches_remain_portable_and_incomplete(tmp_path):
    import frust as ft

    workflow()._write_manifest(tmp_path)
    run=ft.screen.open_run(tmp_path).refresh_analysis()
    profile=run.profile(quantity='electronic',include_invalid=True)
    assert profile.quality_status.eq('incomplete').all()
    assert profile.relative_e_kcal_mol.isna().all()


@pytest.mark.parametrize('method',['uma-gas','r2scan-3c'])
@pytest.mark.parametrize('mixed',[None,'ts7_only','ts8_only'])
def test_mocked_native_children_collect_all_candidates_and_balanced_profiles(tmp_path,method,monkeypatch,mixed):
    import test_modular_workflow_validation as helper
    from test_modular_workflow_validation import run_mocked,prepared

    formulas={'ligand':dict(C=5,H=7,N=1),'catalyst':dict(C=15,H=24,B=1,N=1),
              'dimer':dict(C=30,H=48,B=2,N=2),'int2':dict(C=20,H=29,B=1,N=2),
              'HBpin-ligand':dict(C=11,H=18,B=1,N=1,O=2),
              'HBpin-mol':dict(C=6,H=13,B=1,O=2),'HH':dict(H=2)}
    for state in ['dimer_bh_bridged','dimer_eight_membered']:formulas[state]=formulas['dimer']
    for s in ['TS5','INT4','TS6']:formulas[s]=dict(C=25,H=36,B=1,N=3)
    formulas['INT5']=dict(C=25,H=34,B=1,N=3)
    for s in ['TS7','INT6','TS8']:formulas[s]=dict(C=31,H=47,B=2,N=3,O=2)
    for state,formula in formulas.items():
        monkeypatch.setitem(helper.FORMULAS,state,formula)
        monkeypatch.setitem(helper.ENERGIES,state,-10.)
    monkeypatch.setitem(helper.ENERGIES,'dimer_bh_bridged',-9.8)
    monkeypatch.setitem(helper.ENERGIES,'dimer_eight_membered',-9.7)
    wf=workflow(method,variant='two_saddle')
    if mixed:
        first=wf.systems().iloc[0].to_dict()
        second={**first,'system_name':'second','catalyst_name':'Pip',
                'catalyst_smiles':'Bc1ccccc1N1CCCCC1'}
        wf=_BisarylationWorkflow(systems=pd.DataFrame([first,second]),method=wf.method,
                                 screening=wf.screening,candidate_specs=wf.candidate_specs,
                                 spec_profile='uma-gas',n_confs=4,validation_top_n=2,prune_initial=False)
        wf.pathway_reviews={}
        for system,variant in [(first,'two_saddle'),(second,mixed)]:
            signature=_system_signature(system,2)
            wf.pathway_reviews[signature]=dict(variant=variant,system_signature=signature,
                method_fingerprint=wf.method.fingerprint(),evidence_sha256=['a'*64])
    with patch.object(IntermediateWorkflow,'_prepare_initial_df',prepared):
        run=run_mocked(wf,tmp_path)
    states=run.states();core=states[states.state_id.isin(['TS5','TS6','TS7','TS8','INT4','INT5','INT6'])]
    assert core.groupby('structure_id').size().eq(2).all()
    for result_id in core.result_id:run.set_identity_review(result_id,'approved',note='Mocked identity review')
    profile=run.profile(quantity='electronic',include_invalid=True)
    assert set(profile.profile_state)=={'Dimer','Cat','M','TS5','INT4','TS6','INT5','TS7','INT6','TS8','Product'}
    assert profile.quality_status.eq('ready').all()
    assert profile.relative_e_kcal_mol.notna().all()
    if mixed:
        absent={'INT6','TS8' if mixed=='ts7_only' else 'TS7'}
        assert not absent.intersection(run.profile(system_name='second',quantity='electronic').profile_state)
        assert not absent.intersection(states.loc[states.system_name.eq('second'),'state_id'])
    assert run.profile(quantity='gibbs').relative_g_kcal_mol.notna().all()
    assert run.dimer_references().groupby('catalyst_name').size().eq(3).all()
    assert not run.profile(quantity='electronic',dimer_reference='lowest').empty
    with pytest.raises(ValueError,match='recorded reference only'):
        run.profile(dimer_reference='dimer')


@pytest.mark.parametrize('variant',['two_saddle','ts7_only','ts8_only','unresolved'])
@pytest.mark.parametrize('method',['uma-gas','r2scan-3c'])
def test_composed_cluster_targets_match_local_plan_and_finalizer(tmp_path,variant,method):
    import frust as ft
    from test_screen_end_to_end import _FakeExecutor

    wf=workflow(method,variant);expected={t.target.target_id for t in wf.targets()}
    fake=_FakeExecutor();cluster=ft.ClusterConfig(backend='slurm',partition='test',log_dir=tmp_path/'logs')
    with (patch('frust.workflows.core.create_executor',return_value=fake),
          patch('frust.workflows.screening.create_executor',return_value=fake)):
        submission=wf.submit(out_dir=tmp_path/'submitted',cluster=cluster)
    assert set(submission.child_submissions)=={'transition_states','intermediates','references'}
    manifest=json.loads((tmp_path/'submitted/manifest.json').read_text())
    bindings=manifest['bisarylation']['cases'][0]['bindings']
    assert set(bindings.values())==expected
    assert fake.submissions[-1][0].__name__=='_finalize_submitted_run'
    assert fake.parameters[-1]['slurm_additional_parameters']['dependency'].startswith('afterany:')
