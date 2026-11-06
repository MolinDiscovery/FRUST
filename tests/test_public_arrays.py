import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.cluster import ClusterConfig, Resources
from frust.cluster.chains import submit_chain_jobs
from frust.workflows.screening import _cleanup_screening_targets
from test_screen_end_to_end import _components


class Scheduler:
    def __init__(self):
        self.next_id = 100
        self.executors = []
        self.calls = []

    def create(self, cluster):
        scheduler = self
        class Executor:
            def __init__(self):
                self.parameters = []
            def update_parameters(self, **kwargs):
                self.parameters.append(kwargs)
            def submit(self, function, *args, **kwargs):
                job = str(scheduler.next_id)
                scheduler.next_id += 1
                scheduler.calls.append((function, args, kwargs, job))
                return SimpleNamespace(job_id=job)
            def map_array(self, function, *columns):
                parent = str(scheduler.next_id)
                scheduler.next_id += 1
                arguments = list(zip(*columns))
                jobs = []
                for index, args in enumerate(arguments):
                    job = parent if len(arguments)==1 else f'{parent}_{index}'
                    scheduler.calls.append((function, args, {}, job))
                    jobs.append(SimpleNamespace(job_id=job))
                return jobs
        executor = Executor()
        self.executors.append(executor)
        return executor

    def run(self, start=0):
        errors = []
        for function, args, kwargs, job_id in self.calls[start:]:
            try:
                function(*args, **kwargs)
            except Exception as exc:
                errors.append((job_id,str(exc)))
        return errors


def test_facade_native_batches_outputs_and_collection(tmp_path):
    scheduler = Scheduler()
    prepared = {'tags':['A','B','C'], 'payloads':[{'A':'a'},{'B':'b'},{'C':'c'}]}
    called = []
    def pipeline(mol_struct, n_cores, mem_gb, out_dir, output_parquet, **kwargs):
        called.append((next(iter(mol_struct)),n_cores,mem_gb,out_dir))
        assert output_parquet is None
        return pd.DataFrame({'target':[next(iter(mol_struct))],'calc-NT':[True]})
    with patch('frust.cluster.facade.prepare_pipeline_inputs',return_value=prepared), \
         patch('frust.cluster.pipeline_workflow.load_pipeline',return_value=pipeline), \
         patch('frust.workflows.core.create_executor',side_effect=scheduler.create):
        result = ft.cluster.submit_jobs(csv_path='input.csv',pipeline='run_mols_per_rpos',
                                       out_dir=tmp_path,cluster=ClusterConfig(),resources=Resources(2,4,5),
                                       array=True,array_parallelism=1,targets_per_task=2)
        assert result.job_ids == ['100_0','100_1']
        assert result.array_job_ids == ['100'] and len(result.records)==3
        assert scheduler.run()==[]
    assert [item[0] for item in called]==list('ABC')
    assert all(item[1:3]==(2,3.2) for item in called)
    assert len(pd.read_parquet(result.collection_output))==3
    assert all((tmp_path/tag/'final.parquet').exists() for tag in 'ABC')
    assert result.mode=='run_mols_per_rpos'


def test_facade_retry_preserves_successful_target_outputs(tmp_path):
    scheduler = Scheduler()
    prepared = {'tags':['A','B','C'], 'payloads':[{'A':'a'},{'B':'b'},{'C':'c'}]}
    ready = False
    called = []
    def pipeline(mol_struct, **kwargs):
        tag = next(iter(mol_struct))
        called.append(tag)
        if tag=='B' and not ready:
            raise RuntimeError('pipeline failure')
        return pd.DataFrame({'substrate_name':[tag],'calc-NT':[True]})
    with patch('frust.cluster.facade.prepare_pipeline_inputs',return_value=prepared), \
         patch('frust.cluster.pipeline_workflow.load_pipeline',return_value=pipeline), \
         patch('frust.workflows.core.create_executor',side_effect=scheduler.create):
        kwargs = dict(csv_path='input.csv',pipeline='run_mols_per_rpos',out_dir=tmp_path,
                      cluster=ClusterConfig(),resources=Resources(2,4,5),array=True,array_parallelism=1)
        first = ft.cluster.submit_jobs(**kwargs,targets_per_task=2)
        assert scheduler.run()
        preserved = {tag:(tmp_path/tag/'final.parquet').read_bytes() for tag in ['A','C']}
        prior_calls = len(scheduler.calls)
        ready = True
        retried = ft.cluster.submit_jobs(**kwargs,retry=True,targets=[1])
        assert scheduler.run(prior_calls)==[]
    assert called==['A','B','C','B']
    assert retried.tags==['B'] and retried.collection_output!=first.collection_output
    assert all((tmp_path/tag/'final.parquet').read_bytes()==value for tag,value in preserved.items())


@pytest.mark.parametrize('entry',[submit_chain_jobs, ft.cluster.submit_screen_chain])
def test_legacy_chain_options_rejected_before_input_or_scheduler(tmp_path,entry):
    kwargs = {'preset':'screen_ts_per_rpos','module_path':None,'stage_order':None} if entry is submit_chain_jobs else {}
    with patch('frust.cluster.chains.prepare_chain_inputs') as prepare:
        with pytest.raises(NotImplementedError,match='ft.workflows.screen_ts'):
            entry(csv_path='absent.csv',out_dir=tmp_path/'bad',cluster=ClusterConfig(),
                  array=True,array_parallelism=2,**kwargs)
    prepare.assert_not_called()
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('artifact_policy',['diagnostic','compact'])
def test_screen_arrays_retry_recollects_successes(tmp_path,artifact_policy):
    scheduler = Scheduler()
    wf = ft.workflows.catalyst_screen(dataframe=_components(),level='low_cost',ts_types=['TS1','TS2'])
    root = tmp_path/'screen'
    cluster = ClusterConfig(log_dir=tmp_path/'logs')
    failed_tag = wf.children()['transition_states'].targets()[0].tag
    ready = False
    def calculation(child, target, stages, **kwargs):
        if target.tag == failed_tag and not ready:
            raise RuntimeError('injected failure')
        frame = pd.DataFrame({'substrate_name':[target.tag],'calc-EE':[-1.0],'calc-NT':[True]})
        frame.attrs['frust_results'] = {'columns': {'analysis': {'energy': 'calc-EE'}}}
        return frame
    def finalize(workflow, root, expected_report_paths=None, **kwargs):
        reports = [json.loads(Path(path).read_text()) for path in expected_report_paths]
        status = 'partial' if any(report['n_failures'] for report in reports) else 'success'
        report = {'overall_status':status}
        (root/'run_report.json').write_text(json.dumps(report))
        return report
    from frust.workflows.core import BaseWorkflow
    with patch('frust.workflows.core.create_executor',side_effect=scheduler.create), \
         patch('frust.workflows.screening.create_executor',side_effect=scheduler.create), \
         patch.object(BaseWorkflow,'_run_stage_group',calculation), \
         patch('frust.workflows.screening._finalize_run',side_effect=finalize):
        first = wf.submit(out_dir=root,cluster=cluster,array=True,array_parallelism=2,
                          targets_per_task=2,
                          artifact_policy=artifact_policy)
        assert set(first.child_submissions)=={'transition_states','references'}
        assert len(first.child_submissions['transition_states'].job_ids)==1
        assert first.child_submissions['references'].array_job_ids
        assert scheduler.run()
        ts = first.child_submissions['transition_states']
        preserved_tag = wf.children()['transition_states'].targets()[1].tag
        preserved = (root/'calculations/transition_states'/preserved_tag/'final.parquet')
        if not preserved.exists():
            preserved = Path(ts.save_dirs[1])/'final.parquet'
        before = preserved.read_bytes()
        manifest = (root/'manifest.json').read_bytes()
        prior_calls = len(scheduler.calls)
        ready = True
        retried = wf.submit(out_dir=root,cluster=cluster,array=True,array_parallelism=1,
                            retry=True,artifact_policy=artifact_policy)
        assert scheduler.run(prior_calls)==[]
        assert set(retried.child_submissions)=={'transition_states'}
        assert retried.child_submissions['transition_states'].tags==[failed_tag]
        frame = pd.read_parquet(retried.child_submissions['transition_states'].collection_output)
        assert sorted(frame.substrate_name.tolist())==sorted([failed_tag,preserved_tag])
        assert preserved.read_bytes()==before
        assert (root/'manifest.json').read_bytes()==manifest
        assert json.loads((root/'run_report.json').read_text())['overall_status']=='success'
        _cleanup_screening_targets(wf,root)
        assert Path(ts.submission_path).exists()
        assert Path(retried.child_submissions['transition_states'].submission_path).exists()
        assert list(root.glob('.frust/screen_submissions/*.json'))
        with pytest.raises(ValueError,match='finalized successfully'):
            wf.submit(out_dir=root,cluster=cluster,array=True,array_parallelism=1,
                      retry=True,targets={'transition_states':[0]},
                      artifact_policy=artifact_policy)


def test_empty_selection_and_invalid_limits_do_not_create_jobs(tmp_path):
    wf = ft.workflows.catalyst_screen(dataframe=_components(),level='low_cost')
    scheduler = Scheduler()
    with patch('frust.workflows.core.create_executor',side_effect=scheduler.create):
        empty = wf.submit(out_dir=tmp_path/'empty',cluster=ClusterConfig(),array=True,
                          array_parallelism=1,targets={})
        assert empty.finalization_job_id is None and empty.child_submissions=={}
        with pytest.raises(ValueError,match='positive integer'):
            wf.submit(out_dir=tmp_path/'bad',cluster=ClusterConfig(),array=True,array_parallelism=0)
    assert scheduler.calls==[] and not (tmp_path/'bad').exists()


@pytest.mark.parametrize('cluster',[
    ClusterConfig(extra_slurm_parameters={'dependency':'afterok:123'}),
    ClusterConfig(max_array_size=1),
    ClusterConfig(extra_slurm_parameters={'ntasks':'2'}),
])
def test_screen_rejects_invalid_scheduler_array_options_before_manifest(tmp_path,cluster):
    wf = ft.workflows.catalyst_screen(dataframe=_components(),level='low_cost',ts_types=['TS1','TS2'])
    with patch('frust.workflows.core.create_executor') as create:
        with pytest.raises(ValueError):
            wf.submit(out_dir=tmp_path/'bad',cluster=cluster,array=True,array_parallelism=1)
    create.assert_not_called()
    assert not (tmp_path/'bad').exists()


def test_staged_screen_uses_group_limits_resources_and_control_jobs(tmp_path):
    scheduler = Scheduler()
    wf = ft.workflows.catalyst_screen(dataframe=_components(),level='full',ts_types=['TS1','TS2'])
    names = {child._group_name(group) for child in wf.children().values() for group in child._stage_groups('dft_staged')}
    limits = {name:index+1 for index,name in enumerate(sorted(names))}
    with patch('frust.workflows.core.create_executor',side_effect=scheduler.create), \
         patch('frust.workflows.screening.create_executor',side_effect=scheduler.create):
        result = wf.submit(out_dir=tmp_path/'screen',cluster=ClusterConfig(),execution='dft_staged',
                           array=True,array_parallelism=limits,
                           stage_resources={'dft_freq':Resources(3,7,5)})
    assert set(result.child_submissions)=={'transition_states','references'}
    workers = [call for call in scheduler.calls if call[0].__name__=='_run_stage_array_submitted_job']
    assert workers and all(call[1][11]>=0 for call in workers)
    freq = [call for call in workers if call[1][10]=='dft_freq']
    assert freq and all(call[1][6].n_cores==3 and call[1][6].mem_gb==pytest.approx(5.6) for call in freq)
    for child in result.child_submissions.values():
        by_group = {}
        for record in child.records:
            by_group.setdefault(record.group,[]).append(record)
        for records in by_group.values():
            assert [record.array_index for record in records]==list(range(len(records)))
    finalizer_executor = scheduler.executors[-1]
    assert all('slurm_array_parallelism' not in values for values in finalizer_executor.parameters)
    assert finalizer_executor.parameters[-1]['slurm_additional_parameters']['dependency'].startswith('afterany:')
    assert {values['slurm_array_parallelism'] for executor in scheduler.executors for values in executor.parameters
            if 'slurm_array_parallelism' in values}==set(limits.values())


def test_reference_retry_rebuilds_compact_screen(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from test_modular_workflow_validation import FakeStepper, prepared, workflow
    from frust.workflows.factories import MolsWorkflow, ScreenTSWorkflow

    scheduler = Scheduler()
    wf = workflow(ts_types=["TS1"], dimer_reference="dimer")
    root = tmp_path / "screen"
    references = wf.children()["references"].targets()
    failed = next(target for target in references if target.state_id == "dimer")
    ready = False

    def prepare_reference(child, target, **kwargs):
        if target == failed and not ready:
            raise RuntimeError("injected dimer failure")
        return prepared(child, target, **kwargs)

    monkeypatch.setattr(MolsWorkflow, "_prepare_initial_df", prepare_reference)
    monkeypatch.setattr(ScreenTSWorkflow, "_prepare_initial_df", prepared)
    monkeypatch.setattr("frust.workflows.core.Stepper", FakeStepper)
    monkeypatch.setattr("frust.workflows.core._uma_scope_for_stages", lambda *a, **k: nullcontext())
    monkeypatch.setattr("frust.workflows.core.create_executor", scheduler.create)
    monkeypatch.setattr("frust.workflows.screening.create_executor", scheduler.create)

    settings = dict(out_dir=root, cluster=ClusterConfig(), array=True,
                    array_parallelism=2, artifact_policy="compact")
    first = wf.submit(**settings)
    assert first.child_submissions["references"].tags == [t.tag for t in references]
    assert scheduler.run()
    report = json.loads((root / "run_report.json").read_text())
    assert report["overall_status"] == "partial"
    assert report["terminal_results"]["references"]["missing_targets"] == [failed.target_id]
    assert all(
        branches["references"]["missing_targets"] == [failed.target_id]
        for branches in report["analysis_tiers"]["levels"].values()
    )
    run = ft.screen.open_run(root)
    assert run.barriers().quality_status.tolist() == ["incomplete"]
    manifest = (root / "manifest.json").read_bytes()

    prior_calls = len(scheduler.calls)
    ready = True
    retry = wf.submit(**settings, retry=True)
    assert list(retry.child_submissions) == ["references"]
    assert retry.child_submissions["references"].tags == [failed.tag]
    assert scheduler.run(prior_calls) == []
    assert (root / "manifest.json").read_bytes() == manifest
    report = json.loads((root / "run_report.json").read_text())
    assert report["overall_status"] == "success"
    assert report["terminal_results"]["references"]["n_collected"] == len(references)
    assert report["artifact_cleanup"]["n_removed_targets"] > 0
    reference_dir = root / "calculations" / "references"
    assert set(reference_dir.glob("*.parquet")) == {reference_dir / "merged.parquet"}
    assert not (reference_dir / "entries").exists()
    assert not (reference_dir / "reviews.csv").exists()
    before = {level: run.barriers(level=level) for level in run.available_analysis_levels()}
    relocated = root.rename(tmp_path / "relocated")
    moved = ft.screen.open_run(relocated).refresh_analysis()
    for level, barriers in before.items():
        pd.testing.assert_frame_equal(moved.barriers(level=level), barriers)


def test_retry_rejects_older_screen_manifest(tmp_path):
    wf = ft.workflows.catalyst_screen(dataframe=_components(), level="low_cost")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"run_type": "catalyst_screen", "schema_version": 3}))
    with patch("frust.workflows.core.create_executor") as create:
        with pytest.raises(ValueError, match="Unsupported screen manifest"):
            wf.submit(out_dir=tmp_path, cluster=ClusterConfig(), retry=True)
        create.assert_not_called()
    with pytest.raises(ValueError, match="Unsupported catalyst-screen manifest version"):
        ft.screen.build_analysis(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json"]


def test_active_screen_finalizer_blocks_retries(tmp_path):
    scheduler = Scheduler()
    wf = ft.workflows.catalyst_screen(dataframe=_components(),level='low_cost',ts_types=['TS1'])
    root = tmp_path/'screen'
    with patch('frust.workflows.core.create_executor',side_effect=scheduler.create), \
         patch('frust.workflows.screening.create_executor',side_effect=scheduler.create):
        wf.submit(out_dir=root,cluster=ClusterConfig(),array=True,array_parallelism=1)
        with patch('frust.workflows.screening._job_terminal',return_value=False):
            with pytest.raises(RuntimeError,match='finalizer is active'):
                wf.submit(out_dir=root,cluster=ClusterConfig(),array=True,array_parallelism=1,
                          retry=True,targets={'transition_states':[0]})


def test_default_single_job_screen_preserves_manifest_validated_restarts(tmp_path):
    scheduler = Scheduler()
    wf = ft.workflows.catalyst_screen(dataframe=_components(),level='low_cost',ts_types=['TS1'])
    root = tmp_path/'screen'
    with patch('frust.workflows.core.create_executor',side_effect=scheduler.create), \
         patch('frust.workflows.screening.create_executor',side_effect=scheduler.create):
        first = wf.submit(out_dir=root,cluster=ClusterConfig())
        manifest = (root/'manifest.json').read_bytes()
        second = wf.submit(out_dir=root,cluster=ClusterConfig())
    assert first.child_submissions.keys()==second.child_submissions.keys()
    assert (root/'manifest.json').read_bytes()==manifest
    assert not list(root.glob('.frust/screen_submissions/*.json'))


def test_local_screen_collectors_finish_before_partial_finalizer(tmp_path):
    import time
    import types
    import submitit
    from frust.results import attach_result_contract
    from test_modular_workflow_validation import FORMULAS
    wf = ft.workflows.catalyst_screen(dataframe=_components(),level='low_cost',ts_types=['TS1','TS2'])
    failed_tag = wf.children()['transition_states'].targets()[0].tag
    def fake_group(self, target, stages, **kwargs):
        if target.tag==failed_tag:
            raise RuntimeError('deliberate local screen failure')
        atoms = [atom for atom, count in FORMULAS[target.state_id].items()
                 for _ in range(count)]
        frame = pd.DataFrame([{
            "structure_id": target.target_id,
            "state_id": target.state_id,
            "state_kind": target.state_kind,
            "system_name": target.system.system_name,
            "substrate_name": target.system.substrate_name,
            "catalyst_name": target.system.catalyst_name,
            "rpos": target.rpos,
            "cid": 0,
            "atoms": atoms,
            "xtb_opt-EE": -10.0,
            "xtb_opt-oc": [[0.0, 0.0, 0.0]] * len(atoms),
            "xtb_opt-NT": True,
        }])
        attach_result_contract(frame, target.state_kind, dft=False, calculation_level="low_cost")
        return frame
    for child in wf.children().values():
        child._run_stage_group = types.MethodType(fake_group,child)
    cluster = ClusterConfig(backend='local',log_dir=tmp_path/'logs')
    root = tmp_path/'screen'
    result = wf.submit(out_dir=root,cluster=cluster,array=True,array_parallelism=2,targets_per_task=2)
    finalizer = submitit.LocalJob(folder=cluster.log_dir,job_id=result.finalization_job_id)
    deadline = time.monotonic()+30
    while not finalizer.paths.result_pickle.exists() and time.monotonic()<deadline:
        time.sleep(.1)
    assert finalizer.paths.result_pickle.exists()
    report = json.loads((root/'run_report.json').read_text())
    assert report['overall_status']=='partial'
    assert all(Path(child.collection_report).exists() for child in result.child_submissions.values())
    assert all(Path(child.collection_report).name!='collection_report.json' for child in result.child_submissions.values())
    assert list(root.glob('.frust/completions/*/screen_finalize_0.json'))
