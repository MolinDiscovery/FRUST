import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import submitit

from frust.cluster import ClusterConfig, Resources
from frust.cluster.submission import _array_dependency, _stage_outcome_path
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget
from test_cluster_arrays import ArrayExecutor


def StagedWorkflow(marker=None, fail=False, non_normal=False):
    class FixtureWorkflow(BaseWorkflow):
        def __init__(self, marker=None, fail=False, non_normal=False):
            super().__init__(dft=True)
            self.marker = marker
            self.fail = fail
            self.non_normal = non_normal

        def _build_targets(self):
            return [WorkflowTarget(tag, None) for tag in 'ABC']

        def _stage_defs(self):
            return [StageDef('prepare', 'prepare', kind='prepare'),
                    StageDef('select', 'select', kind='filter'),
                    StageDef('dft_opt', 'opt', kind='calc'),
                    StageDef('dft_freq', 'freq', kind='calc')]

        def _run_stage_group(self, target, stages, input_df, **kwargs):
            stage = stages[-1].id
            first = stages[0].id == 'prepare'
            if first:
                time.sleep(0.1 if target.tag == 'A' else 2.0)
                if target.tag == 'B' and self.fail and not self.marker.exists():
                    raise RuntimeError('injected upstream failure')
                frame = pd.DataFrame({'target':[target.tag], 'prepare-NT':[not (target.tag == 'B' and self.non_normal)]})
            else:
                assert input_df is not None
                frame = input_df.copy()
            frame[f'{stage}-finished'] = time.time_ns()
            return frame
    return FixtureWorkflow(marker, fail, non_normal)

@pytest.mark.parametrize('mode', ['dft_staged', 'fully_staged'])
def test_stage_arrays_keep_indices_resources_and_dependencies(tmp_path, mode):
    wf = StagedWorkflow()
    groups = wf._stage_groups(mode)
    names = [wf._group_name(group) for group in groups]
    workers = [ArrayExecutor([f'{100 + i}_0', f'{100 + i}_1', f'{100 + i}_2']) for i in range(len(groups))]
    collector = ArrayExecutor([])
    limits = {name: i + 1 for i, name in enumerate(names)}
    resources = {name: Resources(i + 1, i + 2, 5) for i, name in enumerate(names)}
    with patch('frust.workflows.core.create_executor', side_effect=[*workers, collector]), patch.object(wf, '_prepare_initial_df') as prepare:
        result = wf.submit(out_dir=tmp_path, cluster=ClusterConfig(), execution=mode,
                           array=True, array_parallelism=limits, stage_resources=resources, targets=[2, 0, 1])
    prepare.assert_not_called()
    assert result.tags == ['C', 'A', 'B']
    for i, (name, worker) in enumerate(zip(names, workers)):
        fn, arguments = worker.calls[0]
        assert fn.__name__ == '_run_stage_array_submitted_job'
        assert [args[1].tag for args in arguments] == result.tags
        assert [args[11] for args in arguments] == [0, 1, 2]
        assert arguments[0][2] == [stage.id for stage in groups[i]]
        assert arguments[0][6].n_cores == resources[name].cpus
        assert arguments[0][6].mem_gb == pytest.approx(resources[name].mem_gb * .8)
        extra = worker.parameters[0]['slurm_additional_parameters']
        assert extra['kill-on-invalid-dep'] == 'yes'
        assert extra.get('dependency') == (None if i == 0 else f'aftercorr:{99 + i}')
        assert worker.parameters[-1]['slurm_array_parallelism'] == limits[name]
        records = [r for r in result.records if r.group == name]
        assert [r.array_index for r in records] == [0, 1, 2]
    assert result.array_job_ids == [str(100 + i) for i in range(len(groups))]
    assert collector.parameters[0]['slurm_additional_parameters']['dependency'] == 'afterany:' + ':'.join(result.array_job_ids)
    assert 'kill-on-invalid-dep' not in collector.parameters[0]['slurm_additional_parameters']


def test_single_target_staged_fallback_and_scalar_limit(tmp_path):
    workers = [ArrayExecutor([str(100 + i)]) for i in range(3)]
    with patch('frust.workflows.core.create_executor', side_effect=workers):
        result = StagedWorkflow().submit(out_dir=tmp_path, cluster=ClusterConfig(), execution='dft_staged',
                                        array=True, array_parallelism=2, targets=[0], collect=False)
    assert result.array_job_ids == []
    assert result.job_ids == ['100', '101', '102']
    assert workers[1].parameters[0]['slurm_additional_parameters']['dependency'] == 'afterok:100'
    assert all(worker.parameters[-1]['slurm_array_parallelism'] == 2 for worker in workers)


def _wait(result, logs):
    job = submitit.LocalJob(folder=logs, job_id=result.collection_job_id)
    deadline = time.monotonic() + 30
    while not job.paths.result_pickle.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    assert job.paths.result_pickle.exists()
    job.result()


@pytest.mark.parametrize('mode', ['dft_staged', 'fully_staged'])
def test_local_staged_order_failure_independent_progress_and_retry(tmp_path, mode):
    marker = tmp_path/'retry-ready'
    wf = StagedWorkflow(marker, fail=True)
    root = tmp_path/'run'
    cluster = ClusterConfig(backend='local', log_dir=tmp_path/'logs')
    names = [wf._group_name(group) for group in wf._stage_groups(mode)]
    limits = {name: 2 if i == 0 else 1 for i, name in enumerate(names)}
    first = wf.submit(out_dir=root, cluster=cluster, execution=mode, array=True,
                      array_parallelism=limits, target_retention='all')
    _wait(first, cluster.log_dir)
    report = json.loads(Path(first.collection_report).read_text())
    assert report['retry_targets'] == ['B']
    b = next(item for item in report['target_results'] if item['target']=='B')
    assert b['status']=='blocked' and b['failed_group']=='init'
    assert [item['status'] for item in b['stage_results']] == ['failed'] + ['blocked'] * (len(names) - 1)
    assert all(r.job_id is None and r.status=='blocked' for r in first.records if r.target=='B' and r.group!='init')
    records = json.loads(Path(first.submission_path).read_text())
    attempt = records['attempt_id']
    times = {(r.group,r.target): json.loads(_stage_outcome_path(root, attempt, r.group, r.batch_index).read_text())
             for r in first.records}
    assert times[names[1],'A']['started_ns'] < times['init','C']['finished_ns']
    final_name = Path(first.records[-1].output_path).name
    for tag in 'AC':
        df = pd.read_parquet(root/tag/final_name)
        assert df['select-finished'].iloc[0] < df['dft_opt-finished'].iloc[0] < df['dft_freq-finished'].iloc[0]
    preserved = (root/'A'/final_name).read_bytes()
    marker.touch()
    retry = wf.submit(out_dir=root, cluster=cluster, execution=mode, array=True,
                      array_parallelism=1, targets=[1], retry=True, target_retention='all',
                      stage_resources={'init':Resources(2,4,5)})
    _wait(retry, cluster.log_dir)
    assert (root/'A'/final_name).read_bytes() == preserved
    assert len(pd.read_parquet(retry.collection_output)) == 1
    final = wf.collect(root, require_normal_termination=True)
    assert sorted(final.target.tolist()) == list('ABC')
    with pytest.raises(ValueError, match='already succeeded'):
        wf.submit(out_dir=root, cluster=cluster, execution=mode, array=True,
                  array_parallelism=1, targets=[0], retry=True)


def test_local_scientific_failure_blocks_descendants(tmp_path):
    wf = StagedWorkflow(non_normal=True)
    cluster = ClusterConfig(backend='local', log_dir=tmp_path/'logs')
    result = wf.submit(out_dir=tmp_path/'run', cluster=cluster, execution='dft_staged',
                       array=True, array_parallelism=2)
    _wait(result, cluster.log_dir)
    report = json.loads(Path(result.collection_report).read_text())
    b = next(item for item in report['target_results'] if item['target']=='B')
    assert b['status']=='blocked' and b['upstream_status']=='non_normal'
    assert (tmp_path/'run/B/init.parquet').exists()
    assert not (tmp_path/'run/B/init.dft_opt.parquet').exists()


def test_invalid_array_identity_is_not_guessed():
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match='indices'):
        _array_dependency([SimpleNamespace(job_id='123_1'), SimpleNamespace(job_id='123_0')])


def test_staged_worker_rejects_stale_checkpoint_before_calculation(tmp_path):
    from frust.workflows.core import ExecutionOptions, _run_stage_array_submitted_job
    from frust.cluster.submission import _atomic_write_submission_json
    wf = StagedWorkflow()
    target = wf.targets()[0]
    directory = tmp_path/target.tag
    directory.mkdir()
    df = pd.DataFrame({'target':['A'],'prepare-NT':[True]})
    df.attrs['frust_submission'] = {'attempt_id':'older','target':'A','group':'init'}
    df.to_parquet(directory/'init.parquet')
    _atomic_write_submission_json(_stage_outcome_path(tmp_path,'new','init',0),{'status':'success'})
    with patch.object(wf,'_run_stage_group') as calculate:
        with pytest.raises(ValueError,match='attribution'):
            _run_stage_array_submitted_job(wf,target,['dft_opt'],'init.parquet',
                                          'init.dft_opt.parquet',directory,ExecutionOptions(),
                                          None,False,'new','dft_opt',0,'init')
    calculate.assert_not_called()
    assert not (directory/'init.dft_opt.parquet').exists()
    assert json.loads(_stage_outcome_path(tmp_path,'new','dft_opt',0).read_text())['status']=='failed'


def test_interrupted_upstream_and_cancelled_descendant_report(tmp_path):
    from frust.cluster.submission import _atomic_write_submission_json, _target_outcomes
    workers = [ArrayExecutor([f'{100+i}_0',f'{100+i}_1',f'{100+i}_2']) for i in range(3)]
    with patch('frust.workflows.core.create_executor', side_effect=workers):
        result = StagedWorkflow().submit(out_dir=tmp_path,cluster=ClusterConfig(),execution='dft_staged',
                                        array=True,array_parallelism=1,collect=False)
    attempt = result.records[0].attempt_id
    _atomic_write_submission_json(_stage_outcome_path(tmp_path,attempt,'init',0),{'status':'running'})
    outcomes = _target_outcomes(tmp_path,workers_finished=True)
    assert outcomes['A']['status']=='blocked'
    assert outcomes['A']['upstream_status']=='interrupted'
    assert [stage['status'] for stage in outcomes['A']['stage_results']]==['interrupted','blocked','blocked']
    assert 'unknown' in outcomes['A']['stage_results'][0]['error']


def test_afterany_collector_preserves_terminal_proof_for_unstarted_jobs(tmp_path):
    from frust.cluster.submission import _completion_path, _record_afterany_completion, _attempt_terminal
    workers = [ArrayExecutor([f'{100+i}_0',f'{100+i}_1',f'{100+i}_2']) for i in range(3)]
    with patch('frust.workflows.core.create_executor',side_effect=workers):
        result = StagedWorkflow().submit(out_dir=tmp_path,cluster=ClusterConfig(),execution='dft_staged',
                                        array=True,array_parallelism=1,collect=False)
    attempt = json.loads(Path(result.submission_path).read_text())
    _record_afterany_completion(tmp_path,attempt['attempt_id'])
    receipt = json.loads(_completion_path(tmp_path,attempt['attempt_id'],'dft_opt',1).read_text())
    assert receipt['proof']=='afterany_collection' and receipt['job_id']=='101_1'
    with patch('frust.cluster.submission._job_terminal',return_value=False):
        assert _attempt_terminal(tmp_path,attempt)


def test_terminal_cancelled_element_missing_from_submitit_accounting(tmp_path):
    from types import SimpleNamespace
    from frust.cluster.submission import _job_terminal
    job = SimpleNamespace(paths=SimpleNamespace(result_pickle=tmp_path/'absent'),state='UNKNOWN')
    api = SimpleNamespace(SlurmJob=lambda **kwargs:job)
    with patch('frust.cluster.executor._load_submitit',return_value=api), \
         patch('frust.cluster.submission.subprocess.run',return_value=SimpleNamespace(stdout='JobState=CANCELLED Reason=DependencyNeverSatisfied')):
        assert _job_terminal({'backend':'slurm','log_dir':str(tmp_path)},'101_1')
