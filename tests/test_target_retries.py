import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import submitit

from frust.cluster import ClusterConfig, Resources
from frust.cluster.submission import _mark_completed, _mutation_lock
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget
from test_cluster_arrays import ArrayExecutor


def _fixture(marker):
    class Tiny(BaseWorkflow):
        def __init__(self):
            super().__init__()
            self.marker = marker
        def _build_targets(self):
            return [WorkflowTarget(tag, None) for tag in 'ABCDE']
        def _stage_defs(self):
            return [StageDef('prepare','prepare',kind='prepare')]
        def _run_stage_group(self, target, *args, **kwargs):
            if not self.marker.exists():
                if target.tag == 'B':
                    raise RuntimeError('deliberate target failure')
                if target.tag == 'D':
                    raise SystemExit(143)
            return pd.DataFrame({'target': [target.tag], 'calc-NT': [True]})
    return Tiny()


def _wait(result, logs):
    job = submitit.LocalJob(folder=logs, job_id=result.collection_job_id)
    deadline = time.monotonic() + 40
    while not job.paths.result_pickle.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    assert job.paths.result_pickle.exists()
    job.result()


def test_local_failure_retry_and_recollection(tmp_path):
    marker = tmp_path / 'retry-ready'
    wf = _fixture(marker)
    root = tmp_path / 'run'
    cluster = ClusterConfig(backend='local',log_dir=tmp_path/'logs')
    first = wf.submit(out_dir=root, cluster=cluster, array=True, array_parallelism=2,
                      targets_per_task=3, target_retention='all')
    _wait(first, cluster.log_dir)
    report = json.loads(Path(first.collection_report).read_text())
    assert report['retry_targets'] == ['B','D','E']
    assert {r['target']:r['status'] for r in report['target_results']} == {
        'A':'success','B':'failed','C':'success','D':'interrupted','E':'unattempted'}
    preserved = (root/'A/final.parquet').read_bytes()
    marker.touch()
    selected = [target for target in wf.targets() if target.tag in report['retry_targets']]
    retried = wf.submit(out_dir=root, cluster=cluster, targets=selected, retry=True,
                        array=True, array_parallelism=2, targets_per_task=1,
                        stage_resources={'single_job':Resources(cpus=2,mem_gb=8,timeout_min=5)})
    _wait(retried, cluster.log_dir)
    assert retried.collection_output != first.collection_output
    assert len(pd.read_parquet(retried.collection_output)) == 3
    assert (root/'A/final.parquet').read_bytes() == preserved
    final = wf.collect(root, require_normal_termination=True)
    assert sorted(final.target.tolist()) == list('ABCDE')
    final_report = json.loads((root/'collection_report.json').read_text())
    assert final_report['retry_targets'] == []
    assert len(final_report['attempt_history']['A']) == 1
    assert len(final_report['attempt_history']['B']) == 2
    metadata = json.loads(Path(retried.submission_path).read_text())
    assert metadata['previous_attempts']['B'] == first.records[0].attempt_id
    assert Path(metadata['archives']['B']).exists()
    with pytest.raises(ValueError,match='already succeeded'):
        wf.submit(out_dir=root,cluster=cluster,retry=True,targets=[0],array=True,array_parallelism=1)


def test_active_attempt_and_mutation_lock_block_overlap(tmp_path):
    wf = _fixture(tmp_path/'ready')
    worker = ArrayExecutor(['123_0','123_1'])
    with patch('frust.workflows.core.create_executor',return_value=worker):
        first = wf.submit(out_dir=tmp_path/'run',cluster=ClusterConfig(), array=True,
                          array_parallelism=1,targets_per_task=3,collect=False)
    with patch('frust.cluster.submission._job_terminal',return_value=False):
        with pytest.raises(RuntimeError,match='overlapping writes'):
            wf.submit(out_dir=tmp_path/'run',cluster=ClusterConfig(),retry=True,targets=[1])
        with pytest.raises(RuntimeError,match='active or unverified'):
            wf.collect(tmp_path/'run')
    with _mutation_lock(tmp_path/'run'):
        with pytest.raises(RuntimeError,match='mutation.lock'):
            wf.submit(out_dir=tmp_path/'run',cluster=ClusterConfig(),targets=[1],retry=True)


def test_stale_output_excluded_and_archived_before_failed_retry(tmp_path):
    wf = _fixture(tmp_path/'ready')
    root = tmp_path/'run'
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['123'])):
        first = wf.submit(out_dir=root,cluster=ClusterConfig(),targets=[1],collect=False)
    attempt = first.records[0].attempt_id
    _mark_completed(root,attempt,'single_job',0)
    df = pd.DataFrame({'target':['B'],'calc-NT':[True]})
    df.attrs['frust_submission'] = {'attempt_id':'older-attempt','target':'B'}
    df.to_parquet(root/'B/final.parquet')
    with pytest.raises(FileNotFoundError):
        wf.collect(root)
    report = json.loads((root/'collection_report.json').read_text())
    assert next(r for r in report['target_results'] if r['target']=='B')['status']=='stale'
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['124'])):
        retry = wf.submit(out_dir=root,cluster=ClusterConfig(),targets=[1],retry=True,collect=False)
    assert not (root/'B/final.parquet').exists()
    metadata = json.loads(Path(retry.submission_path).read_text())
    assert (Path(metadata['archives']['B'])/'final.parquet').exists()
    _mark_completed(root,retry.records[0].attempt_id,'single_job',0)
    with pytest.raises(FileNotFoundError):
        wf.collect(root)


def test_non_normal_and_legacy_outputs_remain_readable(tmp_path):
    wf = _fixture(tmp_path/'ready')
    (tmp_path/'A').mkdir()
    pd.DataFrame({'target':['A'],'calc-NT':[False]}).to_parquet(tmp_path/'A/final.parquet')
    assert len(wf.collect(tmp_path)) == 1
    with pytest.raises(FileNotFoundError):
        wf.collect(tmp_path,require_normal_termination=True)
    assert json.loads((tmp_path/'collection_report.json').read_text())['retry_targets'] == list('ABCDE')


def test_completed_target_with_deleted_output_is_missing(tmp_path):
    wf = _fixture(tmp_path/'ready')
    root = tmp_path/'run'
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['123'])):
        first = wf.submit(out_dir=root,cluster=ClusterConfig(),targets=[0],collect=False)
    attempt = first.records[0].attempt_id
    _mark_completed(root,attempt,'single_job',0)
    batch = root/'.frust/batches'/attempt/'0.json'
    batch.parent.mkdir(parents=True)
    batch.write_text(json.dumps({'targets':[{'target':'A','status':'success'}]}))
    with pytest.raises(FileNotFoundError):
        wf.collect(root)
    report = json.loads((root/'collection_report.json').read_text())
    assert report['target_results'][0]['status'] == 'missing'
    assert 'A' in report['retry_targets']


def test_late_exception_with_normal_output_can_be_retried(tmp_path):
    wf = _fixture(tmp_path/'ready')
    root = tmp_path/'run'
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['123'])):
        first = wf.submit(out_dir=root,cluster=ClusterConfig(),targets=[1],collect=False)
    attempt = first.records[0].attempt_id
    _mark_completed(root,attempt,'single_job',0)
    df = pd.DataFrame({'calc-NT':[True]})
    df.attrs['frust_submission'] = {'attempt_id':attempt,'target':'B'}
    df.to_parquet(root/'B/final.parquet')
    batch = root/'.frust/batches'/attempt/'0.json'
    batch.parent.mkdir(parents=True)
    batch.write_text(json.dumps({'targets':[{'target':'B','status':'failed','error':'late write failure'}]}))
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['124'])):
        retry = wf.submit(out_dir=root,cluster=ClusterConfig(),targets=[1],retry=True,collect=False)
    assert retry.records[0].attempt_id != attempt
    assert not (root/'B/final.parquet').exists()


def test_retry_rejects_chemistry_change_before_archiving(tmp_path):
    wf = _fixture(tmp_path/'ready')
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['123'])):
        first = wf.submit(out_dir=tmp_path/'run',cluster=ClusterConfig(),targets=[1],collect=False)
    _mark_completed(tmp_path/'run',first.records[0].attempt_id,'single_job',0)
    wf.top_n += 1
    with patch('frust.workflows.core.create_executor') as create:
        with pytest.raises(ValueError,match='fingerprint'):
            wf.submit(out_dir=tmp_path/'run',cluster=ClusterConfig(),targets=[1],retry=True,collect=False)
    create.assert_not_called()
    assert not (tmp_path/'run/.frust/history').exists()


@pytest.mark.parametrize('unreadable',[False,True])
def test_non_normal_or_unreadable_output_is_archived(tmp_path,unreadable):
    wf = _fixture(tmp_path/'ready')
    root=tmp_path/'run'
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['123'])):
        first=wf.submit(out_dir=root,cluster=ClusterConfig(),targets=[1],collect=False)
    _mark_completed(root,first.records[0].attempt_id,'single_job',0)
    if unreadable:
        (root/'B/final.parquet').write_text('broken parquet')
    else:
        df=pd.DataFrame({'calc-NT':[False]})
        df.attrs['frust_submission']={'attempt_id':first.records[0].attempt_id,'target':'B'}
        df.to_parquet(root/'B/final.parquet')
    with patch('frust.workflows.core.create_executor',return_value=ArrayExecutor(['124'])):
        retry=wf.submit(out_dir=root,cluster=ClusterConfig(),targets=[1],retry=True,collect=False)
    archive=json.loads(Path(retry.submission_path).read_text())['archives']['B']
    assert (Path(archive)/'final.parquet').exists()
    assert not (root/'B/final.parquet').exists()
