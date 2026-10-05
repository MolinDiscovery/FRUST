from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import json

import pandas as pd
import pytest

from frust.cluster import ClusterConfig
from frust.utils.uma import current_uma_job_scope, uma_job_server_scope
from frust.workflows.core import ExecutionOptions, _run_target_batch_submitted_job
from test_uma_job_scope import _TinyUmaWorkflow, _method, _oet_root, _fake_orca
from test_cluster_arrays import ArrayExecutor, _workflow


def _targets(wf):
    from frust.workflows.core import WorkflowTarget
    return [WorkflowTarget(name, None) for name in ("A", "B", "C")]


def _status(root):
    return json.loads((root / '.frust/batches/attempt/0.json').read_text())


def test_shared_scope_and_real_stepper_multiple_targets(tmp_path):
    from frust.stepper import Stepper
    wf = _TinyUmaWorkflow(method=_method())
    events = []
    @contextmanager
    def server(**kwargs):
        events.append("start")
        try:
            yield SimpleNamespace(bind="127.0.0.1:12345", pid=11, hostname="test", preserve=lambda: None)
        finally:
            events.append("stop")
    def stepper(**kwargs):
        step = Stepper(**kwargs)
        step.orca_fn = _fake_orca
        return step
    with patch('frust.utils.uma.uma_server', server), \
         patch('frust.utils.uma._healthz_ready', return_value=True), \
         patch('frust.workflows.core.Stepper', side_effect=stepper):
        results = _run_target_batch_submitted_job(
            wf, _targets(wf), tmp_path, ExecutionOptions(n_cores=4, mem_gb=2,
            save_output_dir=False, uma_oet_tools=str(_oet_root(tmp_path))), None, 'attempt', 0)
    assert events == ['start', 'stop']
    assert len(results) == 3
    assert all((tmp_path / tag / 'final.parquet').exists() for tag in ('A','B','C'))
    assert _status(tmp_path)['uma_server']['pid'] == 11
    assert _status(tmp_path)['server_startup_s'] >= 0
    assert current_uma_job_scope() is None


@pytest.mark.parametrize('failure', [RuntimeError('bad target'), SystemExit(143)])
def test_failure_and_interruption_preserve_outputs(tmp_path, failure):
    wf = _workflow()
    seen = []
    def run(workflow, target, directory, options, submitted):
        seen.append(target.tag)
        if target.tag == 'B':
            raise failure
        directory.mkdir(exist_ok=True)
        df = pd.DataFrame({'calc-NT': [True]})
        df.to_parquet(directory / 'final.parquet')
        return df
    with patch('frust.workflows.core._run_target_job', run), \
         patch('frust.utils.uma.uma_server') as server:
        with pytest.raises(type(failure)):
            _run_target_batch_submitted_job(wf, wf.targets(), tmp_path, ExecutionOptions(), None, 'attempt', 0)
    server.assert_not_called()
    assert (tmp_path / 'A/final.parquet').exists()
    states = [r['status'] for r in _status(tmp_path)['targets']]
    if isinstance(failure, Exception):
        assert seen == ['A','B','C']
        assert states == ['success','failed','success']
    else:
        assert seen == ['A','B']
        assert states == ['success','interrupted','unattempted']


def test_broken_server_stops_remaining_targets(tmp_path):
    wf = _TinyUmaWorkflow(method=_method())
    events = []
    @contextmanager
    def server(**kwargs):
        try:
            yield SimpleNamespace(bind='127.0.0.1:12345', pid=11, hostname='test')
        finally:
            events.append('stop')
    def run(*args):
        current_uma_job_scope().acquire(log_dir=None, keep_logs='always', use_gpu=False,
                                      server_cores=1, memory_per_thread_mib=500)
        return pd.DataFrame({'calc-NT': [True]})
    with patch('frust.utils.uma.uma_server', server), \
         patch('frust.utils.uma._healthz_ready', return_value=False), \
         patch('frust.workflows.core._run_target_job', run):
        with pytest.raises(RuntimeError, match='stopped'):
            _run_target_batch_submitted_job(wf, _targets(wf), tmp_path, ExecutionOptions(), None, 'attempt', 0)
    assert [r['status'] for r in _status(tmp_path)['targets']] == ['success','unattempted','unattempted']
    assert events == ['stop']


def test_runtime_reuse_compatible_and_outer_owner(tmp_path):
    runtime = _oet_root(tmp_path)
    with uma_job_server_scope(oet_tools=str(runtime)) as owner:
        with uma_job_server_scope(oet_tools=str(runtime), reuse=True) as reused:
            assert owner is reused
        assert current_uma_job_scope() is owner
        with pytest.raises(ValueError, match='different OET runtime'):
            with uma_job_server_scope(oet_tools=str(tmp_path / 'other'), reuse=True):
                pass
    assert current_uma_job_scope() is None


def test_array_uneven_batches_and_resources(tmp_path):
    executor = ArrayExecutor(['123_0','123_1'])
    with patch('frust.workflows.core.create_executor', return_value=executor):
        result = _workflow().submit(out_dir=tmp_path, cluster=ClusterConfig(), array=True,
                                    array_parallelism=2, targets_per_task=2, collect=False)
    fn, args = executor.calls[0]
    assert fn.__name__ == '_run_target_batch_submitted_job'
    assert [[t.tag for t in call[1]] for call in args] == [['A','B'],['C']]
    assert [r.job_id for r in result.records] == ['123_0','123_0','123_1']
    assert len(result.job_ids) == 2 and len(result.tags) == 3
    assert executor.parameters[0]['mem_gb'] == 20


def test_non_normal_result_is_separate_from_exception(tmp_path):
    wf = _workflow()
    with patch('frust.workflows.core._run_target_job', return_value=pd.DataFrame({'calc-NT': [False]})):
        results = _run_target_batch_submitted_job(wf, wf.targets(), tmp_path, ExecutionOptions(), None, 'attempt', 0)
    assert all(result.status == 'non_normal' for result in results)
    assert all(record['error'] is None for record in _status(tmp_path)['targets'])


def test_local_batched_submission_collects_successes(tmp_path):
    import time
    import submitit
    from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget
    class Tiny(BaseWorkflow):
        def _build_targets(self):
            return [WorkflowTarget(tag, None) for tag in ('A', 'B', 'C')]
        def _stage_defs(self):
            return [StageDef('prepare', 'prepare', kind='prepare')]
        def _run_stage_group(self, target, *args, **kwargs):
            if target.tag == 'B':
                raise RuntimeError('injected target exception')
            return pd.DataFrame({'target': [target.tag], 'calc-NT': [True]})
    result = Tiny().submit(out_dir=tmp_path / 'run',
                           cluster=ClusterConfig(backend='local', log_dir=tmp_path / 'logs'),
                           array=True, array_parallelism=1, targets_per_task=2, target_retention='all')
    collector = submitit.LocalJob(folder=tmp_path / 'logs', job_id=result.collection_job_id)
    deadline = time.monotonic() + 30
    while not collector.paths.result_pickle.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert collector.paths.result_pickle.exists()
    collector.result()
    assert sorted(pd.read_parquet(result.collection_output)['target'].tolist()) == ['A', 'C']
    batch_files = sorted((tmp_path / 'run/.frust/batches').glob('*/*.json'))
    assert [[r['status'] for r in json.loads(p.read_text())['targets']] for p in batch_files] == [
        ['success', 'failed'], ['success'],
    ]
