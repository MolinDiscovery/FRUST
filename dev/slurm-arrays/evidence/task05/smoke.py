"""Two tiny stage arrays, failed dependency cancellation, and a full-chain retry.

Use PYTHONPATH=. from the checkout. Run initial, wait for its collector, then
retry, wait for that collector, then verify. No chemistry/server is required.
"""
import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import submitit

from frust.cluster import ClusterConfig, Resources
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget


class StagedSmoke(BaseWorkflow):
    """Scheduler fixture with unequal target times and an external retry switch."""
    def __init__(self, root):
        super().__init__(dft=True)
        self.root = root

    def _build_targets(self):
        return [WorkflowTarget(tag, None) for tag in 'ABC']

    def _stage_defs(self):
        return [StageDef('prepare', 'prepare', kind='prepare'),
                StageDef('dft_freq', 'frequency', kind='filter')]

    def _run_stage_group(self, target, stages, input_df, **kwargs):
        stage = stages[-1].id
        if stage == 'prepare':
            time.sleep({'A':1, 'B':4, 'C':40}[target.tag] if not (self.root/'retry-ready').exists() else 1)
            if target.tag == 'B' and not (self.root/'retry-ready').exists():
                raise RuntimeError('deliberate upstream failure')
            df = pd.DataFrame({'target':[target.tag], 'prepare-NT':[True]})
        else:
            assert input_df is not None
            df = input_df.copy()
            time.sleep(1)
            df['dft_freq-NT'] = True
        df[f'{stage}-finished'] = time.time_ns()
        return df


def snapshot(root, result, name):
    payload = asdict(result)
    payload['revision'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    payload['scheduler'] = {}
    for job_id in result.array_job_ids or result.job_ids:
        payload['scheduler'][str(job_id)] = subprocess.check_output(['scontrol','show','job',str(job_id)],text=True)
    (root/f'{name}.json').write_text(json.dumps(payload,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['initial','retry','verify'],required=True)
    parser.add_argument('--partition',default='kemi1')
    args = parser.parse_args()
    root = args.output
    wf = StagedSmoke(root)
    cluster = ClusterConfig(partition=args.partition,log_dir=root/'logs')
    if args.phase == 'initial':
        root.mkdir(parents=True,exist_ok=False)
        result = wf.submit(out_dir=root/'run',cluster=cluster,execution='dft_staged',
                           array=True,array_parallelism={'init':3,'dft_freq':1},target_retention='all',
                           stage_resources={'init':Resources(1,2,5),'dft_freq':Resources(2,4,5)},
                           collect_resources=Resources(1,2,5))
        snapshot(root,result,'initial')
        print(f'arrays={result.array_job_ids}, collector={result.collection_job_id}')
    elif args.phase == 'retry':
        initial = json.loads((root/'initial.json').read_text())
        report = json.loads(Path(initial['collection_report']).read_text())
        assert report['retry_targets'] == ['B'], report
        (root/'initial-report.json').write_text(json.dumps(report,indent=2)+'\n')
        blocked_job = next(record['job_id'] for record in initial['records'] if record['target']=='B' and record['group']=='dft_freq')
        (root/'blocked-job.txt').write_text(subprocess.check_output(['scontrol','show','job',str(blocked_job)],text=True))
        (root/'retry-ready').touch()
        result = wf.submit(out_dir=root/'run',cluster=cluster,execution='dft_staged',
                           array=True,array_parallelism=1,targets=[1],retry=True,target_retention='all',
                           stage_resources={'init':Resources(1,2,5),'dft_freq':Resources(2,4,5)},
                           collect_resources=Resources(1,2,5))
        snapshot(root,result,'retry')
        print(f'singleton chain={result.job_ids}, collector={result.collection_job_id}')
    else:
        initial = json.loads((root/'initial.json').read_text())
        retry = json.loads((root/'retry.json').read_text())
        old_report = json.loads((root/'initial-report.json').read_text())
        b = next(item for item in old_report['target_results'] if item['target']=='B')
        assert b['status']=='blocked' and b['upstream_status']=='failed'
        a = next(item for item in old_report['target_results'] if item['target']=='A')
        c = next(item for item in old_report['target_results'] if item['target']=='C')
        assert a['stage_results'][1]['started_ns'] < c['stage_results'][0]['finished_ns'], 'Corresponding element did not advance independently'
        frame = wf.collect(root/'run',require_normal_termination=True)
        assert sorted(frame.target.tolist()) == list('ABC')
        report = json.loads((root/'run/collection_report.json').read_text())
        assert report['retry_targets']==[]
        for tag in 'AC':
            df = pd.read_parquet(root/'run'/tag/'init.dft_freq.parquet')
            assert df.attrs['frust_submission']['attempt_id']==initial['records'][0]['attempt_id']
        ids = ','.join(initial['array_job_ids']+retry['job_ids']+
                       [str(initial['collection_job_id']),str(retry['collection_job_id'])])
        accounting = subprocess.check_output(['sacct','-j',ids,'--format=JobID,State,ExitCode,Start,End,ReqCPUS,ReqMem','-P'],text=True)
        blocked_job = next(record['job_id'] for record in initial['records'] if record['target']=='B' and record['group']=='dft_freq')
        blocked_info = (root/'blocked-job.txt').read_text()
        assert 'JobState=CANCELLED' in blocked_info and 'Reason=DependencyNeverSatisfied' in blocked_info
        assert 'ArrayTaskId=1' in blocked_info
        evidence = {'initial':initial,'retry':retry,'initial_report':old_report,'final_report':report,
                    'accounting':accounting,'blocked_scheduler':blocked_info,'submitit_version':submitit.__version__,
                    'slurm_version':subprocess.check_output(['sinfo','--version'],text=True).strip()}
        (root/'verified.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(f'Verified matching dependencies, terminal cancellation, and retry: {root / "verified.json"}')
