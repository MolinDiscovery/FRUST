"""Small Slurm initial -> selected retry -> all-target recollection check.

Run --phase initial, wait for its collector, then --phase retry. After that
collector finishes run --phase verify. No chemistry or UMA service is needed.
"""
import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import submitit

from frust.cluster import ClusterConfig,Resources
from frust.workflows.core import BaseWorkflow,StageDef,WorkflowTarget


class RetryWorkflow(BaseWorkflow):
    """Internal scheduler fixture with an external failure injection switch."""
    def __init__(self,root):
        super().__init__()
        self.root=root
    def _build_targets(self):
        return [WorkflowTarget(tag,None) for tag in 'ABCDE']
    def _stage_defs(self):
        return [StageDef('prepare','prepare',kind='prepare')]
    def _run_stage_group(self,target,*args,**kwargs):
        if not (self.root/'retry-ready').exists():
            if target.tag=='B':
                raise RuntimeError('Deliberate target failure')
            if target.tag=='D':
                raise SystemExit(143)
        return pd.DataFrame({'target':[target.tag],'smoke-NT':[True]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['initial','retry','verify'],required=True)
    parser.add_argument('--partition',default='kemi1')
    args=parser.parse_args()
    root=args.output
    wf=RetryWorkflow(root)
    cluster=ClusterConfig(partition=args.partition,log_dir=root/'logs')
    if args.phase=='initial':
        root.mkdir(parents=True,exist_ok=False)
        result=wf.submit(out_dir=root/'run',cluster=cluster,array=True,array_parallelism=2,
                         targets_per_task=3,target_retention='all',
                         stage_resources={'single_job':Resources(1,2,5)},collect_resources=Resources(1,2,5))
        (root/'initial.json').write_text(json.dumps(asdict(result),indent=2)+'\n')
        print(f'initial={result.job_ids}, collector={result.collection_job_id}')
    elif args.phase=='retry':
        first=json.loads((root/'initial.json').read_text())
        report=json.loads(Path(first['collection_report']).read_text())
        assert report['retry_targets']==['B','D','E'],report
        (root/'initial-report.json').write_text(json.dumps(report,indent=2)+'\n')
        (root/'retry-ready').touch()
        selected=[target for target in wf.targets() if target.tag in report['retry_targets']]
        result=wf.submit(out_dir=root/'run',cluster=cluster,targets=selected,retry=True,
                         array=True,array_parallelism=2,targets_per_task=1,target_retention='all',
                         stage_resources={'single_job':Resources(2,4,5)},collect_resources=Resources(1,2,5))
        (root/'retry.json').write_text(json.dumps(asdict(result),indent=2)+'\n')
        print(f'retry={result.job_ids}, collector={result.collection_job_id}')
    else:
        final=wf.collect(root/'run',require_normal_termination=True)
        assert sorted(final.target.tolist())==list('ABCDE')
        report=json.loads((root/'run/collection_report.json').read_text())
        assert report['retry_targets']==[]
        first=json.loads((root/'initial.json').read_text())
        retry=json.loads((root/'retry.json').read_text())
        for tag in 'AC':
            df=pd.read_parquet(root/'run'/tag/'final.parquet')
            assert df.attrs['frust_submission']['attempt_id']==first['records'][0]['attempt_id']
        evidence={'initial':first,'retry':retry,'initial_report':json.loads((root/'initial-report.json').read_text()),
                  'final_report':report,'submitit_version':submitit.__version__,
                  'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
        ids=','.join(first['array_job_ids']+retry['array_job_ids']+
                     [str(first['collection_job_id']),str(retry['collection_job_id'])])
        evidence['accounting']=subprocess.check_output(['sacct','-j',ids,'--format=JobID,State,ExitCode,AllocCPUS,ReqMem','-P'],text=True)
        (root/'verified.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(f'Verified retry and recollection: {root / "verified.json"}')
