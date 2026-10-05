"""Run four water UMA SP targets in two sequential batches on Slurm.

Use the UMA environment with the site's ORCA environment loaded. Target 1
deliberately fails before its SP; targets 0 and 2 must share one server and
finish. Target 3 tests the uneven final batch and its separate server lifetime.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import submitit

from frust.cluster import ClusterConfig, Resources
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget
from frust.workflows.methods import CalculatorSpec, MethodPlan


class WaterBatchWorkflow(BaseWorkflow):
    """Internal real-calculator fixture for a bounded cluster lifecycle check."""

    workflow_name = "uma_batch_smoke"

    def __init__(self, root):
        common = {
            "uma": "omol@uma-s-1p2p1", "uma_offline": True,
            "uma_keep_logs": "always", "uma_log_dir": str(root / "server-logs"),
        }
        super().__init__(method=MethodPlan('water-batch-smoke', {
            'uma_sp': CalculatorSpec('orca', {'ExtOpt': None}, kwargs=common),
        }))

    def _build_targets(self):
        return [WorkflowTarget(f'water_{i}', {'fail': i == 1}) for i in range(4)]

    def _stage_defs(self):
        return [StageDef('prepare', 'prepare', kind='prepare'), StageDef('uma_sp', 'UMA SP')]

    def _prepare_initial_df(self, target, *, save_dir, options):
        if target.payload['fail']:
            raise RuntimeError('Deliberate middle-target failure')
        return pd.DataFrame({
            'substrate_name': [target.tag], 'atoms': [['O','H','H']],
            'coords_embedded': [[[0.,0.,0.],[0.758602,0.,0.504284],[-0.758602,0.,0.504284]]],
        })


def _verify(root):
    evidence = json.loads((root / 'submitted.json').read_text())
    result = evidence['submission']
    report = json.loads(Path(result['collection_report']).read_text())
    assert report['n_collected'] == 3 and report['n_missing'] == 1, report
    batches = [json.loads(p.read_text()) for p in sorted((root / 'run/.frust/batches').glob('*/*.json'))]
    assert len(batches) == 2
    assert [r['status'] for r in batches[0]['targets']] == ['success','failed','success'], batches
    assert [r['status'] for r in batches[1]['targets']] == ['success'], batches
    server_records = []
    for tag in ('water_0','water_2','water_3'):
        df = pd.read_parquet(root / 'run' / tag / 'final.parquet')
        assert df['uma_sp-NT'].all(), (tag, df.columns.tolist())
        server_records.append(df.attrs['frust_steps']['uma_sp']['input'])
    assert server_records[0]['uma_server_pid'] == server_records[1]['uma_server_pid']
    assert server_records[0]['uma_server_bind'] == server_records[1]['uma_server_bind']
    assert server_records[0]['uma_server_hostname'] == server_records[1]['uma_server_hostname']
    logs = [p.read_text() for p in (root / 'server-logs').glob('oet_uma_server_*.log')]
    assert len(logs) == 2
    assert all(log.count('event=started') == 1 and log.count('event=stopped') == 1 for log in logs)
    evidence.update(batches=batches, collection_report=report, servers=server_records,
                    log_checks={'starts': 2, 'stops': 2})
    ids = ','.join(result['array_job_ids'] + [str(result['collection_job_id'])])
    evidence['accounting'] = subprocess.check_output(
        ['sacct','-j',ids,'--format=JobID,State,ExitCode,AllocCPUS,ReqMem','-P'], text=True)
    (root / 'verified.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(f"Verified shared UMA batch: {root / 'verified.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', default='kemi1')
    parser.add_argument('--runtime', default='/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        _verify(args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        result = WaterBatchWorkflow(args.output).submit(
            out_dir=args.output / 'run',
            cluster=ClusterConfig(partition=args.partition, log_dir=args.output / 'logs'),
            execution='single_job', array=True, array_parallelism=1, targets_per_task=3,
            stage_resources={'single_job': Resources(cpus=4, mem_gb=32, timeout_min=20)},
            collect_resources=Resources(cpus=1, mem_gb=2, timeout_min=5),
            target_retention='all', uma_oet_tools=args.runtime,
        )
        evidence = {
            'revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'submitit_version': submitit.__version__,
            'slurm_version': subprocess.check_output(['sbatch','--version'], text=True).strip(),
            'runtime': args.runtime, 'partition': args.partition, 'submission': asdict(result),
        }
        (args.output / 'submitted.json').write_text(json.dumps(evidence, indent=2)+'\n')
        print(f'workers={result.job_ids}, collector={result.collection_job_id}')
