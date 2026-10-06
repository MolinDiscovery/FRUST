"""Bounded public-API scheduler smoke with synthetic calculation results.

Run initial, retry, reuse, then verify, waiting for each phase's control jobs.
Use PYTHONPATH=. and the UMA environment. --backend local exercises the same
script without Slurm. No external calculator or model service is launched.
Reference entries are synthetic fixtures isolated under this run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import types
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import submitit

import frust as ft
from frust.cluster.pipeline_workflow import _PipelineWorkflow
from frust.results import attach_result_contract
from frust.screen.references import ReferenceLibrary
from frust.utils.mols import create_mol_per_rpos


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def frame(target):
    formulas = {'ligand': {'C':5,'H':7,'N':1}, 'dimer': {'C':16,'H':24,'B':2,'N':2},
                'HBpin-mol': {'C':6,'H':13,'B':1,'O':2}, 'HH': {'H':2},
                'TS1': {'C':13,'H':19,'B':1,'N':2}, 'TS2': {'C':13,'H':19,'B':1,'N':2}}
    energies = {'ligand':-1.0,'dimer':-4.0,'HBpin-mol':-2.0,'HH':-.5,'TS1':-2.8,'TS2':-2.7}
    atoms = [atom for atom, count in formulas[target.state_id].items() for _ in range(count)]
    result = pd.DataFrame({
        'structure_id':[target.target_id], 'state_id':[target.state_id], 'state_kind':[target.state_kind],
        'system_name':[target.system.system_name], 'substrate_name':[target.system.substrate_name],
        'catalyst_name':[target.system.catalyst_name], 'rpos':[target.rpos], 'cid':[0], 'atoms':[atoms],
        'xtb_opt-oc':[[[0.,0.,float(i)] for i in range(len(atoms))]],
        'xtb_opt-EE':[energies[target.state_id]], 'xtb_opt-NT':[True],
    })
    attach_result_contract(result, target.state_kind, dft=False, calculation_level='low_cost')
    result.attrs['scheduler_fixture'] = 'Synthetic energies and geometries; no scientific calculation'
    return result


def screen(root):
    components = pd.DataFrame({'role':['substrate','catalyst'],
                              'smiles':['CN1C=CC=C1','BC1=C(N(C)C)C=CC=C1'],
                              'compound_name':['pyrrole','NMe'],'rpos':['2',None]})
    workflow = ft.workflows.catalyst_screen(dataframe=components, level='low_cost',
                                          ts_types=['TS1','TS2'], dimer_reference='dimer',
                                          reference_store=root/'synthetic_reference_store')
    def calculate(self, target, stages, **kwargs):
        time.sleep(3)
        if target.state_id=='TS1' and not (root/'retry-ready').exists():
            raise RuntimeError('Deliberate Task 07 screen failure')
        return frame(target)
    for child in workflow.children().values():
        child._run_stage_group = types.MethodType(calculate, child)
    return workflow


def pipeline(mol_struct, n_cores, mem_gb, out_dir, output_parquet=None, **kwargs):
    tag = next(iter(mol_struct))
    start = time.time()
    time.sleep(10)
    result = pd.DataFrame({'substrate_name':[tag],'fixture-NT':[True]})
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/f'interval_{tag}.json', {
        'target':tag,'start':start,'finish':time.time(),'cpus':n_cores,'memory_gb':mem_gb,
        'host':socket.gethostname(),'job_id':os.getenv('SLURM_JOB_ID'),
        'array_job_id':os.getenv('SLURM_ARRAY_JOB_ID'),'array_index':os.getenv('SLURM_ARRAY_TASK_ID'),
    })
    if output_parquet is not None:
        result.to_parquet(output_parquet, index=False)
    return result


def facade_worker(self, target, stages, input_df, save_dir, options):
    with patch('frust.cluster.pipeline_workflow.load_pipeline', return_value=pipeline):
        return _PipelineWorkflow._run_stage_group(self, target, stages, input_df, save_dir, options)


def facade(root, cluster, resources, *, array):
    original_init = _PipelineWorkflow.__init__
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._run_stage_group = types.MethodType(facade_worker, self)
    def prepare(*args, **kwargs):
        return create_mol_per_rpos(*args, **kwargs, show_iupac=False)
    with patch.object(_PipelineWorkflow, '__init__', initialize), \
         patch('frust.cluster.inputs.create_mol_per_rpos', side_effect=prepare), \
         patch('frust.cluster.facade.load_pipeline', return_value=pipeline):
        return ft.cluster.submit_jobs(
            csv_path=root/'molecules.csv', pipeline='run_mols_per_rpos',
            out_dir=root/('facade_array' if array else 'facade_individual'),
            cluster=cluster, resources=resources, select_mols=['ligand'],
            array=array, array_parallelism=2 if array else None, targets_per_task=2 if array else 1,
        )


def snapshot(root, name, result, cluster):
    payload = asdict(result)
    if cluster.backend=='slurm':
        ids = []
        for child in getattr(result, 'child_submissions', {}).values():
            ids.extend(child.array_job_ids or child.job_ids)
            ids.append(child.collection_job_id)
        ids.extend(getattr(result, 'array_job_ids', []) or getattr(result, 'job_ids', []))
        ids.extend([getattr(result, 'finalization_job_id', None),getattr(result, 'collection_job_id', None)])
        payload['scheduler'] = {str(job):subprocess.check_output(['scontrol','show','job',str(job)],text=True)
                                for job in ids if job is not None}
    write_json(root/f'{name}.json', payload)
    print(name, getattr(result, 'finalization_job_id', None) or result.job_ids)


def verify(root, backend):
    evidence = {'metadata':json.loads((root/'metadata.json').read_text())}
    evidence['verification_revision'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name in ['facade_array','facade_individual','screen_initial','screen_retry','screen_reuse']:
        evidence[name] = json.loads((root/f'{name}.json').read_text())
    first = json.loads((root/'initial-screen-report.json').read_text())
    assert first['overall_status']=='partial', first
    for name in ['screen','reused_screen']:
        report = json.loads((root/name/'run_report.json').read_text())
        assert report['overall_status']=='success', report
        assert report['artifact_cleanup']['n_removed_targets']>0
        assert not report['artifact_cleanup']['errors']
        assert list((root/name).glob('.frust/screen_submissions/*.json'))
        assert list((root/name).glob('.frust/completions/*/screen_finalize_0.json'))
        evidence[name+'_report'] = report
    retry = evidence['screen_retry']['child_submissions']
    assert set(retry)=={'transition_states'}
    assert retry['transition_states']['tags']==['TS1__pyrrole__NMe__r2']
    reused = evidence['screen_reuse']['child_submissions']
    assert set(reused)=={'transition_states'}
    merged = pd.read_parquet(root/'screen/calculations/transition_states/merged.parquet')
    assert sorted(merged.state_id.tolist())==['TS1','TS2']
    assert merged.set_index('state_id').loc['TS2','xtb_opt-EE']==-2.7
    report = json.loads((root/'screen/calculations/transition_states/collection_report.json').read_text())
    preserved = next(item for item in report['target_results'] if item['target'].startswith('TS2'))
    initial = evidence['screen_initial']['child_submissions']['transition_states']
    assert preserved['attempt_id']==initial['records'][0]['attempt_id']
    assert hashlib.sha256((root/'screen/manifest.json').read_bytes()).hexdigest()==(root/'manifest-sha256').read_text()
    assert all(Path(child['submission_path']).exists() for child in evidence['screen_initial']['child_submissions'].values())
    evidence['retry_branch_report'] = report
    intervals = [json.loads(path.read_text()) for path in (root/'facade_array').glob('*/interval_*.json')]
    assert len(intervals)==3
    assert all(item['cpus']==1 and item['memory_gb']==1.6 for item in intervals)
    events = [(item['start'],1) for item in intervals]+[(item['finish'],-1) for item in intervals]
    active = maximum = 0
    for _, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    assert 1<=maximum<=2
    assert len(pd.read_parquet(evidence['facade_array']['collection_output']))==3
    assert len(list((root/'facade_individual').glob('*.parquet')))==3
    if backend=='slurm':
        assert len(evidence['facade_array']['array_job_ids'])==1
        assert sorted({int(item['array_index']) for item in intervals})==[0,1]
        assert maximum==2, 'No concurrent elements observed; rerun the bounded check'
    evidence['facade_intervals'] = intervals
    evidence['maximum_running_facade_targets'] = maximum
    if backend=='slurm':
        ids = [job for name in ['facade_array','facade_individual','screen_initial','screen_retry','screen_reuse']
               for job in evidence[name]['scheduler']]
        evidence['accounting'] = subprocess.check_output(
            ['sacct','-j',','.join(ids),'--format=JobID,State,ExitCode,Start,End,ReqCPUS,ReqMem','-P'],text=True)
        assert not list(root.glob('**/*_1_log.out')), 'Duplicate Submitit ranks were launched'
    try:
        screen(root).submit(out_dir=root/'screen',
                            cluster=ft.cluster.ClusterConfig(backend=backend,log_dir=root/'logs'),
                            array=True,array_parallelism=1,retry=True,targets={'transition_states':[0]},
                            artifact_policy='screening')
    except ValueError as error:
        assert 'finalized successfully' in str(error)
    else:
        raise AssertionError('Completed screen accepted a destructive retry')
    evidence['completed_screen_retry'] = 'rejected before submission'
    write_json(root/'verified.json', evidence)
    print('Verified public facade, screen failure/retry, reuse, and cleanup:', root/'verified.json')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['initial','retry','reuse','verify'],required=True)
    parser.add_argument('--backend',choices=['slurm','local'],default='slurm')
    parser.add_argument('--partition',default='kemi1')
    args = parser.parse_args()
    root = args.output.resolve()
    resources = ft.cluster.Resources(1,2,5)
    cluster = ft.cluster.ClusterConfig(backend=args.backend,partition=args.partition,log_dir=root/'logs')
    if args.phase=='verify':
        verify(root,args.backend)
    else:
        if args.phase=='initial':
            root.mkdir(parents=True,exist_ok=False)
            write_json(root/'metadata.json',{
                'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'frust_version':'0.1.0','submitit_version':submitit.__version__,'backend':args.backend,
                'partition':args.partition,'resources':asdict(resources),
                'fixture':'Synthetic calculation results; production submission and finalization',
                'slurm_version':subprocess.check_output(['sinfo','--version'],text=True).strip() if args.backend=='slurm' else None,
            })
            pd.DataFrame({'smiles':['CN1C=CC=C1','c1ccccc1','Cc1ccccc1'],
                          'system_name':['pyrrole','benzene','toluene']}).to_csv(root/'molecules.csv',index=False)
            workflow = screen(root)
            library = ReferenceLibrary(root/'synthetic_reference_store').initialize()
            for target in workflow.children()['references'].targets()[:2]:
                library.publish(frame(target),target,workflow.method,calculation_level='low_cost',
                                protocol=workflow._reference_protocol())
            assert set(workflow.plan().query("branch=='references'").action)=={'calculate','reuse'}
            for array in [True,False]:
                result = facade(root,cluster,resources,array=array)
                snapshot(root,'facade_array' if array else 'facade_individual',result,cluster)
        else:
            workflow = screen(root)
        if args.phase=='retry':
            report = json.loads((root/'screen/run_report.json').read_text())
            assert report['overall_status']=='partial'
            write_json(root/'initial-screen-report.json',report)
            (root/'manifest-sha256').write_text(hashlib.sha256((root/'screen/manifest.json').read_bytes()).hexdigest())
            (root/'retry-ready').touch()
        if args.phase=='reuse':
            assert set(workflow.plan().query("branch=='references'").action)=={'reuse'}
        result = workflow.submit(
            out_dir=root/('reused_screen' if args.phase=='reuse' else 'screen'),cluster=cluster,
            execution='single_job',array=True,array_parallelism=1,
            stage_resources={'single_job':resources},collect_resources=resources,finalize_resources=resources,
            artifact_policy='screening',retry=args.phase=='retry',
        )
        snapshot(root,'screen_'+args.phase,result,cluster)
