"""Internal adapter for tracked, sequential legacy pipeline submissions."""
from __future__ import annotations

import inspect

from frust.cluster.inputs import load_pipeline
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget


class _PipelineWorkflow(BaseWorkflow):
    def __init__(self, pipeline, prepared, options):
        super().__init__()
        self.pipeline = pipeline
        self.payloads = tuple(zip(prepared['tags'], prepared['payloads']))
        self.options = options
        self.workflow_name = pipeline

    def _build_targets(self):
        return [WorkflowTarget(tag, payload) for tag, payload in self.payloads]

    def _stage_defs(self):
        return [StageDef('pipeline', 'pipeline', kind='prepare')]

    def _run_stage_group(self, target, stages, input_df, save_dir, options):
        function = load_pipeline(self.pipeline)
        kwargs = dict(self.options, n_cores=options.n_cores, mem_gb=options.mem_gb,
                      out_dir=str(save_dir), output_parquet=None, work_dir=options.work_dir)
        kwargs['ligand_smiles_df' if self.pipeline == 'run_mols' else 'mol_struct'] = target.payload
        signature = inspect.signature(function)
        return function(**{key: value for key, value in kwargs.items() if key in signature.parameters})
