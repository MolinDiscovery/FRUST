"""Private composed bisarylation prototype; production activation stays gated."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import tempfile

from frust.workflows.screening import CatalystScreenWorkflow
from frust.workflows.bisarylation import _plan_bisarylation, _execution_children


class _BisarylationWorkflow(CatalystScreenWorkflow):
    """Exercise native child execution and portable balanced-ledger analysis.

    This private prototype deliberately has no public factory yet. It lacks
    final barrier integration and source-bound
    post-calculation pathway reviews. Scientific profile activation remains
    separate from construction and execution support.
    """

    def __init__(self, *, candidate_specs=None, pathway_reviews=None,
                 random_seed=None, **kwargs):
        if kwargs.get('scope') is not None:
            raise ValueError('The bisarylation prototype owns its pathway scope')
        kwargs.pop('scope', None)
        self._side_corrections = dict(kwargs.get('g_corrections_kcal_mol') or {})
        kwargs['ts_types'] = ('TS5', 'TS6', 'TS7', 'TS8')
        super().__init__(scope='barriers', **kwargs)
        self.scope = 'bisarylation'
        self.candidate_specs = dict(candidate_specs or {})
        self.pathway_reviews = dict(pathway_reviews or {})
        self.random_seed = random_seed
        self._bisarylation_plan = None

    def children(self):
        """Use the same filtered native targets for local and cluster paths."""
        if self._children_cache is None:
            if self.level != 'full' or self.ranking is not None:
                raise ValueError('Prototype analysis currently requires full characterization without ranking')
            self._bisarylation_plan = _plan_bisarylation(
                self.systems(), method_fingerprint=self.method.fingerprint(),
                pathway_reviews=self.pathway_reviews, dimer_reference=self.dimer_reference)
            self._children_cache = _execution_children(
                self._bisarylation_plan, method=self.method, screening=self.screening,
                candidate_specs=self.candidate_specs, spec_profile=self.spec_profile,
                random_seed=self.random_seed, n_confs=self.n_confs,
                validation_top_n=self.validation_top_n or 1,
                top_n=self.top_n, prune_initial=self.prune_initial)
        return dict(self._children_cache)

    def _write_manifest(self, root, *, artifact_policy='standard'):
        """Bind graph reviews, native target identities and numerical profiles."""
        self.children()
        # Reuse the parent schema and compatibility checks without replacing an
        # existing manifest before the composed signature has been checked.
        with tempfile.TemporaryDirectory() as directory:
            super()._write_manifest(Path(directory), artifact_policy=artifact_policy)
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
        manifest['mechanism_id'] = 'frust_bisarylation_prototype::v1'
        manifest['bisarylation'] = dict(
            cases=self._bisarylation_plan['cases'], reference='M',
            pathway_reviews=self.pathway_reviews,
            candidate_specs={s: asdict(spec) for s, spec in self.candidate_specs.items()},
            random_seed=self.random_seed,
            g_corrections_kcal_mol=self._side_corrections,
            limitations=['dynamic profile reference overrides pending', 'candidate barrier integration pending',
                         'prototype manifests and pathway reviews require final integration'])
        manifest['run_signature'] = hashlib.sha256(json.dumps(
            {'base': manifest['run_signature'], 'bisarylation': manifest['bisarylation']},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        destination = Path(root) / 'manifest.json'
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing.get('run_signature') != manifest['run_signature']:
                raise FileExistsError('Run directory contains a different bisarylation plan')
            return
        if any(p.name != '.frust' for p in Path(root).iterdir()):
            raise FileExistsError('Bisarylation run directory is not empty')
        destination.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
