"""Composed native bisarylation execution with independently reviewed pathways."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import tempfile

from frust.workflows.screening import CatalystScreenWorkflow
from frust.workflows.bisarylation import _plan_bisarylation, _execution_children


class _BisarylationWorkflow(CatalystScreenWorkflow):
    """Compose native child execution, review and balanced-ledger analysis."""

    def __init__(self, *, candidate_specs=None, pathway_reviews=None,
                 random_seed=None, **kwargs):
        if kwargs.get('scope') is not None:
            raise ValueError('The bisarylation prototype owns its pathway scope')
        kwargs.pop('scope', None)
        self._side_corrections = dict(kwargs.get('g_corrections_kcal_mol') or {})
        kwargs['ts_types'] = ('TS5', 'TS6', 'TS7', 'TS8')
        super().__init__(scope='barriers', **kwargs)
        if self.level != 'full' and self.validation_top_n is not None:
            raise ValueError('validation_top_n requires a full calculation level')
        if self.dimer_pairing != 'input':
            raise ValueError('Bisarylation currently supports dimer_pairing="input" only')
        self.scope = 'bisarylation'
        self.candidate_specs = dict(candidate_specs or {})
        self.pathway_reviews = dict(pathway_reviews or {})
        self.random_seed = random_seed
        self._bisarylation_plan = None

    def children(self):
        """Use the same filtered native targets for local and cluster paths."""
        if self._children_cache is None:
            self._bisarylation_plan = _plan_bisarylation(
                self.systems(), method_fingerprint=self.method.fingerprint(),
                pathway_reviews=self.pathway_reviews, dimer_reference=self.dimer_reference)
            self._children_cache = _execution_children(
                self._bisarylation_plan, method=self.method, screening=self.screening,
                candidate_specs=self.candidate_specs, spec_profile=self.spec_profile,
                random_seed=self.random_seed, n_confs=self.n_confs,
                validation_top_n=((self.validation_top_n if self.validation_top_n is not None
                                  else self.ranking_top_n if self.ranking is not None else 1)
                                  if self.level == 'full' else None),
                top_n=self.top_n, prune_initial=self.prune_initial,
                calculation_level=self.level,
                ranking_top_n=self.ranking_top_n if self.ranking is not None else None,
                include_dft_rank_sp=self.include_dft_rank_sp, spec_match=self.spec_match)
        return dict(self._children_cache)

    def _write_manifest(self, root, *, artifact_policy='standard'):
        """Bind graph reviews, native target identities and numerical profiles."""
        self.children()
        # Reuse the parent schema and compatibility checks without replacing an
        # existing manifest before the composed signature has been checked.
        with tempfile.TemporaryDirectory() as directory:
            super()._write_manifest(Path(directory), artifact_policy=artifact_policy)
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
        manifest['mechanism_id'] = 'frust_bisarylation::v1'
        manifest['bisarylation'] = dict(
            cases=self._bisarylation_plan['cases'], reference='M',
            pathway_reviews=self.pathway_reviews,
            candidate_specs={s: asdict(spec) for s, spec in self.candidate_specs.items()},
            random_seed=self.random_seed,
            g_corrections_kcal_mol=self._side_corrections,
            limitations=['Scientific pathway coverage is independent of software support.',
                         'Resolved pathways require current result-bound reviews and saved evidence.',
                         'Racemic dimer pairing is not yet supported.'])
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


def bisarylation_screen(*, systems=None, dataframe=None, csv_path=None,
                       method='uma-gas', screening='uma-gas', ranking=None,
                       level='full', spec_profile='auto', spec_match='prefer-exact',
                       candidate_specs=None, pathway_reviews=None, random_seed=None,
                       n_confs=None, top_n=20, validation_top_n=None, ranking_top_n=None,
                       prune_initial=True, dimer_reference='lowest',
                       catalyst_stereoisomers='input', dimer_pairing='input',
                       reference_store=None, reuse_policy='approved',
                       g_corrections_kcal_mol=None) -> CatalystScreenWorkflow:
    """Build and review the second-arylation branch from native chemical inputs.

    Parameters
    ----------
    systems, dataframe, csv_path : pandas.DataFrame, path-like or None, optional
        Supply exactly one input. ``systems`` accepts expanded catalyst/substrate
        pairs; the other inputs accept the usual component screen table. An
        expanded row can additionally specify ``bound_substrate_smiles`` and
        ``bound_rpos`` to distinguish the prebound aryl from the incoming group.
        Without them both aryl groups use the incoming substrate and site.
    method, screening, ranking : method-plan object, str or None, optional
        Final, screening and optional ranking calculators. Geometry construction,
        target identity and requested validation counts stay independent of
        theory. Using UMA guesses for DFT needs an explicitly selected geometry
        profile; no method-specific reference is fabricated.
    level : {"low_cost", "uma_ranked", "dft_ranked", "full"}, optional
        ``low_cost`` retains screening results, the ranked levels retain the
        named single-point ranking, and ``full`` performs released optimization
        and frequency characterization. Scientific identity/pathway decisions
        require ``full``.
    spec_profile : str, optional
        Geometry-reference profile. ``auto`` follows the final method. Missing,
        inactive or quarantined references are never silently activated.
    spec_match : {"prefer-exact", "exact"}, optional
        Prefer the exact environment with allowed same-method fallback, or
        require an exact profile match.
    candidate_specs : mapping or None, optional
        Optional explicit geometry specifications overriding registry selection
        for this call. Reviewed UMA gas references for TS5–TS8 and INT4–INT6
        are selectable without overrides. Other methods require available
        references or an explicitly selected geometry profile.
    pathway_reviews : mapping or None, optional
        Prior system/method-bound hypotheses used to plan applicable states.
        They do not approve the newly calculated results. New runs default to
        unresolved exchange discovery and retain TS7, INT6 and TS8 targets.
        Use the returned run's ``set_pathway_review`` after reviewing its exact
        saddles, endpoint evidence and intermediate identities.
    random_seed : int or None, optional
        Native construction seed, recorded in the portable manifest.
    n_confs, top_n, validation_top_n, ranking_top_n : int or None, optional
        Initial and screening candidate limits and shared validation count.
        ``ranking_top_n`` is an alias of ``validation_top_n``; explicit aliases
        must agree. Every retained validation candidate receives the requested
        characterization stages. The default final count is one per target.
    prune_initial : bool or dict, optional
        Native initial diversity/pruning policy shared with other workflows.
    dimer_reference : {"lowest", "dimer", "dimer_bh_bridged", "dimer_eight_membered"}, optional
        Compare all three qualified dimer types or use the named topology.
        Local precursor barriers are independent of dimer-referenced elevations.
    catalyst_stereoisomers : {"input", "representative", "all"}, optional
        Preserve supplied configurations, choose representatives or expand all
        configurations using the shared screen policy.
    dimer_pairing : {"input"}, optional
        Use dimers of the supplied catalyst configuration. Racemic pairing is
        currently unsupported and raises before calculation or embedding.
    reference_store : path-like or None, optional
        Shared immutable minimum-reference library for ordinary reference roles.
    reuse_policy : {"approved", "auto_valid"}, optional
        Reuse explicitly approved references, or references that meet the shared
        numerical policy. This never approves a TS or pathway automatically.
    g_corrections_kcal_mol : mapping or None, optional
        Explicit per-balanced-state Gibbs corrections. No productive-cycle
        association correction is assumed for this different inventory.

    Returns
    -------
    CatalystScreenWorkflow
        Workflow with matching local ``run`` and cluster ``submit`` definitions.
        Its portable run supports balanced profiles, forward/reverse local
        barriers, all retained TS candidates and result-bound pathway review.

    Notes
    -----
    Software support for ``two_saddle``, ``ts7_only`` and ``ts8_only`` does not
    establish which route a particular system follows. Reviewed scientific
    examples establish two-saddle local exchange; direct single-saddle chemistry
    remains unvalidated. Unreviewed exchange
    barriers remain withheld; unused states are omitted only after the relevant
    per-system hypothesis/decision. A missing bridge does not prove a direct path.
    """
    return _BisarylationWorkflow(systems=systems, dataframe=dataframe, csv_path=csv_path,
        method=method, screening=screening, ranking=ranking, level=level,
        spec_profile=spec_profile, spec_match=spec_match, candidate_specs=candidate_specs,
        pathway_reviews=pathway_reviews, random_seed=random_seed, n_confs=n_confs,
        top_n=top_n, validation_top_n=validation_top_n, ranking_top_n=ranking_top_n,
        prune_initial=prune_initial, dimer_reference=dimer_reference,
        catalyst_stereoisomers=catalyst_stereoisomers, dimer_pairing=dimer_pairing,
        reference_store=reference_store, reuse_policy=reuse_policy,
        g_corrections_kcal_mol=g_corrections_kcal_mol)
