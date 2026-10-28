"""Evidence-scoped bisarylation planning and balanced-ledger policies."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import pandas as pd


BISARYL_STATES = ('TS5', 'INT4', 'TS6', 'INT5', 'TS7', 'INT6', 'TS8')
PATH_STATES = {
    'two_saddle': frozenset(BISARYL_STATES),
    'ts7_only': frozenset({'TS5', 'INT4', 'TS6', 'INT5', 'TS7'}),
    'ts8_only': frozenset({'TS5', 'INT4', 'TS6', 'INT5', 'TS8'}),
    'unresolved': frozenset(BISARYL_STATES),
}
BISARYL_PROFILE_TERMS = {
    'Dimer': {'dimer': 0.5, 'incoming_ligand': 1.0, 'bound_ligand': 1.0, 'HBpin-mol': 1.0},
    'Cat': {'catalyst': 1.0, 'incoming_ligand': 1.0, 'bound_ligand': 1.0, 'HBpin-mol': 1.0},
    'M': {'bound_int2': 1.0, 'incoming_ligand': 1.0, 'HBpin-mol': 1.0, 'HH': 1.0},
    'TS5': {'TS5': 1.0, 'HBpin-mol': 1.0, 'HH': 1.0},
    'INT4': {'INT4': 1.0, 'HBpin-mol': 1.0, 'HH': 1.0},
    'TS6': {'TS6': 1.0, 'HBpin-mol': 1.0, 'HH': 1.0},
    'INT5': {'INT5': 1.0, 'HBpin-mol': 1.0, 'HH': 2.0},
    'TS7': {'TS7': 1.0, 'HH': 2.0},
    'INT6': {'INT6': 1.0, 'HH': 2.0},
    'TS8': {'TS8': 1.0, 'HH': 2.0},
    'Product': {'int2': 1.0, 'bound_product': 1.0, 'HH': 2.0},
}


def _optional(value):
    return None if value is None or pd.isna(value) else value


def _reference_roles(system: Mapping, rpos: int) -> dict[str, dict]:
    """Preserve which aryl supplies the starting catalyst and released product."""
    incoming = system['substrate_smiles']
    bound = _optional(system.get('bound_substrate_smiles')) or incoming
    bound_rpos = _optional(system.get('bound_rpos'))
    if bound_rpos is None and bound == incoming:
        bound_rpos = rpos
    if bound_rpos is None:
        raise ValueError('bound_rpos is required for the bound substrate')
    def component(state, smiles, site=None):
        return dict(state_id=state, substrate_smiles=smiles, rpos=site,
                    catalyst_smiles=system['catalyst_smiles'])
    return {
        'incoming_ligand': component('ligand', incoming),
        'bound_ligand': component('ligand', bound),
        'bound_int2': component('int2', bound, int(bound_rpos)),
        'int2': component('int2', incoming, int(rpos)),
        'bound_product': component('HBpin-ligand', bound, int(bound_rpos)),
    }


def _state_applicability(variant: str) -> dict[str, str]:
    """Keep unresolved discovery distinct from required or inapplicable states."""
    if variant not in PATH_STATES:
        raise ValueError(f'Unknown pathway variant {variant!r}; expected {tuple(PATH_STATES)}')
    return {state: ('not_applicable' if state not in PATH_STATES[variant] else
                    'discovery' if variant == 'unresolved' and state in {'TS7', 'INT6', 'TS8'} else
                    'required') for state in BISARYL_STATES}


def _profile_terms(variant: str) -> dict[str, dict[str, float]]:
    """Return balanced terms without fabricating an unused saddle or bridge."""
    applicability = _state_applicability(variant)
    return {label: dict(terms) for label, terms in BISARYL_PROFILE_TERMS.items()
            if applicability.get(label) != 'not_applicable'}


def _system_signature(system: Mapping, rpos: int) -> str:
    """Bind pathway planning to both aryl identities, sites and stereochemistry."""
    bound = _optional(system.get('bound_substrate_smiles'))
    if bound is None:
        bound = system['substrate_smiles']
        bound_rpos = rpos
    else:
        bound_rpos = _optional(system.get('bound_rpos'))
        if bound_rpos is None:
            raise ValueError('bound_rpos is required with bound_substrate_smiles')
    payload = dict(catalyst_smiles=system['catalyst_smiles'],
                   incoming_substrate_smiles=system['substrate_smiles'],incoming_rpos=int(rpos),
                   bound_substrate_smiles=bound,bound_rpos=int(bound_rpos))
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def _validated_review(review, *, system, rpos, method_fingerprint):
    """Require explicit source evidence for a resolved per-system hypothesis."""
    if review is None:
        return 'unresolved'
    variant = review.get('variant', 'unresolved')
    _state_applicability(variant)
    if variant == 'unresolved':
        return variant
    if review.get('system_signature') != _system_signature(system,rpos):
        raise ValueError('Pathway review does not match the chemical system and reactive sites')
    if review.get('method_fingerprint') != method_fingerprint:
        raise ValueError('Pathway review does not match the calculation method/environment')
    evidence = review.get('evidence_sha256', [])
    if not evidence or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in evidence):
        raise ValueError('A resolved pathway requires immutable reviewed evidence hashes')
    return variant


def _plan_bisarylation(systems, *, method_fingerprint, pathway_reviews=None, dimer_reference='dimer'):
    """Plan native targets and chemical role bindings without embedding.

    The returned private plan is a foundation for composed execution; it is not
    a public calculation workflow or a certificate of pathway validation.
    """
    from frust.structures.planner import normalize_systems, plan_targets
    from frust.tsguess.matching import parse_rpos_value
    from frust.structures.specs import DIMER_STATES

    if dimer_reference not in {'lowest', *DIMER_STATES}:
        raise ValueError('Unknown dimer reference')
    dimer_states = DIMER_STATES if dimer_reference == 'lowest' else (dimer_reference,)

    targets = {}
    cases = []
    reviews = pathway_reviews or {}
    for _, series in normalize_systems(systems).iterrows():
        system = series.to_dict()
        for site in parse_rpos_value(system.get('rpos'), system['substrate_smiles']):
            site = int(site)
            signature = _system_signature(system, site)
            variant = _validated_review(reviews.get(signature), system=system,
                                        rpos=site, method_fingerprint=method_fingerprint)
            applicability = _state_applicability(variant)
            context = {**system, 'rpos': site}
            core = plan_targets(pd.DataFrame([context]), states=BISARYL_STATES)
            bindings = {}
            for target in core:
                if applicability[target.state_id] != 'not_applicable':
                    targets[target.target_id] = target
                    bindings[target.state_id] = target.target_id
            references = _reference_roles(system, site)
            references.update({s: dict(state_id=s, substrate_smiles=system['substrate_smiles'],
                                      catalyst_smiles=system['catalyst_smiles'], rpos=None)
                               for s in (*dimer_states, 'catalyst', 'HBpin-mol', 'HH')})
            for role, component in references.items():
                ref = {**context, **component}
                if component['substrate_smiles'] != system['substrate_smiles']:
                    label = hashlib.sha256(component['substrate_smiles'].encode()).hexdigest()[:12]
                    ref['system_name'] = system['system_name'] + '__bound_' + label
                    ref['substrate_name'] = 'bound_' + label
                target = plan_targets(pd.DataFrame([ref]), states=[component['state_id']])[0]
                targets[target.target_id] = target
                bindings[role] = target.target_id
            bindings['dimer'] = bindings[dimer_states[0]]
            cases.append(dict(system_name=system['system_name'], rpos=site,
                              catalyst_name=system['catalyst_name'], substrate_name=system['substrate_name'],
                              system_signature=signature, variant=variant,
                              dimer_reference=dimer_reference,
                              applicability=applicability, bindings=bindings,
                              method_fingerprint=method_fingerprint))
    return dict(cases=cases, targets=list(targets.values()), method_fingerprint=method_fingerprint)


def _balanced_profile_rows(states, case, *, reference='M', quantity='electronic', corrections=None):
    """Resolve quality-qualified candidates by exact target identity and balance.

    Sparse chemistry-specific provenance belongs in the returned helper table,
    not the main calculation dataframe. Missing required evidence remains
    incomplete; inapplicable pathway states are omitted rather than fabricated.
    """
    from collections import Counter
    from frust.screen.runs import HARTREE_TO_KCAL_MOL, QUALITY_ORDER, _parse_formula

    if quantity not in {'electronic', 'gibbs'}:
        raise ValueError('quantity must be electronic or gibbs')
    energy_column = 'free_energy_hartree' if quantity == 'gibbs' else 'electronic_energy_hartree'
    corrections = corrections or {}

    terms = _profile_terms(case['variant'])
    if reference not in terms:
        raise ValueError(f'Unknown applicable profile reference {reference!r}')
    resolved = {}
    for label, components in terms.items():
        chosen = {}
        issues = []
        for role in components:
            identity = case['bindings'].get(role)
            candidates = states.loc[states.structure_id.eq(identity)] if identity is not None else states.iloc[:0]
            if candidates.empty:
                issues.append('missing:' + role)
                continue
            ranks = candidates.quality_status.map(QUALITY_ORDER).fillna(3)
            if energy_column not in candidates:
                candidates = candidates.assign(**{energy_column: float('nan')})
            best = candidates.loc[ranks.eq(ranks.min())].sort_values(energy_column, na_position='last').iloc[0]
            chosen[role] = best
            if best.quality_status != 'ready':
                issues.append(str(best.quality_status) + ':' + role)
        if len(chosen) != len(components):
            resolved[label] = dict(quality_status='incomplete', quality_issues=issues)
            continue
        protocols = {r['energy_protocol_fingerprint'] for r in chosen.values()}
        if any(pd.isna(protocol) for protocol in protocols):
            issues.append('missing_energy_protocol')
        composition = Counter()
        for role, row in chosen.items():
            formula = row['formula'] if isinstance(row['formula'], Mapping) else _parse_formula(str(row['formula']))
            for atom, count in formula.items():
                composition[atom] += components[role] * count
        if len(protocols) != 1:
            issues.append('incompatible_energy_protocols')
        energies = [row['electronic_energy_hartree'] for row in chosen.values()]
        if any(pd.isna(value) for value in energies):
            issues.append('missing_electronic_energy')
        free_energies = [row.get('free_energy_hartree', float('nan')) for row in chosen.values()]
        if quantity == 'gibbs' and any(pd.isna(value) for value in free_energies):
            issues.append('missing_free_energy')
        resolved[label] = dict(
            quality_status='incomplete' if any(issue in issues for issue in ['missing_electronic_energy','missing_free_energy','missing_energy_protocol']) else
                           'invalid' if 'incompatible_energy_protocols' in issues else
                           max((str(row.quality_status) for row in chosen.values()), key=QUALITY_ORDER.get),
            quality_issues=issues, composition=dict(composition),
            electronic_energy_hartree=sum(components[role] * row.electronic_energy_hartree for role, row in chosen.items()),
            free_energy_hartree=sum(components[role] * row.get('free_energy_hartree',float('nan')) for role,row in chosen.items()),
            result_ids={role:row.result_id for role,row in chosen.items()})
    base = resolved[reference]
    rows = []
    for order, (label, value) in enumerate(resolved.items()):
        issues = list(value['quality_issues'])
        quality = max([value['quality_status'], base['quality_status']], key=QUALITY_ORDER.get)
        if 'composition' in value and 'composition' in base and value['composition'] != base['composition']:
            quality = 'invalid'
            issues.append('unbalanced_composition')
        if base['quality_status'] != 'ready':
            issues.append(base['quality_status'] + ':profile_reference')
        if case['applicability'].get(label) == 'discovery' and quality == 'ready':
            quality = 'review'
            issues.append('unresolved_pathway')
        relative = ((value['electronic_energy_hartree'] - base['electronic_energy_hartree']) * HARTREE_TO_KCAL_MOL
                    if quality == 'ready' else float('nan'))
        relative_g = ((value['free_energy_hartree'] - base['free_energy_hartree']) * HARTREE_TO_KCAL_MOL
                      if quality == 'ready' else float('nan'))
        correction = float(corrections.get(label, 0.)) - float(corrections.get(reference, 0.))
        rows.append(dict(system_name=case['system_name'], rpos=case['rpos'],
                         system_signature=case.get('system_signature'), profile_order=order,
                         variant=case['variant'], profile_state=label, reference=reference,
                         applicability=case['applicability'].get(label, 'required'),
                         relative_e_kcal_mol=relative, quality_status=quality,
                         relative_g_kcal_mol=relative_g, g_correction_kcal_mol=correction,
                         relative_g_corrected_kcal_mol=relative_g+correction,
                         quality_issues=';'.join(issues), result_ids=value.get('result_ids')))
    return pd.DataFrame(rows)


def _execution_children(plan, *, method, screening, candidate_specs=None,
                        spec_profile='auto', random_seed=None, n_confs=None,
                        validation_top_n=1, top_n=20, prune_initial=True):
    """Bind a private reviewed plan to existing homogeneous execution graphs.

    Children retain their ordinary local ``run`` and cluster ``submit`` paths.
    This does not yet supply composed manifests, reference-cache integration or
    a public end-to-end bisarylation factory.
    """
    from frust.workflows.factories import ScreenTSWorkflow, IntermediateWorkflow, MolsWorkflow
    from frust.workflows.methods import preset

    method = preset(method) if isinstance(method, str) else method
    if method.fingerprint() != plan['method_fingerprint']:
        raise ValueError('Execution method does not match the reviewed target plan')
    grouped = {
        'transition_states': [t for t in plan['targets'] if t.state_kind == 'transition_state'],
        'intermediates': [t for t in plan['targets'] if t.state_kind == 'constrained_minimum'],
        'references': [t for t in plan['targets'] if t.state_kind == 'minimum'],
    }
    children = {}
    for branch, targets in grouped.items():
        if not targets:
            continue
        systems = pd.DataFrame([{**t.system.as_dict(), 'rpos': t.rpos} for t in targets])
        states = tuple(dict.fromkeys(t.state_id for t in targets))
        common = dict(dataframe=systems, method=method, screening=screening,
                      calculation_level='full', n_confs=n_confs, top_n=top_n,
                      validation_top_n=validation_top_n, prune_initial=prune_initial,
                      include_dft_rank_sp=False)
        if branch == 'references':
            child = MolsWorkflow(select_mols=states, **common)
        else:
            candidates = {s: spec for s, spec in (candidate_specs or {}).items() if s in states}
            connected = dict(candidate_specs=candidates, spec_profile=spec_profile, random_seed=random_seed)
            child = (ScreenTSWorkflow(ts_types=states, **common, **connected)
                     if branch == 'transition_states' else
                     IntermediateWorkflow(states=states, **common, **connected))
            child.targets()  # Resolve only applicable profiles, before any embedding.
        child._target_cache = list(targets)
        children[branch] = child
    return children
