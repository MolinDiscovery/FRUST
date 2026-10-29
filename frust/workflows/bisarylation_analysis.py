"""Portable, result-bound pathway review and balanced bisarylation barriers."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import pandas as pd

from frust.workflows.bisarylation import _balanced_profile_rows, _profile_terms, _state_applicability, _component_rows


def _review_roles(variant):
    """Return all result roles that contribute to the reviewed exchange segment."""
    if variant == 'unresolved':
        return set()
    terms = _profile_terms(variant)
    labels = ['INT5', 'Product', *[s for s in ('TS7', 'INT6', 'TS8') if s in terms]]
    return set().union(*(set(terms[label]) for label in labels))


def _read_pathway_reviews(root):
    """Read append-only decisions without inventing a default approval."""
    path = Path(root) / 'analysis/pathway_reviews.json'
    return json.loads(path.read_text()).get('records', []) if path.exists() else []


def _review_fingerprint(root):
    """Include actual proof-file contents so cached analysis cannot hide tampering."""
    records = _read_pathway_reviews(root)
    allowed = (Path(root) / 'analysis/pathway_evidence').resolve()
    actual = []
    for record in records:
        for item in record.get('evidence', []):
            path = (Path(root) / item['path']).resolve()
            digest = (hashlib.sha256(path.read_bytes()).hexdigest()
                      if path.is_relative_to(allowed) and path.is_file() else None)
            actual.append([item['path'], digest])
    return hashlib.sha256(json.dumps([records, actual], sort_keys=True).encode()).hexdigest()


def _reaction_value(case, selected, left, right, *, quantity, corrections):
    """Cancel spectators before checking dependencies and evaluating a barrier."""
    from frust.screen.runs import HARTREE_TO_KCAL_MOL, QUALITY_ORDER, _parse_formula

    terms = _profile_terms(case['variant'])
    coefficients = Counter(terms[left])
    coefficients.subtract(terms[right])
    coefficients = {r: c for r, c in coefficients.items() if abs(c) > 1e-12}
    rows, issues, qualities, protocols, inventory = {}, [], [], set(), Counter()
    for role, coefficient in coefficients.items():
        row = selected.get(role)
        if row is None:
            qualities.append('incomplete');issues.append('missing:' + role)
            continue
        rows[role] = row
        qualities.append(str(row.quality_status))
        if row.quality_status != 'ready':
            issues.append(str(row.quality_status) + ':' + role)
        protocol = row.get('energy_protocol_fingerprint')
        if protocol is None or pd.isna(protocol):
            qualities.append('incomplete');issues.append('missing_energy_protocol:' + role)
        else:
            protocols.add(protocol)
        formula = row.formula if isinstance(row.formula, Mapping) else _parse_formula(str(row.formula))
        for atom, count in formula.items():
            inventory[atom] += coefficient * count
        if pd.isna(row.electronic_energy_hartree) or (quantity == 'gibbs' and pd.isna(row.get('free_energy_hartree', float('nan')))):
            qualities.append('incomplete');issues.append('missing_energy:' + role)
    if len(protocols) > 1:
        qualities.append('invalid');issues.append('incompatible_energy_protocols')
    if len(rows) == len(coefficients) and any(abs(v) > 1e-8 for v in inventory.values()):
        qualities.append('invalid');issues.append('unbalanced_composition')
    pathway = case.get('pathway_review_status')
    if {left, right} & {'TS7', 'INT6', 'TS8'} and pathway not in {None, 'approved'}:
        qualities.append('review');issues.append('pathway:' + pathway)
    quality = max(qualities or ['incomplete'], key=QUALITY_ORDER.get)
    e = sum(coefficients[r] * row.electronic_energy_hartree for r, row in rows.items())
    g = sum(coefficients[r] * row.get('free_energy_hartree', float('nan')) for r, row in rows.items())
    correction = float(corrections.get(left, 0.)) - float(corrections.get(right, 0.))
    return dict(quality_status=quality, quality_issues=';'.join(dict.fromkeys(issues)),
        relative_e_kcal_mol=e * HARTREE_TO_KCAL_MOL if quality == 'ready' else float('nan'),
        relative_g_kcal_mol=g * HARTREE_TO_KCAL_MOL if quality == 'ready' else float('nan'),
        relative_g_corrected_kcal_mol=g * HARTREE_TO_KCAL_MOL + correction if quality == 'ready' else float('nan'),
        g_correction_kcal_mol=correction,
        reference_result_ids={r: row.result_id for r, row in rows.items() if coefficients[r] < 0})


def _check_result_bindings(states, case, variant, result_ids, minimum_result_ids=None):
    """Check target ownership and numerical/identity quality for every exact row."""
    _state_applicability(variant)
    required = _review_roles(variant)
    if set(result_ids) != required:
        raise ValueError(f'Pathway result_ids must contain exactly these roles: {sorted(required)}')
    for role, result_id in result_ids.items():
        identity = case['bindings'].get(role)
        matches = states.loc[states.result_id.eq(result_id) & states.structure_id.eq(identity)]
        if len(matches) != 1:
            raise ValueError(f'Pathway role {role!r} lacks its exact current target result')
        if matches.iloc[0].quality_status != 'ready':
            raise ValueError(f'Pathway role {role!r} needs a numerically valid, identity-approved result')
    terms = _profile_terms(variant)
    minima = {'TS5': ['M', 'INT4'], 'TS6': ['INT4', 'INT5'],
              'TS7': ['INT5', 'INT6'] if variant == 'two_saddle' else ['INT5', 'Product'],
              'TS8': ['INT6', 'Product'] if variant == 'two_saddle' else ['INT5', 'Product']}
    for saddle, bindings in (minimum_result_ids or {}).items():
        if saddle not in terms or not saddle.startswith('TS'):
            raise ValueError('Minimum bindings require an applicable saddle')
        allowed = set().union(*(set(terms[label]) for label in minima[saddle]))
        if not set(bindings).issubset(allowed):
            raise ValueError('Minimum bindings must belong to an adjacent balanced minimum')
        for role, result_id in bindings.items():
            matches = states.loc[states.result_id.eq(result_id)
                                 & states.structure_id.eq(case['bindings'].get(role))]
            if len(matches) != 1 or matches.iloc[0].quality_status != 'ready':
                raise ValueError('Minimum binding needs its exact current, qualified target result')


def _review_status(states, case, record, root):
    """Invalidate decisions after changed results, method, identity or evidence."""
    if record is None or record.get('variant') == 'unresolved':
        return 'unreviewed'
    if (record.get('system_signature') != case['system_signature']
            or record.get('method_fingerprint') != case['method_fingerprint']):
        return 'stale'
    try:
        _check_result_bindings(states, case, record['variant'], record['result_ids'],
                               record.get('minimum_result_ids'))
    except (ValueError, KeyError):
        return 'stale'
    evidence = record.get('evidence', [])
    if not evidence:
        return 'missing_evidence'
    allowed = (Path(root) / 'analysis/pathway_evidence').resolve()
    for item in evidence:
        path = (Path(root) / item['path']).resolve()
        if not path.is_relative_to(allowed) or not path.is_file():
            return 'missing_evidence'
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            return 'changed_evidence'
    return 'approved'


def _analysis_case(states, case, root, reviews=None):
    """Use the current reviewed graph; keep an unreviewed plan as a hypothesis."""
    case = deepcopy(case)
    reviews = _read_pathway_reviews(root) if reviews is None else reviews
    matches = [r for r in reviews if r.get('system_signature') == case['system_signature']
               and r.get('system_name') == case['system_name'] and r.get('rpos') == case['rpos']]
    review = matches[-1] if matches else None
    status = _review_status(states, case, review, root)
    case['pathway_review_status'] = status
    case['reviewed_result_ids'] = {}
    case['reviewed_minimum_result_ids'] = {}
    if status == 'approved':
        case['variant'] = review['variant']
        case['applicability'] = _state_applicability(review['variant'])
        case['reviewed_result_ids'] = dict(review['result_ids'])
        case['reviewed_minimum_result_ids'] = deepcopy(review.get('minimum_result_ids', {}))
    return case


def _case_dimer(states, case, dimers):
    """Apply the existing strict dimer-selection policy to exact result identity."""
    from frust.screen.runs import _selected_dimer_reference

    case = deepcopy(case)
    state, result_id, quality, _ = _selected_dimer_reference(dimers, case['catalyst_name'])
    matches = states.loc[states.result_id.eq(result_id)]
    local = states.copy()
    case['dimer_state_id'] = state
    case['dimer_result_id'] = result_id
    case['dimer_reference_quality'] = quality
    if matches.empty:
        case['bindings'].pop('dimer', None)
    else:
        case['bindings']['dimer'] = str(matches.iloc[0].structure_id)
        case.setdefault('reviewed_result_ids', {})['dimer'] = result_id
        local.loc[local.result_id.eq(result_id), 'quality_status'] = quality
    return local, case


def _barrier_rows(states, case, *, quantity='gibbs', corrections=None, candidate=None):
    """Compare each saddle with its own atom-balanced adjacent minima.

    A missing dimer cannot invalidate an otherwise qualified local barrier.
    Dimer-referenced elevations and reverse barriers carry separate quality.
    Candidate overrides do not inherit another saddle's pathway approval.
    """
    corrections = corrections or {}
    choices = dict(case.get('reviewed_result_ids', {}))
    if candidate is not None:
        state = candidate.state_id
        if state in {'TS7', 'TS8'} and choices.get(state) != candidate.result_id:
            case = {**case, 'pathway_review_status': 'unreviewed_candidate'}
        choices[state] = candidate.result_id
    pairs = dict(TS5=('M', 'INT4'), TS6=('INT4', 'INT5'))
    if case['variant'] == 'two_saddle':
        pairs.update(TS7=('INT5', 'INT6'), TS8=('INT6', 'Product'))
    elif case['variant'] == 'ts7_only':
        pairs['TS7'] = ('INT5', 'Product')
    elif case['variant'] == 'ts8_only':
        pairs['TS8'] = ('INT5', 'Product')
    else:
        pairs.update(TS7=('INT5', 'INT6'), TS8=('INT6', 'Product'))
    if candidate is not None:
        pairs = {state: pairs[state]} if state in pairs else {}
    selected = _component_rows(states, case, quantity=quantity, result_ids=choices)
    rows = []
    for state, (precursor, product) in pairs.items():
        minimum_choices = case.get('reviewed_minimum_result_ids', {}).get(state, {})
        local = (_component_rows(states, case, quantity=quantity,
                                 result_ids={**choices, **minimum_choices})
                 if minimum_choices else selected)
        forward = pd.Series(_reaction_value(case, local, state, precursor,
                            quantity=quantity, corrections=corrections))
        reverse = pd.Series(_reaction_value(case, local, state, product,
                            quantity=quantity, corrections=corrections))
        elevation = pd.Series(_reaction_value(case, selected, state, 'Dimer',
                              quantity=quantity, corrections=corrections))
        ts = selected.get(state)
        ts = ts if ts is not None else pd.Series(dtype=object)
        result_id = ts.get('result_id')
        rows.append(dict(system_name=case['system_name'], substrate_name=case['substrate_name'],
            catalyst_name=case['catalyst_name'], rpos=case['rpos'], ts_type=state,
            system_signature=case['system_signature'], variant=case['variant'],
            pathway_review_status=case.get('pathway_review_status', 'unreviewed'),
            ts_result_id=result_id, ts_cid=ts.get('cid'), reference_state=precursor,
            product_state=product, reference_result_ids=json.dumps(forward.reference_result_ids, sort_keys=True),
            product_result_ids=json.dumps(reverse.reference_result_ids, sort_keys=True),
            uses_alternate_minimum=any(choices.get(r) != v for r, v in minimum_choices.items()),
            delta_e_kcal_mol=forward.relative_e_kcal_mol,
            delta_g_kcal_mol=forward.relative_g_kcal_mol,
            delta_g_corrected_kcal_mol=forward.relative_g_corrected_kcal_mol,
            g_correction_kcal_mol=forward.g_correction_kcal_mol,
            reverse_delta_e_kcal_mol=reverse.relative_e_kcal_mol,
            reverse_delta_g_kcal_mol=reverse.relative_g_kcal_mol,
            reverse_delta_g_corrected_kcal_mol=reverse.relative_g_corrected_kcal_mol,
            reverse_quality_status=reverse.quality_status, reverse_quality_issues=reverse.quality_issues,
            dimer_relative_e_kcal_mol=elevation.relative_e_kcal_mol,
            dimer_relative_g_kcal_mol=elevation.relative_g_kcal_mol,
            dimer_elevation_quality_status=elevation.quality_status,
            dimer_state_id=case.get('dimer_state_id'), dimer_result_id=case.get('dimer_result_id'),
            dimer_reference_quality=case.get('dimer_reference_quality', 'incomplete'),
            method_family=ts.get('method_family'), energy_method=ts.get('energy_method'),
            energy_model=ts.get('energy_model'), solvation_model=ts.get('solvation_model'),
            solvent=ts.get('solvent'), energy_protocol_fingerprint=ts.get('energy_protocol_fingerprint'),
            quality_status=forward.quality_status, quality_issues=forward.quality_issues,
            formula_id=f'frust_bisarylation_local_barrier::{state}::v1'))
    return pd.DataFrame(rows)


def _bisarylation_tables(root, states, manifest, dimers, *, calculation_level):
    """Build profiles and selected local barriers with the same reviewed choices."""
    quantity = 'gibbs' if calculation_level == 'full' else 'electronic'
    corrections = manifest['bisarylation'].get('g_corrections_kcal_mol', {})
    reviews = _read_pathway_reviews(root)
    profiles, barriers = [], []
    for original in manifest['bisarylation']['cases']:
        case = _analysis_case(states, original, root, reviews)
        local, case = _case_dimer(states, case, dimers)
        profiles.append(_balanced_profile_rows(local, case, reference=manifest['bisarylation']['reference'],
            quantity=quantity, corrections=corrections, result_ids=case['reviewed_result_ids']))
        barriers.append(_barrier_rows(local, case, quantity=quantity, corrections=corrections))
    return pd.concat(barriers, ignore_index=True), pd.concat(profiles, ignore_index=True)


def _candidate_barriers(root, states, manifest, dimers, selected):
    """Retain every applicable TS candidate, including unreviewed or failed ones."""
    rows = []
    corrections = manifest['bisarylation'].get('g_corrections_kcal_mol', {})
    for original in manifest['bisarylation']['cases']:
        case = _analysis_case(states, original, root)
        local, case = _case_dimer(states, case, dimers)
        for state in ('TS5', 'TS6', 'TS7', 'TS8'):
            if case['applicability'][state] == 'not_applicable':
                continue
            candidates = local.loc[local.structure_id.eq(case['bindings'].get(state))]
            selected_ids = set(selected.loc[selected.system_signature.eq(case['system_signature'])
                                           & selected.ts_type.eq(state), 'ts_result_id'])
            for _, candidate in candidates.iterrows():
                table = _barrier_rows(local, case, corrections=corrections, candidate=candidate)
                row = table.iloc[0].to_dict()
                row.update(cid=candidate.cid, selected=candidate.result_id in selected_ids,
                    ts_review_status=candidate.review_status, ts_quality_issues=candidate.quality_issues,
                    n_imag=candidate.n_imag)
                rows.append(row)
    return pd.DataFrame(rows, columns=[*selected.columns, 'cid', 'selected', 'ts_review_status',
                                      'ts_quality_issues', 'n_imag'])
