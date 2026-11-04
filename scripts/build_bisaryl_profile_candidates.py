"""Export reviewed native geometries as inactive UMA profile candidates.

The evidence directory is an explicit local input. No local paths or raw run
artifacts are embedded in the generated package module. Activation is a separate
review after reconstruction and transfer checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from frust.tsguess2.topologies import (
    BISARYL_CH_TOPOLOGY, BISARYL_H2_TOPOLOGY, BISARYL_EXCHANGE_TOPOLOGIES,
    BISARYL_MINIMUM_TOPOLOGIES,
)


def main() -> None:
    """Verify source bindings and recompute every exported core distance."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_root',type=Path)
    parser.add_argument('--output',type=Path,default=Path('frust/tsguess2/profiles/uma_bisaryl_gas.py'))
    args=parser.parse_args();root=args.evidence_root;payload={}
    for state in ['TS5','TS6','TS7','TS8','INT4','INT5','INT6']:
        path=root/('stage02/resumed_20261028/ts6_reviewed_native_candidate.json' if state=='TS6' else 'stage03/INT6_C7_reviewed_native_candidate.json' if state=='INT6' else f'stage03/{state}_reviewed_native_candidate.json')
        candidate=json.loads(path.read_text());source=Path(candidate['source_dataframe'])
        if not source.is_absolute():source=root.parent/source
        elif not source.exists() and 'runs' in source.parts:
            source=root.parent.joinpath(*source.parts[source.parts.index('runs'):])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==candidate['source_sha256']
        if state=='TS6':decision=json.loads((root/'stage02/resumed_20261028/scientific_decision.json').read_text());assert 'passed' in str(decision.get('gate',decision.get('decision',''))).lower() or decision.get('accepted_reference_candidate')==path.name
        elif state in {'INT4','INT5'}:assert json.loads((root/f'stage03/{state}_native_minimum_decision.json').read_text())['gate']=='passed_native_minimum_identity'
        elif state!='INT6':assert json.loads((root/f'stage03/{state}_scientific_decision.json').read_text())['gate']=='passed_practical_reference_identity'
        df=pd.read_parquet(source);row=df.loc[df.cid==candidate['cid']].iloc[0] if 'cid' in candidate else df.iloc[0]
        coordinate_column=candidate.get('coordinate_column','min_opt-oc' if state=='INT6' else 'uma_ts_opt-oc');vibration_column=candidate.get('vibrations_column','frequency-vibs' if state=='INT6' else 'uma_freq-vibs')
        xyz=np.stack(row[coordinate_column]);stored=np.array(candidate['source_geometry']['coords']);assert np.abs(xyz-stored).max()<1e-7
        negatives=tuple(float(v['frequency']) for v in row[vibration_column] if v['frequency']<0)
        assert len(negatives)==(0 if state.startswith('INT') else 1)
        topology={'TS5':BISARYL_CH_TOPOLOGY,'TS6':BISARYL_H2_TOPOLOGY,**BISARYL_EXCHANGE_TOPOLOGIES,**BISARYL_MINIMUM_TOPOLOGIES}[state];roles=candidate['source_geometry']['roles'];coordinates={r:tuple(map(float,x)) for r,x in candidate['role_coordinates'].items()}
        assert all(np.abs(np.array(x)-xyz[roles[r]]).max()<1e-7 for r,x in coordinates.items())
        values={c.name:float(np.linalg.norm(xyz[roles[c.roles[0]]]-xyz[roles[c.roles[1]]])) for c in topology.constraints}
        for c in candidate['constraints']:
            assert abs(float(c['value'])-np.linalg.norm(xyz[roles[c['roles'][0]]]-xyz[roles[c['roles'][1]]]))<1e-7
        notes='Native TMP / two alpha-1-methylpyrrole-derived groups. Exact source geometry, modes, both reaction directions and independently converged zero-negative minima reviewed. Inactive pending final profile regeneration and transfer review.'
        if state=='TS6':notes+=' Original strict IRCs reached their caps; their final gradients meet ordinary thresholds and the practical connection was accepted.'
        if state=='TS8':notes+=' Original uphill backward kick excluded. Corrected 0.2mEh downhill converged IRC plus separately bound 117-cycle endpoint relaxation establishes the practical bridge basin connection; relaxation cycles are not IRC.'
        if state=='INT6':notes='Native TS7-derived C7 bridge minimum, independently converged with zero negative frequencies and complete stable-fragment identity. TS8 reaches a distinct C8 conformer; do not splice conformer energies. Inactive pending regeneration/transfer review.'
        if state=='INT4':notes='Native 65-atom bisaryl N-H/B-H adduct, four independent chemical constructions and two candidates through released minimum optimization/frequencies. Complete intended graph and zero negative frequencies reviewed. Inactive pending final regeneration/transfer review.'
        if state=='INT5':notes='Native 63-atom neutral diaryl catalyst, four independent chemical constructions and two candidates through released minimum optimization/frequencies. Complete intended graph and zero negative frequencies reviewed. Free H2 is outside this local geometry. Inactive pending final regeneration/transfer review.'
        notes+=f' Source row cid{int(row.cid)}; coordinates column {coordinate_column}; vibrations column {vibration_column}.'
        payload[state]=dict(role_coordinates=coordinates,constraint_values=values,source_sha256=candidate['source_sha256'],coordinates_column=coordinate_column,vibrations_column=vibration_column,negative_frequencies=negatives,notes=notes)
    header='''"""Reviewed native bisaryl UMA references awaiting production activation.

Generated by scripts/build_bisaryl_profile_candidates.py from explicitly bound
review evidence. Existing TS1–TS4/INT3 profile revisions are unchanged.
"""
from frust.tsguess2.models import GeometryKey, ReferenceRecord, StateGeometrySpec

GEOMETRY_KEY = GeometryKey(method="omol-uma-s-1p2p1")

_RECORDS = '''
    tail='''

GEOMETRIES = {
    state: StateGeometrySpec(
        state=state, geometry_key=GEOMETRY_KEY, revision=1, status="candidate",
        role_coordinates=record["role_coordinates"],
        constraint_values=record["constraint_values"],
        reference=ReferenceRecord(
            substrate_name="1-methylpyrrole", catalyst_name="TMP",
            method="UMA-S 1.2.1 (OMol)", basis=None,
            solvation_model=None, solvent=None,
            coordinates_column=record["coordinates_column"],
            vibrations_column=record["vibrations_column"],
            negative_frequencies=record["negative_frequencies"],
            mode_reviewed=True, source_sha256=record["source_sha256"],
            notes=record["notes"]),
    ) for state, record in _RECORDS.items()
}
'''
    args.output.write_text(header+pformat(payload,sort_dicts=True,width=100)+tail)
    print(f'Wrote {len(payload)} inactive candidates to {args.output}')


if __name__=='__main__':main()
