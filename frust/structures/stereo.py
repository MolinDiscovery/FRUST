"""Geometry checks of mapped persistent tetrahedral carbon stereocentres."""
from __future__ import annotations

import numpy as np
from rdkit import Chem


def _stereo_spec(mol):
    graph = Chem.AddHs(Chem.Mol(mol))
    centres = [atom.GetIdx() for atom in graph.GetAtoms()
               if atom.GetAtomicNum() == 6 and atom.GetChiralTag() in
               (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)]
    return {'molblock': Chem.MolToMolBlock(graph), 'centres': centres}


def check_geometry(spec, coords, atoms=None):
    """Check mapped tetrahedral orientation without comparing CIP labels.

    Parameters
    ----------
    spec : dict
        Builder-recorded hydrogen-explicit graph and persistent carbon indices.
    coords : array-like or None
        Cartesian coordinates in the builder's original atom order.
    atoms : sequence of str or None, optional
        Atomic symbols used to verify the coordinate mapping.

    Returns
    -------
    dict
        ``status`` is ``preserved`` for matching orientation, ``inverted`` for
        reversal, ``unresolved`` for a planar centre, ``unavailable`` for missing
        or inconsistent evidence, or ``not_applicable`` when no centres exist.
        ``centres`` gives each mapped atom's outcome. This does not review TS
        modes or establish reaction-path connectivity.
    """
    if not spec or not spec.get('centres'):
        return {'status': 'not_applicable', 'centres': []}
    unavailable = {'status': 'unavailable', 'centres': []}
    try:
        mol = Chem.MolFromMolBlock(spec['molblock'], removeHs=False, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        xyz = np.asarray(list(coords), dtype=float)
        if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
            return unavailable
        if atoms is not None and list(atoms) != [a.GetSymbol() for a in mol.GetAtoms()]:
            return unavailable
        expected = {i: mol.GetAtomWithIdx(i).GetChiralTag() for i in spec['centres']}
        conf = Chem.Conformer(len(xyz))
        for i, point in enumerate(xyz):
            conf.SetAtomPosition(i, point)
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)
        results = []
        for i in spec['centres']:
            neighbours = [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()]
            if len(neighbours) != 4:
                status = 'unavailable'
            else:
                vectors = xyz[neighbours[:3]] - xyz[neighbours[3]]
                volume = abs(np.linalg.det(vectors)) / np.prod(np.linalg.norm(vectors, axis=1))
                observed = mol.GetAtomWithIdx(i).GetChiralTag()
                status = ('unresolved' if volume < 0.02 or observed == Chem.ChiralType.CHI_UNSPECIFIED
                          else 'preserved' if observed == expected[i] else 'inverted')
            results.append({'atom_index': i, 'status': status})
        order = ['unavailable', 'inverted', 'unresolved', 'preserved']
        return {'status': min((r['status'] for r in results), key=order.index), 'centres': results}
    except (ValueError, TypeError, AttributeError, RuntimeError, KeyError, IndexError):
        return unavailable


def _check_row_stereo(row, spec, coords):
    """Require mapped evidence when an input graph requests a carbon configuration."""
    if spec is None:
        for key in ("smiles", "input_smiles", "catalyst_smiles"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                continue
            mol = Chem.MolFromSmiles(value)
            if mol is not None and any(atom.GetAtomicNum() == 6 and atom.GetChiralTag() in
                    (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    for atom in mol.GetAtoms()):
                return {"status": "unavailable", "centres": []}
    return check_geometry(spec, coords, row.get("atoms"))
