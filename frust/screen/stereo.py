"""Explicit stable carbon configurations for catalyst screen inputs."""
from __future__ import annotations

import pandas as pd
from rdkit import Chem
from itertools import product


def _molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid catalyst SMILES: {smiles!r}")
    return mol


def _constitution(mol):
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return Chem.MolToSmiles(copy)


def _mirror(mol):
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        if atom.GetAtomicNum() == 6:
            atom.InvertChirality()
    Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    return copy


def _label(mol):
    return '_'.join(label for idx, label in Chem.FindMolChiralCenters(mol, includeUnassigned=False)
                    if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 and label in {'R', 'S'}) or 'achiral'


def _configurations(smiles, *, racemic=False):
    mol = _molecule(smiles)
    centres = [idx for idx, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
               if label == '?' and mol.GetAtomWithIdx(idx).GetAtomicNum() == 6]
    if len(centres) > 10:
        raise ValueError('Catalyst stereo enumeration exceeded supported bound')
    configs = {}
    for orientations in product((Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                 Chem.ChiralType.CHI_TETRAHEDRAL_CCW), repeat=len(centres)):
        item = Chem.Mol(mol)
        for idx, orientation in zip(centres, orientations):
            item.GetAtomWithIdx(idx).SetChiralTag(orientation)
        Chem.AssignStereochemistry(item, cleanIt=True, force=True)
        configs[Chem.MolToSmiles(item)] = item
    if racemic:
        for item in list(configs.values()):
            mirror = _mirror(item)
            configs[Chem.MolToSmiles(mirror)] = mirror
    return configs


def resolve_stereoisomers(systems, *, catalyst_stereoisomers='input', racemic=False):
    """Resolve stable carbon configurations in an expanded systems table.

    Parameters
    ----------
    systems : pandas.DataFrame
        Explicit substrate/catalyst systems, optionally with
        ``catalyst_parent_name`` and ``catalyst_parent_smiles`` for CSV round trips.
    catalyst_stereoisomers : {"input", "representative", "all"}, optional
        ``input`` preserves supplied identities; ``all`` resolves unspecified
        tetrahedral carbon centres; ``representative`` keeps one configuration
        per mirror pair while retaining distinct diastereomers. Assigned inputs
        always retain their identities.
    racemic : bool, optional
        Include mirror counterparts of assigned inputs in the resolved pool.
        Used for reference planning independently of monomer TS coverage.

    Returns
    -------
    pandas.DataFrame
        Resolved systems with explicit parent association. Other selection
        metadata is inherited unchanged, rather than recalculated.

    Raises
    ------
    ValueError
        If enumeration exceeds its bound or input contains inconsistent parent
        constitutions or conflicting identity labels. Other unspecified stereo
        is retained and reported in ``attrs['frust_stereoisomers']``.
    """
    if catalyst_stereoisomers not in {'input', 'representative', 'all'}:
        raise ValueError("catalyst_stereoisomers must be 'input', 'representative', or 'all'")
    if catalyst_stereoisomers == 'input' and not racemic:
        return systems.copy()
    rows = []
    parents = {}
    names = {}
    inferred_parents = {}
    for row in systems.itertuples():
        constitution = _constitution(_molecule(str(row.catalyst_smiles)))
        explicit_parent = getattr(row, 'catalyst_parent_name', None)
        if isinstance(explicit_parent, str) and explicit_parent:
            inferred_parents.setdefault(constitution, explicit_parent)
        else:
            inferred_parents.setdefault(constitution, str(row.catalyst_name))
    supplied = {}
    for row in systems.itertuples():
        parent = getattr(row, 'catalyst_parent_name', None)
        if not isinstance(parent, str) or not parent:
            parent = inferred_parents[_constitution(_molecule(str(row.catalyst_smiles)))]
        supplied[(parent, Chem.MolToSmiles(_molecule(str(row.catalyst_smiles))))] = str(row.catalyst_name)
    for _, row in systems.iterrows():
        name, smiles = str(row.catalyst_name), str(row.catalyst_smiles)
        original = _molecule(smiles)
        constitution = _constitution(original)
        parent = row.get('catalyst_parent_name')
        if not isinstance(parent, str) or not parent:
            parent = inferred_parents[constitution]
        parent_smiles = row.get('catalyst_parent_smiles')
        if isinstance(parent_smiles, str) and _constitution(_molecule(parent_smiles)) != constitution:
            raise ValueError(f'Conflicting parent constitution for {name}')
        if parent in parents and parents[parent] != constitution:
            raise ValueError(f'Conflicting parent constitution for {parent}')
        parents[parent] = constitution
        configs = _configurations(smiles, racemic=racemic)
        specified = not any(label == '?' for _, label in Chem.FindMolChiralCenters(original, includeUnassigned=True))
        if catalyst_stereoisomers == 'input' and not racemic:
            configs = {Chem.MolToSmiles(original): original}
        if catalyst_stereoisomers == 'representative' and not specified:
            configs = {key: mol for key, mol in configs.items()
                       if Chem.MolToSmiles(_mirror(mol)) not in configs or key <= Chem.MolToSmiles(_mirror(mol))}
        for canonical, mol in sorted(configs.items()):
            explicit_name = (name if canonical == Chem.MolToSmiles(original) else supplied.get((parent, canonical)))
            if explicit_name is None:
                explicit_name = name if len(configs) == 1 and specified else f'{parent}__{_label(mol)}'
            if explicit_name in names and names[explicit_name] != canonical:
                raise ValueError(f'Conflicting catalyst identity {explicit_name}')
            names[explicit_name] = canonical
            item = row.to_dict()
            item.update(catalyst_name=explicit_name, catalyst_smiles=canonical,
                        catalyst_parent_name=parent, catalyst_parent_smiles=constitution)
            if explicit_name != name:
                item['system_name'] = f'{row.system_name}__{explicit_name[len(parent)+2:]}' if name == parent else f'{row.substrate_name}__{explicit_name}'
            rows.append(item)
    unique = {}
    for row in rows:
        unique.setdefault((row['system_name'], row['catalyst_smiles'], str(row.get('rpos'))), row)
    out = pd.DataFrame(unique.values()).reset_index(drop=True)
    out.attrs.update(systems.attrs)
    out.attrs['frust_stereoisomers'] = {
        'policy': catalyst_stereoisomers, 'racemic': racemic,
        'unresolved_other_stereo': [
            {'catalyst_name': str(row.catalyst_name), 'type': str(info.type),
             'index': int(info.centeredOn)}
            for row in systems.drop_duplicates('catalyst_name').itertuples()
            for info in Chem.FindPotentialStereo(_molecule(str(row.catalyst_smiles)))
            if info.specified == Chem.StereoSpecified.Unspecified
            and (info.type != Chem.StereoType.Atom_Tetrahedral
                 or _molecule(str(row.catalyst_smiles)).GetAtomWithIdx(info.centeredOn).GetAtomicNum() != 6)
        ]}

    return out
