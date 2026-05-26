"""Embedding helpers for assembled TS guess molecules."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom
from rdkit.Geometry.rdGeometry import Point3D


def embed_with_coord_map(
    mol: Chem.Mol,
    coord_map: dict[int, tuple[float, float, float]],
    *,
    n_confs: int = 1,
    n_cores: int = 1,
    allowed_contact_pairs: set[tuple[int, int]] | None = None,
) -> tuple[Chem.Mol, list[int]]:
    """Embed conformers while fixing role atoms to TS-core coordinates.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with explicit hydrogens.
    coord_map : dict
        Mapping from atom index to Cartesian coordinates.
    n_confs : int, optional
        Number of conformers to embed.
    n_cores : int, optional
        Number of RDKit threads.
    allowed_contact_pairs : set of tuple of int, optional
        Nonbonded atom pairs expected to be close in the TS core.

    Returns
    -------
    tuple
        Embedded molecule and conformer IDs.
    """
    working = Chem.Mol(mol)
    requested_confs = int(n_confs)
    candidate_confs = requested_confs
    rdkit_coord_map = {
        int(idx): Point3D(float(x), float(y), float(z))
        for idx, (x, y, z) in coord_map.items()
    }
    use_coord_map = len(Chem.GetMolFrags(working)) == 1
    allowed_pairs = _normalize_pairs(allowed_contact_pairs or set())

    with rdBase.BlockLogs():
        cids = list(
            rdDistGeom.EmbedMultipleConfs(
                working,
                numConfs=candidate_confs,
                maxAttempts=0,
                randomSeed=0xF00D,
                useRandomCoords=False,
                pruneRmsThresh=-1.0,
                coordMap=rdkit_coord_map if use_coord_map else {},
                ignoreSmoothingFailures=True,
                enforceChirality=True,
                useSmallRingTorsions=True,
                numThreads=int(n_cores),
            )
        )
    if not cids:
        with rdBase.BlockLogs():
            cids = list(
                rdDistGeom.EmbedMultipleConfs(
                    working,
                    numConfs=candidate_confs,
                    maxAttempts=0,
                    randomSeed=0xF00D,
                    useRandomCoords=True,
                    pruneRmsThresh=-1.0,
                    coordMap=rdkit_coord_map if use_coord_map else {},
                    ignoreSmoothingFailures=True,
                    enforceChirality=True,
                    useSmallRingTorsions=True,
                    numThreads=int(n_cores),
                )
            )
    if not cids:
        raise ValueError("RDKit embedding produced no conformers for the TS guess")
    if not use_coord_map:
        _place_fragments_by_anchors(working, cids, rdkit_coord_map, allowed_pairs)
        cids = _rank_conformers_by_clashes(working, cids, allowed_pairs)[:requested_confs]
    return working, cids


def _place_fragments_by_anchors(
    mol: Chem.Mol,
    cids: list[int],
    coord_map: dict[int, Point3D],
    allowed_contact_pairs: set[tuple[int, int]],
) -> None:
    """Rigidly place disconnected fragments from role-atom anchors.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule containing one or more disconnected fragments.
    cids : list of int
        Conformer IDs to update in place.
    coord_map : dict
        Mapping from atom index to target coordinates.
    allowed_contact_pairs : set of tuple of int
        Nonbonded atom pairs expected to be close in the TS core.
    """
    fragments = tuple(tuple(int(idx) for idx in fragment) for fragment in Chem.GetMolFrags(mol))
    targets = {
        int(idx): np.array([point.x, point.y, point.z], dtype=float)
        for idx, point in coord_map.items()
    }
    anchors_by_fragment = tuple(
        tuple(idx for idx in fragment if idx in targets)
        for fragment in fragments
    )
    for cid in cids:
        conf = mol.GetConformer(int(cid))
        all_coords = _conformer_coordinates(conf, mol.GetNumAtoms())
        for fragment, anchors in zip(fragments, anchors_by_fragment):
            if not anchors:
                continue
            source = np.array([all_coords[idx] for idx in anchors], dtype=float)
            target = np.array([targets[idx] for idx in anchors], dtype=float)
            rotation, translation = _rigid_transform(source, target)
            for atom_idx in fragment:
                placed = all_coords[int(atom_idx)] @ rotation.T + translation
                conf.SetAtomPosition(
                    int(atom_idx),
                    Point3D(float(placed[0]), float(placed[1]), float(placed[2])),
                )
        _snap_anchor_positions(conf, coord_map)
        placed_coords = _conformer_coordinates(conf, mol.GetNumAtoms())
        placed_coords = _relieve_anchor_preserving_clashes(
            mol,
            placed_coords,
            fragments,
            anchors_by_fragment,
            allowed_contact_pairs,
        )
        _set_conformer_coordinates(conf, placed_coords)
        _snap_anchor_positions(conf, coord_map)


def _conformer_coordinates(conf: Chem.Conformer, n_atoms: int) -> np.ndarray:
    """Return conformer coordinates as a dense NumPy array.

    Parameters
    ----------
    conf : rdkit.Chem.Conformer
        Conformer to read.
    n_atoms : int
        Number of atoms to extract.

    Returns
    -------
    numpy.ndarray
        Coordinate array with shape ``(n_atoms, 3)``.
    """
    return np.array(
        [
            [conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y, conf.GetAtomPosition(idx).z]
            for idx in range(n_atoms)
        ],
        dtype=float,
    )


def _set_conformer_coordinates(conf: Chem.Conformer, coords: np.ndarray) -> None:
    """Set conformer coordinates from a dense NumPy array.

    Parameters
    ----------
    conf : rdkit.Chem.Conformer
        Conformer to update.
    coords : numpy.ndarray
        Coordinate array with shape ``(n_atoms, 3)``.
    """
    for atom_idx, coord in enumerate(coords):
        conf.SetAtomPosition(
            int(atom_idx),
            Point3D(float(coord[0]), float(coord[1]), float(coord[2])),
        )


def _snap_anchor_positions(conf: Chem.Conformer, coord_map: dict[int, Point3D]) -> None:
    """Set anchor atom coordinates exactly to their target positions.

    Parameters
    ----------
    conf : rdkit.Chem.Conformer
        Conformer to update.
    coord_map : dict
        Mapping from atom index to target coordinates.
    """
    for atom_idx, position in coord_map.items():
        conf.SetAtomPosition(int(atom_idx), position)


def _relieve_anchor_preserving_clashes(
    mol: Chem.Mol,
    coords: np.ndarray,
    fragments: tuple[tuple[int, ...], ...],
    anchors_by_fragment: tuple[tuple[int, ...], ...],
    allowed_contact_pairs: set[tuple[int, int]],
) -> np.ndarray:
    """Rotate anchored fragments to reduce inter-fragment clashes.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule whose fragments are being placed.
    coords : numpy.ndarray
        Current coordinates.
    fragments : tuple of tuple of int
        Atom indices for each disconnected fragment.
    anchors_by_fragment : tuple of tuple of int
        Anchor atom indices for each fragment.
    allowed_contact_pairs : set of tuple of int
        Nonbonded atom pairs expected to be close in the TS core.

    Returns
    -------
    numpy.ndarray
        Clash-relieved coordinates.
    """
    out = np.array(coords, dtype=float, copy=True)
    rotatable_fragments = [
        (frag_idx, fragment, anchors)
        for frag_idx, (fragment, anchors) in enumerate(zip(fragments, anchors_by_fragment))
        if 1 <= len(anchors) <= 2 and len(fragment) > len(anchors)
    ]
    if not rotatable_fragments:
        return out

    radii = _covalent_radii(mol)
    ordered_fragments = sorted(rotatable_fragments, key=lambda item: -len(item[1]))
    for _ in range(2):
        for frag_idx, fragment, anchors in ordered_fragments:
            fragment_indices = np.array(fragment, dtype=int)
            other_indices = np.array(
                [
                    atom_idx
                    for other_idx, other_fragment in enumerate(fragments)
                    if other_idx != frag_idx
                    for atom_idx in other_fragment
                ],
                dtype=int,
            )
            origin = out[int(anchors[0])].copy()
            relative = out[fragment_indices] - origin
            best_score: tuple[float, int] | None = None
            best_coords: np.ndarray | None = None
            for rotation in _anchor_preserving_rotations(out, anchors):
                candidate = relative @ rotation.T + origin
                score = _clash_score(
                    candidate,
                    fragment_indices,
                    out,
                    other_indices,
                    radii,
                    allowed_contact_pairs,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_coords = candidate
            if best_coords is not None:
                out[fragment_indices] = best_coords
    return out


def _anchor_preserving_rotations(
    coords: np.ndarray,
    anchors: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Return rotations that preserve one or two anchor positions.

    Parameters
    ----------
    coords : numpy.ndarray
        Current coordinates.
    anchors : tuple of int
        One or two anchor atom indices.

    Returns
    -------
    tuple of numpy.ndarray
        Candidate rotations.
    """
    if len(anchors) == 1:
        return _rotation_candidates()
    if len(anchors) != 2:
        return (np.eye(3),)
    axis = coords[int(anchors[1])] - coords[int(anchors[0])]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return (np.eye(3),)
    unit_axis = axis / norm
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    return tuple(_rotation_around_axis(unit_axis, angle) for angle in angles)


def _covalent_radii(mol: Chem.Mol) -> np.ndarray:
    """Return covalent radii for all atoms in a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to inspect.

    Returns
    -------
    numpy.ndarray
        Covalent radius for each atom.
    """
    periodic_table = Chem.GetPeriodicTable()
    return np.array(
        [periodic_table.GetRcovalent(atom.GetAtomicNum()) for atom in mol.GetAtoms()],
        dtype=float,
    )


def _clash_score(
    candidate: np.ndarray,
    fragment_indices: np.ndarray,
    all_coords: np.ndarray,
    other_indices: np.ndarray,
    radii: np.ndarray,
    allowed_contact_pairs: set[tuple[int, int]],
) -> tuple[float, int]:
    """Score close contacts between one candidate fragment and all others.

    Parameters
    ----------
    candidate : numpy.ndarray
        Candidate coordinates for the moving fragment.
    fragment_indices : numpy.ndarray
        Atom indices in the moving fragment.
    all_coords : numpy.ndarray
        Current coordinates for all atoms.
    other_indices : numpy.ndarray
        Atom indices outside the moving fragment.
    radii : numpy.ndarray
        Covalent radii by atom index.
    allowed_contact_pairs : set of tuple of int
        Nonbonded atom pairs expected to be close in the TS core.

    Returns
    -------
    tuple
        Weighted clash score and number of close contacts.
    """
    score = 0.0
    close_contacts = 0
    for local_idx, atom_idx in enumerate(fragment_indices):
        pair_allowed = np.array(
            [
                _normalize_pair(int(atom_idx), int(other_idx)) in allowed_contact_pairs
                for other_idx in other_indices
            ],
            dtype=bool,
        )
        scored_indices = other_indices[~pair_allowed]
        if len(scored_indices) == 0:
            continue
        cutoff = 1.15 * (radii[int(atom_idx)] + radii[scored_indices])
        distances = np.linalg.norm(candidate[local_idx] - all_coords[scored_indices], axis=1)
        overlap = cutoff - distances
        clashing = overlap > 0.0
        if not np.any(clashing):
            continue
        close_contacts += int(np.count_nonzero(clashing))
        score += float(np.sum(overlap[clashing] ** 2))
        severe = distances < 0.7
        if np.any(severe):
            score += float(np.sum(10.0 + 100.0 * (0.7 - distances[severe]) ** 2))
    return score, close_contacts


def _rank_conformers_by_clashes(
    mol: Chem.Mol,
    cids: list[int],
    allowed_contact_pairs: set[tuple[int, int]],
) -> list[int]:
    """Rank conformers from lowest to highest unexpected close-contact score.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule containing embedded conformers.
    cids : list of int
        Conformer IDs to rank.
    allowed_contact_pairs : set of tuple of int
        Nonbonded atom pairs expected to be close in the TS core.

    Returns
    -------
    list of int
        Ranked conformer IDs.
    """
    scores = [
        (
            _conformer_clash_score(
                mol,
                _conformer_coordinates(mol.GetConformer(int(cid)), mol.GetNumAtoms()),
                allowed_contact_pairs,
            ),
            int(cid),
        )
        for cid in cids
    ]
    return [cid for _, cid in sorted(scores, key=lambda item: item[0])]


def _conformer_clash_score(
    mol: Chem.Mol,
    coords: np.ndarray,
    allowed_contact_pairs: set[tuple[int, int]],
) -> tuple[float, int]:
    """Score unexpected close contacts across one conformer.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to score.
    coords : numpy.ndarray
        Conformer coordinates.
    allowed_contact_pairs : set of tuple of int
        Nonbonded atom pairs expected to be close in the TS core.

    Returns
    -------
    tuple
        Weighted clash score and number of close contacts.
    """
    radii = _covalent_radii(mol)
    bonded_pairs = {
        _normalize_pair(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    }
    score = 0.0
    close_contacts = 0
    for atom_i in range(mol.GetNumAtoms()):
        for atom_j in range(atom_i + 1, mol.GetNumAtoms()):
            pair = (atom_i, atom_j)
            if pair in bonded_pairs or pair in allowed_contact_pairs:
                continue
            distance = float(np.linalg.norm(coords[atom_i] - coords[atom_j]))
            cutoff = 1.15 * (radii[atom_i] + radii[atom_j])
            overlap = cutoff - distance
            if overlap <= 0.0:
                continue
            close_contacts += 1
            score += overlap**2
            if distance < 0.7:
                score += 10.0 + 100.0 * (0.7 - distance) ** 2
    return score, close_contacts


def _normalize_pairs(pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Return atom-index pairs in sorted order.

    Parameters
    ----------
    pairs : set of tuple of int
        Atom-index pairs.

    Returns
    -------
    set of tuple of int
        Normalized atom-index pairs.
    """
    return {_normalize_pair(left, right) for left, right in pairs}


def _normalize_pair(left: int, right: int) -> tuple[int, int]:
    """Return one atom-index pair in sorted order.

    Parameters
    ----------
    left : int
        First atom index.
    right : int
        Second atom index.

    Returns
    -------
    tuple of int
        Sorted atom-index pair.
    """
    return (int(left), int(right)) if int(left) < int(right) else (int(right), int(left))


@lru_cache(maxsize=1)
def _rotation_candidates() -> tuple[np.ndarray, ...]:
    """Return deterministic rotation candidates for clash relief.

    Returns
    -------
    tuple of numpy.ndarray
        Candidate 3D rotation matrices.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    rotations: dict[tuple[float, ...], np.ndarray] = {}
    for alpha in angles:
        for beta in angles:
            for gamma in angles:
                rotation = _rotation_z(gamma) @ _rotation_y(beta) @ _rotation_x(alpha)
                key = tuple(np.round(rotation, 12).ravel())
                rotations[key] = rotation
    return tuple(rotations.values())


def _rotation_x(angle: float) -> np.ndarray:
    """Return a rotation matrix around the x-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_y(angle: float) -> np.ndarray:
    """Return a rotation matrix around the y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _rotation_z(angle: float) -> np.ndarray:
    """Return a rotation matrix around the z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _rotation_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a rotation matrix around an arbitrary unit vector.

    Parameters
    ----------
    axis : numpy.ndarray
        Unit vector defining the rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    x, y, z = (float(value) for value in axis)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ]
    )


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the least-squares rigid transform from source to target points.

    Parameters
    ----------
    source : numpy.ndarray
        Source coordinates with shape ``(n_points, 3)``.
    target : numpy.ndarray
        Target coordinates with shape ``(n_points, 3)``.

    Returns
    -------
    tuple
        Rotation matrix and translation vector. Coordinates are transformed as
        ``coords @ rotation.T + translation``.
    """
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_shifted = source - source_center
    target_shifted = target - target_center
    if len(source) == 1 or np.linalg.norm(source_shifted) < 1e-12:
        rotation = np.eye(3)
    else:
        covariance = source_shifted.T @ target_shifted
        left, _, right_t = np.linalg.svd(covariance)
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1, :] *= -1.0
            rotation = right_t.T @ left.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation
