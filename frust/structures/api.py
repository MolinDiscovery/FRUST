"""Calculation-free public structure-generation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from frust.schema import stamp_schema
from frust.structures.builders import build
from frust.structures.models import StructureTarget
from frust.structures.planner import molecule_states, normalize_systems, plan_targets
from frust.utils.dataframes import merge_dataframe_attrs

_CANONICAL_STRUCTURE_COLUMNS = (
    "system_name",
    "state_id",
    "state_kind",
    "rpos",
    "atoms",
    "coords_embedded",
)


def create_mols(
    systems: str | Path | pd.DataFrame,
    *,
    states: str | Iterable[str] = "all",
    n_confs: int | None = 1,
    n_cores: int = 1,
) -> pd.DataFrame:
    """Create embedded catalytic-cycle molecule structures without calculations.

    Parameters
    ----------
    systems : str, pathlib.Path, or pandas.DataFrame
        Expanded systems from :func:`frust.screen.expand`, a component table
        accepted by :func:`frust.screen.read`, or a compatible CSV path.
    states : str or iterable of str, optional
        Molecule states to generate. Accepted individual states are
        ``"dimer"``, ``"HH"``, ``"ligand"``, ``"catalyst"``, ``"int1"``,
        ``"int2"``, ``"HBpin-ligand"``, and ``"HBpin-mol"``. The shortcuts
        ``"all"``, ``"uniques"``, and ``"generics"`` have the same meanings
        as ``ft.workflows.mols(..., select_mols=...)``.
    n_confs : int or None, optional
        Number of embedded conformers per target. If ``None``, use FRUST's
        rotatable-bond heuristic.
    n_cores : int, optional
        RDKit embedding threads.

    Returns
    -------
    pandas.DataFrame
        Canonical embedded structures. The dataframe contains
        ``system_name``, ``state_id``, ``state_kind``, ``rpos``, ``atoms``,
        and ``coords_embedded`` and contains no xTB or DFT result columns.

    Examples
    --------
    >>> import frust as ft
    >>> systems = ft.screen.expand(ft.screen.read("screen.csv"))
    >>> mols = ft.structures.create_mols(
    ...     systems,
    ...     states=["HH", "int1", "int2"],
    ...     n_confs=1,
    ... )
    >>> mols[["system_name", "state_id", "rpos", "cid"]]

    Notes
    -----
    This function plans :class:`StructureTarget` objects and delegates to the
    same typed builder used by ``ft.workflows.mols(...).run()``. It performs
    structure generation and embedding only.
    """
    normalized = normalize_systems(systems)
    targets = plan_targets(normalized, states=molecule_states(states))
    return _create_from_targets(
        targets,
        n_confs=n_confs,
        n_cores=n_cores,
        source="frust.structures.create_mols",
    )


def create_int3_guesses(
    systems: str | Path | pd.DataFrame,
    *,
    n_confs: int | None = 1,
    n_cores: int = 1,
    spec_profile: str = "wb97xd3-631g/gas",
    spec_match: str = "prefer-exact",
) -> pd.DataFrame:
    """Create embedded INT3 guesses without running calculations.

    Parameters
    ----------
    systems : str, pathlib.Path, or pandas.DataFrame
        Expanded systems from :func:`frust.screen.expand`, a component table
        accepted by :func:`frust.screen.read`, or a compatible CSV path.
    n_confs : int or None, optional
        Number of embedded conformers per INT3 target. If ``None``, use the
        connected-graph builder's conformer-count heuristic.
    n_cores : int, optional
        RDKit embedding threads.
    spec_profile : str, optional
        Method/environment geometry profile.
    spec_match : {"prefer-exact", "exact"}, optional
        Geometry-profile matching policy.

    Returns
    -------
    pandas.DataFrame
        Canonical embedded INT3 structures with ``state_id="INT3"`` and
        ``state_kind="constrained_minimum"``. No xTB or DFT stages are run.

    Examples
    --------
    >>> import frust as ft
    >>> systems = ft.screen.expand(ft.screen.read("screen.csv"))
    >>> int3 = ft.structures.create_int3_guesses(systems, n_confs=1)
    >>> int3[["system_name", "state_id", "state_kind", "rpos", "cid"]]

    Notes
    -----
    This function delegates to the same typed INT3 target builder used by
    ``ft.workflows.int3(...).run()``.
    """
    normalized = normalize_systems(systems)
    targets = plan_targets(normalized, states=["INT3"])
    return _create_from_targets(
        targets,
        n_confs=n_confs,
        n_cores=n_cores,
        source="frust.structures.create_int3_guesses",
        spec_profile=spec_profile,
        spec_match=spec_match,
    )


def _create_from_targets(
    targets: Sequence[StructureTarget] | Iterable[StructureTarget],
    *,
    n_confs: int | None,
    n_cores: int,
    source: str,
    spec_profile: str = "wb97xd3-631g/gas",
    spec_match: str = "prefer-exact",
) -> pd.DataFrame:
    """Build and concatenate typed targets without executing workflow stages."""
    target_list = list(targets)
    if n_confs is not None and int(n_confs) < 1:
        raise ValueError("n_confs must be at least 1 or None")
    if int(n_cores) < 1:
        raise ValueError("n_cores must be at least 1")
    if not all(isinstance(target, StructureTarget) for target in target_list):
        raise TypeError("structure generation requires typed StructureTarget objects")

    frames = []
    for target in target_list:
        build_kwargs = {
            "n_confs": n_confs,
            "n_cores": int(n_cores),
            "memory_gb": 4,
            "debug": False,
        }
        if target.builder_spec.startswith("connected_graph::"):
            build_kwargs.update(
                {"spec_profile": spec_profile, "spec_match": spec_match}
            )
        frames.append(build(target, **build_kwargs))
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.attrs.update(
            merge_dataframe_attrs(
                frames,
                source_files=[target.target_id for target in target_list],
            )
        )
    else:
        out = pd.DataFrame(columns=list(_CANONICAL_STRUCTURE_COLUMNS))

    missing = [column for column in _CANONICAL_STRUCTURE_COLUMNS if column not in out]
    if missing:
        raise RuntimeError(
            "typed structure builder omitted canonical columns: " + ", ".join(missing)
        )

    generation_metadata = {
        "schema_version": 1,
        "source": source,
        "calculation_free": True,
        "requested_n_confs": n_confs,
        "n_cores": int(n_cores),
        "n_targets": len(target_list),
        "states": list(dict.fromkeys(target.state_id for target in target_list)),
    }
    if any(target.builder_spec.startswith("connected_graph::") for target in target_list):
        generation_metadata.update(
            {"spec_profile": spec_profile, "spec_match": spec_match}
        )
    out.attrs["frust_structure_generation"] = generation_metadata
    stamp_schema(out)
    return out
