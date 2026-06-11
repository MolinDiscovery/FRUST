"""Read, expand, and generate substrate/catalyst screens."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from frust.tsguess import create_ts_guess_dataframes as create_tsguess_dataframes
from frust.tsguess2 import create_ts_guess_dataframes as create_tsguess2_dataframes

ROLE_ALIASES = {"substrate": "substrate", "sub": "substrate", "catalyst": "catalyst", "cat": "catalyst"}
BASE_COLUMNS = {"role", "smiles", "compound_name", "rpos"}


def read(input_data: str | Path | pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Read a substrate/catalyst screen definition.

    Parameters
    ----------
    input_data : str, pathlib.Path, or pandas.DataFrame
        CSV path or dataframe containing at least ``role`` and ``smiles``.
    strict : bool, optional
        If ``True``, catalyst rows with ``rpos`` values are rejected. Otherwise
        they are warned about and ignored.

    Returns
    -------
    pandas.DataFrame
        Normalized component table.
    """
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)

    missing = sorted({"role", "smiles"} - set(df.columns))
    if missing:
        raise ValueError("screen input is missing required columns: " + ", ".join(missing))

    out = df.copy()
    out["role"] = out["role"].map(_normalize_role)
    if out["role"].isna().any():
        bad = sorted(set(df.loc[out["role"].isna(), "role"].astype(str)))
        raise ValueError(f"Unrecognized screen role values: {bad}")
    if out["smiles"].isna().any():
        raise ValueError("screen input contains missing SMILES values")

    if "rpos" not in out.columns:
        out["rpos"] = pd.NA
    catalyst_rpos = out["role"].eq("catalyst") & out["rpos"].map(lambda value: not _is_missing(value))
    if catalyst_rpos.any():
        message = "Catalyst rows do not use rpos values; ignoring catalyst rpos entries"
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
        out.loc[catalyst_rpos, "rpos"] = pd.NA

    if "compound_name" not in out.columns:
        out["compound_name"] = pd.NA
    out["compound_name"] = _fill_component_names(out)

    columns = ["role", "smiles", "compound_name", "rpos"]
    extra = [col for col in out.columns if col not in columns]
    return out[columns + extra]


def expand(components: pd.DataFrame) -> pd.DataFrame:
    """Expand a normalized screen into substrate-catalyst systems.

    Parameters
    ----------
    components : pandas.DataFrame
        Component table from :func:`read`.

    Returns
    -------
    pandas.DataFrame
        One row per substrate-catalyst system.
    """
    normalized = read(components)
    substrates = normalized[normalized["role"].eq("substrate")]
    catalysts = normalized[normalized["role"].eq("catalyst")]
    if substrates.empty:
        raise ValueError("screen contains no substrate rows")
    if catalysts.empty:
        raise ValueError("screen contains no catalyst rows")

    rows: list[dict[str, Any]] = []
    for _, substrate in substrates.iterrows():
        for _, catalyst in catalysts.iterrows():
            substrate_name = str(substrate["compound_name"])
            catalyst_name = str(catalyst["compound_name"])
            row = {
                "system_name": f"{substrate_name}__{catalyst_name}",
                "substrate_name": substrate_name,
                "catalyst_name": catalyst_name,
                "substrate_smiles": str(substrate["smiles"]),
                "catalyst_smiles": str(catalyst["smiles"]),
                "smiles": str(substrate["smiles"]),
                "rpos": substrate["rpos"],
            }
            row.update(_prefixed_metadata(substrate, "substrate"))
            row.update(_prefixed_metadata(catalyst, "catalyst"))
            rows.append(row)
    return pd.DataFrame(rows)


def create_ts_guesses(
    systems: pd.DataFrame,
    *,
    ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
    n_confs: int | None = 1,
    n_cores: int = 1,
    validate: bool = True,
    backend: str = "tsguess2",
) -> dict[str, pd.DataFrame]:
    """Create grouped TS guess dataframes for an expanded screen.

    Parameters
    ----------
    systems : pandas.DataFrame
        Expanded systems from :func:`expand`.
    ts_types : tuple or list of str, optional
        TS types to generate.
    n_confs : int or None, optional
        Number of embedded conformers per generated TS guess. If ``None``,
        choose a count from the assembled molecule's rotatable-bond count.
    n_cores : int, optional
        RDKit embedding threads.
    validate : bool, optional
        Validate required columns before generation.
    backend : {"tsguess2", "tsguess"}, optional
        TS guess backend. ``"tsguess2"`` uses the SMILES-roundtrip backend.
        ``"tsguess"`` uses the original role-assembly backend.

    Returns
    -------
    dict
        Mapping from TS type to dataframe.
    """
    backend_key = str(backend).strip().lower()
    if backend_key == "tsguess2":
        create = create_tsguess2_dataframes
    elif backend_key == "tsguess":
        create = create_tsguess_dataframes
    else:
        raise ValueError("backend must be one of 'tsguess2' or 'tsguess'")

    return create(
        systems,
        ts_types=ts_types,
        n_confs=n_confs,
        n_cores=n_cores,
        validate=validate,
    )


def _normalize_role(value: object) -> str | None:
    return ROLE_ALIASES.get(str(value).strip().lower())


def _fill_component_names(df: pd.DataFrame) -> pd.Series:
    counters = {"substrate": 0, "catalyst": 0}
    names: list[str] = []
    for _, row in df.iterrows():
        existing = row.get("compound_name")
        if not _is_missing(existing):
            names.append(str(existing))
            continue
        role = str(row["role"])
        names.append(f"{role}_{counters[role]:03d}")
        counters[role] += 1
    return pd.Series(names, index=df.index)


def _prefixed_metadata(row: pd.Series, prefix: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for col, value in row.items():
        if col in BASE_COLUMNS:
            continue
        metadata[f"{prefix}_{col}"] = value
    return metadata


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
