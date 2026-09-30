"""Artifact policies and compact scientific result projections."""

from __future__ import annotations

import copy
from typing import Literal

import pandas as pd

from frust.results import COMPACT_FREQUENCY_COLUMN, frequency_values
from frust.schema import normal_termination_columns

ArtifactPolicy = Literal["standard", "screening"]

_IDENTITY_COLUMNS = (
    "structure_id",
    "state_id",
    "state_kind",
    "system_name",
    "substrate_name",
    "catalyst_name",
    "compound_name",
    "custom_name",
    "molecule_role",
    "structure_type",
    "rpos",
    "cid",
    "parent_uma_result_id",
    "charge",
    "multiplicity",
    "smiles",
    "input_smiles",
)
_STRUCTURE_COLUMNS = (
    "atoms",
    "connectivity_bonds",
    "constraint_roles",
    "constraint_spec",
    "ts_spec_id",
    "tsguess_backend",
)


def validate_artifact_policy(value: str) -> ArtifactPolicy:
    """Validate and return an artifact policy.

    Parameters
    ----------
    value : {"standard", "screening"}
        ``"standard"`` keeps the audit-oriented result. ``"screening"``
        retains compact scientific values and final geometry.

    Returns
    -------
    {"standard", "screening"}
        The validated policy.
    """
    if value not in {"standard", "screening"}:
        raise ValueError("artifact_policy must be 'standard' or 'screening'")
    return value  # type: ignore[return-value]


def compact_result_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Project a successful workflow result onto the screening schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Normally terminated workflow result with a ``frust_results`` contract.

    Returns
    -------
    pandas.DataFrame
        Compact result containing identity, final structure, scalar energies,
        full frequency values, termination flags, and provenance attrs. Full
        UMA transition-state rows also retain final mode vectors so their
        reaction coordinate remains reviewable.
    """
    contract = copy.deepcopy(df.attrs.get("frust_results", {}))
    columns = contract.get("columns", {}) if isinstance(contract, dict) else {}
    keep: list[str] = [
        column
        for column in (*_IDENTITY_COLUMNS, *_STRUCTURE_COLUMNS)
        if column in df.columns
    ]

    for purpose in ("analysis", "ranking", "optimized", "frequency"):
        mapping = columns.get(purpose, {}) if isinstance(columns, dict) else {}
        if not isinstance(mapping, dict):
            continue
        for key, column in mapping.items():
            if key == "frequencies":
                continue
            if str(column) in df.columns and str(column) not in keep:
                keep.append(str(column))

    has_frequency_contract = isinstance(columns, dict) and isinstance(
        columns.get("frequency"),
        dict,
    )
    frequency_rows = (
        [frequency_values(row) for _, row in df.iterrows()]
        if has_frequency_contract
        else [[] for _ in range(len(df))]
    )
    has_frequencies = any(frequency_rows)
    keep_uma_modes = (
        contract.get("profile") == "transition_state"
        and contract.get("calculation_level") == "full"
        and contract.get("dft") is False
        and "uma_freq-vibs" in df.columns
    )
    if keep_uma_modes:
        keep.append("uma_freq-vibs")
    if has_frequencies:
        df = df.copy()
        frequency_column = columns["frequency"].get(
            "frequencies", COMPACT_FREQUENCY_COLUMN
        )
        df[frequency_column] = frequency_rows
        keep.append(frequency_column)

    retained_prefixes = {
        column.rsplit("-", 1)[0]
        for column in keep
        if "-" in column
    }
    for column in normal_termination_columns(df):
        if column.rsplit("-", 1)[0] in retained_prefixes and column not in keep:
            keep.append(column)

    compact = df.loc[:, keep].copy()
    compact.attrs = copy.deepcopy(df.attrs)
    if isinstance(contract, dict):
        contract["schema_version"] = max(int(contract.get("schema_version", 0)), 5)
        contract["artifact_policy"] = "screening"
        if has_frequencies:
            frequency_contract = contract.setdefault("columns", {}).setdefault(
                "frequency", {}
            )
            frequency_contract["frequencies"] = frequency_column
        compact.attrs["frust_results"] = contract
    compact.attrs["frust_artifacts"] = {
        "schema_version": 1,
        "policy": "screening",
        "final_geometry": True,
        "frequency_values": bool(has_frequencies),
        "vibration_displacements": keep_uma_modes,
        "calculator_files": False,
    }
    return compact
