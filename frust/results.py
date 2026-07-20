"""Semantic access to canonical FRUST result columns."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from frust.schema import output_column

ResultProfile = Literal["minimum", "transition_state", "constrained_minimum"]
ResultPurpose = Literal["analysis", "ranking", "optimized", "frequency"]


def result_contract(
    profile: ResultProfile,
    *,
    dft: bool,
    include_terminal_solv_sp: bool = True,
) -> dict[str, object]:
    """Return the canonical semantic-column contract for a workflow profile.

    Parameters
    ----------
    profile : {"minimum", "transition_state", "constrained_minimum"}
        Workflow chemistry/result profile.
    dft : bool
        Whether the workflow includes DFT refinement.
    include_terminal_solv_sp : bool, optional
        Whether the DFT workflow includes a final solvent single point. When
        ``False``, the final DFT frequency-stage electronic energy is the
        analysis energy because all DFT stages already include solvent.

    Returns
    -------
    dict
        Versioned mapping from semantic purposes to canonical columns.
    """
    if profile == "minimum":
        ranking_stage = "xtb_opt"
        optimized_stage = "dft_opt" if dft else "xtb_opt"
    elif profile == "transition_state":
        ranking_stage = "dft_rank_sp"
        optimized_stage = "dft_ts_opt" if dft else "xtb_opt"
    elif profile == "constrained_minimum":
        ranking_stage = "dft_rank_sp"
        optimized_stage = "dft_opt" if dft else "xtb_opt"
    else:
        raise ValueError(f"Unknown result profile {profile!r}")
    analysis_stage = (
        "dft_solv_sp"
        if dft and include_terminal_solv_sp
        else "dft_freq"
        if dft
        else ranking_stage
    )
    columns: dict[str, dict[str, str]] = {
        "analysis": {
            "electronic_energy": output_column(analysis_stage, "electronic_energy")
        },
        "ranking": {
            "electronic_energy": output_column(ranking_stage, "electronic_energy")
        },
        "optimized": {"coords": output_column(optimized_stage, "opt_coords")},
    }
    if dft:
        columns["frequency"] = {
            "gibbs_energy": output_column("dft_freq", "gibbs_energy"),
            "electronic_energy": output_column("dft_freq", "electronic_energy"),
        }
    return {
        "schema_version": 2,
        "profile": profile,
        "dft": bool(dft),
        "columns": columns,
    }


def attach_result_contract(
    df: pd.DataFrame,
    profile: ResultProfile,
    *,
    dft: bool,
    include_terminal_solv_sp: bool = True,
) -> pd.DataFrame:
    """Attach compact semantic result metadata to a dataframe in place.

    Parameters
    ----------
    df : pandas.DataFrame
        Workflow result dataframe.
    profile : {"minimum", "transition_state", "constrained_minimum"}
        Workflow chemistry/result profile.
    dft : bool
        Whether the workflow includes DFT refinement.
    include_terminal_solv_sp : bool, optional
        Whether a separate final solvent single point was calculated.

    Returns
    -------
    pandas.DataFrame
        The same dataframe with ``frust_results`` metadata.
    """
    df.attrs["frust_results"] = result_contract(
        profile,
        dft=dft,
        include_terminal_solv_sp=include_terminal_solv_sp,
    )
    return df


def result_column(
    df: pd.DataFrame,
    key: str = "electronic_energy",
    *,
    purpose: ResultPurpose = "analysis",
    require_present: bool = True,
) -> str:
    """Resolve a result column by meaning instead of workflow-specific name.

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical workflow result with ``frust_results`` metadata.
    key : str, optional
        Result meaning, such as ``"electronic_energy"``, ``"gibbs_energy"``,
        or ``"coords"``.
    purpose : {"analysis", "ranking", "optimized", "frequency"}, optional
        Analysis chooses the final comparable energy, ranking chooses the
        explicitly configured cutoff energy, optimized chooses final geometry,
        and frequency chooses thermochemistry outputs.
    require_present : bool, optional
        Raise when the resolved column has not yet been calculated.

    Returns
    -------
    str
        Canonical dataframe column name.
    """
    contract = df.attrs.get("frust_results")
    if not isinstance(contract, dict):
        raise ValueError(
            "dataframe has no canonical result contract; use ft.upgrade_dataframe(...) "
            "for legacy results"
        )
    try:
        column = str(contract["columns"][purpose][key])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"result contract has no {purpose!r} result named {key!r}"
        ) from exc
    if require_present and column not in df.columns:
        raise ValueError(f"canonical result column {column!r} has not been calculated")
    return column


def get_result(
    df: pd.DataFrame,
    key: str = "electronic_energy",
    *,
    purpose: ResultPurpose = "analysis",
) -> pd.Series:
    """Return a semantic result series from a canonical workflow dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical workflow result.
    key : str, optional
        Result meaning to retrieve.
    purpose : {"analysis", "ranking", "optimized", "frequency"}, optional
        Semantic use of the result.

    Returns
    -------
    pandas.Series
        Resolved result values.
    """
    return df[result_column(df, key, purpose=purpose)]
