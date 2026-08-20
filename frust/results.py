"""Semantic access to canonical FRUST result columns."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from frust.schema import output_column

ResultProfile = Literal["minimum", "transition_state", "constrained_minimum"]
ResultPurpose = Literal["analysis", "ranking", "optimized", "frequency"]


def result_contract(
    profile: ResultProfile,
    *,
    dft: bool,
    include_terminal_solv_sp: bool = True,
    thermochemistry: Any | None = None,
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
    thermochemistry : ThermochemistrySpec or None, optional
        Explicit molecular free-energy assembly recipe recorded in the result
        contract.

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
    contract = {
        "schema_version": 2,
        "profile": profile,
        "dft": bool(dft),
        "columns": columns,
    }
    if thermochemistry is not None:
        to_dict = getattr(thermochemistry, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("thermochemistry must provide to_dict()")
        contract["thermochemistry"] = to_dict()
        contract["schema_version"] = 3
    return contract


def attach_result_contract(
    df: pd.DataFrame,
    profile: ResultProfile,
    *,
    dft: bool,
    include_terminal_solv_sp: bool = True,
    thermochemistry: Any | None = None,
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
    thermochemistry : ThermochemistrySpec or None, optional
        Explicit molecular free-energy assembly recipe to record.

    Returns
    -------
    pandas.DataFrame
        The same dataframe with ``frust_results`` metadata.
    """
    df.attrs["frust_results"] = result_contract(
        profile,
        dft=dft,
        include_terminal_solv_sp=include_terminal_solv_sp,
        thermochemistry=thermochemistry,
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


def free_energy_components(
    df: pd.DataFrame,
    *,
    thermochemistry: Any | None = None,
) -> pd.DataFrame:
    """Return auditable free-energy components for each dataframe row.

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical DFT workflow result.
    thermochemistry : ThermochemistrySpec or mapping or None, optional
        Explicit recipe. When omitted, use the recipe recorded in the
        dataframe's ``frust_results`` contract.

    Returns
    -------
    pandas.DataFrame
        Component energies in Hartree. ``free_energy_hartree`` is either the
        direct frequency Gibbs energy or the analysis electronic energy plus
        the frequency-stage thermal correction.
    """
    recipe = _thermochemistry_mapping(df, thermochemistry)
    mode = str(recipe.get("mode", "")).strip().lower()
    frequency_ge = get_result(df, "gibbs_energy", purpose="frequency")
    frequency_ee = get_result(df, "electronic_energy", purpose="frequency")
    analysis_ee = get_result(df, "electronic_energy", purpose="analysis")
    thermal = frequency_ge - frequency_ee
    if mode == "frequency_gibbs":
        free_energy = frequency_ge
    elif mode == "electronic_plus_thermal":
        free_energy = analysis_ee + thermal
    else:
        raise ValueError(
            "thermochemistry mode must be 'frequency_gibbs' or "
            "'electronic_plus_thermal'"
        )
    return pd.DataFrame(
        {
            "analysis_electronic_energy_hartree": analysis_ee,
            "frequency_electronic_energy_hartree": frequency_ee,
            "frequency_gibbs_energy_hartree": frequency_ge,
            "thermal_correction_hartree": thermal,
            "free_energy_hartree": free_energy,
            "thermochemistry_mode": mode,
        },
        index=df.index,
    )


def get_free_energy(
    df: pd.DataFrame,
    *,
    thermochemistry: Any | None = None,
) -> pd.Series:
    """Return assembled molecular free energies in Hartree.

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical DFT workflow result.
    thermochemistry : ThermochemistrySpec or mapping or None, optional
        Explicit recipe. When omitted, use the recipe recorded in the result
        contract.

    Returns
    -------
    pandas.Series
        One assembled free energy per dataframe row, in Hartree.
    """
    return free_energy_components(
        df,
        thermochemistry=thermochemistry,
    )["free_energy_hartree"]


def _thermochemistry_mapping(
    df: pd.DataFrame,
    thermochemistry: Any | None,
) -> dict[str, Any]:
    """Resolve an explicit or result-contract thermochemistry mapping."""
    value = thermochemistry
    if value is None:
        contract = df.attrs.get("frust_results", {})
        value = contract.get("thermochemistry") if isinstance(contract, dict) else None
    if value is None:
        raise ValueError(
            "No thermochemistry recipe is recorded; use a MethodPlan with a "
            "ThermochemistrySpec or pass thermochemistry explicitly"
        )
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError("thermochemistry must be a mapping or provide to_dict()")
