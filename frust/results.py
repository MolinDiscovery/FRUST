"""Semantic access to canonical FRUST result columns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

import pandas as pd

from frust.schema import output_column

ResultProfile = Literal["minimum", "transition_state", "constrained_minimum"]
ResultPurpose = Literal["analysis", "ranking", "optimized", "frequency"]
COMPACT_FREQUENCY_COLUMN = "dft_freq-frequencies_cm1"


def result_contract(
    profile: ResultProfile,
    *,
    dft: bool,
    calculation_level: str | None = None,
    include_terminal_solv_sp: bool | None = None,
    screening_opt_stage: str = "xtb_opt",
    include_dft_rank_sp: bool = True,
    thermochemistry: Any | None = None,
    full_method: Literal["dft", "uma"] = "dft",
    ranking_stage: str | None = None,
) -> dict[str, object]:
    """Return the canonical semantic-column contract for a workflow profile.

    Parameters
    ----------
    profile : {"minimum", "transition_state", "constrained_minimum"}
        Workflow chemistry/result profile.
    dft : bool
        Whether the workflow includes DFT refinement.
    calculation_level : {"low_cost", "uma_ranked", "dft_ranked", "full"} or None,
        optional
        Explicit workflow depth. ``"low_cost"`` resolves analysis to the
        selected g-xTB or UMA screening optimization energy,
        ``"uma_ranked"`` to a UMA single point on that geometry,
        ``"dft_ranked"`` to the DFT single point on that geometry, and
        ``"full"`` to the final method's energy and frequency results.
    include_terminal_solv_sp : bool or None, optional
        ``True`` selects ``dft_solv_sp`` or ``uma_solv_sp`` analysis energy;
        ``False`` selects the frequency electronic energy. ``None`` defaults
        to ``True`` for DFT and ``False`` for UMA, preserving existing contracts.
    screening_opt_stage : {"xtb_opt", "uma_opt"}, optional
        Optimization stage providing the low-cost energy and geometry.
    include_dft_rank_sp : bool, optional
        Whether a full run includes a separate DFT ranking single point.
    thermochemistry : ThermochemistrySpec or None, optional
        Explicit molecular free-energy assembly recipe recorded for a
        ``"full"`` result. Lower calculation levels have no frequency result,
        so their contracts omit this field.
    full_method : {"dft", "uma"}, optional
        Calculator family used for a ``"full"`` result. ``"uma"`` resolves
        final TS geometry to ``uma_ts_opt``, final minimum geometry to
        ``uma_min_opt``, and thermochemistry to ``uma_freq``.
    ranking_stage : str or None, optional
        Explicit ranking calculator stage for a ``"full"`` result. Use
        ``"uma_rank_sp"`` when UMA reranks g-xTB optimized geometries.
        ``None`` preserves the existing DFT-ranking switch and screening
        optimization behavior.

    Returns
    -------
    dict
        Versioned mapping from semantic purposes to canonical columns.
    """
    if include_terminal_solv_sp is None:
        include_terminal_solv_sp = full_method == "dft"
    if calculation_level is None:
        calculation_level = (
            "full"
            if dft
            else (
                "dft_ranked"
                if profile in {"transition_state", "constrained_minimum"}
                else "low_cost"
            )
        )
    calculation_level = str(calculation_level).strip().lower()
    if calculation_level not in {"low_cost", "uma_ranked", "dft_ranked", "full"}:
        raise ValueError(
            "calculation_level must be 'low_cost', 'uma_ranked', "
            "'dft_ranked', or 'full'"
        )
    if full_method not in {"dft", "uma"}:
        raise ValueError("full_method must be 'dft' or 'uma'")
    if ranking_stage not in {None, "dft_rank_sp", "uma_rank_sp"}:
        raise ValueError("ranking_stage must be 'dft_rank_sp' or 'uma_rank_sp'")
    if calculation_level == "uma_ranked" and ranking_stage == "dft_rank_sp":
        raise ValueError("uma_ranked requires a UMA ranking stage")
    if calculation_level == "dft_ranked" and ranking_stage == "uma_rank_sp":
        raise ValueError("dft_ranked requires a DFT ranking stage")
    has_full = calculation_level == "full"
    has_full_dft = has_full and full_method == "dft"
    if has_full_dft and ranking_stage == "uma_rank_sp" and include_dft_rank_sp:
        raise ValueError(
            "UMA ranking for a full DFT result requires include_dft_rank_sp=False"
        )
    if screening_opt_stage not in {"xtb_opt", "uma_opt"}:
        raise ValueError("screening_opt_stage must be 'xtb_opt' or 'uma_opt'")
    if calculation_level == "uma_ranked":
        resolved_ranking_stage = "uma_rank_sp"
    elif calculation_level == "dft_ranked":
        resolved_ranking_stage = "dft_rank_sp"
    elif ranking_stage is not None:
        resolved_ranking_stage = ranking_stage
    elif has_full_dft and include_dft_rank_sp:
        resolved_ranking_stage = "dft_rank_sp"
    else:
        resolved_ranking_stage = screening_opt_stage
    if profile == "minimum":
        optimized_stage = (
            ("dft_opt" if has_full_dft else "uma_min_opt")
            if has_full else screening_opt_stage
        )
    elif profile == "transition_state":
        optimized_stage = (
            ("dft_ts_opt" if has_full_dft else "uma_ts_opt")
            if has_full else screening_opt_stage
        )
    elif profile == "constrained_minimum":
        optimized_stage = (
            ("dft_opt" if has_full_dft else "uma_min_opt")
            if has_full else screening_opt_stage
        )
    else:
        raise ValueError(f"Unknown result profile {profile!r}")
    frequency_stage = "dft_freq" if has_full_dft else "uma_freq"
    analysis_stage = (
        ("dft_solv_sp" if has_full_dft else "uma_solv_sp")
        if has_full and include_terminal_solv_sp
        else frequency_stage if has_full else resolved_ranking_stage
    )
    columns: dict[str, dict[str, str]] = {
        "analysis": {
            "electronic_energy": output_column(analysis_stage, "electronic_energy")
        },
        "ranking": {
            "electronic_energy": output_column(
                resolved_ranking_stage, "electronic_energy"
            )
        },
        "optimized": {"coords": output_column(optimized_stage, "opt_coords")},
    }
    if has_full:
        columns["frequency"] = {
            "gibbs_energy": output_column(frequency_stage, "gibbs_energy"),
            "electronic_energy": output_column(frequency_stage, "electronic_energy"),
            "frequencies": f"{frequency_stage}-frequencies_cm1",
        }
    contract = {
        "schema_version": (
            5
            if full_method == "dft"
            and calculation_level != "uma_ranked"
            and ranking_stage is None
            else 6
        ),
        "profile": profile,
        "dft": has_full_dft,
        "calculation_level": calculation_level,
        "columns": columns,
    }
    if has_full and thermochemistry is not None:
        to_dict = getattr(thermochemistry, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("thermochemistry must provide to_dict()")
        contract["thermochemistry"] = to_dict()
    return contract


def attach_result_contract(
    df: pd.DataFrame,
    profile: ResultProfile,
    *,
    dft: bool,
    calculation_level: str | None = None,
    include_terminal_solv_sp: bool | None = None,
    screening_opt_stage: str = "xtb_opt",
    include_dft_rank_sp: bool = True,
    thermochemistry: Any | None = None,
    full_method: Literal["dft", "uma"] = "dft",
    ranking_stage: str | None = None,
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
    calculation_level : {"low_cost", "uma_ranked", "dft_ranked", "full"} or None,
        optional
        Explicit workflow depth recorded in the canonical contract.
    include_terminal_solv_sp : bool or None, optional
        Whether a separate final solvent single point was calculated.
        ``None`` defaults to ``True`` for DFT and ``False`` for UMA.
    screening_opt_stage : {"xtb_opt", "uma_opt"}, optional
        Screening optimization stage used for lower-tier result columns.
    include_dft_rank_sp : bool, optional
        Whether a full run includes the DFT ranking single point.
    thermochemistry : ThermochemistrySpec or None, optional
        Explicit molecular free-energy assembly recipe to record.
    full_method : {"dft", "uma"}, optional
        Calculator family used for final geometries and frequencies.
    ranking_stage : str or None, optional
        Explicit ranking stage, such as ``"uma_rank_sp"`` for a hybrid run.

    Returns
    -------
    pandas.DataFrame
        The same dataframe with ``frust_results`` metadata.
    """
    df.attrs["frust_results"] = result_contract(
        profile,
        dft=dft,
        calculation_level=calculation_level,
        include_terminal_solv_sp=include_terminal_solv_sp,
        screening_opt_stage=screening_opt_stage,
        include_dft_rank_sp=include_dft_rank_sp,
        thermochemistry=thermochemistry,
        full_method=full_method,
        ranking_stage=ranking_stage,
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


def frequency_values(row: Mapping[str, Any] | pd.Series) -> list[float]:
    """Return vibration frequencies without requiring displacement vectors.

    Parameters
    ----------
    row : mapping or pandas.Series
        One result row. Screening rows use
        ``dft_freq-frequencies_cm1`` or ``uma_freq-frequencies_cm1``;
        standard rows may instead contain a full ``*-vibs`` normal-mode
        column.

    Returns
    -------
    list of float
        Frequencies in inverse centimetres, preserving dataframe order.
    """
    compact_columns = [
        column
        for column in row.keys()
        if str(column) == COMPACT_FREQUENCY_COLUMN
        or str(column).endswith("-frequencies_cm1")
    ]
    for column in reversed(compact_columns):
        value = row[column]
        if isinstance(value, (list, tuple, np.ndarray)):
            return [float(item) for item in value]

    vibration_columns = [
        column for column in row.keys() if str(column).endswith("-vibs")
    ]
    for column in reversed(vibration_columns):
        value = row[column]
        if not isinstance(value, (list, tuple, np.ndarray)):
            continue
        frequencies: list[float] = []
        for mode in value:
            if isinstance(mode, Mapping) and mode.get("frequency") is not None:
                frequencies.append(float(mode["frequency"]))
        if frequencies:
            return frequencies
    return []


def free_energy_components(
    df: pd.DataFrame,
    *,
    thermochemistry: Any | None = None,
) -> pd.DataFrame:
    """Return auditable free-energy components for each dataframe row.

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical full workflow result with frequency thermochemistry.
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
    frequency_ge_column = result_column(
        df, "gibbs_energy", purpose="frequency", require_present=False
    )
    frequency_ee_column = result_column(
        df, "electronic_energy", purpose="frequency", require_present=False
    )
    frequency_ge = pd.to_numeric(
        df.get(frequency_ge_column, pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    frequency_ee = pd.to_numeric(
        df.get(frequency_ee_column, pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    analysis_ee = pd.to_numeric(
        get_result(df, "electronic_energy", purpose="analysis"), errors="coerce"
    )
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
