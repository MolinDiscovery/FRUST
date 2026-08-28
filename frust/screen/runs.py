"""Portable catalyst-screen run analysis and scientific review."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from frust.results import free_energy_components, get_result
from frust.schema import normal_termination_columns
from frust.screen._quality import minimum_vibration_status
from frust.structures.specs import DIMER_STATES


HARTREE_TO_KCAL_MOL = 627.5094740631
DIMER_TIE_TOLERANCE_KCAL_MOL = 0.01
QUALITY_ORDER = {"ready": 0, "review": 1, "invalid": 2, "incomplete": 3}
PROFILE_ORDER = ["Dimer", "Cat", "TS1", "int1", "TS2", "int2", "TS3", "INT3", "TS4", "Product"]
PROFILE_TERMS: dict[str, dict[str, float]] = {
    "Dimer": {"dimer": 0.5, "ligand": 1.0, "HBpin-mol": 1.0},
    "Cat": {"catalyst": 1.0, "ligand": 1.0, "HBpin-mol": 1.0},
    "TS1": {"TS1": 1.0, "HBpin-mol": 1.0},
    "int1": {"int1": 1.0, "HBpin-mol": 1.0},
    "TS2": {"TS2": 1.0, "HBpin-mol": 1.0},
    "int2": {"int2": 1.0, "HH": 1.0, "HBpin-mol": 1.0},
    "TS3": {"TS3": 1.0, "HH": 1.0},
    "INT3": {"INT3": 1.0, "HH": 1.0},
    "TS4": {"TS4": 1.0, "HH": 1.0},
    "Product": {"dimer": 0.5, "HBpin-ligand": 1.0, "HH": 1.0},
}


def open_run(path: str | Path) -> "ScreenRun":
    """Open a portable catalyst-screen result bundle.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory containing ``manifest.json`` and the calculation bundle.

    Returns
    -------
    ScreenRun
        Lazy analysis interface independent of the original cluster paths.
    """
    return ScreenRun(path)


class ScreenRun:
    """Portable catalyst-screen analysis and review interface."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.manifest_path = self.path / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Catalyst-screen manifest not found: {self.manifest_path}")
        self.analysis_dir = self.path / "analysis"
        self.reviews_path = self.analysis_dir / "reviews.csv"

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the portable run manifest."""
        payload = json.loads(self.manifest_path.read_text())
        if payload.get("run_type") != "catalyst_screen":
            raise ValueError(f"Unsupported run type {payload.get('run_type')!r}")
        return payload

    def refresh_analysis(self) -> "ScreenRun":
        """Rebuild all derived tables from portable calculation results."""
        build_analysis(self.path)
        return self

    def summary(self) -> pd.DataFrame:
        """Return compact completion and quality counts."""
        self._ensure_analysis()
        states = self.states()
        barriers = self.barriers()
        rows: list[dict[str, Any]] = []
        for artifact, table in (("states", states), ("barriers", barriers)):
            counts = table.get("quality_status", pd.Series(dtype=str)).value_counts()
            for status in ("ready", "review", "invalid", "incomplete"):
                rows.append(
                    {
                        "artifact": artifact,
                        "quality_status": status,
                        "count": int(counts.get(status, 0)),
                        "calculation_level": self.manifest.get(
                            "calculation_level", "full"
                        ),
                        "screening": self.manifest.get("screening", {}).get("name"),
                        "method": self.manifest.get("method", {}).get("name"),
                        "ranking_solvation": _solvation_label(
                            self.manifest.get("ranking_solvation", {})
                        ),
                        "electronic_barriers": True,
                        "gibbs_barriers": self.manifest.get(
                            "calculation_level", "full"
                        ) == "full",
                    }
                )
        return pd.DataFrame(rows)

    def available_analysis_levels(self) -> tuple[str, ...]:
        """Return independently selected result tiers stored in this run."""
        return tuple(
            str(level)
            for level in self.manifest.get(
                "analysis_levels",
                [self.manifest.get("calculation_level", "full")],
            )
        )

    def states(self, *, level: str | None = None) -> pd.DataFrame:
        """Return one auditable row per calculated or expected state.

        Parameters
        ----------
        level : {"low_cost", "dft_ranked", "full"} or None, optional
            Nested analysis tier. ``None`` returns the terminal level requested
            when the workflow was submitted.
        """
        self._ensure_analysis()
        return self._level_table("states", level=level)

    def dimer_references(self, *, level: str | None = None) -> pd.DataFrame:
        """Return dimer candidates and the selected reference per catalyst.

        Parameters
        ----------
        level : {"low_cost", "dft_ranked", "full"} or None, optional
            Nested analysis tier. Each tier performs its own conformer and
            dimer-topology selection.

        Returns
        -------
        pandas.DataFrame
            One row per candidate topology and catalyst. ``selected`` marks
            the exact result row used by barrier and profile equations.
            ``selection_quality_status`` applies to the complete per-catalyst
            selection and is ``"incomplete"`` or ``"invalid"`` when strict
            ``dimer_reference="lowest"`` selection cannot be made.
        """
        self._ensure_analysis()
        return self._level_table("dimer_references", level=level)

    def barriers(self, *, level: str | None = None) -> pd.DataFrame:
        """Return the tidy TS1--TS4 barriers for one result tier.

        Parameters
        ----------
        level : {"low_cost", "dft_ranked", "full"} or None, optional
            Nested analysis tier. ``None`` preserves the historical behavior
            and returns the terminal requested level.
        """
        self._ensure_analysis()
        return self._level_table("barriers", level=level)

    def compare_barriers(
        self,
        *,
        levels: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return screening and full barriers side by side.

        Parameters
        ----------
        levels : sequence of str or None, optional
            Levels to compare. ``None`` uses every available tier in workflow
            order.

        Returns
        -------
        pandas.DataFrame
            One row per system, reactive position, and TS type. Energy,
            quality, and selected-dimer columns carry a level suffix, for
            example ``delta_e_dft_ranked_kcal_mol`` and
            ``delta_g_full_kcal_mol``.
        """
        requested = (
            self.available_analysis_levels()
            if levels is None
            else tuple(str(level) for level in levels)
        )
        keys = [
            "system_name",
            "substrate_name",
            "catalyst_name",
            "rpos",
            "ts_type",
        ]
        result: pd.DataFrame | None = None
        for level in requested:
            table = self.barriers(level=level)
            value_columns = [
                "delta_e_kcal_mol",
                "delta_g_kcal_mol",
                "delta_g_corrected_kcal_mol",
                "quality_status",
                "quality_issues",
                "dimer_state_id",
                "dimer_result_id",
                "dimer_reference_quality",
            ]
            selected = table[[*keys, *value_columns]].copy()
            selected = selected.rename(
                columns={
                    "delta_e_kcal_mol": f"delta_e_{level}_kcal_mol",
                    "delta_g_kcal_mol": f"delta_g_{level}_kcal_mol",
                    "delta_g_corrected_kcal_mol": (
                        f"delta_g_corrected_{level}_kcal_mol"
                    ),
                    **{
                        column: f"{column}_{level}"
                        for column in value_columns[3:]
                    },
                }
            )
            result = (
                selected
                if result is None
                else result.merge(selected, on=keys, how="outer", validate="one_to_one")
            )
        return pd.DataFrame(columns=keys) if result is None else result

    def profile(
        self,
        *,
        system_name: str | None = None,
        rpos: int | None = None,
        include_invalid: bool = False,
        quantity: Literal["electronic", "gibbs"] = "gibbs",
        level: str | None = None,
    ) -> pd.DataFrame:
        """Return one ordered balanced catalytic-cycle profile."""
        self._ensure_analysis()
        selected_level = self._resolve_analysis_level(level)
        path = self.analysis_dir / (
            "profiles.parquet" if level is None else "profiles_by_level.parquet"
        )
        if not path.exists():
            raise ValueError("This run used scope='barriers'; no full-cycle profile exists")
        table = pd.read_parquet(path)
        if level is not None:
            table = table[table["analysis_level"].eq(selected_level)].drop(
                columns="analysis_level"
            )
        if system_name is not None:
            table = table[table["system_name"].eq(str(system_name))]
        if rpos is not None:
            table = table[table["rpos"].eq(int(rpos))]
        if not include_invalid:
            table = table[~table["quality_status"].isin(["invalid", "incomplete"])]
        if quantity not in {"electronic", "gibbs"}:
            raise ValueError("quantity must be 'electronic' or 'gibbs'")
        if quantity == "gibbs" and selected_level != "full":
            raise ValueError("Gibbs profiles require level='full'; use quantity='electronic'")
        result = table.sort_values(
            ["system_name", "rpos", "profile_order"]
        ).reset_index(drop=True)
        result.attrs["energy_column"] = (
            "relative_g_corrected_kcal_mol"
            if quantity == "gibbs"
            else "relative_e_kcal_mol"
        )
        return result

    def plot_profile(
        self,
        *,
        system_name: str,
        rpos: int,
        include_invalid: bool = False,
        quantity: Literal["electronic", "gibbs"] = "gibbs",
        level: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot one balanced profile with FRUST's energy-profile renderer."""
        from frust.vis import plot_energy_profile

        profile = self.profile(
            system_name=system_name,
            rpos=rpos,
            include_invalid=include_invalid,
            quantity=quantity,
            level=level,
        )
        if profile.empty:
            raise ValueError("No plottable profile states match the requested system and rpos")
        energy_column = str(profile.attrs["energy_column"])
        states = list(zip(profile["profile_state"], profile[energy_column]))
        return plot_energy_profile(states, **kwargs)

    def review_queue(self) -> pd.DataFrame:
        """Return TS results needing manual imaginary-mode review."""
        if self.manifest.get("calculation_level", "full") != "full":
            return self.states().iloc[0:0].copy()
        states = self.states()
        return states[
            states["state_kind"].eq("transition_state")
            & states["review_status"].eq("unreviewed")
        ].reset_index(drop=True)

    def plot_vibration(self, result_id: str, *, mode: int = 0, **kwargs: Any) -> Any:
        """Display a result's selected vibration mode from raw portable data."""
        from frust.vis import plot_vibs

        row = self._raw_result(result_id)
        return plot_vibs(row, row_index=0, vId=mode, **kwargs)

    def set_review(
        self,
        result_id: str,
        decision: Literal["approved", "rejected"],
        *,
        note: str = "",
        reviewer: str = "",
    ) -> None:
        """Record a scientific review of a transition state's imaginary mode.

        A transition state with one imaginary frequency is not necessarily the
        intended reaction coordinate. Use this method after inspecting its
        normal mode with :meth:`plot_vibration`.

        An ``"approved"`` decision records that the mode represents the
        intended reaction. The state becomes ``"ready"`` when it has no other
        vibration flags; otherwise it remains ``"review"`` for further
        inspection. A ``"rejected"`` decision marks the state and dependent
        barriers or profile states as ``"invalid"``. This method records a
        decision only: it does not alter the calculated geometry, energy, or
        frequencies.

        Reviews are tied to the immutable ``result_id``, which includes the
        result geometry, energy, frequencies, and normal-mode content. A
        recalculation therefore receives a new ID and requires a new review.

        Parameters
        ----------
        result_id : str
            Identifier of the full-level transition-state result to review.
        decision : {"approved", "rejected"}
            ``"approved"`` confirms that the imaginary mode follows the
            intended reaction coordinate. ``"rejected"`` records that it does
            not.
        note : str, optional
            Scientific rationale for the decision, for example ``"Mode follows
            the B-H/C-H transfer coordinate."``.
        reviewer : str, optional
            Name or initials of the person who made the decision.

        Raises
        ------
        ValueError
            If the run is not full-level, the decision is unsupported, or the
            result is not a transition state.
        KeyError
            If ``result_id`` is not part of this run.
        """
        if self.manifest.get("calculation_level", "full") != "full":
            raise ValueError("TS vibration review requires level='full'")
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")
        states = self.states()
        matches = states[states["result_id"].eq(str(result_id))]
        if matches.empty:
            raise KeyError(f"Unknown result ID {result_id!r}")
        if not matches.iloc[0]["state_kind"] == "transition_state":
            raise ValueError("Manual mode review applies only to transition states")
        reviews = _read_reviews(self.reviews_path)
        review = pd.DataFrame(
            [{
                "result_id": result_id,
                "decision": decision,
                "note": str(note),
                "reviewer": str(reviewer),
                "reviewed_at": _utc_now(),
            }]
        )
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([reviews, review], ignore_index=True).to_csv(self.reviews_path, index=False)
        self.refresh_analysis()

    def _ensure_analysis(self) -> None:
        required = [
            self.analysis_dir / "states.parquet",
            self.analysis_dir / "dimer_references.parquet",
            self.analysis_dir / "barriers.parquet",
            self.analysis_dir / "states_by_level.parquet",
            self.analysis_dir / "dimer_references_by_level.parquet",
            self.analysis_dir / "barriers_by_level.parquet",
        ]
        if any(not path.exists() for path in required):
            self.refresh_analysis()

    def _resolve_analysis_level(self, level: str | None) -> str:
        """Validate and resolve a requested nested analysis tier."""
        resolved = (
            str(self.manifest.get("calculation_level", "full"))
            if level is None
            else str(level)
        )
        available = self.available_analysis_levels()
        if resolved not in available:
            raise ValueError(
                f"Analysis level {resolved!r} is unavailable; expected one of "
                f"{list(available)}"
            )
        return resolved

    def _level_table(self, artifact: str, *, level: str | None) -> pd.DataFrame:
        """Load a terminal or explicitly selected by-level analysis table."""
        if level is None:
            return pd.read_parquet(self.analysis_dir / f"{artifact}.parquet")
        resolved = self._resolve_analysis_level(level)
        table = pd.read_parquet(self.analysis_dir / f"{artifact}_by_level.parquet")
        return table[table["analysis_level"].eq(resolved)].drop(
            columns="analysis_level"
        ).reset_index(drop=True)

    def _raw_result(self, result_id: str) -> pd.DataFrame:
        manifest = self.manifest
        recipe = manifest["method"].get("thermochemistry")
        for source, frame in _load_raw_frames(self.path, manifest):
            enriched = _state_rows(
                frame,
                source=source,
                recipe=recipe,
                reviews={},
                calculation_level=manifest.get("calculation_level", "full"),
                method=manifest.get("method", {}),
            )
            mask = enriched["result_id"].eq(str(result_id))
            if mask.any():
                position = int(np.flatnonzero(mask.to_numpy())[0])
                return frame.iloc[[position]].reset_index(drop=True)
        raise KeyError(f"Raw result for {result_id!r} is not present")


def build_analysis(run_dir: str | Path) -> dict[str, Any]:
    """Build portable state, barrier, profile, and report artifacts."""
    root = Path(run_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = analysis_dir / "reviews.csv"
    reviews = _latest_reviews(_read_reviews(reviews_path))
    calculation_level = manifest.get("calculation_level", "full")
    states, dimer_references, barriers, profiles = _build_level_tables(
        root,
        manifest,
        calculation_level=calculation_level,
        calculation_results=manifest.get("calculation_results", {}),
        reviews=reviews,
    )
    states.to_parquet(analysis_dir / "states.parquet", index=False)
    dimer_references.to_parquet(
        analysis_dir / "dimer_references.parquet",
        index=False,
    )
    barriers.to_parquet(analysis_dir / "barriers.parquet", index=False)
    if manifest.get("scope") == "full_cycle":
        profiles.to_parquet(analysis_dir / "profiles.parquet", index=False)

    analysis_levels = tuple(
        str(level)
        for level in manifest.get("analysis_levels", [calculation_level])
    )
    level_results = manifest.get("tier_calculation_results", {})
    states_by_level: list[pd.DataFrame] = []
    dimers_by_level: list[pd.DataFrame] = []
    barriers_by_level: list[pd.DataFrame] = []
    profiles_by_level: list[pd.DataFrame] = []
    for level in analysis_levels:
        if level == calculation_level:
            level_tables = (states, dimer_references, barriers, profiles)
        else:
            level_tables = _build_level_tables(
                root,
                manifest,
                calculation_level=level,
                calculation_results=level_results.get(level, {}),
                reviews=reviews,
            )
        for table, destination in zip(
            level_tables,
            (
                states_by_level,
                dimers_by_level,
                barriers_by_level,
                profiles_by_level,
            ),
        ):
            labelled = table.copy()
            labelled.insert(0, "analysis_level", level)
            destination.append(labelled)
    _concat_level_tables(states_by_level).to_parquet(
        analysis_dir / "states_by_level.parquet",
        index=False,
    )
    _concat_level_tables(dimers_by_level).to_parquet(
        analysis_dir / "dimer_references_by_level.parquet",
        index=False,
    )
    _concat_level_tables(barriers_by_level).to_parquet(
        analysis_dir / "barriers_by_level.parquet",
        index=False,
    )
    if manifest.get("scope") == "full_cycle":
        _concat_level_tables(profiles_by_level).to_parquet(
            analysis_dir / "profiles_by_level.parquet",
            index=False,
        )

    if not reviews_path.exists():
        _empty_reviews().to_csv(reviews_path, index=False)
    report = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "manifest_hash": _json_hash(manifest),
        "n_states": int(len(states)),
        "n_dimer_candidates": int(len(dimer_references)),
        "n_barriers": int(len(barriers)),
        "n_profile_states": int(len(profiles)),
        "state_quality": _counts(states, "quality_status"),
        "dimer_reference_quality": _counts(
            dimer_references.drop_duplicates("catalyst_name"),
            "selection_quality_status",
        ),
        "barrier_quality": _counts(barriers, "quality_status"),
        "analysis_levels": list(analysis_levels),
    }
    (analysis_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_level_tables(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    calculation_level: str,
    calculation_results: Mapping[str, Any],
    reviews: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all analysis tables for one independently selected tier."""
    level_manifest = dict(manifest)
    level_manifest["calculation_level"] = calculation_level
    recipe = (
        manifest["method"].get("thermochemistry")
        if calculation_level == "full"
        else None
    )
    frames = []
    for source, frame in _load_raw_frames(
        root,
        manifest,
        calculation_results=calculation_results,
    ):
        state_rows = _state_rows(
            frame,
            source=source,
            recipe=recipe,
            reviews=reviews,
            calculation_level=calculation_level,
            method=manifest.get("method", {}),
        )
        if not state_rows.empty:
            frames.append(state_rows)
    states = pd.concat(frames, ignore_index=True) if frames else _empty_states()
    dimer_references = _build_dimer_references(states, level_manifest)
    barriers = _build_barriers(states, level_manifest, dimer_references)
    profiles = (
        _build_profiles(states, level_manifest, dimer_references)
        if manifest.get("scope") == "full_cycle"
        else pd.DataFrame()
    )
    return states, dimer_references, barriers, profiles


def _concat_level_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate labelled analysis-level tables with stable empty behavior."""
    populated = [table for table in tables if not table.empty]
    if populated:
        return pd.concat(populated, ignore_index=True)
    return tables[0].iloc[0:0].copy() if tables else pd.DataFrame()


def _load_raw_frames(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    calculation_results: Mapping[str, Any] | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    paths = (
        manifest.get("calculation_results", {})
        if calculation_results is None
        else calculation_results
    )
    for source, relative in paths.items():
        if not relative:
            continue
        path = root / str(relative)
        if path.exists():
            frames.append((str(source), pd.read_parquet(path)))
    return frames


def _state_rows(
    df: pd.DataFrame,
    *,
    source: str,
    recipe: Mapping[str, Any] | None,
    reviews: Mapping[str, str],
    calculation_level: str,
    method: Mapping[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return _empty_states()
    electronic = get_result(df, "electronic_energy", purpose="analysis")
    full = calculation_level == "full"
    components = (
        free_energy_components(df, thermochemistry=recipe)
        if full
        else pd.DataFrame(index=df.index)
    )
    protocol = _energy_protocol(df, method, calculation_level)
    protocol_fingerprint = _json_hash(
        {
            "calculation_level": calculation_level,
            "calculator": protocol.get("calculator"),
        }
    )
    nt_columns = normal_termination_columns(df)
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(df.iterrows()):
        component = components.iloc[position] if full else pd.Series(dtype=object)
        state_id = str(row.get("state_id", row.get("structure_type", "")))
        default_kind = "transition_state" if state_id.startswith("TS") else "minimum"
        state_kind = str(row.get("state_kind", default_kind))
        vibration = (
            _vibration_status(row, state_kind)
            if full
            else _not_applicable_vibration_status()
        )
        failed_nt = [
            column
            for column in nt_columns
            if not _truthy(row.get(column))
        ]
        issues = list(vibration["issues"])
        if not nt_columns:
            issues.append("missing_normal_termination_status")
        if failed_nt:
            issues.append("non_normal_termination:" + ",".join(failed_nt))
        electronic_energy = electronic.iloc[position]
        free_energy = component.get("free_energy_hartree", np.nan)
        if pd.isna(electronic_energy):
            issues.append("missing_electronic_energy")
        if full and pd.isna(free_energy):
            issues.append("missing_free_energy")
        result_id = _result_id(
            row,
            state_id,
            electronic_energy,
            free_energy,
            vibration,
        )
        review_status = (
            reviews.get(result_id, "unreviewed")
            if full and state_kind == "transition_state"
            else "not_required"
        )
        if pd.isna(electronic_energy) or (full and pd.isna(free_energy)) or not nt_columns:
            quality = "incomplete"
        elif failed_nt or not vibration["valid"] or review_status == "rejected":
            quality = "invalid"
        elif full and (
            "weak_minimum_imag" in vibration["flags"]
            or (
                state_kind == "transition_state"
                and (review_status == "unreviewed" or vibration["flags"])
            )
        ):
            quality = "review"
        else:
            quality = "ready"
        rows.append(
            {
                "result_id": result_id,
                "source": source,
                "source_row": position,
                "system_name": row.get("system_name"),
                "substrate_name": row.get("substrate_name"),
                "catalyst_name": row.get("catalyst_name"),
                "rpos": row.get("rpos"),
                "state_id": state_id,
                "state_kind": state_kind,
                "structure_id": row.get("structure_id"),
                "cid": row.get("cid"),
                "formula": _formula(row.get("atoms", [])),
                "calculation_level": calculation_level,
                "geometry_stage": protocol.get("geometry_stage"),
                "energy_stage": protocol.get("analysis_stage"),
                "energy_method": protocol.get("calculator", {}).get("method"),
                "energy_basis": protocol.get("calculator", {}).get("basis"),
                "solvation_model": protocol.get("calculator", {}).get(
                    "solvation_model"
                ),
                "solvent": protocol.get("calculator", {}).get("solvent"),
                "energy_protocol_fingerprint": protocol_fingerprint,
                "electronic_energy_hartree": electronic_energy,
                "analysis_electronic_energy_hartree": electronic_energy,
                "frequency_electronic_energy_hartree": component.get(
                    "frequency_electronic_energy_hartree", np.nan
                ),
                "frequency_gibbs_energy_hartree": component.get(
                    "frequency_gibbs_energy_hartree", np.nan
                ),
                "thermal_correction_hartree": component.get(
                    "thermal_correction_hartree", np.nan
                ),
                "free_energy_hartree": free_energy,
                "thermochemistry_mode": component.get("thermochemistry_mode"),
                "n_imag": vibration["n_imag"],
                "imaginary_frequencies_cm1": vibration["imaginary_frequencies"],
                "vibration_flags": ";".join(vibration["flags"]),
                "review_status": review_status,
                "quality_status": quality,
                "quality_issues": ";".join(issues),
            }
        )
    result = pd.DataFrame(rows)
    result["rpos"] = pd.to_numeric(result["rpos"], errors="coerce").astype("Int64")
    return result


def _build_dimer_references(
    states: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    """Select one exact dimer result row for each catalyst."""
    mode = str(manifest.get("dimer_reference", "dimer"))
    candidates = tuple(
        str(value)
        for value in manifest.get(
            "dimer_candidates",
            DIMER_STATES if mode == "lowest" else (mode,),
        )
    )
    full = manifest.get("calculation_level", "full") == "full"
    energy_column = (
        "free_energy_hartree" if full else "electronic_energy_hartree"
    )
    selection_quantity = "gibbs" if full else "electronic"
    catalysts = list(
        dict.fromkeys(
            str(target["catalyst_name"])
            for target in manifest.get("analysis_targets", [])
        )
    )
    if not catalysts and not states.empty and "catalyst_name" in states:
        catalysts = [
            str(value)
            for value in states.loc[
                states["state_id"].isin(candidates), "catalyst_name"
            ].dropna().unique()
        ]

    rows: list[dict[str, Any]] = []
    for catalyst in catalysts:
        candidate_rows: list[dict[str, Any]] = []
        selection_problems: list[str] = []
        for state_id in candidates:
            matches = states[
                states["state_id"].eq(state_id)
                & states["catalyst_name"].eq(catalyst)
            ]
            energy_bearing = matches[matches[energy_column].notna()]
            qualified = (
                energy_bearing[
                    energy_bearing["quality_status"].isin(["ready", "review"])
                ]
                if mode == "lowest"
                else energy_bearing
            )
            if qualified.empty:
                if not matches.empty and matches["quality_status"].eq("invalid").any():
                    candidate_quality = "invalid"
                    selection_problems.append(f"invalid:{state_id}")
                else:
                    candidate_quality = "incomplete"
                    selection_problems.append(f"missing:{state_id}")
                chosen = None
            else:
                chosen = qualified.sort_values(
                    [energy_column, "result_id"],
                    kind="stable",
                ).iloc[0]
                candidate_quality = str(chosen["quality_status"])
            candidate_rows.append(
                {
                    "catalyst_name": catalyst,
                    "dimer_reference": mode,
                    "state_id": state_id,
                    "result_id": None if chosen is None else chosen["result_id"],
                    "cid": None if chosen is None else chosen.get("cid"),
                    "selection_quantity": selection_quantity,
                    "selection_energy_hartree": (
                        np.nan if chosen is None else chosen[energy_column]
                    ),
                    "relative_energy_kcal_mol": np.nan,
                    "selected": False,
                    "candidate_quality_status": candidate_quality,
                    "selection_quality_status": "incomplete",
                    "selection_issues": "",
                    "energy_gap_to_next_kcal_mol": np.nan,
                }
            )

        complete = not selection_problems
        if complete:
            available = [
                row for row in candidate_rows
                if not pd.isna(row["selection_energy_hartree"])
            ]
            minimum = min(float(row["selection_energy_hartree"]) for row in available)
            for row in available:
                row["relative_energy_kcal_mol"] = (
                    float(row["selection_energy_hartree"]) - minimum
                ) * HARTREE_TO_KCAL_MOL
            tied = [
                row for row in available
                if float(row["relative_energy_kcal_mol"])
                <= DIMER_TIE_TOLERANCE_KCAL_MOL
            ]
            order = {state_id: index for index, state_id in enumerate(DIMER_STATES)}
            selected_row = min(tied, key=lambda row: order.get(row["state_id"], 999))
            selected_row["selected"] = True
            ranked = sorted(
                float(row["selection_energy_hartree"])
                for row in available
            )
            gap = (
                (ranked[1] - ranked[0]) * HARTREE_TO_KCAL_MOL
                if len(ranked) > 1
                else np.nan
            )
            selection_quality = _combined_quality(
                [str(row["candidate_quality_status"]) for row in available]
            )
            if len(tied) > 1:
                selection_problems.append("near_degenerate_dimer_reference")
        else:
            gap = np.nan
            selection_quality = _combined_quality(
                [str(row["candidate_quality_status"]) for row in candidate_rows]
            )
        issues = ";".join(dict.fromkeys(selection_problems))
        for row in candidate_rows:
            row["selection_quality_status"] = selection_quality
            row["selection_issues"] = issues
            row["energy_gap_to_next_kcal_mol"] = gap
        rows.extend(candidate_rows)
    return pd.DataFrame(rows, columns=_dimer_reference_columns())


def _selected_dimer_reference(
    dimer_references: pd.DataFrame,
    catalyst_name: str,
) -> tuple[str | None, str | None, str, list[str]]:
    """Return the selected state/result and strict selection provenance."""
    matches = dimer_references[
        dimer_references["catalyst_name"].eq(catalyst_name)
    ]
    if matches.empty:
        return None, None, "incomplete", ["missing:dimer_reference"]
    quality = str(matches.iloc[0]["selection_quality_status"])
    issues = [
        issue
        for issue in str(matches.iloc[0]["selection_issues"]).split(";")
        if issue
    ]
    selected = matches[matches["selected"]]
    if selected.empty:
        return None, None, quality, issues or [f"{quality}:dimer_reference"]
    row = selected.iloc[0]
    return str(row["state_id"]), str(row["result_id"]), quality, issues


def _build_barriers(
    states: pd.DataFrame,
    manifest: Mapping[str, Any],
    dimer_references: pd.DataFrame,
) -> pd.DataFrame:
    corrections = {
        str(key): float(value)
        for key, value in manifest.get("g_corrections_kcal_mol", {}).items()
    }
    full = manifest.get("calculation_level", "full") == "full"
    targets = manifest.get("analysis_targets", [])
    rows: list[dict[str, Any]] = []
    for target in targets:
        ts_type = str(target["state_id"])
        system = str(target["system_name"])
        substrate = str(target["substrate_name"])
        catalyst = str(target["catalyst_name"])
        rpos = int(target["rpos"])
        dimer_state, dimer_result_id, dimer_quality, dimer_issues = (
            _selected_dimer_reference(dimer_references, catalyst)
        )
        terms = {ts_type: 1.0, "ligand": -1.0}
        if dimer_state is not None:
            terms[dimer_state] = -0.5
        if ts_type in {"TS3", "TS4"}:
            terms.update({"HBpin-mol": -1.0, "HH": 1.0})
        selected, problems = _resolve_terms(
            states,
            terms,
            system_name=system,
            substrate_name=substrate,
            catalyst_name=catalyst,
            rpos=rpos,
            result_ids=(
                {dimer_state: dimer_result_id}
                if dimer_state is not None and dimer_result_id is not None
                else None
            ),
        )
        if dimer_state is None:
            problems.extend(dimer_issues)
        dependency_issues: list[str] = list(dimer_issues) if dimer_state else []
        if dimer_state and dimer_quality != "ready":
            dependency_issues.append(
                f"dependency_{dimer_quality}:dimer_reference"
            )
        correction = corrections.get(ts_type, 0.0)
        delta_e = np.nan
        delta_g = np.nan
        corrected_delta_g = np.nan
        if problems:
            quality = (
                "incomplete"
                if any(problem.startswith("missing") for problem in problems)
                else "invalid"
            )
        else:
            protocols = {
                str(selected[state]["energy_protocol_fingerprint"])
                for state in terms
            }
            if len(protocols) != 1:
                problems.append("mixed_electronic_energy_protocols")
                quality = "invalid"
            delta_e = sum(
                float(selected[state]["electronic_energy_hartree"]) * coefficient
                for state, coefficient in terms.items()
            )
            delta_e *= HARTREE_TO_KCAL_MOL
            if full:
                free_energies = [selected[state]["free_energy_hartree"] for state in terms]
                if any(pd.isna(value) for value in free_energies):
                    problems.append("missing_free_energy")
                    quality = "incomplete"
                else:
                    delta_g = sum(
                        float(selected[state]["free_energy_hartree"]) * coefficient
                        for state, coefficient in terms.items()
                    ) * HARTREE_TO_KCAL_MOL
                    corrected_delta_g = delta_g + correction
            if not problems:
                quality = _combined_quality(
                    [
                        *[str(selected[state]["quality_status"]) for state in terms],
                        dimer_quality,
                    ]
                )
            dependency_issues.extend(_dependency_quality_issues(selected, terms))
        rows.append(
            {
                "system_name": system,
                "substrate_name": substrate,
                "catalyst_name": catalyst,
                "rpos": rpos,
                "ts_type": ts_type,
                "dimer_state_id": dimer_state,
                "dimer_result_id": dimer_result_id,
                "dimer_reference_quality": dimer_quality,
                "delta_e_kcal_mol": delta_e,
                "delta_g_kcal_mol": delta_g,
                "g_correction_kcal_mol": correction if full else np.nan,
                "delta_g_corrected_kcal_mol": corrected_delta_g,
                "quality_status": quality,
                "quality_issues": ";".join(
                    dict.fromkeys([*problems, *dependency_issues])
                ),
                "formula_id": f"frust_ts_barrier::{ts_type}::v2",
            }
        )
    return pd.DataFrame(rows)


def _build_profiles(
    states: pd.DataFrame,
    manifest: Mapping[str, Any],
    dimer_references: pd.DataFrame,
) -> pd.DataFrame:
    corrections = {
        str(key): float(value)
        for key, value in manifest.get("g_corrections_kcal_mol", {}).items()
    }
    full = manifest.get("calculation_level", "full") == "full"
    unique_targets: dict[tuple[str, int], Mapping[str, Any]] = {}
    for target in manifest.get("analysis_targets", []):
        unique_targets[(str(target["system_name"]), int(target["rpos"]))] = target
    rows: list[dict[str, Any]] = []
    for (system, rpos), target in unique_targets.items():
        substrate = str(target["substrate_name"])
        catalyst = str(target["catalyst_name"])
        dimer_state, dimer_result_id, dimer_quality, dimer_issues = (
            _selected_dimer_reference(dimer_references, catalyst)
        )
        resolved: dict[
            str,
            tuple[float, float | None, str, Counter[str], list[str]] | None,
        ] = {}
        for label, base_terms in PROFILE_TERMS.items():
            terms = dict(base_terms)
            uses_dimer = "dimer" in terms
            if uses_dimer:
                coefficient = terms.pop("dimer")
                if dimer_state is None:
                    resolved[label] = None
                    continue
                terms[dimer_state] = coefficient
            selected, problems = _resolve_terms(
                states,
                terms,
                system_name=system,
                substrate_name=substrate,
                catalyst_name=catalyst,
                rpos=rpos,
                result_ids=(
                    {dimer_state: dimer_result_id}
                    if uses_dimer
                    and dimer_state is not None
                    and dimer_result_id is not None
                    else None
                ),
            )
            if problems:
                resolved[label] = None
                continue
            protocols = {
                str(selected[state]["energy_protocol_fingerprint"])
                for state in terms
            }
            if len(protocols) != 1:
                resolved[label] = None
                continue
            electronic_energy = sum(
                float(selected[state]["electronic_energy_hartree"]) * coefficient
                for state, coefficient in terms.items()
            )
            free_energy = (
                sum(
                    float(selected[state]["free_energy_hartree"]) * coefficient
                    for state, coefficient in terms.items()
                )
                if full
                else None
            )
            quality_values = [
                str(selected[state]["quality_status"]) for state in terms
            ]
            if uses_dimer:
                quality_values.append(dimer_quality)
            quality = _combined_quality(quality_values)
            dependency_issues = _dependency_quality_issues(selected, terms)
            if uses_dimer:
                dependency_issues.extend(dimer_issues)
                if dimer_quality != "ready":
                    dependency_issues.append(
                        f"dependency_{dimer_quality}:dimer_reference"
                    )
            composition = Counter()
            for state, coefficient in terms.items():
                for element, count in _parse_formula(str(selected[state]["formula"])).items():
                    composition[element] += count * coefficient
            resolved[label] = (
                electronic_energy,
                free_energy,
                quality,
                composition,
                dependency_issues,
            )
        reference = resolved.get("Dimer")
        for order, label in enumerate(PROFILE_ORDER):
            value = resolved.get(label)
            issues: list[str] = []
            if value is None or reference is None:
                relative_e = np.nan
                relative_g = np.nan
                corrected_relative_g = np.nan
                quality = (
                    dimer_quality
                    if dimer_state is None
                    else "incomplete"
                )
                issues.append("missing_profile_dependency")
                issues.extend(dimer_issues)
            else:
                (
                    electronic_energy,
                    free_energy,
                    value_quality,
                    composition,
                    value_issues,
                ) = value
                (
                    ref_electronic_energy,
                    ref_free_energy,
                    ref_quality,
                    ref_composition,
                    ref_issues,
                ) = reference
                quality = _combined_quality([value_quality, ref_quality])
                issues.extend(value_issues)
                issues.extend(ref_issues)
                if not _same_composition(composition, ref_composition):
                    quality = "invalid"
                    issues.append("unbalanced_composition")
                relative_e = (
                    electronic_energy - ref_electronic_energy
                ) * HARTREE_TO_KCAL_MOL
                relative_g = (
                    (free_energy - ref_free_energy) * HARTREE_TO_KCAL_MOL
                    if free_energy is not None and ref_free_energy is not None
                    else np.nan
                )
                corrected_relative_g = (
                    relative_g + corrections.get(label, 0.0)
                    if full
                    else np.nan
                )
            rows.append(
                {
                    "system_name": system,
                    "substrate_name": substrate,
                    "catalyst_name": catalyst,
                    "rpos": rpos,
                    "dimer_state_id": dimer_state,
                    "dimer_result_id": dimer_result_id,
                    "dimer_reference_quality": dimer_quality,
                    "profile_state": label,
                    "profile_order": order,
                    "relative_e_kcal_mol": relative_e,
                    "relative_g_kcal_mol": relative_g,
                    "g_correction_kcal_mol": (
                        corrections.get(label, 0.0) if full else np.nan
                    ),
                    "relative_g_corrected_kcal_mol": corrected_relative_g,
                    "quality_status": quality,
                    "quality_issues": ";".join(dict.fromkeys(issues)),
                    "mechanism_id": "frust_balanced_cycle::v3",
                }
            )
    return pd.DataFrame(rows)


def _resolve_terms(
    states: pd.DataFrame,
    terms: Mapping[str, float],
    *,
    system_name: str,
    substrate_name: str,
    catalyst_name: str,
    rpos: int,
    result_ids: Mapping[str, str] | None = None,
) -> tuple[dict[str, pd.Series], list[str]]:
    selected: dict[str, pd.Series] = {}
    problems: list[str] = []
    for state_id in terms:
        matches = states[states["state_id"].eq(state_id)]
        if result_ids is not None and state_id in result_ids:
            matches = matches[matches["result_id"].eq(result_ids[state_id])]
        if state_id in {"HH", "HBpin-mol"}:
            pass
        elif state_id == "ligand":
            matches = matches[matches["substrate_name"].eq(substrate_name)]
        elif state_id in {*DIMER_STATES, "catalyst"}:
            matches = matches[matches["catalyst_name"].eq(catalyst_name)]
        elif state_id == "HBpin-ligand":
            matches = matches[
                matches["substrate_name"].eq(substrate_name)
                & _rpos_mask(matches["rpos"], rpos)
            ]
        else:
            matches = matches[
                matches["system_name"].eq(system_name)
                & _rpos_mask(matches["rpos"], rpos)
            ]
        if len(matches) == 0:
            problems.append(f"missing:{state_id}")
        elif len(matches) > 1:
            problems.append(f"ambiguous:{state_id}")
        else:
            selected[state_id] = matches.iloc[0]
    return selected, problems


def _dependency_quality_issues(
    selected: Mapping[str, pd.Series],
    state_ids: Mapping[str, float],
) -> list[str]:
    """Return compact quality provenance for non-ready dependencies."""
    issues: list[str] = []
    for state_id in state_ids:
        row = selected.get(state_id)
        if row is None:
            continue
        quality = str(row.get("quality_status", "incomplete"))
        if quality != "ready":
            issues.append(f"dependency_{quality}:{state_id}")
    return issues


def _vibration_status(row: pd.Series, state_kind: str) -> dict[str, Any]:
    columns = [column for column in row.index if str(column).endswith("-vibs")]
    vibrations = next(
        (
            row[column]
            for column in reversed(columns)
            if isinstance(row[column], (list, tuple, np.ndarray)) and len(row[column]) > 0
        ),
        None,
    )
    if vibrations is None:
        return {
            "valid": False,
            "n_imag": pd.NA,
            "imaginary_frequencies": "",
            "flags": [],
            "issues": ["missing_vibrations"],
        }
    frequencies = [float(mode["frequency"]) for mode in vibrations]
    negative = [frequency for frequency in frequencies if frequency < 0]
    positive = [frequency for frequency in frequencies if frequency >= 0]
    expected = 1 if state_kind == "transition_state" else 0
    if expected == 0:
        minimum = minimum_vibration_status(frequencies)
        flags = list(minimum["flags"])
        if positive and min(positive) < 10.0:
            flags.append("very_low_pos")
        return {
            "valid": minimum["valid"],
            "n_imag": minimum["n_imag"],
            "imaginary_frequencies": ",".join(
                f"{frequency:.2f}"
                for frequency in minimum["negative_frequencies_cm1"]
            ),
            "flags": flags,
            "issues": minimum["issues"],
        }
    flags: list[str] = []
    if expected == 1 and len(negative) == 1 and abs(negative[0]) < 50.0:
        flags.append("weak_imag")
    if positive and min(positive) < 10.0:
        flags.append("very_low_pos")
    issues = (
        []
        if len(negative) == expected
        else [f"expected_{expected}_imag_found_{len(negative)}"]
    )
    return {
        "valid": len(negative) == expected,
        "n_imag": len(negative),
        "imaginary_frequencies": ",".join(f"{frequency:.2f}" for frequency in negative),
        "flags": flags,
        "issues": issues,
    }


def _not_applicable_vibration_status() -> dict[str, Any]:
    """Return the explicit no-frequency status for electronic-only levels."""
    return {
        "valid": True,
        "n_imag": pd.NA,
        "imaginary_frequencies": "",
        "flags": [],
        "issues": [],
    }


def _result_id(
    row: pd.Series,
    state_id: str,
    electronic_energy: Any,
    free_energy: Any,
    vibration: Mapping[str, Any],
) -> str:
    payload = {
        "state_id": state_id,
        "system_name": row.get("system_name"),
        "substrate_name": row.get("substrate_name"),
        "catalyst_name": row.get("catalyst_name"),
        "rpos": _json_value(row.get("rpos")),
        "atoms": _json_value(row.get("atoms")),
        "coords": _json_value(_last_coords(row)),
        "electronic_energy_hartree": _json_value(electronic_energy),
        "free_energy_hartree": _json_value(free_energy),
        "imaginary_frequencies": vibration["imaginary_frequencies"],
        "vibrations": _json_value(_last_vibrations(row)),
    }
    return "result_" + _json_hash(payload)[:16]


def _last_coords(row: pd.Series) -> Any:
    columns = [
        column
        for column in row.index
        if "coords" in str(column) or str(column).endswith("-oc")
    ]
    for column in reversed(columns):
        value = row[column]
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return None


def _last_vibrations(row: pd.Series) -> Any:
    columns = [column for column in row.index if str(column).endswith("-vibs")]
    for column in reversed(columns):
        value = row[column]
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            return value
    return None


def _combined_quality(values: Sequence[str]) -> str:
    return max(values, key=lambda value: QUALITY_ORDER.get(value, 3)) if values else "incomplete"


def _formula(atoms: Any) -> str:
    if atoms is None or atoms is pd.NA:
        values: list[Any] = []
    else:
        values = list(atoms)
    counts = Counter(str(atom) for atom in values)
    order = ["C", "H"] + sorted(element for element in counts if element not in {"C", "H"})
    return "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in order
        if counts.get(element)
    )


def _parse_formula(formula: str) -> Counter[str]:
    import re

    counts: Counter[str] = Counter()
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[element] += int(number or 1)
    return counts


def _same_composition(left: Counter[str], right: Counter[str]) -> bool:
    keys = set(left) | set(right)
    return all(abs(float(left[key]) - float(right[key])) < 1e-8 for key in keys)


def _rpos_mask(series: pd.Series, value: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").eq(int(value))


def _truthy(value: Any) -> bool:
    return False if value is None or pd.isna(value) else bool(value)


def _empty_states() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "result_id", "source", "source_row", "system_name", "substrate_name",
            "catalyst_name", "rpos", "state_id", "state_kind", "structure_id",
            "cid", "formula", "electronic_energy_hartree",
            "free_energy_hartree", "quality_status",
        ]
    )


def _dimer_reference_columns() -> list[str]:
    """Return the stable dimer-reference audit-table schema."""
    return [
        "catalyst_name",
        "dimer_reference",
        "state_id",
        "result_id",
        "cid",
        "selection_quantity",
        "selection_energy_hartree",
        "relative_energy_kcal_mol",
        "selected",
        "candidate_quality_status",
        "selection_quality_status",
        "selection_issues",
        "energy_gap_to_next_kcal_mol",
    ]


def _energy_protocol(
    df: pd.DataFrame,
    method: Mapping[str, Any],
    calculation_level: str,
) -> dict[str, Any]:
    """Resolve energy and geometry provenance from a canonical result."""
    contract = df.attrs.get("frust_results", {})
    protocol = contract.get("energy_protocol") if isinstance(contract, Mapping) else None
    if isinstance(protocol, Mapping):
        return dict(protocol)
    columns = contract.get("columns", {}) if isinstance(contract, Mapping) else {}
    analysis_column = (
        columns.get("analysis", {}).get("electronic_energy")
        if isinstance(columns, Mapping)
        else None
    )
    optimized_column = (
        columns.get("optimized", {}).get("coords")
        if isinstance(columns, Mapping)
        else None
    )
    analysis_stage = str(analysis_column or "").removesuffix("-EE")
    geometry_stage = str(optimized_column or "").removesuffix("-oc")
    stages = method.get("stages", {}) if isinstance(method, Mapping) else {}
    calculator = stages.get(analysis_stage, {}) if isinstance(stages, Mapping) else {}
    return {
        "calculation_level": calculation_level,
        "analysis_stage": analysis_stage,
        "geometry_stage": geometry_stage,
        "calculator": dict(calculator) if isinstance(calculator, Mapping) else {},
    }


def _solvation_label(value: Any) -> str:
    """Return a compact human-facing label for manifest solvation metadata."""
    if isinstance(value, Mapping) and value.get("applied") is False:
        return "not applicable"
    if not isinstance(value, Mapping) or not value.get("solvent"):
        return "gas"
    model = str(value.get("model") or "implicit").upper()
    return f"{model}({value['solvent']})"


def _empty_reviews() -> pd.DataFrame:
    return pd.DataFrame(columns=["result_id", "decision", "note", "reviewer", "reviewed_at"])


def _read_reviews(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("") if path.exists() else _empty_reviews()


def _latest_reviews(reviews: pd.DataFrame) -> dict[str, str]:
    if reviews.empty:
        return {}
    latest = reviews.drop_duplicates("result_id", keep="last")
    return dict(zip(latest["result_id"], latest["decision"]))


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).items()}


def _json_hash(value: Any) -> str:
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if value is pd.NA:
        return None
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _json_value(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
