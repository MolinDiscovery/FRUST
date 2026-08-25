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

from frust.results import free_energy_components
from frust.schema import normal_termination_columns


HARTREE_TO_KCAL_MOL = 627.5094740631
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
    "Product": {"catalyst": 1.0, "HBpin-ligand": 1.0, "HH": 1.0},
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
                    }
                )
        return pd.DataFrame(rows)

    def states(self) -> pd.DataFrame:
        """Return one auditable row per calculated or expected state."""
        self._ensure_analysis()
        return pd.read_parquet(self.analysis_dir / "states.parquet")

    def barriers(self) -> pd.DataFrame:
        """Return the tidy TS1--TS4 barrier table, including flagged rows."""
        self._ensure_analysis()
        return pd.read_parquet(self.analysis_dir / "barriers.parquet")

    def profile(
        self,
        *,
        system_name: str | None = None,
        rpos: int | None = None,
        include_invalid: bool = False,
    ) -> pd.DataFrame:
        """Return one ordered balanced catalytic-cycle profile."""
        self._ensure_analysis()
        path = self.analysis_dir / "profiles.parquet"
        if not path.exists():
            raise ValueError("This run used scope='barriers'; no full-cycle profile exists")
        table = pd.read_parquet(path)
        if system_name is not None:
            table = table[table["system_name"].eq(str(system_name))]
        if rpos is not None:
            table = table[table["rpos"].eq(int(rpos))]
        if not include_invalid:
            table = table[~table["quality_status"].isin(["invalid", "incomplete"])]
        return table.sort_values(["system_name", "rpos", "profile_order"]).reset_index(drop=True)

    def plot_profile(
        self,
        *,
        system_name: str,
        rpos: int,
        include_invalid: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot one balanced profile with FRUST's energy-profile renderer."""
        from frust.vis import plot_energy_profile

        profile = self.profile(
            system_name=system_name,
            rpos=rpos,
            include_invalid=include_invalid,
        )
        if profile.empty:
            raise ValueError("No plottable profile states match the requested system and rpos")
        states = list(zip(profile["profile_state"], profile["relative_g_kcal_mol"]))
        return plot_energy_profile(states, **kwargs)

    def review_queue(self) -> pd.DataFrame:
        """Return TS results needing manual imaginary-mode review."""
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
        """Persist a TS-mode review and refresh dependent analysis."""
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
        required = [self.analysis_dir / "states.parquet", self.analysis_dir / "barriers.parquet"]
        if any(not path.exists() for path in required):
            self.refresh_analysis()

    def _raw_result(self, result_id: str) -> pd.DataFrame:
        manifest = self.manifest
        recipe = manifest["method"]["thermochemistry"]
        for source, frame in _load_raw_frames(self.path, manifest):
            enriched = _state_rows(frame, source=source, recipe=recipe, reviews={})
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
    recipe = manifest["method"]["thermochemistry"]

    frames = [
        _state_rows(frame, source=source, recipe=recipe, reviews=reviews)
        for source, frame in _load_raw_frames(root, manifest)
    ]
    states = pd.concat(frames, ignore_index=True) if frames else _empty_states()
    states.to_parquet(analysis_dir / "states.parquet", index=False)

    barriers = _build_barriers(states, manifest)
    barriers.to_parquet(analysis_dir / "barriers.parquet", index=False)

    profiles = pd.DataFrame()
    if manifest.get("scope") == "full_cycle":
        profiles = _build_profiles(states, manifest)
        profiles.to_parquet(analysis_dir / "profiles.parquet", index=False)

    if not reviews_path.exists():
        _empty_reviews().to_csv(reviews_path, index=False)
    report = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "manifest_hash": _json_hash(manifest),
        "n_states": int(len(states)),
        "n_barriers": int(len(barriers)),
        "n_profile_states": int(len(profiles)),
        "state_quality": _counts(states, "quality_status"),
        "barrier_quality": _counts(barriers, "quality_status"),
    }
    (analysis_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _load_raw_frames(root: Path, manifest: Mapping[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for source, relative in manifest.get("calculation_results", {}).items():
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
    recipe: Mapping[str, Any],
    reviews: Mapping[str, str],
) -> pd.DataFrame:
    if df.empty:
        return _empty_states()
    components = free_energy_components(df, thermochemistry=recipe)
    nt_columns = normal_termination_columns(df)
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(df.iterrows()):
        component = components.iloc[position]
        state_id = str(row.get("state_id", row.get("structure_type", "")))
        default_kind = "transition_state" if state_id.startswith("TS") else "minimum"
        state_kind = str(row.get("state_kind", default_kind))
        vibration = _vibration_status(row, state_kind)
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
        if pd.isna(component["free_energy_hartree"]):
            issues.append("missing_free_energy")
        result_id = _result_id(row, state_id, component, vibration)
        review_status = (
            reviews.get(result_id, "unreviewed")
            if state_kind == "transition_state"
            else "not_required"
        )
        if pd.isna(component["free_energy_hartree"]) or not nt_columns:
            quality = "incomplete"
        elif failed_nt or not vibration["valid"] or review_status == "rejected":
            quality = "invalid"
        elif state_kind == "transition_state" and (
            review_status == "unreviewed" or vibration["flags"]
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
                "analysis_electronic_energy_hartree": component[
                    "analysis_electronic_energy_hartree"
                ],
                "frequency_electronic_energy_hartree": component[
                    "frequency_electronic_energy_hartree"
                ],
                "frequency_gibbs_energy_hartree": component["frequency_gibbs_energy_hartree"],
                "thermal_correction_hartree": component["thermal_correction_hartree"],
                "free_energy_hartree": component["free_energy_hartree"],
                "thermochemistry_mode": component["thermochemistry_mode"],
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


def _build_barriers(states: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    corrections = {
        str(key): float(value)
        for key, value in manifest.get("corrections_kcal_mol", {}).items()
    }
    targets = manifest.get("analysis_targets", [])
    rows: list[dict[str, Any]] = []
    for target in targets:
        ts_type = str(target["state_id"])
        system = str(target["system_name"])
        substrate = str(target["substrate_name"])
        catalyst = str(target["catalyst_name"])
        rpos = int(target["rpos"])
        terms = {ts_type: 1.0, "ligand": -1.0, "dimer": -0.5}
        if ts_type in {"TS3", "TS4"}:
            terms.update({"HBpin-mol": -1.0, "HH": 1.0})
        selected, problems = _resolve_terms(
            states,
            terms,
            system_name=system,
            substrate_name=substrate,
            catalyst_name=catalyst,
            rpos=rpos,
        )
        correction = corrections.get(ts_type, 0.0)
        if problems:
            value = np.nan
            quality = (
                "incomplete"
                if any(problem.startswith("missing") for problem in problems)
                else "invalid"
            )
        else:
            value = sum(
                float(selected[state]["free_energy_hartree"]) * coefficient
                for state, coefficient in terms.items()
            )
            value = value * HARTREE_TO_KCAL_MOL + correction
            quality = _combined_quality([str(selected[state]["quality_status"]) for state in terms])
        rows.append(
            {
                "system_name": system,
                "substrate_name": substrate,
                "catalyst_name": catalyst,
                "rpos": rpos,
                "ts_type": ts_type,
                "barrier_kcal_mol": value,
                "correction_kcal_mol": correction,
                "quality_status": quality,
                "quality_issues": ";".join(problems),
                "formula_id": f"frust_ts_barrier::{ts_type}::v1",
            }
        )
    return pd.DataFrame(rows)


def _build_profiles(states: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    corrections = {
        str(key): float(value)
        for key, value in manifest.get("corrections_kcal_mol", {}).items()
    }
    unique_targets: dict[tuple[str, int], Mapping[str, Any]] = {}
    for target in manifest.get("analysis_targets", []):
        unique_targets[(str(target["system_name"]), int(target["rpos"]))] = target
    rows: list[dict[str, Any]] = []
    for (system, rpos), target in unique_targets.items():
        substrate = str(target["substrate_name"])
        catalyst = str(target["catalyst_name"])
        resolved: dict[str, tuple[float, str, Counter[str]] | None] = {}
        for label, terms in PROFILE_TERMS.items():
            selected, problems = _resolve_terms(
                states,
                terms,
                system_name=system,
                substrate_name=substrate,
                catalyst_name=catalyst,
                rpos=rpos,
            )
            if problems:
                resolved[label] = None
                continue
            energy = sum(
                float(selected[state]["free_energy_hartree"]) * coefficient
                for state, coefficient in terms.items()
            )
            quality = _combined_quality([str(selected[state]["quality_status"]) for state in terms])
            composition = Counter()
            for state, coefficient in terms.items():
                for element, count in _parse_formula(str(selected[state]["formula"])).items():
                    composition[element] += count * coefficient
            resolved[label] = (energy, quality, composition)
        reference = resolved.get("Dimer")
        for order, label in enumerate(PROFILE_ORDER):
            value = resolved.get(label)
            issues: list[str] = []
            if value is None or reference is None:
                relative = np.nan
                quality = "incomplete"
                issues.append("missing_profile_dependency")
            else:
                energy, quality, composition = value
                ref_energy, _, ref_composition = reference
                if not _same_composition(composition, ref_composition):
                    quality = "invalid"
                    issues.append("unbalanced_composition")
                relative = (energy - ref_energy) * HARTREE_TO_KCAL_MOL + corrections.get(label, 0.0)
            rows.append(
                {
                    "system_name": system,
                    "substrate_name": substrate,
                    "catalyst_name": catalyst,
                    "rpos": rpos,
                    "profile_state": label,
                    "profile_order": order,
                    "relative_g_kcal_mol": relative,
                    "correction_kcal_mol": corrections.get(label, 0.0),
                    "quality_status": quality,
                    "quality_issues": ";".join(issues),
                    "mechanism_id": "frust_balanced_cycle::v1",
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
) -> tuple[dict[str, pd.Series], list[str]]:
    selected: dict[str, pd.Series] = {}
    problems: list[str] = []
    for state_id in terms:
        matches = states[states["state_id"].eq(state_id)]
        if state_id in {"HH", "HBpin-mol"}:
            pass
        elif state_id == "ligand":
            matches = matches[matches["substrate_name"].eq(substrate_name)]
        elif state_id in {"dimer", "catalyst"}:
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


def _result_id(
    row: pd.Series,
    state_id: str,
    components: pd.Series,
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
        "free_energy_hartree": _json_value(components["free_energy_hartree"]),
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
            "cid", "formula", "free_energy_hartree", "quality_status",
        ]
    )


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
