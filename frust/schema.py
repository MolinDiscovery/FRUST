"""DataFrame schema helpers for FRUST results."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

OUTPUT_SUFFIXES = {
    "electronic_energy": "EE",
    "normal_termination": "NT",
    "opt_coords": "oc",
    "gibbs_energy": "GE",
}

LEGACY_OUTPUT_SUFFIXES = {v: k for k, v in OUTPUT_SUFFIXES.items()}

ENERGY_SUFFIXES = ("-EE", "_energy", "-electronic_energy", "-GE", "-gibbs_energy")
NORMAL_TERMINATION_SUFFIXES = ("-NT", "-normal_termination")
OPT_COORD_SUFFIXES = ("-oc", "-opt_coords")
SCHEMA_VERSION = 3

LEGACY_MOLECULE_STATE_RENAMES = {
    "int2": "int1",
    "mol2": "int2",
}

CANONICAL_STAGE_PREFIXES = {
    "DFT-pre-SP": "dft_rank_sp",
    "DFT-pre-Opt": "dft_preopt",
    "DFT-Opt": "dft_opt",
    "Hess": "dft_hessian",
    "Freq": "dft_freq",
    "DFT-solv": "dft_solv_sp",
    "DFT-SP-solvent": "dft_solv_sp",
}


@dataclass(frozen=True)
class StructureMetadata:
    """Parsed structure identity independent of display/file names."""

    structure_id: str
    custom_name: str
    substrate_name: str
    structure_type: str
    molecule_role: str
    rpos: Any
    smiles: str | None = None
    input_smiles: str | None = None


def output_column(prefix: str, key: str) -> str:
    """Build a dataframe output column with the canonical short suffix."""
    return f"{prefix}-{OUTPUT_SUFFIXES.get(key, key)}"


def energy_columns(df: pd.DataFrame) -> list[str]:
    """Return energy-like columns in dataframe order."""
    return [c for c in df.columns if str(c).endswith(ENERGY_SUFFIXES)]


def normal_termination_columns(df: pd.DataFrame) -> list[str]:
    """Return normal-termination columns in dataframe order."""
    return [c for c in df.columns if str(c).endswith(NORMAL_TERMINATION_SUFFIXES)]


def latest_opt_coords_column(prefix: str, df: pd.DataFrame) -> str | None:
    """Find the optimized coordinate column matching a vibration prefix."""
    for suffix in ("oc", "opt_coords"):
        col = f"{prefix}{suffix}"
        if col in df.columns:
            return col
    return None


def canonical_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy output suffixes to the compact schema."""
    renamed: dict[str, str] = {}
    for col in df.columns:
        text = str(col)
        for old, new in OUTPUT_SUFFIXES.items():
            old_suffix = f"-{old}"
            if text.endswith(old_suffix):
                renamed[col] = text[: -len(old)] + new
                break
    return df.rename(columns=renamed) if renamed else df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old FRUST dataframe columns to the current schema.

    This is intentionally conservative: it does not add ``ligand_name`` back.
    It only maps old data into canonical names so older parquet files can still
    be read by current utilities.
    """
    out = canonical_output_columns(df.copy())
    if "substrate_name" not in out.columns and "ligand_name" in out.columns:
        out = out.rename(columns={"ligand_name": "substrate_name"})
    elif "ligand_name" in out.columns:
        out = out.drop(columns=["ligand_name"])
    return out


def state_kind_for(state_id: object) -> str:
    """Return the canonical chemical kind for a state identifier.

    Parameters
    ----------
    state_id : object
        State label such as ``"int1"``, ``"int2"``, ``"TS3"``, or
        ``"INT3"``.

    Returns
    -------
    str
        ``"minimum"``, ``"transition_state"``, or
        ``"constrained_minimum"``.
    """
    text = str(state_id).upper()
    if text.startswith("TS"):
        return "transition_state"
    if text == "INT3":
        return "constrained_minimum"
    return "minimum"


def canonical_state_columns(
    df: pd.DataFrame,
    *,
    state_id: str | None = None,
    state_kind: str | None = None,
    structure_id: str | None = None,
) -> pd.DataFrame:
    """Add canonical structure identity columns to a FRUST dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Structure or result dataframe.
    state_id, state_kind, structure_id : str or None, optional
        Explicit values supplied by a typed structure target. Missing values
        are inferred from legacy ``structure_type`` and ``molecule_role``.

    Returns
    -------
    pandas.DataFrame
        Copy containing ``structure_id``, ``state_id``, and ``state_kind``.
    """
    out = normalize_dataframe(df)
    if state_id is not None:
        out["state_id"] = str(state_id)
    elif "state_id" not in out:
        if "structure_type" in out:
            state_values = out["structure_type"].astype(str)
            if "molecule_role" in out:
                state_values = state_values.where(
                    ~state_values.str.upper().eq("MOL"), out["molecule_role"].astype(str)
                )
            out["state_id"] = state_values
        else:
            out["state_id"] = "structure"
    if state_kind is not None:
        out["state_kind"] = str(state_kind)
    elif "state_kind" not in out:
        out["state_kind"] = out["state_id"].map(state_kind_for)
    if structure_id is not None:
        out["structure_id"] = str(structure_id)
    elif "structure_id" not in out:
        out["structure_id"] = [f"{state}:{index}" for index, state in enumerate(out["state_id"])]
    return out


def stamp_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp canonical schema provenance on a dataframe in place.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to mark as a canonical FRUST result.

    Returns
    -------
    pandas.DataFrame
        The same dataframe with ``frust_schema`` metadata.
    """
    df.attrs["frust_schema"] = {
        "name": "frust-results",
        "version": SCHEMA_VERSION,
    }
    return df


def upgrade_dataframe(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Upgrade legacy workflow outputs to the canonical result schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Legacy or current FRUST dataframe.
    strict : bool, optional
        If ``True``, ambiguous legacy prefixes such as ``"DFT-SP"`` and
        mixed-state ``"OptTS"`` raise an error instead of guessing. If
        ``False``, ambiguous columns are retained and a warning is emitted.

    Returns
    -------
    pandas.DataFrame
        Upgraded copy with canonical identity columns and stage prefixes.

    Examples
    --------
    >>> upgraded = upgrade_dataframe(old_df)
    >>> upgraded[["structure_id", "state_id", "state_kind", "dft_solv_sp-EE"]]
    """
    schema_metadata = getattr(df, "attrs", {}).get("frust_schema", {})
    raw_source_version = (
        schema_metadata.get("version", 0) if isinstance(schema_metadata, dict) else 0
    )
    try:
        source_version = int(raw_source_version)
    except (TypeError, ValueError):
        source_version = 0
    out = canonical_state_columns(df)
    out.attrs.update(getattr(df, "attrs", {}))
    if source_version < 3:
        out = _upgrade_legacy_molecule_state_names(out)
    workflow = str(out.attrs.get("frust_workflow", {}).get("workflow", ""))
    kinds = set(out["state_kind"].dropna().astype(str))
    renamed: dict[Any, str] = {}
    for column in out.columns:
        text = str(column)
        prefix, separator, suffix = text.rpartition("-")
        if not separator:
            continue
        canonical_prefix = CANONICAL_STAGE_PREFIXES.get(prefix)
        if prefix == "DFT-SP":
            if workflow in {"mols", "raw_mols"}:
                canonical_prefix = "dft_solv_sp"
            else:
                _ambiguous_legacy_column(text, strict=strict)
        elif prefix == "OptTS":
            if kinds == {"transition_state"}:
                canonical_prefix = "dft_ts_opt"
            elif kinds <= {"minimum", "constrained_minimum"}:
                canonical_prefix = "dft_opt"
            else:
                _ambiguous_legacy_column(text, strict=strict)
        if canonical_prefix:
            renamed[column] = f"{canonical_prefix}-{suffix}"
    if renamed:
        collisions = set(renamed.values()) & (set(out.columns) - set(renamed))
        if collisions:
            raise ValueError(f"schema upgrade would overwrite canonical columns: {sorted(collisions)}")
        out = out.rename(columns=renamed)
    stamp_schema(out)
    if len(kinds) == 1:
        from frust.results import attach_result_contract

        profile = next(iter(kinds))
        refined_prefixes = ("dft_opt-", "dft_ts_opt-", "dft_freq-", "dft_solv_sp-")
        dft = any(str(column).startswith(refined_prefixes) for column in out.columns)
        attach_result_contract(out, profile, dft=dft)
    return out


def _upgrade_legacy_molecule_state_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename pre-v3 catalytic-cycle state identifiers.

    Parameters
    ----------
    df : pandas.DataFrame
        Legacy dataframe whose ``int2`` means current ``int1`` and whose
        ``mol2`` means current ``int2``.

    Returns
    -------
    pandas.DataFrame
        Copy with state labels and embedded identifiers migrated together.
    """
    out = df.copy()
    for column in ("state_id", "molecule_role"):
        if column in out:
            out[column] = out[column].replace(LEGACY_MOLECULE_STATE_RENAMES)

    if "structure_id" in out:
        out["structure_id"] = out["structure_id"].map(_upgrade_legacy_structure_id)
    if "custom_name" in out:
        out["custom_name"] = out["custom_name"].map(_upgrade_legacy_custom_name)

    builder = out.attrs.get("frust_builder")
    if isinstance(builder, dict):
        migrated_builder = dict(builder)
        state_id = migrated_builder.get("state_id")
        if state_id in LEGACY_MOLECULE_STATE_RENAMES:
            migrated_builder["state_id"] = LEGACY_MOLECULE_STATE_RENAMES[state_id]
        target_id = migrated_builder.get("target_id")
        if target_id is not None:
            migrated_builder["target_id"] = _upgrade_legacy_structure_id(target_id)
        builder_spec = str(migrated_builder.get("builder_spec", ""))
        builder_match = re.match(r"^cycle::(int2|mol2)(::.+)$", builder_spec)
        if builder_match:
            migrated_builder["builder_spec"] = (
                f"cycle::{LEGACY_MOLECULE_STATE_RENAMES[builder_match.group(1)]}"
                f"{builder_match.group(2)}"
            )
        out.attrs["frust_builder"] = migrated_builder
    return out


def _upgrade_legacy_structure_id(value: object) -> object:
    """Rename a legacy molecule state embedded in a structure identifier."""
    if not isinstance(value, str):
        return value
    pattern = r":(int2|mol2)(?=:r\d+$|$)"
    return re.sub(
        pattern,
        lambda match: f":{LEGACY_MOLECULE_STATE_RENAMES[match.group(1)]}",
        value,
    )


def _upgrade_legacy_custom_name(value: object) -> object:
    """Rename a legacy molecule state embedded in a generated display name."""
    if not isinstance(value, str):
        return value
    pattern = r"(?P<prefix>^|_)(?P<state>int2|mol2)(?=_rpos\(|$)"
    return re.sub(
        pattern,
        lambda match: (
            f"{match.group('prefix')}"
            f"{LEGACY_MOLECULE_STATE_RENAMES[match.group('state')]}"
        ),
        value,
    )


def _ambiguous_legacy_column(column: str, *, strict: bool) -> None:
    message = (
        f"Cannot safely map legacy result column {column!r}; add workflow/state "
        "provenance or call upgrade_dataframe(..., strict=False) to retain it"
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def infer_group_columns(df: pd.DataFrame) -> list[str]:
    """Choose columns that identify one chemical object for lowest filtering."""
    if "structure_id" in df.columns:
        return ["structure_id"]
    preferred = [
        "system_name",
        "substrate_name",
        "catalyst_name",
        "structure_type",
        "molecule_role",
        "rpos",
    ]
    return [col for col in preferred if col in df.columns]


def parse_structure_name(name: str, smiles: str | None = None) -> StructureMetadata:
    """Parse legacy structure names when no structured metadata is available."""
    text = str(name)

    wrapped = re.match(
        r"^(?P<stype>(?:TS|INT)\d*)\((?P<body>.+)_rpos\((?P<rpos>\d+)\)\)$",
        text,
    )
    if wrapped:
        stype = wrapped.group("stype").upper()
        substrate = wrapped.group("body")
        rpos = int(wrapped.group("rpos"))
        return StructureMetadata(
            structure_id=f"{stype}:{substrate}:r{rpos}",
            custom_name=text,
            substrate_name=substrate,
            structure_type=stype,
            molecule_role="ts" if stype.startswith("TS") else stype.lower(),
            rpos=rpos,
            smiles=smiles,
            input_smiles=smiles,
        )

    rpos_match = re.match(r"^(?P<base>.+)_(?P<role>[^_]+)_rpos\((?P<rpos>\d+)\)$", text)
    if rpos_match:
        base = rpos_match.group("base")
        role = rpos_match.group("role")
        rpos = int(rpos_match.group("rpos"))
        substrate = _substrate_from_base(base)
        return StructureMetadata(
            structure_id=f"MOL:{substrate}:{role}:r{rpos}",
            custom_name=text,
            substrate_name=substrate,
            structure_type="MOL",
            molecule_role=role,
            rpos=rpos,
            smiles=smiles,
            input_smiles=smiles or base,
        )

    if "_" in text:
        base, role_or_name = text.rsplit("_", 1)
        role = role_or_name if role_or_name in _KNOWN_ROLES else "structure"
        substrate = role_or_name if role == "structure" else _substrate_from_base(base)
    else:
        role = text if text in _KNOWN_ROLES else "structure"
        substrate = text

    return StructureMetadata(
        structure_id=f"MOL:{substrate}:{role}",
        custom_name=text,
        substrate_name=substrate,
        structure_type="MOL",
        molecule_role=role,
        rpos=pd.NA,
        smiles=smiles,
        input_smiles=smiles,
    )


def metadata_from_mapping(
    metadata: dict[str, Any] | None,
    *,
    fallback_name: str,
    smiles: str | None = None,
) -> StructureMetadata:
    """Build complete metadata from an optional partial mapping."""
    if not metadata:
        return parse_structure_name(fallback_name, smiles=smiles)

    parsed = parse_structure_name(str(metadata.get("custom_name", fallback_name)), smiles=smiles)
    substrate = metadata.get("substrate_name", parsed.substrate_name)
    rpos = metadata.get("rpos", parsed.rpos)
    if rpos is None:
        rpos = pd.NA
    return StructureMetadata(
        structure_id=str(metadata.get("structure_id", parsed.structure_id)),
        custom_name=str(metadata.get("custom_name", fallback_name)),
        substrate_name=str(substrate),
        structure_type=str(metadata.get("structure_type", parsed.structure_type)).upper(),
        molecule_role=str(metadata.get("molecule_role", parsed.molecule_role)),
        rpos=int(rpos) if rpos is not pd.NA and not pd.isna(rpos) else pd.NA,
        smiles=metadata.get("smiles", smiles),
        input_smiles=metadata.get("input_smiles", metadata.get("smiles", smiles)),
    )


_KNOWN_ROLES = {
    "dimer",
    "HH",
    "ligand",
    "catalyst",
    "int1",
    "int2",
    "HBpin-ligand",
    "HBpin-mol",
    "ts",
    "structure",
}


def _substrate_from_base(base: str) -> str:
    """Best-effort substrate label from old SMILES-prefixed molecule keys."""
    if "_" in base:
        return base.rsplit("_", 1)[-1]
    return base
