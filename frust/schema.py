"""DataFrame schema helpers for FRUST results."""

from __future__ import annotations

import re
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


def infer_group_columns(df: pd.DataFrame) -> list[str]:
    """Choose columns that identify one chemical object for lowest filtering."""
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
    "int2",
    "mol2",
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
