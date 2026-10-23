"""Single construction facade for modern FRUST structure targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from frust.schema import canonical_state_columns, stamp_schema
from frust.screen import create_ts_guesses
from frust.stepper import Stepper
from frust.structures.models import StructureTarget
from frust.utils.mols import create_mol_per_rpos


def build(
    target: StructureTarget,
    *,
    n_confs: int | None,
    n_cores: int,
    memory_gb: int,
    debug: bool,
    save_dir: Path | None = None,
    stepper_cls: type[Stepper] = Stepper,
    ts_guess_factory: Any = create_ts_guesses,
    mol_factory: Any = create_mol_per_rpos,
    spec_profile: str = "wb97xd3-631g/gas",
    spec_match: str = "prefer-exact",
) -> pd.DataFrame:
    """Construct an initial dataframe for one typed target.

    Parameters
    ----------
    target : StructureTarget
        Lightweight target returned by :func:`plan_targets`.
    n_confs, n_cores, memory_gb, debug
        Runtime embedding options.
    save_dir : pathlib.Path or None, optional
        Directory used for the inspectable pre-calculation TS/INT guess.
    stepper_cls, ts_guess_factory, mol_factory : callable, optional
        Injectable collaborators used by workflow tests.
    spec_profile : str, optional
        Method/environment geometry profile for connected constrained states.
    spec_match : {"prefer-exact", "exact"}, optional
        Geometry-profile matching policy.

    Returns
    -------
    pandas.DataFrame
        Canonical initial structure dataframe.
    """
    if target.builder_spec.startswith("cycle::"):
        df = _build_cycle(
            target,
            n_confs=n_confs,
            n_cores=n_cores,
            memory_gb=memory_gb,
            debug=debug,
            stepper_cls=stepper_cls,
            mol_factory=mol_factory,
        )
    elif target.builder_spec.startswith("connected_graph::"):
        df = _build_connected(
            target,
            n_confs=n_confs,
            n_cores=n_cores,
            ts_guess_factory=ts_guess_factory,
            spec_profile=spec_profile,
            spec_match=spec_match,
        )
        if save_dir is not None:
            df.to_parquet(save_dir / "structure_guess.parquet")
    else:
        raise ValueError(f"Unsupported structure builder {target.builder_spec!r}")

    out = canonical_state_columns(
        df,
        state_id=target.state_id,
        state_kind=target.state_kind,
        structure_id=target.target_id,
    )
    out["system_name"] = target.system.system_name
    out["substrate_name"] = target.system.substrate_name
    out["catalyst_name"] = target.system.catalyst_name
    if target.builder_spec.startswith("cycle::"):
        out["structure_type"] = "MOL"
        out["molecule_role"] = target.state_id
    if target.rpos is not None:
        out["rpos"] = int(target.rpos)
    elif "rpos" not in out:
        out["rpos"] = pd.NA
    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["frust_builder"] = {
        "schema_version": 3,
        "builder_spec": target.builder_spec,
        "target_id": target.target_id,
        "state_id": target.state_id,
        "state_kind": target.state_kind,
        "stereochemistry": dict(target.system.metadata),
    }
    if "frust_stereo" not in out.attrs and "smiles" in out and not out.empty:
        from rdkit import Chem
        from frust.structures.stereo import _stereo_spec

        mol = Chem.MolFromSmiles(str(out.iloc[0]["smiles"]))
        if mol is not None:
            out.attrs["frust_stereo"] = _stereo_spec(mol)
    from frust.structures.stereo import check_geometry

    spec = out.attrs.get("frust_stereo")
    if spec and spec["centres"]:
        out.attrs["frust_stereo_embedding"] = [
            check_geometry(spec, row.get("coords_embedded"), row.get("atoms"))
            for _, row in out.iterrows()
        ]
    if spec is not None:
        out.attrs["frust_stereo"] = {"targets": {target.target_id: spec}}
    stamp_schema(out)
    return out


def _build_cycle(
    target: StructureTarget,
    *,
    n_confs: int | None,
    n_cores: int,
    memory_gb: int,
    debug: bool,
    stepper_cls: type[Stepper],
    mol_factory: Any,
) -> pd.DataFrame:
    row = {
        "smiles": target.system.substrate_smiles,
        "substrate_name": target.system.substrate_name,
        "compound_name": target.system.substrate_name,
        "catalyst_smiles": target.system.catalyst_smiles,
        "system_name": target.system.system_name,
    }
    if target.rpos is not None:
        row["rpos"] = int(target.rpos)
    if target.system.metadata.get("dimer_pairing") == "racemic":
        from frust.transformers import _build_dimer_pair

        mol = _build_dimer_pair(target.state_id, target.system.catalyst_smiles,
                               target.system.metadata["dimer_partner_smiles"])
        payload = {target.tag: (mol, {"molecule_role": target.state_id,
                                     "structure_type": "MOL"})}
    else:
        payload = mol_factory(
            pd.DataFrame([row]),
            return_format="dict",
            select_mols=[target.state_id],
            show_iupac=False,
        )
    step = stepper_cls(
        step_type="MOLS",
        n_cores=n_cores,
        memory_gb=memory_gb,
        debug=debug,
        save_output_dir=False,
    )
    frame = step.build_initial_df(payload, n_confs=n_confs, n_cores=n_cores)
    from frust.structures.stereo import _stereo_spec

    if isinstance(payload, dict) and payload:
        from rdkit.Chem.rdchem import Mol

        value = next(iter(payload.values()))
        mol = value[0] if isinstance(value, tuple) else value
        if isinstance(mol, Mol):
            frame.attrs["frust_stereo"] = _stereo_spec(mol)
    return frame


def _build_connected(
    target: StructureTarget,
    *,
    n_confs: int | None,
    n_cores: int,
    ts_guess_factory: Any,
    spec_profile: str,
    spec_match: str,
) -> pd.DataFrame:
    if target.rpos is None:
        raise ValueError(f"Connected-graph target {target.target_id!r} requires rpos")
    row = target.system.as_dict()
    row.update({"smiles": target.system.substrate_smiles, "rpos": int(target.rpos)})
    grouped = ts_guess_factory(
        pd.DataFrame([row]),
        ts_types=[target.state_id],
        n_confs=n_confs,
        n_cores=n_cores,
        backend="tsguess2",
        spec_profile=spec_profile,
        spec_match=spec_match,
    )
    return grouped[target.state_id]
