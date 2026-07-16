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
    }
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
    return step.build_initial_df(payload, n_confs=n_confs, n_cores=n_cores)


def _build_connected(
    target: StructureTarget,
    *,
    n_confs: int | None,
    n_cores: int,
    ts_guess_factory: Any,
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
    )
    return grouped[target.state_id]
