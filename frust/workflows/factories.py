"""Public workflow factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from frust.cluster.naming import sanitize_tag
from frust.screen import create_ts_guesses
from frust.screen import expand as expand_screen
from frust.screen import read as read_screen
from frust.schema import parse_structure_name
from frust.stepper import Stepper
from frust.tsguess.matching import parse_rpos_value
from frust.tsguess.specs import BUILTIN_TS_SPECS
from frust.utils.io import read_ts_type_from_xyz
from frust.utils.mols import create_mol_per_rpos, create_ts_per_rpos
from frust.workflows.core import BaseWorkflow, ExecutionOptions, StageDef, WorkflowTarget
from frust.workflows.methods import MethodPlan


SplitMode = Literal["per_input", "per_rpos"]


class MolsWorkflow(BaseWorkflow):
    """Workflow for catalytic-cycle molecular states."""

    workflow_name = "mols"

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        smiles: list[str] | None = None,
        split: SplitMode = "per_rpos",
        select_mols: str | list[str] = "all",
        method: MethodPlan | str | None = None,
        n_confs: int | None = None,
        top_n: int = 10,
        dft: bool = False,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.dataframe = dataframe
        self.smiles = smiles
        self.split = split
        self.select_mols = select_mols

    def _input_df(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe.copy()
        if self.csv_path is not None:
            return pd.read_csv(self.csv_path)
        if self.smiles is not None:
            return pd.DataFrame({"smiles": list(self.smiles)})
        raise ValueError("MolsWorkflow requires csv_path, dataframe, or smiles")

    def _build_targets(self) -> list[WorkflowTarget]:
        df = self._input_df()
        if "smiles" not in df.columns:
            raise ValueError("mols workflow input must contain a 'smiles' column")
        if self.split == "per_input":
            targets = []
            for idx, row in df.iterrows():
                name = row.get("compound_name") or row.get("substrate_name") or f"row_{idx:03d}"
                targets.append(
                    WorkflowTarget(
                        tag=sanitize_tag(str(name)),
                        payload=pd.DataFrame([row]),
                        metadata={"input_index": int(idx), "smiles": row["smiles"]},
                    )
                )
            return targets
        if self.split != "per_rpos":
            raise ValueError("split must be 'per_input' or 'per_rpos'")

        jobs = create_mol_per_rpos(
            df,
            return_format="list",
            select_mols=self.select_mols,
        )
        return [
            WorkflowTarget(
                tag=sanitize_tag(list(job.keys())[0]),
                payload=job,
                metadata={"kind": "molecule"},
            )
            for job in jobs
        ]

    def _prepare_initial_df(
        self,
        target: WorkflowTarget,
        *,
        save_dir: Path | None,
        options: ExecutionOptions,
    ) -> pd.DataFrame:
        del save_dir
        payload = target.payload
        if isinstance(payload, pd.DataFrame):
            payload = create_mol_per_rpos(
                payload,
                return_format="dict",
                select_mols=self.select_mols,
            )
        step = Stepper(
            step_type="MOLS",
            n_cores=options.n_cores,
            memory_gb=options.mem_gb,
            debug=options.debug,
            save_output_dir=False,
        )
        return step.build_initial_df(
            payload,
            n_confs=self.n_confs,
            n_cores=options.n_cores,
        )

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        del target
        return "MOLS"

    def _stage_defs(self) -> list[StageDef]:
        stages = [
            StageDef("prepare", "prepare", kind="prepare"),
            StageDef("xtb_preopt", "xtb_preopt", n_cores=2),
            StageDef("xtb_sp", "xtb_sp", n_cores=2),
            StageDef("xtb_opt", "xtb_opt", lowest=self.top_n, n_cores=2),
        ]
        if self.dft:
            stages.extend(
                [
                    StageDef("dft_pre_sp", "DFT-pre-SP"),
                    StageDef("dft_opt", "DFT-Opt", lowest=1),
                    StageDef("solv", "DFT-SP"),
                ]
            )
        else:
            stages.append(StageDef("filter", "filter", kind="filter"))
        return stages


class ScreenTSWorkflow(BaseWorkflow):
    """Workflow for substrate/catalyst screen TS guesses."""

    workflow_name = "screen_ts"

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
        method: MethodPlan | str | None = None,
        n_confs: int | None = None,
        top_n: int = 10,
        dft: bool = True,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.dataframe = dataframe
        self.ts_types = tuple(str(ts_type).upper() for ts_type in ts_types)

    def _systems(self) -> pd.DataFrame:
        system_cols = {"system_name", "substrate_smiles", "catalyst_smiles", "rpos"}
        if self.dataframe is not None and system_cols.issubset(self.dataframe.columns):
            return self.dataframe.copy()
        source = self.dataframe if self.dataframe is not None else self.csv_path
        if source is None:
            raise ValueError("screen_ts workflow requires csv_path or dataframe")
        return expand_screen(read_screen(source))

    def _build_targets(self) -> list[WorkflowTarget]:
        unknown = sorted(set(self.ts_types) - set(BUILTIN_TS_SPECS))
        if unknown:
            supported = ", ".join(sorted(BUILTIN_TS_SPECS))
            raise ValueError(f"Unsupported screen TS types {unknown}. Supported: {supported}")
        systems = self._systems()
        targets: list[WorkflowTarget] = []
        for _, system in systems.iterrows():
            rpos_values = parse_rpos_value(system.get("rpos"), str(system["substrate_smiles"]))
            for ts_type in self.ts_types:
                for rpos in rpos_values:
                    target = system.copy()
                    target["rpos"] = int(rpos)
                    target["ts_type"] = ts_type
                    tag = sanitize_tag(f"{ts_type}__{system['system_name']}__r{int(rpos)}")
                    targets.append(
                        WorkflowTarget(
                            tag=tag,
                            payload=pd.DataFrame([target]),
                            metadata={
                                "ts_type": ts_type,
                                "system_name": system["system_name"],
                                "rpos": int(rpos),
                            },
                        )
                    )
        return targets

    def _prepare_initial_df(
        self,
        target: WorkflowTarget,
        *,
        save_dir: Path | None,
        options: ExecutionOptions,
    ) -> pd.DataFrame:
        screen_target = target.payload
        ts_type = str(screen_target["ts_type"].iloc[0]).upper()
        guesses = create_ts_guesses(
            screen_target,
            ts_types=[ts_type],
            n_confs=self.n_confs,
            n_cores=options.n_cores,
        )
        df = guesses[ts_type]
        if save_dir is not None:
            df.to_parquet(save_dir / "ts_guess.parquet")
        return df

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        metadata = target.metadata or {}
        return metadata.get("ts_type")

    def _stage_defs(self) -> list[StageDef]:
        stages = _ts_init_stages(self.top_n)
        if self.dft:
            stages.extend(_ts_dft_stages())
        else:
            stages.append(StageDef("filter", "filter", kind="filter"))
        return stages


class LegacyTSWorkflow(BaseWorkflow):
    """Workflow for legacy transformer TS/INT inputs."""

    workflow_name = "legacy_ts"

    def __init__(
        self,
        *,
        csv_path: str | Path,
        ts_xyz: str | Path,
        int3: bool = False,
        method: MethodPlan | str | None = None,
        n_confs: int | None = None,
        top_n: int = 10,
        dft: bool = True,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.ts_xyz = ts_xyz
        self.int3 = int3

    def _build_targets(self) -> list[WorkflowTarget]:
        df = pd.read_csv(self.csv_path)
        jobs = create_ts_per_rpos(df, str(self.ts_xyz), return_format="list")
        return [
            WorkflowTarget(
                tag=sanitize_tag(list(job.keys())[0]),
                payload=job,
                metadata={"structure_type": self._default_step_type(job)},
            )
            for job in jobs
        ]

    def _default_step_type(self, job: dict[str, Any]) -> str:
        if self.int3:
            return "INT3"
        try:
            return read_ts_type_from_xyz(str(self.ts_xyz)).upper()
        except Exception:
            name = list(job.keys())[0]
            return parse_structure_name(name).structure_type.upper()

    def _prepare_initial_df(
        self,
        target: WorkflowTarget,
        *,
        save_dir: Path | None,
        options: ExecutionOptions,
    ) -> pd.DataFrame:
        del save_dir
        step = Stepper(
            step_type=self._step_type_for_target(target),
            n_cores=options.n_cores,
            memory_gb=options.mem_gb,
            debug=options.debug,
            save_output_dir=False,
        )
        return step.build_initial_df(
            target.payload,
            n_confs=self.n_confs,
            n_cores=options.n_cores,
            ts_type=self._step_type_for_target(target),
            ts_optimize=not options.debug,
        )

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        return (target.metadata or {}).get("structure_type")

    def _stage_defs(self) -> list[StageDef]:
        stages = _ts_init_stages(self.top_n)
        if not self.dft:
            stages.append(StageDef("filter", "filter", kind="filter"))
            return stages
        if self.int3:
            stages.extend(
                [
                    StageDef("dft_opt", "OptTS", lowest=1),
                    StageDef("freq", "Freq"),
                    StageDef("solv", "DFT-solv"),
                ]
            )
        else:
            stages.extend(_ts_dft_stages())
        return stages


def mols(
    *,
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    smiles: list[str] | None = None,
    split: SplitMode = "per_rpos",
    select_mols: str | list[str] = "all",
    method: MethodPlan | str | None = None,
    n_confs: int | None = None,
    top_n: int = 10,
    dft: bool = False,
) -> MolsWorkflow:
    """Create a molecule workflow."""
    return MolsWorkflow(
        csv_path=csv_path,
        dataframe=dataframe,
        smiles=smiles,
        split=split,
        select_mols=select_mols,
        method=method,
        n_confs=n_confs,
        top_n=top_n,
        dft=dft,
    )


def screen_ts(
    *,
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
    method: MethodPlan | str | None = None,
    n_confs: int | None = None,
    top_n: int = 10,
    dft: bool = True,
) -> ScreenTSWorkflow:
    """Create a screen TS workflow."""
    return ScreenTSWorkflow(
        csv_path=csv_path,
        dataframe=dataframe,
        ts_types=ts_types,
        method=method,
        n_confs=n_confs,
        top_n=top_n,
        dft=dft,
    )


def legacy_ts(
    *,
    csv_path: str | Path,
    ts_xyz: str | Path,
    method: MethodPlan | str | None = None,
    n_confs: int | None = None,
    top_n: int = 10,
    dft: bool = True,
) -> LegacyTSWorkflow:
    """Create a legacy TS workflow from a transformer template XYZ file."""
    return LegacyTSWorkflow(
        csv_path=csv_path,
        ts_xyz=ts_xyz,
        method=method,
        n_confs=n_confs,
        top_n=top_n,
        dft=dft,
    )


def int3(
    *,
    csv_path: str | Path,
    ts_xyz: str | Path,
    method: MethodPlan | str | None = None,
    n_confs: int | None = None,
    top_n: int = 10,
    dft: bool = True,
) -> LegacyTSWorkflow:
    """Create a legacy INT3 workflow from a transformer template XYZ file."""
    workflow = LegacyTSWorkflow(
        csv_path=csv_path,
        ts_xyz=ts_xyz,
        int3=True,
        method=method,
        n_confs=n_confs,
        top_n=top_n,
        dft=dft,
    )
    workflow.workflow_name = "int3"
    return workflow


def _ts_init_stages(top_n: int) -> list[StageDef]:
    """Return common constrained TS initialization stages."""
    return [
        StageDef("prepare", "prepare", kind="prepare"),
        StageDef("xtb_preopt", "xtb_preopt", constraint=True, n_cores=2),
        StageDef("xtb_sp", "xtb_sp", n_cores=2),
        StageDef("xtb_opt", "xtb_opt", constraint=True, lowest=top_n, n_cores=2),
        StageDef("dft_pre_sp", "DFT-pre-SP"),
        StageDef("dft_pre_opt", "DFT-pre-Opt", constraint=True, lowest=1),
    ]


def _ts_dft_stages() -> list[StageDef]:
    """Return common TS DFT continuation stages."""
    return [
        StageDef("hess", "Hess", read_files=["input.hess"]),
        StageDef("optts", "OptTS", use_last_hess=True),
        StageDef("freq", "Freq"),
        StageDef("solv", "DFT-solv"),
    ]
