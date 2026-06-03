"""Concrete workflow factories for FRUST chemistry workflows.

The public functions in this module create workflow objects but do not run
calculators. A workflow object first expands user input into lightweight
``WorkflowTarget`` objects, then ``BaseWorkflow.run`` or ``BaseWorkflow.submit``
prepares structures and executes the stage graph.
"""

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


def _row_label(row: pd.Series, position: int, columns: tuple[str, ...]) -> str:
    """Return the first non-empty label value from a dataframe row."""
    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return f"row_{position:03d}"


def _unique_sanitized_tags(labels: list[str]) -> list[str]:
    """Return scheduler-safe unique tags while preserving input order."""
    used: set[str] = set()
    next_suffix: dict[str, int] = {}
    tags: list[str] = []
    for label in labels:
        base = sanitize_tag(label)
        candidate = base
        suffix = next_suffix.get(base, 1)
        while candidate in used:
            candidate = f"{base}_{suffix:03d}"
            suffix += 1
        next_suffix[base] = suffix
        used.add(candidate)
        tags.append(candidate)
    return tags


def _molecule_stage_defs(*, top_n: int, dft: bool) -> list[StageDef]:
    """Return the shared molecule stage graph."""
    stages = [
        StageDef("prepare", "prepare", kind="prepare"),
        StageDef("xtb_preopt", "xtb_preopt", n_cores=2),
        StageDef("xtb_sp", "xtb_sp", n_cores=2),
        StageDef("xtb_opt", "xtb_opt", lowest=top_n, n_cores=2),
    ]
    if dft:
        stages.extend(
            [
                StageDef("dft_pre_sp", "DFT-pre-SP"),
                StageDef("dft_opt", "DFT-Opt", lowest=1),
                StageDef("freq", "Freq"),
                StageDef("solv", "DFT-SP"),
            ]
        )
    else:
        stages.append(StageDef("filter", "filter", kind="filter"))
    return stages


class MolsWorkflow(BaseWorkflow):
    """Workflow for catalytic-cycle molecular states.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        CSV file containing at least a ``smiles`` column. Optional columns such
        as ``compound_name``, ``substrate_name``, and ``rpos`` are used to label
        and expand targets.
    dataframe : pandas.DataFrame or None, optional
        In-memory input table with the same columns as ``csv_path``.
    smiles : list of str or None, optional
        Direct list of SMILES strings for quick molecule workflows.
    split : {"per_input", "per_rpos"}, optional
        Target expansion mode. ``"per_input"`` creates one target per input row.
        ``"per_rpos"`` expands catalytic-cycle molecule structures per reactive
        position using :func:`frust.utils.mols.create_mol_per_rpos`.
    select_mols : str or list of str, optional
        Molecule subset forwarded to ``create_mol_per_rpos``. Common values are
        ``"all"``, ``"uniques"``, and ``"generics"``.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain stage keys this molecule workflow does not use; call
        ``wf.show_stages()`` to inspect the active stages.
    n_confs : int or None, optional
        Conformer count passed to ``Stepper.build_initial_df``.
    top_n : int, optional
        Number of rows kept after ranking/filtering stages.
    dft : bool, optional
        If ``True``, add DFT optimization, frequency, and solvent stages. If
        ``False``, end with a lowest-energy filter after xTB stages.

    Notes
    -----
    The default stage graph is ``prepare -> xtb_preopt -> xtb_sp -> xtb_opt``.
    DFT workflows then run ``dft_pre_sp -> dft_opt -> freq -> solv``;
    non-DFT workflows run a final ``filter`` stage.
    """

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
        """Return the molecule workflow input table.

        Returns
        -------
        pandas.DataFrame
            Copy of the input dataframe, CSV contents, or a dataframe built from
            ``smiles``.

        Raises
        ------
        ValueError
            If no input source was supplied.
        """
        if self.dataframe is not None:
            return self.dataframe.copy()
        if self.csv_path is not None:
            return pd.read_csv(self.csv_path)
        if self.smiles is not None:
            return pd.DataFrame({"smiles": list(self.smiles)})
        raise ValueError("MolsWorkflow requires csv_path, dataframe, or smiles")

    def _build_targets(self) -> list[WorkflowTarget]:
        """Build molecule workflow targets.

        Returns
        -------
        list of WorkflowTarget
            ``per_input`` targets carry one input row. ``per_rpos`` targets carry
            one prepared molecule payload returned by ``create_mol_per_rpos``.
        """
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
        """Embed one molecule target into the initial FRUST dataframe.

        Parameters
        ----------
        target : WorkflowTarget
            Molecule target selected for execution.
        save_dir : pathlib.Path or None
            Unused for molecule preparation.
        options : ExecutionOptions
            Runtime options controlling conformer embedding.

        Returns
        -------
        pandas.DataFrame
            Initial molecule dataframe with atoms and ``coords_embedded``.
        """
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
        """Return the Stepper type for molecule calculations."""
        del target
        return "MOLS"

    def _stage_defs(self) -> list[StageDef]:
        """Return molecule workflow stages."""
        return _molecule_stage_defs(top_n=self.top_n, dft=self.dft)


class RawMolsWorkflow(BaseWorkflow):
    """Workflow for explicit molecule SMILES without FRUST cycle expansion.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        CSV file containing one exact molecule per row in a ``smiles`` column.
    dataframe : pandas.DataFrame or None, optional
        In-memory input table with the same columns as ``csv_path``.
    smiles : list of str or None, optional
        Direct list of exact molecule SMILES strings.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain TS-specific calculator keys, but ``raw_mols`` only uses the
        molecule stages shown by ``wf.show_stages()``.
    n_confs : int or None, optional
        Conformer count passed to ``Stepper.build_initial_df``.
    top_n : int, optional
        Number of rows kept after ranking/filtering stages.
    dft : bool, optional
        If ``True``, add DFT optimization, frequency, and solvent stages. If
        ``False``, end with a lowest-energy filter after xTB stages.

    Notes
    -----
    This workflow treats each input SMILES as the structure to calculate. It
    does not call ``create_mol_per_rpos`` and does not support ``select_mols``.
    With ``dft=True``, the active DFT stages are ``dft_pre_sp -> dft_opt ->
    freq -> solv``. The ``freq`` stage is a normal minimum-frequency check used
    for thermochemistry; TS-specific ``hess`` and ``optts`` stages are not run.
    """

    workflow_name = "raw_mols"

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        smiles: list[str] | None = None,
        method: MethodPlan | str | None = None,
        n_confs: int | None = None,
        top_n: int = 10,
        dft: bool = False,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.dataframe = dataframe
        self.smiles = smiles

    def _input_df(self) -> pd.DataFrame:
        """Return the raw molecule workflow input table."""
        if self.dataframe is not None:
            return self.dataframe.copy()
        if self.csv_path is not None:
            return pd.read_csv(self.csv_path)
        if self.smiles is not None:
            return pd.DataFrame({"smiles": list(self.smiles)})
        raise ValueError("RawMolsWorkflow requires csv_path, dataframe, or smiles")

    def _normalized_input_df(self) -> pd.DataFrame:
        """Return validated raw molecule inputs with stable labels."""
        df = self._input_df().copy()
        if "smiles" not in df.columns:
            raise ValueError("raw_mols workflow input must contain a 'smiles' column")
        if df["smiles"].isna().any():
            raise ValueError("raw_mols workflow input contains missing SMILES values")

        if "compound_name" in df.columns:
            if "substrate_name" not in df.columns:
                df["substrate_name"] = df["compound_name"]
            else:
                existing = df["substrate_name"].astype("string").fillna("").str.strip()
                missing = existing.eq("")
                df.loc[missing, "substrate_name"] = df.loc[missing, "compound_name"]

        if "substrate_name" not in df.columns:
            df["substrate_name"] = [
                _row_label(row, pos, ("compound_name", "name", "custom_name"))
                for pos, (_, row) in enumerate(df.iterrows())
            ]
        return df

    def _build_targets(self) -> list[WorkflowTarget]:
        """Build one raw molecule target per input row."""
        df = self._normalized_input_df()
        labels = [
            _row_label(
                row,
                pos,
                ("compound_name", "substrate_name", "name", "custom_name"),
            )
            for pos, (_, row) in enumerate(df.iterrows())
        ]
        tags = _unique_sanitized_tags(labels)
        targets: list[WorkflowTarget] = []
        for pos, ((idx, row), tag) in enumerate(zip(df.iterrows(), tags)):
            payload = pd.DataFrame([row]).reset_index(drop=True)
            targets.append(
                WorkflowTarget(
                    tag=tag,
                    payload=payload,
                    metadata={
                        "kind": "raw_molecule",
                        "input_index": idx,
                        "input_position": int(pos),
                        "smiles": row["smiles"],
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
        """Embed one exact molecule target into the initial FRUST dataframe."""
        del save_dir
        step = Stepper(
            step_type="MOLS",
            n_cores=options.n_cores,
            memory_gb=options.mem_gb,
            debug=options.debug,
            save_output_dir=False,
        )
        return step.build_initial_df(
            target.payload,
            n_confs=self.n_confs,
            n_cores=options.n_cores,
        )

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        """Return the Stepper type for raw molecule calculations."""
        del target
        return "MOLS"

    def _stage_defs(self) -> list[StageDef]:
        """Return raw molecule workflow stages."""
        return _molecule_stage_defs(top_n=self.top_n, dft=self.dft)


class ScreenTSWorkflow(BaseWorkflow):
    """Workflow for substrate/catalyst transition-state screens.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        Component CSV accepted by :func:`frust.screen.read`. The table normally
        contains substrate and catalyst rows with ``role``, ``smiles``, optional
        ``compound_name``, and optional substrate ``rpos``.
    dataframe : pandas.DataFrame or None, optional
        Component dataframe accepted by ``frust.screen.read`` or an already
        expanded systems dataframe containing ``system_name``,
        ``substrate_smiles``, ``catalyst_smiles``, and ``rpos``.
    ts_types : tuple or list of str, optional
        Built-in TS types to generate. Supported values are ``"TS1"``,
        ``"TS2"``, ``"TS3"``, and ``"TS4"``.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain molecule-specific calculator keys this TS workflow does not use;
        call ``wf.show_stages()`` to inspect the active stages.
    n_confs : int or None, optional
        Number of TS guess conformers generated per target. ``None`` uses the
        screen TS conformer heuristic.
    top_n : int, optional
        Number of rows kept after constrained xTB optimization.
    dft : bool, optional
        If ``True``, add Hessian, ``OptTS``, frequency, and solvent DFT stages.
        If ``False``, stop after xTB ranking/filtering.

    Notes
    -----
    Targets are the Cartesian product of expanded systems, requested TS types,
    and reactive positions. Each target prepares TS guesses through
    :func:`frust.screen.create_ts_guesses`, writes ``ts_guess.parquet`` when an
    output directory is available, and then runs the shared TS stage graph.
    """

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
        """Return expanded substrate/catalyst systems.

        Returns
        -------
        pandas.DataFrame
            Expanded systems dataframe. Already-expanded dataframes are copied
            directly; component tables are normalized with ``frust.screen.read``
            and expanded with ``frust.screen.expand``.
        """
        system_cols = {"system_name", "substrate_smiles", "catalyst_smiles", "rpos"}
        if self.dataframe is not None and system_cols.issubset(self.dataframe.columns):
            return self.dataframe.copy()
        source = self.dataframe if self.dataframe is not None else self.csv_path
        if source is None:
            raise ValueError("screen_ts workflow requires csv_path or dataframe")
        return expand_screen(read_screen(source))

    def _build_targets(self) -> list[WorkflowTarget]:
        """Build one TS target per system, TS type, and reactive position.

        Returns
        -------
        list of WorkflowTarget
            Targets whose payload is a one-row systems dataframe with resolved
            ``ts_type`` and integer ``rpos``.
        """
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
        """Generate TS guesses for one screen target.

        Parameters
        ----------
        target : WorkflowTarget
            One system, TS type, and reactive position.
        save_dir : pathlib.Path or None
            Target output directory. When provided, the raw TS guesses are also
            written as ``ts_guess.parquet`` before calculator stages start.
        options : ExecutionOptions
            Runtime options controlling TS guess conformer generation.

        Returns
        -------
        pandas.DataFrame
            TS guess dataframe for the target's TS type.
        """
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
        """Return the target TS type for Stepper dispatch."""
        metadata = target.metadata or {}
        return metadata.get("ts_type")

    def _stage_defs(self) -> list[StageDef]:
        """Return screen TS workflow stages."""
        stages = _ts_init_stages(self.top_n)
        if self.dft:
            stages.extend(_ts_dft_stages())
        else:
            stages.append(StageDef("filter", "filter", kind="filter"))
        return stages


class LegacyTSWorkflow(BaseWorkflow):
    """Workflow for legacy transformer TS and INT template inputs.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Ligand/substrate CSV containing SMILES and optional ``rpos`` values.
    ts_xyz : str or pathlib.Path
        Template XYZ file used by the legacy transformer functions. The TS type
        is inferred from the XYZ comment line unless ``int3=True`` is set.
    int3 : bool, optional
        Treat the workflow as an INT3 workflow instead of a TS workflow.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain stage keys this legacy workflow does not use; call
        ``wf.show_stages()`` to inspect the active stages.
    n_confs : int or None, optional
        Conformer count passed to ``Stepper.build_initial_df``.
    top_n : int, optional
        Number of rows kept after constrained xTB optimization.
    dft : bool, optional
        If ``True``, add TS or INT3 DFT continuation stages. If ``False``, stop
        after xTB ranking/filtering.

    Notes
    -----
    This workflow preserves the older template-transformer path based on
    :func:`frust.utils.mols.create_ts_per_rpos`. For new substrate/catalyst
    screens, prefer :func:`screen_ts`.
    """

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
        """Build legacy transformer targets from the input CSV and template."""
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
        """Infer the Stepper type for a legacy transformed target.

        Parameters
        ----------
        job : dict
            Single transformed structure payload.

        Returns
        -------
        str
            ``"INT3"`` when requested, otherwise the TS type inferred from the
            template XYZ file or generated structure name.
        """
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
        """Embed one legacy TS or INT target into an initial dataframe.

        Parameters
        ----------
        target : WorkflowTarget
            Legacy transformed target.
        save_dir : pathlib.Path or None
            Unused for legacy preparation.
        options : ExecutionOptions
            Runtime options controlling embedding and TS optimization.

        Returns
        -------
        pandas.DataFrame
            Initial TS or INT dataframe for calculator stages.
        """
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
        """Return the TS or INT structure type stored in target metadata."""
        return (target.metadata or {}).get("structure_type")

    def _stage_defs(self) -> list[StageDef]:
        """Return legacy TS or INT3 workflow stages."""
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
    """Create a molecule-state workflow.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        CSV file with a ``smiles`` column. Optional ``rpos`` values control
        reactive-position expansion when ``split="per_rpos"``.
    dataframe : pandas.DataFrame or None, optional
        In-memory input table with the same columns as ``csv_path``.
    smiles : list of str or None, optional
        Quick input for simple molecule workflows.
    split : {"per_input", "per_rpos"}, optional
        ``"per_input"`` submits/runs one target per input row. ``"per_rpos"``
        expands FRUST catalytic-cycle molecule structures per reactive position.
    select_mols : str or list of str, optional
        Molecule subset to generate for ``per_rpos`` targets. Common values are
        ``"all"``, ``"uniques"``, and ``"generics"``.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain stage keys this molecule workflow does not use; call
        ``wf.show_stages()`` to inspect the active stages.
    n_confs : int or None, optional
        Conformer count for initial dataframe preparation.
    top_n : int, optional
        Number of rows retained by ranking/filtering stages.
    dft : bool, optional
        Include DFT optimization, frequency, and solvent stages when ``True``.

    Returns
    -------
    MolsWorkflow
        Workflow object. Call ``wf.targets()`` to inspect targets, ``wf.run(...)``
        for local execution, or ``wf.submit(...)`` for cluster submission.

    Examples
    --------
    >>> import frust as ft
    >>> wf = ft.workflows.mols(
    ...     csv_path="molecules.csv",
    ...     split="per_rpos",
    ...     select_mols=["int2", "mol2"],
    ...     method="r2scan-3c",
    ...     dft=True,
    ... )
    >>> wf.targets()[:2]
    """
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


def raw_mols(
    *,
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    smiles: list[str] | None = None,
    method: MethodPlan | str | None = None,
    n_confs: int | None = None,
    top_n: int = 10,
    dft: bool = False,
) -> RawMolsWorkflow:
    """Create a raw molecule workflow.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        CSV file containing one exact molecule per row in a ``smiles`` column.
    dataframe : pandas.DataFrame or None, optional
        In-memory input table with the same columns as ``csv_path``.
    smiles : list of str or None, optional
        Direct list of exact molecule SMILES strings.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain TS-specific calculator keys, but ``raw_mols`` only uses the
        molecule stages shown by ``wf.show_stages()``.
    n_confs : int or None, optional
        Conformer count for initial dataframe preparation.
    top_n : int, optional
        Number of rows retained by ranking/filtering stages.
    dft : bool, optional
        Include DFT optimization, frequency, and solvent stages when ``True``.

    Returns
    -------
    RawMolsWorkflow
        Workflow object. Call ``wf.targets()`` to inspect one target per input
        molecule, ``wf.run(...)`` for local execution, or ``wf.submit(...)`` for
        cluster submission.

    Examples
    --------
    >>> import frust as ft
    >>> wf = ft.workflows.raw_mols(
    ...     csv_path="raw_dimers.csv",
    ...     method="r2scan-3c",
    ...     dft=True,
    ... )
    >>> [target.tag for target in wf.targets()]
    >>> wf.show_stages()[["group", "stage", "engine"]]
    """
    return RawMolsWorkflow(
        csv_path=csv_path,
        dataframe=dataframe,
        smiles=smiles,
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
    """Create a substrate/catalyst transition-state screen workflow.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        Component CSV accepted by ``ft.screen.read(...)``. It should contain
        substrate and catalyst rows with ``role`` and ``smiles`` columns.
    dataframe : pandas.DataFrame or None, optional
        Component dataframe or already-expanded systems dataframe.
    ts_types : tuple or list of str, optional
        Built-in TS types to generate, usually some subset of ``"TS1"``,
        ``"TS2"``, ``"TS3"``, and ``"TS4"``.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages).
    n_confs : int or None, optional
        Number of TS guess conformers generated per target.
    top_n : int, optional
        Number of xTB-ranked TS guesses kept before DFT stages.
    dft : bool, optional
        Include Hessian, ``OptTS``, frequency, and solvent stages when ``True``.

    Returns
    -------
    ScreenTSWorkflow
        Workflow object whose targets are combinations of system, TS type, and
        reactive position.

    Examples
    --------
    >>> import frust as ft
    >>> wf = ft.workflows.screen_ts(
    ...     csv_path="screen.csv",
    ...     ts_types=["TS1", "TS4"],
    ...     method=ft.workflows.methods.preset("r2scan-3c"),
    ...     dft=True,
    ... )
    >>> df = wf.run(targets=[0], out_dir="debug/screen_ts", execution="dft_staged")
    """
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
    """Create a legacy template-based TS workflow.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Ligand/substrate CSV containing a ``smiles`` column and optional
        ``rpos`` values.
    ts_xyz : str or pathlib.Path
        Template XYZ file used by the legacy TS transformer. The template
        comment line is used to infer the TS type when possible.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages).
    n_confs : int or None, optional
        Conformer count for initial TS embedding.
    top_n : int, optional
        Number of constrained xTB structures kept before DFT stages.
    dft : bool, optional
        Include Hessian, ``OptTS``, frequency, and solvent stages when ``True``.

    Returns
    -------
    LegacyTSWorkflow
        Workflow object for the older transformer/template TS path.

    Examples
    --------
    >>> import frust as ft
    >>> wf = ft.workflows.legacy_ts(
    ...     csv_path="ligands.csv",
    ...     ts_xyz="templates/TS3.xyz",
    ...     method="r2scan-3c",
    ... )
    >>> wf.targets()[:1]
    """
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
    """Create a legacy template-based INT3 workflow.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Ligand/substrate CSV containing a ``smiles`` column and optional
        ``rpos`` values.
    ts_xyz : str or pathlib.Path
        INT3-compatible template XYZ file used by the legacy transformer path.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages).
    n_confs : int or None, optional
        Conformer count for initial INT3 embedding.
    top_n : int, optional
        Number of xTB-ranked structures kept before DFT stages.
    dft : bool, optional
        Include INT3 DFT optimization, frequency, and solvent stages when
        ``True``.

    Returns
    -------
    LegacyTSWorkflow
        Workflow object whose ``workflow_name`` is set to ``"int3"``.

    Examples
    --------
    >>> import frust as ft
    >>> wf = ft.workflows.int3(
    ...     csv_path="ligands.csv",
    ...     ts_xyz="templates/INT3.xyz",
    ...     method="r2scan-3c",
    ... )
    >>> result = wf.submit(out_dir="runs/int3", cluster=cluster)
    """
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
    """Return common constrained TS initialization stages.

    Parameters
    ----------
    top_n : int
        Number of constrained xTB-optimized rows to keep before DFT
        pre-optimization.

    Returns
    -------
    list of StageDef
        ``prepare``, constrained GFNFF preoptimization, xTB ranking,
        constrained xTB optimization, DFT pre-SP, and constrained DFT pre-opt.
    """
    return [
        StageDef("prepare", "prepare", kind="prepare"),
        StageDef("xtb_preopt", "xtb_preopt", constraint=True, n_cores=2),
        StageDef("xtb_sp", "xtb_sp", n_cores=2),
        StageDef("xtb_opt", "xtb_opt", constraint=True, lowest=top_n, n_cores=2),
        StageDef("dft_pre_sp", "DFT-pre-SP"),
        StageDef("dft_pre_opt", "DFT-pre-Opt", constraint=True, lowest=1),
    ]


def _ts_dft_stages() -> list[StageDef]:
    """Return common TS DFT continuation stages.

    Returns
    -------
    list of StageDef
        Hessian, ``OptTS``, final frequency, and solvent single-point stages.
    """
    return [
        StageDef("hess", "Hess", read_files=["input.hess"]),
        StageDef("optts", "OptTS", use_last_hess=True),
        StageDef("freq", "Freq"),
        StageDef("solv", "DFT-solv"),
    ]
