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
from frust.stepper import Stepper
from frust.tsguess.matching import parse_rpos_value
from frust.tsguess.specs import BUILTIN_TS_SPECS
from frust.tsguess2.specs import BUILTIN_TS_SPECS_V2
from frust.tsguess2.specs import resolve_profile_specs
from frust.tsguess3.specs import BUILTIN_TS_SPECS_V3
from frust.structures import (
    StructureTarget,
    build as build_structure,
    molecule_states,
    normalize_systems,
    plan_targets,
)
from frust.utils.mols import create_mol_per_rpos
from frust.utils.pruning import normalize_pruning_options
from frust.workflows.core import BaseWorkflow, ExecutionOptions, StageDef, WorkflowTarget
from frust.workflows.methods import MethodPlan
from frust.workflows.spec_profiles import profile_for_geometry_stage


SplitMode = Literal["per_input", "per_rpos"]


def _screen_ts_specs_for_backend(backend: str) -> dict[str, Any]:
    """Return supported screen TS specs for a backend name."""
    backend_key = str(backend).strip().lower()
    if backend_key == "tsguess2":
        return BUILTIN_TS_SPECS_V2
    if backend_key == "tsguess3":
        return BUILTIN_TS_SPECS_V3
    if backend_key == "tsguess":
        return BUILTIN_TS_SPECS
    raise ValueError("ts_backend must be one of 'tsguess2', 'tsguess3', or 'tsguess'")


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


def _with_initial_prune(
    stages: list[StageDef],
    prune_initial: bool | dict[str, Any] | None,
) -> list[StageDef]:
    """Insert an optional initial pruning stage after preparation."""
    pruning_options = normalize_pruning_options(prune_initial)
    if pruning_options is None:
        return stages
    return [
        stages[0],
        StageDef(
            "initial_prune",
            "initial_prune",
            kind="prune",
            prune_options=pruning_options,
        ),
        *stages[1:],
    ]


def _molecule_stage_defs(
    *,
    top_n: int,
    dft: bool,
    include_terminal_solv_sp: bool = True,
    prune_initial: bool | dict[str, Any] | None = False,
) -> list[StageDef]:
    """Return the shared molecule stage graph."""
    stages = [
        StageDef("prepare", "prepare", kind="prepare"),
        StageDef("xtb_preopt", "xtb_preopt", n_cores=2),
        StageDef("xtb_sp", "xtb_sp", n_cores=2),
        StageDef("xtb_opt", "xTB optimization", lowest=top_n, rank_by="xtb_opt", n_cores=2),
    ]
    if dft:
        stages.extend(
            [
                StageDef("dft_rank_sp", "DFT ranking single point"),
                StageDef("dft_opt", "DFT minimum optimization", lowest=1, rank_by="dft_opt"),
                StageDef("dft_freq", "DFT frequencies"),
            ]
        )
        if include_terminal_solv_sp:
            stages.append(StageDef("dft_solv_sp", "DFT solvent single point"))
    else:
        stages.append(StageDef("filter", "filter", kind="filter", lowest=1, rank_by="xtb_opt"))
    return _with_initial_prune(stages, prune_initial)


class MolsWorkflow(BaseWorkflow):
    """Workflow for catalytic-cycle molecular states.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        Component CSV with ``compound_name``, ``role``, and ``smiles`` columns,
        an expanded screen table, or a substrate-only table with ``smiles``.
        Component input may vary both substrates and catalysts.
    dataframe : pandas.DataFrame or None, optional
        In-memory input table with the same columns as ``csv_path``.
    smiles : list of str or None, optional
        Direct list of SMILES strings for quick molecule workflows.
    split : {"per_input", "per_rpos"}, optional
        Target expansion mode. ``"per_input"`` creates one target per input row.
        ``"per_rpos"`` expands catalytic-cycle molecule structures per reactive
        position using :func:`frust.utils.mols.create_mol_per_rpos`.
    select_mols : str or list of str, optional
        Molecules to generate. Accepted individual states are ``"dimer"``,
        ``"HH"``, ``"ligand"``, ``"catalyst"``, ``"int1"``, ``"int2"``,
        ``"HBpin-ligand"``, and ``"HBpin-mol"``. ``"int1"`` is the
        charge-separated catalyst/substrate adduct formerly called ``"int2"``;
        ``"int2"`` is the neutral adduct formerly called ``"mol2"``.
        ``"all"`` selects every state; ``"uniques"`` selects ``"ligand"``,
        ``"int1"``, ``"int2"``, and ``"HBpin-ligand"``; ``"generics"``
        selects ``"dimer"``, ``"HH"``, ``"catalyst"``, and ``"HBpin-mol"``.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow),
        ``"r2scan-3c-solv"`` and ``"wb97xd3-631g-solv"``
        (solvent-inclusive DFT stages without a terminal solvent SP), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain stage keys this molecule workflow does not use; call
        ``wf.show_stages()`` to inspect the active stages.
    n_confs : int or None, optional
        Conformer count passed to ``Stepper.build_initial_df``.
    top_n : int, optional
        Number of rows kept after ranking/filtering stages.
    dft : bool, optional
        If ``True``, add DFT optimization and frequency stages. Gas-phase
        presets also add a terminal solvent single point; solvent-inclusive
        presets do not. If ``False``, end with a lowest-energy filter after xTB
        stages.
    prune_initial : bool or dict, optional
        If ``False``, leave the initial conformer ensemble unchanged. If
        ``True``, insert PRISM pruning immediately after ``prepare`` using the
        default modes ``("moi", "rmsd")``. A dictionary enables pruning and
        overrides default pruning options, for example
        ``{"modes": ("moi",), "moi_max_deviation": 0.03}``.

    Notes
    -----
    The default stage graph is ``prepare -> xtb_preopt -> xtb_sp -> xtb_opt``.
    DFT workflows normally run ``dft_rank_sp -> dft_opt -> dft_freq ->
    dft_solv_sp``. The solvent-inclusive presets omit ``dft_solv_sp`` because
    all their DFT stages already include SMD chloroform. Non-DFT workflows run
    a final ``filter`` stage.
    """

    workflow_name = "mols"
    result_profile = "minimum"

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
        top_n: int = 20,
        dft: bool = True,
        prune_initial: bool | dict[str, Any] = True,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.dataframe = dataframe
        self.smiles = smiles
        self.split = split
        self.select_mols = select_mols
        self.prune_initial = prune_initial

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

        systems = normalize_systems(df)
        return plan_targets(
            systems,
            states=molecule_states(self.select_mols),
            builder_options={"select_mols": self.select_mols},
        )

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
        if isinstance(target, StructureTarget):
            return build_structure(
                target,
                n_confs=self.n_confs,
                n_cores=options.n_cores,
                memory_gb=options.mem_gb,
                debug=options.debug,
                stepper_cls=Stepper,
                ts_guess_factory=create_ts_guesses,
                mol_factory=create_mol_per_rpos,
            )
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
        return _molecule_stage_defs(
            top_n=self.top_n,
            dft=self.dft,
            include_terminal_solv_sp=self.method.include_terminal_solv_sp,
            prune_initial=self.prune_initial,
        )


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
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow),
        ``"r2scan-3c-solv"`` and ``"wb97xd3-631g-solv"``
        (solvent-inclusive DFT stages without a terminal solvent SP), and
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
    prune_initial : bool or dict, optional
        If ``False``, leave the initial conformer ensemble unchanged. If
        ``True``, insert PRISM pruning immediately after ``prepare`` using the
        default modes ``("moi", "rmsd")``. A dictionary enables pruning and
        overrides default pruning options, for example
        ``{"modes": ("moi",), "moi_max_deviation": 0.03}``.

    Notes
    -----
    This workflow treats each input SMILES as the structure to calculate. It
    does not call ``create_mol_per_rpos`` and does not support ``select_mols``.
    With ``dft=True``, gas-phase presets use ``dft_rank_sp -> dft_opt ->
    dft_freq -> dft_solv_sp``. The solvent-inclusive presets stop at
    ``dft_freq``. That stage is a normal minimum-frequency check used for
    thermochemistry; TS-specific ``dft_hessian`` and ``dft_ts_opt`` stages are
    not run.
    """

    workflow_name = "raw_mols"
    result_profile = "minimum"

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
        prune_initial: bool | dict[str, Any] = False,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.dataframe = dataframe
        self.smiles = smiles
        self.prune_initial = prune_initial

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
        return _molecule_stage_defs(
            top_n=self.top_n,
            dft=self.dft,
            include_terminal_solv_sp=self.method.include_terminal_solv_sp,
            prune_initial=self.prune_initial,
        )


class ScreenTSWorkflow(BaseWorkflow):
    """Implementation object returned by :func:`screen_ts`.

    Construct screen workflows through :func:`screen_ts`, which owns the
    user-facing parameter documentation.
    """

    workflow_name = "screen_ts"
    result_profile = "transition_state"

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
        ts_backend: str = "tsguess2",
        spec_profile: str = "auto",
        spec_match: str = "prefer-exact",
        method: MethodPlan | str | None = None,
        n_confs: int | None = None,
        top_n: int = 20,
        dft: bool = True,
        prune_initial: bool | dict[str, Any] = True,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.dataframe = dataframe
        self.ts_types = tuple(str(ts_type).upper() for ts_type in ts_types)
        self.ts_backend = str(ts_backend).strip().lower()
        self.spec_profile = str(spec_profile).strip().lower()
        self.spec_match = str(spec_match).strip().lower()
        self.prune_initial = prune_initial

    def _resolved_spec_profile(self) -> str:
        """Return the explicit or method-derived tsguess2 profile."""
        if self.spec_profile != "auto":
            return self.spec_profile
        return profile_for_geometry_stage(self.method, "dft_ts_opt")

    @property
    def resolved_spec_profile(self) -> str:
        """Return the geometry profile selected for TS construction."""
        return self._resolved_spec_profile()

    def _structure_build_kwargs(self) -> dict[str, Any]:
        """Return method-aware connected-structure build options."""
        if self.ts_backend != "tsguess2":
            return {}
        return {
            "spec_profile": self._resolved_spec_profile(),
            "spec_match": self.spec_match,
        }

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
        supported_specs = {
            key: value
            for key, value in _screen_ts_specs_for_backend(self.ts_backend).items()
            if key.startswith("TS")
        }
        unknown = sorted(set(self.ts_types) - set(supported_specs))
        if unknown:
            supported = ", ".join(sorted(supported_specs))
            raise ValueError(f"Unsupported screen TS types {unknown}. Supported: {supported}")
        systems = self._systems()
        if self.ts_backend == "tsguess2":
            resolve_profile_specs(
                self.ts_types,
                self._resolved_spec_profile(),
                match=self.spec_match,
            )
            return plan_targets(systems, states=self.ts_types)
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
            written as ``structure_guess.parquet`` by the modern backend (or
            ``ts_guess.parquet`` by compatibility backends) before calculator
            stages start.
        options : ExecutionOptions
            Runtime options controlling TS guess conformer generation.

        Returns
        -------
        pandas.DataFrame
            TS guess dataframe for the target's TS type.
        """
        if isinstance(target, StructureTarget):
            return build_structure(
                target,
                n_confs=self.n_confs,
                n_cores=options.n_cores,
                memory_gb=options.mem_gb,
                debug=options.debug,
                save_dir=save_dir,
                stepper_cls=Stepper,
                ts_guess_factory=create_ts_guesses,
                spec_profile=self._resolved_spec_profile(),
                spec_match=self.spec_match,
            )
        screen_target = target.payload
        ts_type = str(screen_target["ts_type"].iloc[0]).upper()
        guesses = create_ts_guesses(
            screen_target,
            ts_types=[ts_type],
            n_confs=self.n_confs,
            n_cores=options.n_cores,
            backend=self.ts_backend,
            spec_profile=self._resolved_spec_profile(),
            spec_match=self.spec_match,
        )
        df = guesses[ts_type]
        if save_dir is not None:
            df.to_parquet(save_dir / "ts_guess.parquet")
        return df

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        """Return the target TS type for Stepper dispatch."""
        metadata = target.metadata or {}
        return metadata.get("ts_type") or metadata.get("state_id")

    def _stage_defs(self) -> list[StageDef]:
        """Return screen TS workflow stages."""
        stages = _ts_screening_stages(self.top_n, prune_initial=self.prune_initial)
        stages.append(_dft_rank_sp_stage())
        if self.dft:
            stages.extend(
                _ts_dft_refinement_stages(
                    include_terminal_solv_sp=self.method.include_terminal_solv_sp,
                )
            )
        else:
            stages.append(
                StageDef(
                    "filter", "filter", kind="filter",
                    lowest=1, rank_by="dft_rank_sp",
                )
            )
        return stages


class Int3Workflow(BaseWorkflow):
    """Modern constrained-minimum workflow for the INT3 state.

    This workflow uses the same typed system planner and connected-graph
    structure builder as ``screen_ts`` while retaining its own minimum-specific
    stage graph and result profile.
    """

    workflow_name = "int3"
    result_profile = "constrained_minimum"

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        spec_profile: str = "auto",
        spec_match: str = "prefer-exact",
        method: MethodPlan | str | None = None,
        n_confs: int | None = None,
        top_n: int = 20,
        dft: bool = True,
        prune_initial: bool | dict[str, Any] = True,
    ) -> None:
        super().__init__(method=method, n_confs=n_confs, top_n=top_n, dft=dft)
        self.csv_path = csv_path
        self.dataframe = dataframe
        self.spec_profile = str(spec_profile).strip().lower()
        self.spec_match = str(spec_match).strip().lower()
        self.prune_initial = prune_initial

    def _resolved_spec_profile(self) -> str:
        """Return the explicit or method-derived INT3 geometry profile."""
        if self.spec_profile != "auto":
            return self.spec_profile
        return profile_for_geometry_stage(self.method, "dft_opt")

    @property
    def resolved_spec_profile(self) -> str:
        """Return the geometry profile selected for INT3 construction."""
        return self._resolved_spec_profile()

    def _structure_build_kwargs(self) -> dict[str, Any]:
        """Return method-aware connected-structure build options."""
        return {
            "spec_profile": self._resolved_spec_profile(),
            "spec_match": self.spec_match,
        }

    def _systems(self) -> pd.DataFrame:
        """Return normalized explicit systems for INT3 construction."""
        source = self.dataframe if self.dataframe is not None else self.csv_path
        if source is None:
            raise ValueError("int3 workflow requires csv_path or dataframe")
        return normalize_systems(source)

    def _build_targets(self) -> list[StructureTarget]:
        """Plan one lightweight INT3 target per system and reactive position."""
        resolve_profile_specs(
            ["INT3"],
            self._resolved_spec_profile(),
            match=self.spec_match,
        )
        return plan_targets(self._systems(), states=["INT3"])

    def _prepare_initial_df(
        self,
        target: WorkflowTarget,
        *,
        save_dir: Path | None,
        options: ExecutionOptions,
    ) -> pd.DataFrame:
        """Construct one INT3 guess with the shared connected-graph builder."""
        if not isinstance(target, StructureTarget):
            raise TypeError("Int3Workflow requires typed StructureTarget objects")
        return build_structure(
            target,
            n_confs=self.n_confs,
            n_cores=options.n_cores,
            memory_gb=options.mem_gb,
            debug=options.debug,
            save_dir=save_dir,
            stepper_cls=Stepper,
            ts_guess_factory=create_ts_guesses,
            spec_profile=self._resolved_spec_profile(),
            spec_match=self.spec_match,
        )

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        """Return the constrained-minimum Stepper type."""
        del target
        return "INT3"

    def _stage_defs(self) -> list[StageDef]:
        """Return the dedicated INT3 screening and refinement graph."""
        stages = _ts_screening_stages(self.top_n, prune_initial=self.prune_initial)
        stages.append(_dft_rank_sp_stage())
        if self.dft:
            stages.extend(
                _int3_dft_refinement_stages(
                    include_terminal_solv_sp=self.method.include_terminal_solv_sp,
                )
            )
        else:
            stages.append(
                StageDef(
                    "filter",
                    "filter",
                    kind="filter",
                    lowest=1,
                    rank_by="dft_rank_sp",
                )
            )
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
    top_n: int = 20,
    dft: bool = True,
    prune_initial: bool | dict[str, Any] = True,
) -> MolsWorkflow:
    """Create a molecule-state workflow.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        Component CSV with ``compound_name``, ``role``, and ``smiles`` columns,
        or a substrate-only CSV with ``smiles``. Component tables use the same
        substrate/catalyst expansion as ``screen_ts``.
    dataframe : pandas.DataFrame or None, optional
        In-memory input table with the same columns as ``csv_path``.
    smiles : list of str or None, optional
        Quick input for simple molecule workflows.
    split : {"per_input", "per_rpos"}, optional
        ``"per_input"`` submits/runs one target per input row. ``"per_rpos"``
        expands FRUST catalytic-cycle molecule structures per reactive position.
    select_mols : str or list of str, optional
        Molecules to generate for ``per_rpos`` targets. Accepted individual
        states are ``"dimer"``, ``"HH"``, ``"ligand"``, ``"catalyst"``,
        ``"int1"``, ``"int2"``, ``"HBpin-ligand"``, and ``"HBpin-mol"``.
        ``"int1"`` is the charge-separated catalyst/substrate adduct formerly
        called ``"int2"``; ``"int2"`` is the neutral adduct formerly called
        ``"mol2"``. ``"all"`` selects every state; ``"uniques"`` selects
        ``"ligand"``, ``"int1"``, ``"int2"``, and ``"HBpin-ligand"``;
        ``"generics"`` selects ``"dimer"``, ``"HH"``, ``"catalyst"``, and
        ``"HBpin-mol"``.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow),
        ``"r2scan-3c-solv"`` and ``"wb97xd3-631g-solv"``
        (solvent-inclusive DFT stages without a terminal solvent SP), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages). A preset may
        contain stage keys this molecule workflow does not use; call
        ``wf.show_stages()`` to inspect the active stages.
    n_confs : int or None, optional
        Conformer count for initial dataframe preparation.
    top_n : int, optional
        Number of rows retained by ranking/filtering stages.
    dft : bool, optional
        Include DFT optimization and frequency stages when ``True``. Gas-phase
        presets also include a terminal solvent single point; solvent-inclusive
        presets do not.
    prune_initial : bool or dict, optional
        If ``False``, leave the initial conformer ensemble unchanged. If
        ``True``, insert PRISM pruning immediately after ``prepare`` using the
        default modes ``("moi", "rmsd")``. A dictionary enables pruning and
        overrides default pruning options, for example
        ``{"modes": ("moi",), "moi_max_deviation": 0.03}``.

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
    ...     select_mols=["int1", "int2"],
    ...     method="r2scan-3c",
    ...     dft=True,
    ... )
    >>> wf.targets()[:2]

    Variable catalyst example:

    >>> import pandas as pd
    >>> components = pd.DataFrame({
    ...     "compound_name": ["furan", "NMe"],
    ...     "role": ["substrate", "catalyst"],
    ...     "smiles": ["C1=CC=CO1", "BC1=C(N(C)C)C=CC=C1"],
    ... })
    >>> wf = ft.workflows.mols(dataframe=components, select_mols=["int1", "int2"])
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
        prune_initial=prune_initial,
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
    prune_initial: bool | dict[str, Any] = False,
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
    prune_initial : bool or dict, optional
        If ``False``, leave the initial conformer ensemble unchanged. If
        ``True``, insert PRISM pruning immediately after ``prepare`` using the
        default modes ``("moi", "rmsd")``. A dictionary enables pruning and
        overrides default pruning options, for example
        ``{"modes": ("moi",), "moi_max_deviation": 0.03}``.

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
        prune_initial=prune_initial,
    )


def screen_ts(
    *,
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
    ts_backend: str = "tsguess2",
    method: MethodPlan | str | None = None,
    spec_profile: str = "auto",
    spec_match: str = "prefer-exact",
    n_confs: int | None = None,
    top_n: int = 20,
    dft: bool = True,
    prune_initial: bool | dict[str, Any] = True,
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
    ts_backend : {"tsguess2", "tsguess3", "tsguess"}, optional
        TS guess backend. ``"tsguess2"`` is the default SMILES-roundtrip
        backend; ``"tsguess3"`` adds fragment-aware TS3/TS4 embedding;
        ``"tsguess"`` preserves the original assembly backend.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow),
        ``"r2scan-3c-solv"`` and ``"wb97xd3-631g-solv"``
        (solvent-inclusive DFT stages without a terminal solvent SP), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages).
    spec_profile : str, optional
        Geometry-reference profile used by ``tsguess2``. ``"auto"`` selects
        the profile from the DFT TS-optimization method and environment. Pass
        an explicit profile such as ``"r2scan-3c/smd(chloroform)"`` for a
        custom method plan.
    spec_match : {"prefer-exact", "exact"}, optional
        Profile matching policy. ``"prefer-exact"`` may use the other
        environment from the same method family when an exact reference is
        missing; ``"exact"`` requires the requested environment. Neither
        policy crosses method families.
    n_confs : int or None, optional
        Number of TS guess conformers generated per target.
    top_n : int, optional
        Number of low-cost optimized TS guesses kept before the DFT pre-SP
        cutoff.
    dft : bool, optional
        If ``True``, include constrained DFT preoptimization, Hessian,
        ``OptTS``, and frequency stages. Gas-phase presets also include a
        terminal solvent single point; solvent-inclusive presets do not. If
        ``False``, stop after the DFT pre-SP cutoff and keep the lowest-energy
        row.
    prune_initial : bool or dict, optional
        Defaults to ``True``, which inserts PRISM pruning immediately after
        ``prepare`` with ``modes=("moi", "rmsd")``,
        ``moi_max_deviation=0.01``, ``rmsd_max_rmsd=0.5``,
        ``heavy_atoms_only=True``, and
        ``graph_source="connectivity_bonds"``. If ``False``, leave the initial
        TS conformer ensemble unchanged. Pass a dictionary to override the
        defaults, for example
        ``{"modes": ("moi", "rmsd"), "moi_max_deviation": 0.02,
        "rmsd_max_rmsd": 0.5, "rmsd_max_dev": 1.0}``. The two RMSD thresholds
        are in Angstrom. If ``rmsd_max_dev`` is omitted or ``None``, PRISM uses
        ``2 * rmsd_max_rmsd``.

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
        ts_backend=ts_backend,
        method=method,
        spec_profile=spec_profile,
        spec_match=spec_match,
        n_confs=n_confs,
        top_n=top_n,
        dft=dft,
        prune_initial=prune_initial,
    )


def int3(
    *,
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    method: MethodPlan | str | None = None,
    spec_profile: str = "auto",
    spec_match: str = "prefer-exact",
    n_confs: int | None = None,
    top_n: int = 20,
    dft: bool = True,
    prune_initial: bool | dict[str, Any] = True,
) -> Int3Workflow:
    """Create a modern, dedicated INT3 constrained-minimum workflow.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        Component screen CSV, expanded-system CSV, or substrate-only CSV.
    dataframe : pandas.DataFrame or None, optional
        In-memory input in any of the same forms as ``csv_path``.
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Accepts ``None`` for the
        default ``"wb97xd3-631g"`` preset, a preset string, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages).
    spec_profile : str, optional
        Geometry-reference profile used for INT3 construction. ``"auto"``
        selects it from the DFT optimization method and environment.
    spec_match : {"prefer-exact", "exact"}, optional
        Profile matching policy. ``"prefer-exact"`` may fall back to the
        other environment from the same method family; ``"exact"`` requires
        the requested environment.
    n_confs : int or None, optional
        Conformer count for initial INT3 embedding.
    top_n : int, optional
        Number of low-cost optimized structures kept before the DFT pre-SP
        cutoff.
    dft : bool, optional
        If ``True``, include constrained DFT preoptimization, INT3 DFT
        optimization, and frequency stages. Gas-phase presets also include a
        terminal solvent single point; solvent-inclusive presets do not. If
        ``False``, stop after the DFT pre-SP cutoff and keep the lowest-energy
        row.
    prune_initial : bool or dict, optional
        If ``False``, leave the initial INT3 conformer ensemble unchanged. If
        ``True``, insert PRISM pruning immediately after ``prepare`` using the
        default modes ``("moi", "rmsd")``. A dictionary enables pruning and
        overrides default pruning options, for example
        ``{"modes": ("moi", "rmsd", "rot_corr_rmsd")}``.

    Returns
    -------
    Int3Workflow
        Dedicated INT3 workflow with its own stage graph and canonical result
        profile.

    Examples
    --------
    >>> import frust as ft
    >>> wf = ft.workflows.int3(
    ...     csv_path="screen.csv",
    ...     method="r2scan-3c",
    ... )
    >>> result = wf.submit(out_dir="runs/int3", cluster=cluster)
    """
    return Int3Workflow(
        csv_path=csv_path,
        dataframe=dataframe,
        method=method,
        spec_profile=spec_profile,
        spec_match=spec_match,
        n_confs=n_confs,
        top_n=top_n,
        dft=dft,
        prune_initial=prune_initial,
    )


def _ts_screening_stages(
    top_n: int,
    *,
    prune_initial: bool | dict[str, Any] | None = False,
) -> list[StageDef]:
    """Return common constrained TS screening stages.

    Parameters
    ----------
    top_n : int
        Number of constrained low-cost optimized rows to keep before the DFT
        single-point cutoff.

    Returns
    -------
    list of StageDef
        ``prepare``, constrained GFNFF preoptimization, low-cost ranking, and
        constrained low-cost optimization.
    """
    stages = [
        StageDef("prepare", "prepare", kind="prepare"),
        StageDef("xtb_preopt", "xtb_preopt", constraint=True, n_cores=2),
        StageDef("xtb_sp", "xtb_sp", n_cores=2),
        StageDef(
            "xtb_opt", "constrained xTB optimization", constraint=True,
            lowest=top_n, rank_by="xtb_opt", n_cores=2,
        ),
    ]
    return _with_initial_prune(stages, prune_initial)


def _dft_rank_sp_stage() -> StageDef:
    """Return the DFT single-point cutoff stage shared by TS workflows."""
    return StageDef("dft_rank_sp", "DFT ranking single point")


def _dft_preopt_stage() -> StageDef:
    """Return the constrained DFT preoptimization stage."""
    return StageDef(
        "dft_preopt", "constrained DFT preoptimization", constraint=True,
        lowest=1, rank_by="dft_preopt",
    )


def _ts_dft_refinement_stages(*, include_terminal_solv_sp: bool = True) -> list[StageDef]:
    """Return common TS DFT refinement stages.

    Returns
    -------
    list of StageDef
        Constrained DFT preoptimization, Hessian, ``OptTS``, final frequency,
        and, by default, a solvent single-point stage.
    """
    stages = [
        _dft_preopt_stage(),
        StageDef("dft_hessian", "DFT Hessian", read_files=["input.hess"]),
        StageDef("dft_ts_opt", "DFT transition-state optimization", use_last_hess=True),
        StageDef("dft_freq", "DFT frequencies"),
    ]
    if include_terminal_solv_sp:
        stages.append(StageDef("dft_solv_sp", "DFT solvent single point"))
    return stages


def _int3_dft_refinement_stages(*, include_terminal_solv_sp: bool = True) -> list[StageDef]:
    """Return INT3 DFT refinement stages after the DFT single-point cutoff."""
    stages = [
        _dft_preopt_stage(),
        StageDef("dft_opt", "DFT minimum optimization", lowest=1, rank_by="dft_opt"),
        StageDef("dft_freq", "DFT frequencies"),
    ]
    if include_terminal_solv_sp:
        stages.append(StageDef("dft_solv_sp", "DFT solvent single point"))
    return stages
