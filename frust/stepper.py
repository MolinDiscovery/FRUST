# frustactivation/stepper.py
import logging
import sys
import os
from pathlib import Path
import re
import textwrap
from typing import Callable
import numpy as np
import pandas as pd
from pandas import Series
from inspect import signature
from frust.utils.dirs import make_step_dir, prepare_base_dir
from frust.utils.slurm import detect_job_id
from frust.schema import (
    energy_columns,
    infer_group_columns,
    metadata_from_mapping,
    normalize_dataframe,
    output_column,
)

def _make_logger(name: str, debug: bool) -> logging.Logger:
    """Create an instance-local logger so debug settings don't leak globally."""
    inst_logger = logging.getLogger(name)
    inst_logger.propagate = False
    inst_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not inst_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        inst_logger.addHandler(handler)

    return inst_logger


def _logger_name(step_type: str | None, job_id: int | None) -> str:
    """Build a readable, stable logger name for one Stepper instance."""
    step_label = (step_type or "generic").upper()
    job_label = f"job{job_id}" if job_id is not None else "jobunknown"
    return f"{__name__}.{step_label}.{job_label}"

class Stepper:
    """Chain xTB and ORCA calculations over dataframe-based conformer tables.

    `Stepper` is the low-level workflow layer used after structures have
    already been generated and embedded. It operates on pandas DataFrames,
    adds calculation outputs as new columns, and optionally manages run
    directories and saved engine files.

    The class does not generate molecules itself. For higher-level workflows
    that create TS or ground-state structures from ligand inputs, prefer the
    pipeline functions in :mod:`frust.pipes`.
    """

    def __init__(
        self,
        step_type: str | None = None,
        output_base: Path | str | None = None,
        job_id: int | None = None,
        debug: bool = False,
        dump_each_step: bool = False,
        n_cores: int = 8,
        memory_gb: float = 20.0,
        save_calc_dirs: bool = False,
        save_output_dir: bool = True,
        work_dir: str | None = None,        
        **kwargs
    ):
        """Configure a Stepper instance.

        Parameters
        ----------
        step_type : str or None, optional
            Workflow label used for built-in constrained TS/INT calculations.
            Supported constrained values are ``TS1``, ``TS2``, ``TS3``,
            ``TS4``, and ``INT3``. If ``None``, unconstrained workflows still
            work, but ``constraint=True`` in :meth:`xtb` or :meth:`orca` will
            raise an error.
        output_base : Path or str or None, optional
            Base directory under which saved outputs are created. The final run
            directory is created lazily on first save request, not at
            construction time.
        job_id : int or None, optional
            Optional job identifier used in output-directory naming and logger
            naming. If omitted, :func:`frust.utils.slurm.detect_job_id` is used
            to infer one when possible.
        debug : bool, optional
            If ``True``, use mock xTB and ORCA backends instead of the real
            engines and enable debug logging.
        dump_each_step : bool, optional
            Reserved for dataframe dumping workflows. Stored on the instance
            but not currently consumed by :class:`Stepper` itself.
        n_cores : int, optional
            Default core count passed to engine calls unless overridden on an
            individual :meth:`xtb` or :meth:`orca` call.
        memory_gb : float, optional
            Default memory setting in gigabytes used for ORCA calls.
        save_calc_dirs : bool, optional
            If ``True``, preserve full calculation directories for each row
            when output saving is enabled.
        save_output_dir : bool, optional
            If ``True``, enable output-directory creation and file saving.
            When ``False``, calculations still run but no output tree is
            created unless a caller explicitly requests per-step saving.
        work_dir : str or None, optional
            Scratch or working directory passed to the engine wrappers. If
            omitted, ``$SCRATCH`` is used when available, otherwise the current
            directory.

        Raises
        ------
        TypeError
            If unexpected keyword arguments are supplied.
        """
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Stepper keyword arguments: {unknown}")

        self.step_type      = step_type.upper() if step_type is not None else None
        self.debug          = debug
        job_id              = detect_job_id(job_id, True)
        self.job_id         = job_id
        self.logger         = _make_logger(_logger_name(self.step_type, self.job_id), debug)
        self.dump_each_step = dump_each_step
        self.n_cores        = n_cores
        self.memory_gb      = memory_gb
        self.save_calc_dirs = save_calc_dirs
        self.save_output    = save_output_dir
        self.output_base    = Path(output_base) if output_base is not None else None

        self.base_dir = self.output_base if self.output_base is not None else Path(".")

        if self.debug:
            from tooltoad.xtb import mock_xtb_calculate
            from tooltoad.orca import mock_orca_calculate
            from tooltoad.gxtb import mock_gxtb_calculate
            self.xtb_fn  = mock_xtb_calculate
            self.orca_fn = mock_orca_calculate
            self.gxtb_fn = mock_gxtb_calculate
        else:
            from tooltoad.xtb import xtb_calculate
            from tooltoad.orca import orca_calculate
            from tooltoad.gxtb import gxtb_calculate
            self.xtb_fn  = xtb_calculate
            self.orca_fn = orca_calculate
            self.gxtb_fn = gxtb_calculate

        if work_dir:
            self.work_dir = work_dir
            os.makedirs(self.work_dir, exist_ok=True)
            self.logger.info(f"Working dir: {self.work_dir}")
        else:
            try:
                self.work_dir = os.environ["SCRATCH"]
            except Exception:
                self.work_dir = "."
            os.makedirs(self.work_dir, exist_ok=True)
            self.logger.info(f"Working dir: {self.work_dir}")

    def _ensure_base_dir(self) -> Path:
        """Create the output directory lazily on first use."""
        if self.save_output and not getattr(self, "_base_dir_prepared", False):
            self.base_dir = prepare_base_dir(self.output_base, self.job_id)
            self._base_dir_prepared = True
        return self.base_dir

    @staticmethod
    def _coord_columns(df: pd.DataFrame) -> list[str]:
        preferred = []
        if "coords_embedded" in df.columns:
            preferred.append("coords_embedded")
        preferred.extend(
            [
                c
                for c in df.columns
                if "coords" in c and c not in preferred and not c.endswith("-opt_coords")
            ]
        )
        preferred.extend([c for c in df.columns if c.endswith("-opt_coords") or c.endswith("-oc")])
        return preferred

    @classmethod
    def _last_coord_col(cls, df: pd.DataFrame) -> str:
        """Return the most specific available coordinate column."""
        cols = cls._coord_columns(df)
        if not cols:
            raise ValueError(
                "DataFrame must contain coordinates in 'coords_embedded' or a '*-oc'/'*-opt_coords' column"
            )
        return cols[-1]

    def _step_type_upper(self) -> str:
        """Normalize step_type for callers that only need TS dispatch."""
        return (self.step_type or "").upper()

    def _validate_constraint_request(self, df: pd.DataFrame) -> str:
        """Validate that constraint mode is fully specified."""
        step_type = self._step_type_upper()
        if not step_type:
            raise ValueError(
                "`constraint=True` requires `Stepper(step_type=...)` so the correct TS/INT constraints can be selected"
            )
        if step_type not in {"TS1", "TS2", "TS3", "TS4", "INT3"}:
            raise ValueError(
                f"`constraint=True` is only supported for TS1/TS2/TS3/TS4/INT3, got {self.step_type!r}"
            )
        if "constraint_atoms" not in df.columns:
            raise ValueError(
                "`constraint=True` requires a 'constraint_atoms' column in the input DataFrame"
            )
        return step_type

    @staticmethod
    def _validate_required_columns(
        df: pd.DataFrame,
        *,
        needs_grouping: bool = False,
        needs_hessian: bool = False,
    ) -> None:
        missing = [col for col in ("atoms",) if col not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {', '.join(missing)}"
            )

        Stepper._last_coord_col(df)

        if needs_grouping and "substrate_name" not in df.columns:
            raise ValueError(
                "`lowest=` requires a 'substrate_name' column so conformers can be grouped before filtering"
            )

        if needs_hessian and not any(col.endswith(".hess") for col in df.columns):
            raise ValueError(
                "`use_last_hess=True` requires a prior '*.hess' column in the input DataFrame"
            )

    @staticmethod
    def _row_name(row: Series, index: object) -> str:
        """Pick a stable human-readable row label from available metadata."""
        for key in ("custom_name", "substrate_name", "moltype"):
            value = row.get(key)
            if value is not None and not pd.isna(value):
                return str(value)
        return f"row_{index}"

    @staticmethod
    def _row_conf_id(row: Series, index: object) -> str:
        """Pick a stable conformer/run identifier when cid is unavailable."""
        value = row.get("cid")
        if value is not None and not pd.isna(value):
            return str(value)
        return str(index)

    @staticmethod
    def _constraint_atoms(row: Series, min_size: int = 6) -> list[int]:
        """Validate and return constraint atoms for TS/INT workflows."""
        atoms = row.get("constraint_atoms")
        if atoms is None:
            raise ValueError("Missing 'constraint_atoms' for a constrained row")
        if not isinstance(atoms, (list, tuple, np.ndarray)):
            if pd.isna(atoms):
                raise ValueError("Missing 'constraint_atoms' for a constrained row")
            raise ValueError("'constraint_atoms' must be a sequence of atom indices")
        atoms = list(atoms)
        if len(atoms) < min_size:
            raise ValueError(
                f"'constraint_atoms' must contain at least {min_size} entries for constrained workflows"
            )
        return atoms

    def build_initial_df(self, embedded_dict: dict) -> pd.DataFrame:
        """
        Turn a dictionary of embedded‐conformer data into a tidy DataFrame.

        The dictionary keys can be either:
          1) TS names of the form 'TS(..._rpos(N))' (with a 4‐ or 5‐tuple value)
          2) plain molecule names (without '_rpos(...)') and a 2‐tuple value

        For TS entries:
            key = 'TS(molname_rpos(N))'
            value = (Mol_with_H, cids, keep_idxs, smiles[, energies])

        For plain‐mol entries:
            key = 'some_name'
            value = (Mol, cids)

        In legacy-input cases, structure metadata is parsed from the key as a
        fallback. New generated records should carry metadata directly.

        Returns
        -------
        pd.DataFrame
            One row per conformer with columns
              - structure_id        (stable structure identifier)
              - custom_name         (the original display/file key)
              - substrate_name      (parsed substrate identity)
              - structure_type      (MOL, TS1, TS2, TS3, TS4, INT3)
              - molecule_role       (ts, ligand, int2, mol2, ...)
              - rpos                (int or None)
              - constraint_atoms    (list[int] or NA)
              - cid                 (conformer ID)
              - smiles              (str or None)
              - atoms               (list of atomic symbols)
              - coords_embedded     (list of (x,y,z) tuples)
              - energy_uff          (float or None)
        """
        rows: list[dict] = []

        for name, val in embedded_dict.items():
            if len(val) == 2:
                mol, cids = val
                keep_idxs = None
                smi = None
                energies: list[tuple[float,int]] = []
                metadata = None
            elif len(val) == 3 and isinstance(val[2], dict):
                mol, cids, metadata = val
                keep_idxs = None
                smi = metadata.get("smiles") or metadata.get("input_smiles")
                energies = []
            elif len(val) == 4:
                mol, cids, keep_idxs, smi = val
                energies = []
                metadata = None
            elif len(val) == 5:
                mol, cids, keep_idxs, smi, energies = val
                metadata = None
            else:
                raise ValueError(f"Bad tuple length for {name}")

            meta = metadata_from_mapping(metadata, fallback_name=name, smiles=smi)

            e_map: dict[int,float] = {cid_val: e_val for (e_val, cid_val) in energies} if energies else {}

            atom_syms = [atom.GetSymbol() for atom in mol.GetAtoms()]

            for cid in cids:
                conf = mol.GetConformer(cid)
                coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

                rows.append({
                    "structure_id":     meta.structure_id,
                    "custom_name":      meta.custom_name,
                    "substrate_name":   meta.substrate_name,
                    "structure_type":   meta.structure_type,
                    "molecule_role":    meta.molecule_role,
                    "rpos":             meta.rpos,
                    "constraint_atoms": keep_idxs,
                    "cid":              cid,
                    "smiles":           meta.smiles or smi,
                    "input_smiles":     meta.input_smiles or smi,
                    "atoms":            atom_syms,
                    "coords_embedded":  coords,
                    "energy_uff":       e_map.get(cid, None)
                })

        return pd.DataFrame(rows)

    def _run_engine(
            self,
            df: pd.DataFrame,
            engine_fn: Callable[..., dict],
            prefix: str,
            build_inputs: Callable[[Series], dict],
            save_step: bool,
            lowest: int | None,
            save_files: list[str] | None = None,
            use_last_hess: bool = False,
        ) -> pd.DataFrame:
        """
        Generic runner for xTB or ORCA or any other engine.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing at least ``atoms`` and one coordinate
            column. Depending on the options used, additional columns may be
            required, such as ``substrate_name`` for ``lowest=`` filtering or a
            ``*.hess`` column for Hessian reuse.
        engine_fn : callable
            Backend calculation function such as the wrapped xTB or ORCA
            driver.
        prefix : str
            Prefix used to name output columns for this calculation stage.
        build_inputs : callable
            Row-wise callback that returns backend-specific keyword arguments.
            These inputs are merged with the generic engine inputs assembled by
            :class:`Stepper`.
        save_step : bool
            If ``True``, preserve the full saved directory for each processed
            row.
        lowest : int or None
            If set, keep only the lowest-energy rows per structure grouping before
            passing data to the engine.
        save_files : list of str or None, optional
            Specific files to save from the engine output when partial saving is
            requested.
        use_last_hess : bool, optional
            If ``True``, reuse the most recent ``*.hess`` column from the
            dataframe by passing it back into the engine as an input file.

        Returns
        -------
        pandas.DataFrame
            A copy of the input dataframe with new stage-prefixed output
            columns added.

        Notes
        -----
        - Rows with missing coordinates are skipped and receive
          ``*-NT=False`` and ``*-EE=NaN``.
        - Engine exceptions are caught, logged, and stored as a stage-specific
          ``*-error`` column instead of aborting the whole dataframe run.
        - Output directories are created lazily and only when saving is
          requested.
        """
        import pandas as _pd

        df_out    = normalize_dataframe(df)
        df_out.attrs.update(getattr(df, "attrs", {}))
        self._validate_required_columns(
            df_out,
            needs_grouping=lowest is not None,
            needs_hessian=use_last_hess,
        )
        coord_col = self._last_coord_col(df_out)
        all_row_data: list[dict[str, object]] = []

        if lowest is not None and lowest < 1:
            self.logger.warning(f"ignoring lowest={lowest!r}, must be ≥1")
            lowest = None

        if lowest:
            e_cols = energy_columns(df_out)
            if not e_cols:
                raise ValueError("cannot apply `lowest=` filter: no energy column found")
            last_energy = e_cols[-1]

            group_keys = infer_group_columns(df_out)
            if not group_keys:
                raise ValueError("cannot apply `lowest=` filter: no structure identity columns found")

            sort_keys = group_keys + [last_energy]
            df_out = (
                df_out
                .sort_values(sort_keys, na_position="last")
                .groupby(group_keys, dropna=False)
                .head(lowest)
            )

        for i, row in df_out.iterrows():
            coords = row[coord_col]
            # --- skip any row with no coords ---
            # if coords is None or (_pd.isna(coords) if not isinstance(coords, (list, tuple)) else False):
            if coords is None or (_pd.isna(coords) if not isinstance(coords, (list, tuple, np.ndarray)) else False):            
                all_row_data.append({
                    output_column(prefix, "normal_termination"): False,
                    output_column(prefix, "electronic_energy"):  np.nan
                })
                continue

            row_save_files = save_files
            save_full_calc = (self.save_calc_dirs and self.save_output) or save_step
            save_partial   = row_save_files is not None and not save_step

            if save_full_calc or save_partial:
                self._ensure_base_dir()
                row_name = self._row_name(row, i)
                row_conf_id = self._row_conf_id(row, i)
                dir_name = f"{row_name}_{row_conf_id}"
                engine_base = self.base_dir / prefix
                engine_base.mkdir(parents=True, exist_ok=True)
                created_dir = str(make_step_dir(engine_base, dir_name))
            else:
                created_dir = None

            if save_full_calc:
                calc_dir   = created_dir
                save_dir   = created_dir
                row_save_files = None
            elif save_partial:
                calc_dir   = None
                save_dir   = created_dir
            else:
                calc_dir   = None
                save_dir   = None
            
            if use_last_hess:
                last_hess_col = [col for col in df.columns if col.endswith(".hess")][-1]
                last_hess = {"private_input.hess": row[last_hess_col]}

            base_args = {
                "atoms":  row["atoms"],
                "coords": [list(c) for c in coords],
                "n_cores": self.n_cores,
                "scr": self.work_dir,
                "data2file": last_hess if use_last_hess else None
            }

            if calc_dir is not None:
                base_args["calc_dir"] = calc_dir
            if save_dir is not None:
                base_args["save_dir"] = save_dir
            if row_save_files is not None:
                base_args["save_files"] = row_save_files

            inputs = {**base_args, **build_inputs(row)}

            # Filter inputs to only those accepted by engine_fn (avoids unexpected kwargs)
            try:
                allowed = set(signature(engine_fn).parameters.keys())
                filtered = {k: v for k, v in inputs.items() if k in allowed}
                if self.debug:
                    dropped = sorted(set(inputs) - set(filtered))
                    if dropped:
                        self.logger.debug(f"[{prefix}] dropped unsupported kwargs: {dropped}")
                inputs = filtered
            except Exception:
                # If introspection fails for any reason, fall back to original inputs
                pass

            # step 3: run engine, catch exceptions
            row_name = self._row_name(row, i)
            self.logger.info(f"[{prefix}] row {i} ({row_name})…")
            try:
                out = engine_fn(**inputs) or {}
            except Exception as e:
                self.logger.exception(f"[{prefix}] row {i} ({row_name}) failed: {e}")
                out = {
                    "normal_termination": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            finally:
                if save_step and save_dir:
                    try:
                        import shutil
                        from pathlib import Path

                        row_conf_id = self._row_conf_id(row, i)
                        dir_name = f"{row_name}_{row_conf_id}"

                        candidates = list(Path(self.work_dir).rglob(dir_name))
                        src = max(candidates, key=lambda p: len(p.parts)) if candidates else None

                        if src is None:
                            self.logger.warning(f"No calc dir named '{dir_name}' found under {self.work_dir}")
                        else:
                            save_path = Path(save_dir)
                            if src.resolve() == save_path.resolve():
                                continue
                            for p in src.iterdir():
                                dst = save_path / p.name
                                if p.is_dir():
                                    shutil.copytree(p, dst, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(p, dst)
                    except Exception as e:
                        self.logger.error(f"Failed to stage files from '{src}' to '{save_dir}': {e}")

            # step 4: enforce defaults
            out.setdefault("normal_termination", False)
            if out["normal_termination"]:
                out.setdefault("electronic_energy", np.nan)
            # step 5: pick only the keys we know how to handle
            allowed = {"normal_termination", "electronic_energy", "opt_coords", "vibs", "error"}
            def _filelike(k: str) -> bool:
                # Accept only filename-ish keys (no spaces), or our private_* stash
                if not isinstance(k, str) or " " in k:
                    return False
                if k.startswith("private_"):
                    return True
                return bool(re.search(
                    r"\.(hess|xyz|inp|out|log|gbw|molden|wfn|txt|json)$", k
                ))

            for k in out.keys():
                if _filelike(k):
                    allowed.add(k)

            if "vibs" in out and "gibbs_energy" in out:
                allowed.add("gibbs_energy")

            if self.debug:
                ignored = sorted([k for k in out.keys()
                                if '.' in k and k not in allowed])
                if ignored:
                    self.logger.debug(f"[{prefix}] ignoring non-file dot-keys: {ignored}")

            row_data: dict[str, object] = {}
            for key in allowed:
                if key in out:
                    col = output_column(prefix, key)
                    row_data[col] = out[key]

            all_row_data.append(row_data)

        # --- now assemble all_row_data back into df_out ---
        all_cols = sorted({c for rd in all_row_data for c in rd})
        column_arrays = {col: [] for col in all_cols}

        for rd in all_row_data:
            for col in all_cols:
                if col in rd:
                    column_arrays[col].append(rd[col])
                else:
                    # fill defaults for skipped rows
                    if col.endswith("-NT") or col.endswith("-normal_termination"):
                        column_arrays[col].append(False)
                    elif col.endswith("-EE") or col.endswith("-GE") or col.endswith("-electronic_energy") or col.endswith("-gibbs_energy"):
                        column_arrays[col].append(np.nan)
                    elif col.endswith("-error"):
                        column_arrays[col].append(None)
                    else:
                        column_arrays[col].append(None)

        for col, vals in column_arrays.items():
            df_out[col] = vals

        steps = dict(df_out.attrs.get("frust_steps", {}))
        steps[prefix] = {"engine": prefix.split("-", 1)[0], "columns": sorted(column_arrays)}
        df_out.attrs["frust_steps"] = steps

        return df_out

    def xtb(
        self,
        df: pd.DataFrame,
        name: str = "xtb",
        options: dict | None = None,
        detailed_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
        lowest: int | None = None,
        n_cores: int | None = None,
    ) -> pd.DataFrame:
        """Embed multiple conformers with xTB and optionally optimize and/or
        compute frequencies.

        Args:
            df (pd.DataFrame): A DataFrame containing conformers. Required
                columns:
                - ``atoms``: list of atomic symbols
                - one coordinate column, typically ``coords_embedded`` or a
                  prior ``*-oc`` column
                - ``constraint_atoms`` when ``constraint=True``
                - ``substrate_name`` when ``lowest`` is used
            name (str): Base name for the xTB step, used to prefix result
                columns.
            options (dict, optional): xTB driver options, e.g. ``{'gfn': 2,
                'opt': None}``. Defaults to ``{'gfn': 0}``.
            detailed_inp_str (str, optional): Additional xTB input block
                (cards) to include. Defaults to ``""``.
            constraint (bool, optional): If ``True``, applies predefined
                distance and angle constraints for supported ``step_type``
                values. Requires ``Stepper(step_type=...)`` with one of
                ``TS1``, ``TS2``, ``TS3``, ``TS4``, or ``INT3``. Defaults to
                ``False``.
            save_step (bool, optional): If ``True``, saves calculation
                directories for each conformer. Defaults to ``False``.
            lowest (int or None, optional): If set, retains only the
                lowest-energy ``N`` conformers per structure group. Defaults
                to ``None``.
            n_cores (int or None, optional): If set, overrides the Stepper’s
                default core count for this xTB call only. Defaults to
                ``None``.

        Returns:
            pd.DataFrame: The input DataFrame augmented with stage-prefixed xTB
            result columns, typically including:
                - ``{prefix}-NT``
                - ``{prefix}-EE``
                - ``{prefix}-oc`` for optimization jobs
                - ``{prefix}-vibs`` and ``{prefix}-GE`` for
                  frequency jobs
                - ``{prefix}-error`` when a row-level engine failure occurs
                - saved file-content columns when the backend returns them
        """
        opts = dict(options) if options else {"gfn": 0}
        if constraint:
            self._validate_constraint_request(df)
        keys = list(opts)
        if name != "xtb":
            prefix = name
        else:
            level = keys[0]
            opt_flag = next((k for k in keys[1:] if k in ("opt", "ohess")), None)
            prefix = f"{name}-{level}" + (f"-{opt_flag}" if opt_flag else "")

        def build_xtb(row: pd.Series) -> dict:
            inp: dict[str, object] = {"options": opts}
            step_type = self._step_type_upper()

            # Per-call override: only affects xTB, not ORCA.
            if n_cores is not None:
                inp["n_cores"] = int(n_cores)

            # Only add the user‐provided card if it's non‐empty
            base_str = detailed_inp_str.strip()
            if base_str:
                inp["detailed_input_str"] = base_str

            block = None

            if step_type == "TS1" and constraint:
                B, N, H, C = 0, 1, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[B]}, {atom[H]}, 2.07696
                distance: {atom[N]}, {atom[H]}, 1.5127 
                distance: {atom[H]}, {atom[C]}, 1.29095
                distance: {atom[B]}, {atom[C]}, 1.68461
                distance: {atom[B]}, {atom[N]}, 3.06223
                angle: {atom[N]}, {atom[H]}, {atom[C]}, 170.1342
                angle: {atom[H]}, {atom[C]}, {atom[B]}, 87.4870
                $end
                """).strip()

            if step_type == "TS2" and constraint:
                BCat10, N17, H40, H41, C46 = 0, 1, 4, 3, 5  # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H41]}, 1.656
                distance: {atom[N17]}, {atom[H40]}, 1.961
                distance: {atom[BCat10]}, {atom[N17]}, 3.080
                angle: {atom[BCat10]}, {atom[H41]}, {atom[N17]}, 86.58
                $end
                """).strip()

            if step_type == "TS3" and constraint:
                BCat10, H11, BPin22, H21, C = 0, 2, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[H21]}, {atom[BCat10]}, 1.376
                distance: {atom[H21]}, {atom[BPin22]}, 1.264
                distance: {atom[H21]}, {atom[C]}, 2.477
                distance: {atom[BCat10]}, {atom[C]}, 1.616
                distance: {atom[BPin22]}, {atom[C]}, 2.180
                distance: {atom[BPin22]}, {atom[BCat10]}, 2.007
                angle: {atom[BCat10]}, {atom[H21]}, {atom[BPin22]}, 98.89
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin22]}, 61.75
                $end
                """).strip()

            if step_type == "TS4" and constraint:
                BCat11, H12, H13, BPin37, C = 0, 2, 3, 4, 5 # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat11]}, {atom[BPin37]}, 2.219
                distance: {atom[BPin37]}, {atom[H13]}, 1.868
                distance: {atom[C]}, {atom[H13]}, 2.489
                distance: {atom[BCat11]}, {atom[H13]}, 1.216
                distance: {atom[BCat11]}, {atom[C]}, 1.946
                distance: {atom[BPin37]}, {atom[C]}, 1.585
                angle: {atom[BCat11]}, {atom[H13]}, {atom[BPin37]}, 89.48
                angle: {atom[BCat11]}, {atom[C]}, {atom[BPin37]}, 77.13
                $end
                """).strip()

            if step_type == "INT3" and constraint:
                BCat10, BPin42, H11, C = 0, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H11]}, 1.279
                distance: {atom[BCat10]}, {atom[C]}, 1.688
                distance: {atom[BPin42]}, {atom[H11]}, 1.378
                distance: {atom[BPin42]}, {atom[C]}, 1.749
                angle: {atom[BCat10]}, {atom[H11]}, {atom[BPin42]}, 89.85
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin42]}, 66.22
                $end
                """).strip()

            if block:
                if "detailed_input_str" in inp:
                    inp["detailed_input_str"] += "\n\n" + block
                else:
                    inp["detailed_input_str"] = block

            return inp

        result = self._run_engine(
            df, self.xtb_fn, prefix, build_xtb, save_step, lowest
        )
        result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
            {"engine": "xtb", "options": opts}
        )
        return result

    def gxtb(
        self,
        df: pd.DataFrame,
        name: str = "gxtb",
        options: dict | None = None,
        detailed_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
        lowest: int | None = None,
        n_cores: int | None = None,
    ) -> pd.DataFrame:
        """Run g-xTB v2 calculations through Tooltoad's g-xTB calculator."""
        opts = dict(options) if options else {}
        if constraint:
            self._validate_constraint_request(df)
        keys = list(opts)
        if name != "gxtb":
            prefix = name
        else:
            opt_flag = next((k for k in keys if k in ("opt", "ohess")), None)
            prefix = f"{name}" + (f"-{opt_flag}" if opt_flag else "")

        def build_gxtb(row: pd.Series) -> dict:
            inp: dict[str, object] = {"options": opts}
            step_type = self._step_type_upper()

            if n_cores is not None:
                inp["n_cores"] = int(n_cores)

            base_str = detailed_inp_str.strip()
            if base_str:
                inp["detailed_input_str"] = base_str

            block = None

            if step_type == "TS1" and constraint:
                B, N, H, C = 0, 1, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[B]}, {atom[H]}, 2.07696
                distance: {atom[N]}, {atom[H]}, 1.5127
                distance: {atom[H]}, {atom[C]}, 1.29095
                distance: {atom[B]}, {atom[C]}, 1.68461
                distance: {atom[B]}, {atom[N]}, 3.06223
                angle: {atom[N]}, {atom[H]}, {atom[C]}, 170.1342
                angle: {atom[H]}, {atom[C]}, {atom[B]}, 87.4870
                $end
                """).strip()

            if step_type == "TS2" and constraint:
                BCat10, N17, H40, H41, C46 = 0, 1, 4, 3, 5  # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H41]}, 1.656
                distance: {atom[N17]}, {atom[H40]}, 1.961
                distance: {atom[BCat10]}, {atom[N17]}, 3.080
                angle: {atom[BCat10]}, {atom[H41]}, {atom[N17]}, 86.58
                $end
                """).strip()

            if step_type == "TS3" and constraint:
                BCat10, H11, BPin22, H21, C = 0, 2, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[H21]}, {atom[BCat10]}, 1.376
                distance: {atom[H21]}, {atom[BPin22]}, 1.264
                distance: {atom[H21]}, {atom[C]}, 2.477
                distance: {atom[BCat10]}, {atom[C]}, 1.616
                distance: {atom[BPin22]}, {atom[C]}, 2.180
                distance: {atom[BPin22]}, {atom[BCat10]}, 2.007
                angle: {atom[BCat10]}, {atom[H21]}, {atom[BPin22]}, 98.89
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin22]}, 61.75
                $end
                """).strip()

            if step_type == "TS4" and constraint:
                BCat11, H12, H13, BPin37, C = 0, 2, 3, 4, 5  # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat11]}, {atom[BPin37]}, 2.219
                distance: {atom[BPin37]}, {atom[H13]}, 1.868
                distance: {atom[C]}, {atom[H13]}, 2.489
                distance: {atom[BCat11]}, {atom[H13]}, 1.216
                distance: {atom[BCat11]}, {atom[C]}, 1.946
                distance: {atom[BPin37]}, {atom[C]}, 1.585
                angle: {atom[BCat11]}, {atom[H13]}, {atom[BPin37]}, 89.48
                angle: {atom[BCat11]}, {atom[C]}, {atom[BPin37]}, 77.13
                $end
                """).strip()

            if step_type == "INT3" and constraint:
                BCat10, BPin42, H11, C = 0, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H11]}, 1.279
                distance: {atom[BCat10]}, {atom[C]}, 1.688
                distance: {atom[BPin42]}, {atom[H11]}, 1.378
                distance: {atom[BPin42]}, {atom[C]}, 1.749
                angle: {atom[BCat10]}, {atom[H11]}, {atom[BPin42]}, 89.85
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin42]}, 66.22
                $end
                """).strip()

            if block:
                if "detailed_input_str" in inp:
                    inp["detailed_input_str"] += "\n\n" + block
                else:
                    inp["detailed_input_str"] = block

            return inp

        result = self._run_engine(
            df, self.gxtb_fn, prefix, build_gxtb, save_step, lowest
        )
        result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
            {"engine": "gxtb", "options": opts}
        )
        return result


    def orca(
        self,
        df: pd.DataFrame,
        name: str = "orca",
        options: dict | None = None,
        xtra_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
        save_files: list[str] | None = None,
        lowest: int | None = None,
        uma: str | None = None,
        read_files: list | None = None,
        use_last_hess: bool = False,
        n_cores: int | None = None,
        uma_server: bool = True,
        uma_device: str = "cpu",
        uma_cache_dir: str | None = None,
        uma_offline: bool = False,
        uma_server_cores: int | None = None,
        uma_memory_per_thread_mib: int = 500,
        uma_keep_logs: bool | str = "on_failure",
        uma_log_dir: str | None = None,
        gxtb: bool = False,
        gxtb_exe: str | None = None,
        gxtb_ext_params: str | None = None,
        **kw        
    ) -> pd.DataFrame:
        """Run ORCA calculations and attach results to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame of conformers to compute. Must include:
                - ``atoms``: list of atomic symbols
                - one coordinate column, typically ``coords_embedded`` or a
                  prior ``*-oc`` column
                - ``constraint_atoms`` when ``constraint=True``
                - ``substrate_name`` when ``lowest`` is used
                - a prior ``*.hess`` column when ``use_last_hess=True``
            name (str): Base name for the ORCA step, used to prefix result
                columns.
            options (dict): ORCA input keywords, e.g.
                ``{'wB97X-D3': None, '6-31G**': None, 'OptTS': None,
                'Freq': None}``.
            xtra_inp_str (str, optional): Additional ORCA input block such as
                CPCM settings or custom geometry directives. Defaults to
                ``""``.
            constraint (bool, optional): If ``True``, applies predefined
                distance and angle constraints for supported ``step_type``
                values. Requires ``Stepper(step_type=...)`` with one of
                ``TS1``, ``TS2``, ``TS3``, ``TS4``, or ``INT3``. Defaults to
                ``False``.
            save_step (bool, optional): If ``True``, saves ORCA run
                directories for inspection. Defaults to ``False``.
            save_files (list[str] or None, optional): Specific ORCA output
                files to retain when partial saving is enabled. If omitted and
                instance-level output saving is enabled, defaults to
                ``["orca.out"]``.
            lowest (int or None, optional): If set, keeps only the
                lowest-energy conformer per structure group. Defaults to
                ``None``.
            uma (str or None, optional): Optional UMA task/profile identifier
                used to inject ORCA external optimization settings. Defaults to
                ``None``.
            read_files (list or None, optional): Deposit contents from files in
                the work directory directly into the dataframe, e.g.
                ``["input.hess"]``.
            use_last_hess (bool, optional): If ``True``, scan the dataframe for
                the most recent ``*.hess`` column and feed it back to ORCA as
                ``private_input.hess``. Defaults to ``False``.
            n_cores (int or None, optional): If set, overrides the Stepper's
                default core count for this ORCA call only. Defaults to
                ``None``.
            uma_server (bool, optional): If ``True``, runs UMA through OET's
                server/client mode. If ``False``, uses standalone ``oet_uma``.
                Defaults to ``True``.
            uma_device (str, optional): OET UMA device argument, typically
                ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
            uma_cache_dir (str or None, optional): Optional FairChem cache
                directory passed to OET UMA. Defaults to ``None``.
            uma_offline (bool, optional): If ``True``, asks OET UMA to use
                offline mode. Defaults to ``False``.
            uma_server_cores (int or None, optional): Total core budget passed
                to ``oet_server --nthreads``. Defaults to this ORCA call's
                core count.
            uma_memory_per_thread_mib (int, optional): Memory budget passed to
                ``oet_server --memory-per-thread``. Defaults to ``500``.
            uma_keep_logs (bool or str, optional): Server-log retention policy:
                ``"on_failure"`` preserves logs only when the UMA-backed ORCA
                step fails, ``True``/``"always"`` keeps logs, and
                ``False``/``"never"`` removes them. Defaults to
                ``"on_failure"``.
            uma_log_dir (str or None, optional): Directory for preserved UMA
                server logs. If omitted, transient logs are written to a temp
                directory and preserved failures are copied to ``UMA-logs``.
            gxtb (bool, optional): If ``True``, runs ORCA with OET g-xTB v2 as
                an external method provider. ORCA still owns ``Opt``,
                ``OptTS``, ``NEB-TS``, and related run types. Defaults to
                ``False``.
            gxtb_exe (str or None, optional): Optional path to the g-xTB v2
                ``xtb`` executable. If omitted, ``GXTB_EXE`` is used.
            gxtb_ext_params (str or None, optional): Extra parameters appended
                to OET ``oet_gxtb`` through ORCA ``Ext_Params``.

        Returns:
            pd.DataFrame: The input DataFrame extended with stage-prefixed ORCA
            output columns, typically including:
                - ``{prefix}-NT``
                - ``{prefix}-EE``
                - ``{prefix}-oc`` for optimization jobs
                - ``{prefix}-vibs`` and ``{prefix}-GE`` for
                  frequency jobs
                - ``{prefix}-error`` when a row-level engine failure occurs
                - saved file-content columns when ORCA returns them
        """
        opts = dict(options or {})
        if kw:
            unknown = ", ".join(sorted(kw))
            raise TypeError(f"Unexpected orca() keyword arguments: {unknown}")
        if uma is not None and gxtb:
            raise ValueError("orca() cannot use both UMA and g-xTB external methods")
        if gxtb and "Freq" in opts:
            raise ValueError(
                "ORCA Freq is not compatible with g-xTB ExtOpt. "
                "Use NumFreq for finite-difference frequencies with external g-xTB gradients."
            )
        if gxtb and "ExtOpt" not in opts:
            opts = {"ExtOpt": None, **opts}
        if constraint:
            self._validate_constraint_request(df)
        if save_files is None and self.save_output:
            save_files = ["orca.out"]
        keys = list(opts)
        if len(keys) < 1:
            raise ValueError("`options` must include at least one ORCA method key")

        if name != "orca":
            prefix = name
        elif len(keys) == 1:
            prefix = f"{name}-{keys[0]}"
        else:
            func, basis = keys[0], keys[1]
            opt_flag = next((k for k in keys[2:] if k in ("OptTS", "Freq", "NumFreq", "NoSym")), None)
            prefix = f"{name}-{func}-{basis}" + (f"-{opt_flag}" if opt_flag else "")

        def build_orca(row: Series) -> dict:
            step_type = self._step_type_upper()
            inp = {
                "options": opts,
                "xtra_inp_str": xtra_inp_str.strip(),
                "memory": self.memory_gb,
                "n_cores": int(n_cores) if n_cores is not None else self.n_cores,
                "read_files": read_files,
            }

            if "Freq" in opts:
                block = textwrap.dedent("""
                    %geom
                      Calc_Hess true
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if use_last_hess:
                block = textwrap.dedent(
                    """
                    %geom
                      inhess Read
                      InHessName "private_input.hess"
                    end
                    """
                ).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS1":
                atom = self._constraint_atoms(row)
                B, N, H, C = atom[0], atom[1], atom[4], atom[5]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {B} {H} 2.07696 C}}
                      {{B {N} {H} 1.5127 C}}
                      {{B {H} {C} 1.29095 C}}
                      {{B {B} {C} 1.68461 C}}
                      {{B {B} {N} 3.06223 C}}
                      {{A {N} {H} {C} C}}
                      {{A {H} {C} {B} 87.4870 C}}
                    end
                    end
                """).strip() # {{A {N} {H} {C} 170.1342 C}}
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS2":
                atom = self._constraint_atoms(row)
                BCat, N17, H40, H41 = atom[0], atom[1], atom[4], atom[3]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {BCat} {H41} 1.656 C}}
                      {{B {N17} {H40} 1.961 C}}
                      {{B {BCat} {N17} 3.080 C}}
                      {{A {BCat} {H41} {N17} 86.58 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS3":
                atom = self._constraint_atoms(row)
                BCat, H11, BPin, H21, C = atom[0], atom[2], atom[3], atom[4], atom[5]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {H21} {BCat} 1.376 C}}
                      {{B {H21} {BPin} 1.264 C}}
                      {{B {H21} {C} 2.477 C}}
                      {{B {BCat} {C} 1.616 C}}
                      {{B {BPin} {C} 2.180 C}}
                      {{B {BPin} {BCat} 2.007 C}}
                      {{A {BCat} {H21} {BPin} 98.89 C}}
                      {{A {BCat} {C} {BPin} 61.75 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS4":
                atom = self._constraint_atoms(row)
                BCat, H12, H13, BPin, C = atom[0], atom[2], atom[3], atom[4], atom[5]  # noqa: F841
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {BCat} {BPin} 2.219 C}}
                      {{B {BPin} {H13} 1.868 C}}
                      {{B {C} {H13} 2.489 C}}
                      {{B {BCat} {H13} 1.216 C}}
                      {{B {BCat} {C} 1.946 C}}
                      {{B {BPin} {C} 1.585 C}}
                      {{A {BCat} {H13} {BPin} 89.48 C}}
                      {{A {BCat} {C} {BPin} 77.13 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "INT3":
                atom = self._constraint_atoms(row)
                BCat, BPin, H11, C = atom[0], atom[3], atom[4], atom[5]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {BCat} {H11} 1.279 C}}
                      {{B {BCat} {C} 1.688 C}}
                      {{B {BPin} {H11} 1.378 C}}
                      {{B {BPin} {C} 1.749 C}}
                      {{A {BCat} {H11} {BPin} 89.85 C}}
                      {{A {BCat} {C} {BPin} 66.22 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            return inp

        if uma is None and not gxtb:
            result = self._run_engine(df, self.orca_fn, prefix, build_orca, save_step, lowest, save_files, use_last_hess)
            result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
                {"engine": "orca", "options": opts}
            )
            return result

        if gxtb:
            from frust.utils.gxtb import gxtb_orca_block

            client_block = gxtb_orca_block(gxtb_exe=gxtb_exe, ext_params=gxtb_ext_params)

            def build_orca_gxtb(row: Series) -> dict:
                inp = build_orca(row)
                xin = inp.get("xtra_inp_str", "")
                inp["xtra_inp_str"] = (xin + "\n\n" + client_block).strip() if xin else client_block
                return inp

            result = self._run_engine(
                df, self.orca_fn, prefix, build_orca_gxtb, save_step, lowest, save_files
            )
            result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
                {
                    "engine": "orca",
                    "options": opts,
                    "gxtb": True,
                    "gxtb_exe": gxtb_exe,
                }
            )
            return result
        
        from frust.utils.uma import parse_uma_spec, uma_orca_block, uma_server as run_uma_server

        spec = parse_uma_spec(
            uma,
            device=uma_device,
            cache_dir=uma_cache_dir,
            offline=uma_offline,
        )

        def run_with_uma_block(client_block: str) -> pd.DataFrame:
            orig_build = build_orca

            def build_orca_uma(row: Series) -> dict:
                inp = orig_build(row)
                xin = inp.get("xtra_inp_str", "")
                inp["xtra_inp_str"] = (xin + "\n\n" + client_block).strip() if xin else client_block
                return inp

            result = self._run_engine(df, self.orca_fn, prefix, build_orca_uma, save_step, lowest, save_files)
            result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
                {
                    "engine": "orca",
                    "options": opts,
                    "uma": uma,
                    "uma_task": spec.task,
                    "uma_model": spec.model,
                    "uma_server": uma_server,
                }
            )
            return result

        def uma_result_failed(result: pd.DataFrame) -> bool:
            nt_col = output_column(prefix, "normal_termination")
            err_col = output_column(prefix, "error")
            if nt_col in result and result[nt_col].eq(False).any():
                return True
            if err_col in result and result[err_col].notna().any():
                return True
            return False

        if not uma_server:
            client_block = uma_orca_block(spec, server=False)
            return run_with_uma_block(client_block)

        effective_cores = int(n_cores) if n_cores is not None else self.n_cores
        server_cores = uma_server_cores if uma_server_cores is not None else effective_cores
        with run_uma_server(
            log_dir=uma_log_dir,
            keep_logs=uma_keep_logs,
            use_gpu=uma_device == "cuda",
            server_cores=server_cores,
            memory_per_thread_mib=uma_memory_per_thread_mib,
        ) as server_handle:
            client_block = uma_orca_block(spec, server=True, bind=server_handle.bind)
            result = run_with_uma_block(client_block)
            if uma_keep_logs == "on_failure" and uma_result_failed(result):
                server_handle.preserve()
            return result
