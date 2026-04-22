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
            self.xtb_fn  = mock_xtb_calculate
            self.orca_fn = mock_orca_calculate
        else:
            from tooltoad.xtb import xtb_calculate
            from tooltoad.orca import orca_calculate
            self.xtb_fn  = xtb_calculate
            self.orca_fn = orca_calculate

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
        preferred.extend([c for c in df.columns if c.endswith("-opt_coords")])
        preferred.extend(
            [c for c in df.columns if "coords" in c and c not in preferred]
        )
        return preferred

    @classmethod
    def _last_coord_col(cls, df: pd.DataFrame) -> str:
        """Return the most specific available coordinate column."""
        cols = cls._coord_columns(df)
        if not cols:
            raise ValueError(
                "DataFrame must contain coordinates in 'coords_embedded' or a '*-opt_coords' column"
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

        if needs_grouping and "ligand_name" not in df.columns:
            raise ValueError(
                "`lowest=` requires a 'ligand_name' column so conformers can be grouped before filtering"
            )

        if needs_hessian and not any(col.endswith(".hess") for col in df.columns):
            raise ValueError(
                "`use_last_hess=True` requires a prior '*.hess' column in the input DataFrame"
            )

    @staticmethod
    def _row_name(row: Series, index: object) -> str:
        """Pick a stable human-readable row label from available metadata."""
        for key in ("custom_name", "ligand_name", "moltype"):
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

        In the TS case we parse out `ligand_name` and `rpos` from the key;
        in the plain‐mol case we set ligand_name=key and rpos=None.

        Returns
        -------
        pd.DataFrame
            One row per conformer with columns
              - custom_name         (the original dict key)
              - ligand_name         (parsed or just the key)
              - rpos                (int or None)
              - constraint_atoms    (list[int] or empty list)
              - cid                 (conformer ID)
              - smiles              (str or None)
              - atoms               (list of atomic symbols)
              - coords_embedded     (list of (x,y,z) tuples)
              - energy_uff          (float or None)
        """
        rows: list[dict] = []

        pattern = re.compile(
            r'^(?:(?P<prefix>(?:TS|INT)\d*|Mols)\()?'
            r'(?P<ligand>.+?)_rpos\('        
            r'(?P<rpos>\d+)\)\)?$'
        )

        for name, val in embedded_dict.items():
            prefix = None
            if len(val) == 2:
                mol, cids = val
                keep_idxs = None
                smi = None
                energies: list[tuple[float,int]] = []
            elif len(val) == 4:
                mol, cids, keep_idxs, smi = val
                energies = []
            elif len(val) == 5:
                mol, cids, keep_idxs, smi, energies = val
            else:
                raise ValueError(f"Bad tuple length for {name}")

            m = pattern.match(name)
            if m:
                prefix = m.group("prefix")
                raw = m.group("ligand")
                ligand_name = raw.split("_", 1)[1] if "_" in raw else raw
                rpos = int(m.group("rpos"))
            else:
                if "_" in name:
                    ligand_name = name.split("_", 1)[1]
                else:
                    ligand_name = name
                rpos = pd.NA

            e_map: dict[int,float] = {cid_val: e_val for (e_val, cid_val) in energies} if energies else {}

            atom_syms = [atom.GetSymbol() for atom in mol.GetAtoms()]

            for cid in cids:
                conf = mol.GetConformer(cid)
                coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

                rows.append({
                    "custom_name":      name,
                    "ligand_name":      ligand_name,
                    "rpos":             rpos,
                    "constraint_atoms": keep_idxs,
                    "cid":              cid,
                    "smiles":           smi,
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

        - Rows with missing coords (None or NaN) are auto‐skipped and get
        normal_termination=False / electronic_energy=NaN.
        - Only rows with valid coords are passed to engine_fn.
        """
        import pandas as _pd

        df_out    = df.copy()
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
            energy_cols = [c for c in df_out.columns if c.endswith("_energy")]
            if not energy_cols:
                raise ValueError("cannot apply `lowest=` filter: no *_energy column found")
            last_energy = energy_cols[-1]

            # build the list of grouping keys: always ligand_name, optionally rpos
            group_keys = ["ligand_name"]
            if "rpos" in df_out.columns:
                group_keys.append("rpos")

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
                    f"{prefix}-normal_termination": False,
                    f"{prefix}-electronic_energy":  np.nan
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
                    col = f"{prefix}-{key}"
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
                    if col.endswith("-normal_termination"):
                        column_arrays[col].append(False)
                    elif col.endswith("-electronic_energy") or col.endswith("-gibbs_energy"):
                        column_arrays[col].append(np.nan)
                    elif col.endswith("-error"):
                        column_arrays[col].append(None)
                    else:
                        column_arrays[col].append(None)

        for col, vals in column_arrays.items():
            df_out[col] = vals

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
            df (pd.DataFrame): A DataFrame containing embedded conformers.
                Required columns:
                - 'coords_embedded': list of 3D coordinate tuples for each
                    conformer.
                - 'atoms': list of atomic symbols.
                - 'constraint_atoms' (optional): list of atom indices to
                    constrain during optimization.
            name (str): Base name for the xTB step, used to prefix result
                columns.
            options (dict, optional): xTB driver options, e.g. {'gfn': 2,
                'opt': None}. Defaults to {'gfn': 0}.
            detailed_inp_str (str, optional): Additional xTB input block
                (cards) to include. Defaults to "".
            constraint (bool, optional): If True, applies predefined
                distance/angle constraints for TS steps. Defaults to False.
            save_step (bool, optional): If True, saves calculation
                directories for each conformer. Defaults to False.
            lowest (int or None, optional): If set, retains only the
                lowest-energy N conformers per ligand/rpos group. Defaults
                to None.
            n_cores (int or None, optional): If set, overrides the Stepper’s
                default core count **for xTB only**. ORCA continues to use
                `self.n_cores`. Defaults to None.

        Returns:
            pd.DataFrame: The input DataFrame augmented with columns for each
            xTB result:
                - '{name}-{method}-normal_termination' (bool)
                - '{name}-{method}-electronic_energy' (float)
                - '{name}-{method}-opt_coords' (list of coords) if
                optimization run
                - '{name}-{method}-vibs' (vibrational modes) if frequencies
                computed
                - '{name}-{method}-gibbs_energy' (float) if frequencies
                computed
        """
        opts = dict(options) if options else {"gfn": 0}
        if constraint:
            self._validate_constraint_request(df)
        keys = list(opts)
        level = keys[0]
        # check for optimization flag among the remaining keys
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

        return self._run_engine(
            df, self.xtb_fn, prefix, build_xtb, save_step, lowest
        )


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
        **kw        
    ) -> pd.DataFrame:
        """Run ORCA calculations (SP, OptTS, Freq) and attach results to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame of conformers to compute. Must include:
                - 'coords_embedded': list of 3D coordinate tuples for each conformer.
                - 'atoms': list of atomic symbols.
            name (str): Base name for the ORCA step, used to prefix result columns.
            options (dict): ORCA input keywords, e.g. {'wB97X-D3': None, '6-31G**': None, 'OptTS': None, 'Freq': None}.
            xtra_inp_str (str, optional): Additional ORCA input block (e.g. CPCM or Calc_Hess). Defaults to "".
            constraint (bool, optional): If True, applies predefined distance/angle constraints for TS steps. Defaults to False.
            save_step (bool, optional): If True, saves ORCA run directories for inspection. Defaults to False.
            lowest (int or None, optional): If set, keeps only the lowest-energy conformer per ligand/rpos group. Defaults to None.
            read_files (list or None, optional): Deposit contents from files in the work_dir directly in the dataframe, \
            e.g ["input.hess"].
            use_last_hess (bool, optional): If True, will scan the dataframe for a *.hess column with contents fror a .hess file. In order to do this \
            a frequency calculation must be done with the `read_files` argument set to ["input.hess"] in order to save the \
            hessian into the dataframe.

        Returns:
            pd.DataFrame: The input DataFrame extended with ORCA output columns:
                - '{name}-{method}-normal_termination' (bool)
                - '{name}-{method}-electronic_energy' (float)
                - '{name}-{method}-opt_coords' (list of coords) if optimization run
                - '{name}-{method}-vibs' (vibrational modes) if frequencies computed
                - '{name}-{method}-gibbs_energy' (float) if frequencies computed
        """
        opts = options or {}
        if kw:
            unknown = ", ".join(sorted(kw))
            raise TypeError(f"Unexpected orca() keyword arguments: {unknown}")
        if constraint:
            self._validate_constraint_request(df)
        if save_files is None and self.save_output:
            save_files = ["orca.out"]
        keys = list(opts)
        if len(keys) < 1:
            raise ValueError("`options` must include at least one ORCA method key")

        if len(keys) == 1:
            prefix = f"{name}-{keys[0]}"
        else:
            func, basis = keys[0], keys[1]
            opt_flag = next((k for k in keys[2:] if k in ("OptTS", "Freq", "NoSym")), None)
            prefix = f"{name}-{func}-{basis}" + (f"-{opt_flag}" if opt_flag else "")

        def build_orca(row: Series) -> dict:
            step_type = self._step_type_upper()
            inp = {
                "options": opts,
                "xtra_inp_str": xtra_inp_str.strip(),
                "memory": self.memory_gb,
                "n_cores": self.n_cores,
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
                      {{A {N} {H} {C} 170.1342 C}}
                      {{A {H} {C} {B} 87.4870 C}}
                    end
                    end
                """).strip()
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

        if uma is None:
            return self._run_engine(df, self.orca_fn, prefix, build_orca, save_step, lowest, save_files, use_last_hess)
        
        from frust.utils.uma import _uma_server
        from frust.config import UMA_TOOLS as TOOLS
        with _uma_server(task=uma, log_dir="UMA-logs") as (port, _slog):
            client_block = f"""
    %method
    ProgExt "{TOOLS}/umaclient.sh"
    Ext_Params "-b 127.0.0.1:{port}"
    end
    %output
    Print[P_EXT_OUT] 1
    Print[P_EXT_GRAD] 1
    end
    """.strip()
            orig_build = build_orca
            def build_orca_uma(row: Series) -> dict:
                inp = orig_build(row)
                xin = inp.get("xtra_inp_str", "")
                inp["xtra_inp_str"] = (xin + "\n\n" + client_block).strip() if xin else client_block
                return inp
            return self._run_engine(df, self.orca_fn, prefix, build_orca_uma, save_step, lowest, save_files)
