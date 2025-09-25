# frustactivation/stepper.py
import logging
logger = logging.getLogger(__name__)

import sys, os
from pathlib import Path
import re
import textwrap
from typing import Callable
import numpy as np
import pandas as pd
from pandas import Series
import submitit

from frust.utils.dirs import make_step_dir, prepare_base_dir
from frust.utils.slurm import detect_job_id

class Stepper:
    def __init__(
        self,
        ligands_smiles: list[str],
        step_type: str | None = None,
        output_base: Path | str | None = None,
        job_id: int | None = None,
        debug: bool = False,
        live: bool = False,
        dump_each_step: bool = False,
        n_cores: int = 8,
        memory_gb: float = 20.0,
        save_calc_dirs: bool = False,
        save_output_dir: bool = True,
        **kwargs
    ):
        self.ligands_smiles = ligands_smiles
        self.step_type      = step_type
        self.debug          = debug
        self.live           = live
        job_id              = detect_job_id(job_id, live and not debug)
        self.dump_each_step = dump_each_step
        self.n_cores        = n_cores
        self.memory_gb      = memory_gb
        self.save_calc_dirs = save_calc_dirs
        self.save_output    = save_output_dir

        self.base_dir = Path(output_base) if output_base is not None else Path(".")
        if save_output_dir:
            self.base_dir = prepare_base_dir(output_base, ligands_smiles, job_id)

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

        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter(
                "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(fmt)
            logger.addHandler(handler)

        try:
            self.work_dir = os.environ["SCRATCH"]
        except:
            self.work_dir = "."
        os.makedirs(self.work_dir, exist_ok=True)
        logger.info(f"Working dir: {self.work_dir}")  

    @staticmethod
    def _last_coord_col(df: pd.DataFrame) -> str:
        """Return the name of the last column containing 'coords'."""
        cols = [c for c in df.columns if "coords" in c]
        if not cols:
            raise ValueError("No column containing 'coords' found")
        return cols[-1]

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

            if self.step_type == None:
                try:
                    if prefix:
                        self.step_type = prefix
                    else:
                        self.step_type = "MOLS"
                except Exception:
                    self.step_type = "unknown"
                    logger.warning("\n\nwarning: No calculation type identified.\nwarning: This is fine if you don't calculate a transition state.\n")

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
        ) -> pd.DataFrame:
        """
        Generic runner for xTB or ORCA or any other engine.

        - Rows with missing coords (None or NaN) are auto‐skipped and get
        normal_termination=False / electronic_energy=NaN.
        - Only rows with valid coords are passed to engine_fn.
        """
        import pandas as _pd

        df_out    = df.copy()
        coord_col = self._last_coord_col(df_out)
        all_row_data: list[dict[str, object]] = []

        if lowest is not None and lowest < 1:
            logger.warning(f"ignoring lowest={lowest!r}, must be ≥1")
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

            save_full_calc = (self.save_calc_dirs and self.save_output) or save_step
            save_partial   = save_files is not None and not save_step

            if save_full_calc or save_partial:
                dir_name    = f"{row['custom_name']}_{row['cid']}"
                engine_base = self.base_dir / prefix
                engine_base.mkdir(parents=True, exist_ok=True)
                created_dir = str(make_step_dir(engine_base, dir_name))
            else:
                created_dir = None

            if save_full_calc:
                calc_dir   = created_dir
                save_dir   = created_dir
                save_files = None
            elif save_partial:
                calc_dir   = None
                save_dir   = created_dir
            else:
                calc_dir   = None
                save_dir   = None

            base_args = {
                "atoms":  row["atoms"],
                "coords": [list(c) for c in coords],
                "n_cores": self.n_cores,
                "scr": self.work_dir
            }

            if calc_dir is not None:
                base_args["calc_dir"] = calc_dir
            if save_dir is not None:
                base_args["save_dir"] = save_dir
            if save_files is not None:
                base_args["save_files"] = save_files

            inputs = {**base_args, **build_inputs(row)}
            
            # step 3: run engine, catch exceptions
            logger.info(f"[{prefix}] row {i} ({row['custom_name']})…")
            try:
                out = engine_fn(**inputs) or {}
            except Exception as e:
                if self.debug:
                    print(f"  → engine error: {e}")
                out = {"normal_termination": False}

            # step 4: enforce defaults
            out.setdefault("normal_termination", False)
            if out["normal_termination"]:
                out.setdefault("electronic_energy", np.nan)

            # step 5: pick only the keys we know how to handle
            allowed = {"normal_termination", "electronic_energy", "opt_coords", "vibs"}
            if "vibs" in out and "gibbs_energy" in out:
                allowed.add("gibbs_energy")

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
        constraint: bool = True,
        save_step = False,
        lowest: int | None = None,
    ) -> pd.DataFrame:
        """Embed multiple conformers with xTB and optionally optimize and/or compute frequencies.

        Args:
            df (pd.DataFrame): A DataFrame containing embedded conformers. Required columns:
                - 'coords_embedded': list of 3D coordinate tuples for each conformer.
                - 'atoms': list of atomic symbols.
                - 'constraint_atoms' (optional): list of atom indices to constrain during optimization.
            name (str): Base name for the xTB step, used to prefix result columns.
            options (dict, optional): xTB driver options, e.g. {'gfn': 2, 'opt': None}. Defaults to {'gfn': 0}.
            detailed_inp_str (str, optional): Additional xTB input block (cards) to include. Defaults to "".
            constraint (bool, optional): If True, applies predefined distance/angle constraints for TS steps. Defaults to False.
            save_step (bool, optional): If True, saves calculation directories for each conformer. Defaults to False.
            lowest (int or None, optional): If set, retains only the lowest-energy N conformers per ligand/rpos group. Defaults to None.

        Returns:
            pd.DataFrame: The input DataFrame augmented with columns for each xTB result:
                - '{name}-{method}-normal_termination' (bool)
                - '{name}-{method}-electronic_energy' (float)
                - '{name}-{method}-opt_coords' (list of coords) if optimization run
                - '{name}-{method}-vibs' (vibrational modes) if frequencies computed
                - '{name}-{method}-gibbs_energy' (float) if frequencies computed
        """        
        opts = options or {"gfn": 0}
        keys = list(opts)
        level = keys[0]
        # check for optimization flag among the remaining keys
        opt_flag = next((k for k in keys[1:] if k in ("opt", "ohess")), None)
        prefix = f"{name}-{level}" + (f"-{opt_flag}" if opt_flag else "")
        
        def build_xtb(row: pd.Series) -> dict:
            inp: dict[str, object] = {"options": opts}

            # Only add the user‐provided card if it's non‐empty
            base_str = detailed_inp_str.strip()
            if base_str:
                inp["detailed_input_str"] = base_str

            block = None

            if self.step_type.upper() == "TS1" and constraint:
                B, N, H, C = 0, 1, 4, 5
                atom = [x+1 for x in row["constraint_atoms"]]
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

            # Old TS2
            # if self.step_type.upper() == "TS2" and constraint:
            #     BCat, BPin, H, C = 0, 3, 4, 5
            #     atom = [x+1 for x in row["constraint_atoms"]]
            #     block = textwrap.dedent(f"""
            #     $constrain
            #       force constant=50
            #       distance: {atom[BCat]}, {atom[H]}, 1.335
            #       distance: {atom[BPin]}, {atom[H]}, 2.168
            #       distance: {atom[H]}, {atom[C]}, 2.424
            #       distance: {atom[BCat]}, {atom[C]}, 1.335
            #       distance: {atom[BPin]}, {atom[C]}, 1.956
            #       distance: {atom[BPin]}, {atom[H]}, 2.168
            #       distance: {atom[BCat]}, {atom[BPin]}, 2.063
            #       angle: {atom[BPin]}, {atom[C]}, {atom[BCat]}, 65.36
            #       angle: {atom[BPin]}, {atom[H]}, {atom[BCat]}, 67.39
            #     $end
            #     """).strip()

            if self.step_type.upper() == "TS2" and constraint:
                BCat10, N17, H40, H41, C46 = 0, 1, 4, 3, 5
                atom = [x+1 for x in row["constraint_atoms"]]
                block = textwrap.dedent(f"""
                $constrain
                  force constant=50
                  distance: {atom[BCat10]}, {atom[H41]}, 1.656
                  distance: {atom[N17]}, {atom[H40]}, 1.961
                  distance: {atom[BCat10]}, {atom[N17]}, 3.080
                  angle: {atom[BCat10]}, {atom[H41]}, {atom[N17]}, 86.58
                $end
                """).strip()

            if self.step_type.upper() == "TS3" and constraint:
                BCat10, H11, BPin22, H21, C = 0, 2, 3, 4, 5
                atom = [x+1 for x in row["constraint_atoms"]]
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

            if self.step_type.upper() == "TS4" and constraint:
                BCat11, H12, H13, BPin37, C = 0, 2, 3, 4, 5
                atom = [x+1 for x in row["constraint_atoms"]]
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

            if self.step_type.upper() == "INT3" and constraint:
                print("noob")
                BCat10, BPin42, H11, C = 0, 3, 4, 5
                atom = [x+1 for x in row["constraint_atoms"]]
                print(atom)
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

            if "detailed_input_str" in inp:
                inp["detailed_input_str"] += "\n\n" + block
            else:
                inp["detailed_input_str"] = block

            return inp

        return self._run_engine(df, self.xtb_fn, prefix, build_xtb, save_step, lowest)

    def orca(
        self,
        df: pd.DataFrame,
        name: str = "orca",
        options: dict | None = None,
        xtra_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
        save_files: list[str] | None = ["orca.out"],
        lowest: int | None = None,
        uma: str | None = None,
        uma_client_block: str | None = None,
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

        Returns:
            pd.DataFrame: The input DataFrame extended with ORCA output columns:
                - '{name}-{method}-normal_termination' (bool)
                - '{name}-{method}-electronic_energy' (float)
                - '{name}-{method}-opt_coords' (list of coords) if optimization run
                - '{name}-{method}-vibs' (vibrational modes) if frequencies computed
                - '{name}-{method}-gibbs_energy' (float) if frequencies computed
        """
        opts = options or {}
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
            inp = {
                "options": opts,
                "xtra_inp_str": xtra_inp_str.strip(),
                "memory": self.memory_gb,
                "n_cores": self.n_cores,
            }

            if "Freq" in opts:
                block = textwrap.dedent("""
                    %geom
                      Calc_Hess true
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and self.step_type.upper() == "TS1":
                atom = row["constraint_atoms"]
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

            if constraint and self.step_type.upper() == "TS2":
                atom = row["constraint_atoms"]
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

            if constraint and self.step_type.upper() == "TS3":
                atom = row["constraint_atoms"]
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

            if constraint and self.step_type.upper() == "TS4":
                atom = row["constraint_atoms"]
                BCat, H12, H13, BPin, C = atom[0], atom[2], atom[3], atom[4], atom[5]
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

            if constraint and self.step_type.upper() == "INT3":
                atom = row["constraint_atoms"]
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
            return self._run_engine(df, self.orca_fn, prefix, build_orca, save_step, lowest, save_files)
        
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