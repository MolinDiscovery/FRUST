# frustactivation/stepper.py
import logging
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
import re
import textwrap
from typing import Callable
import numpy as np
import pandas as pd
from pandas import Series

from frust.utils.dirs import make_step_dir, prepare_base_dir
from frust.utils.slurm import detect_job_id

# indices in the “constraint_atoms” list
B, N, H, C = 0, 1, 4, 5

class Stepper:
    def __init__(
        self,
        ligands_smiles: list[str],
        output_base: Path | str | None = None,
        job_id: int | None = None,
        debug: bool = False,
        live: bool = False,
        dump_each_step: bool = False,
        n_cores: int = 8,
        memory_gb: float = 20.0,
        save_calc_dirs: bool = True,
        save_output_dir: bool = True,
        **kwargs
    ):
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

    @staticmethod
    def _last_coord_col(df: pd.DataFrame) -> str:
        """Return the name of the last column containing 'coords'."""
        cols = [c for c in df.columns if "coords" in c]
        if not cols:
            raise ValueError("No column containing 'coords' found")
        return cols[-1]

    @staticmethod
    def build_initial_df(embedded_dict: dict) -> pd.DataFrame:
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
        pattern = re.compile(r'^(?:TS\()?(?P<ligand>.+?)_rpos\((?P<rpos>\d+)\)\)?$')

        ligand_mol_block = None

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
                ligand_name = m.group("ligand")
                rpos = int(m.group("rpos"))
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
    ) -> pd.DataFrame:
        """
        Generic runner for xTB or ORCA or any other engine.

        For each row:
          - Build a calc_dir = base_dir/prefix/row_name
          - Call engine_fn(atoms=…, coords=…, n_cores=…, calc_dir=…, **build_inputs(row))
          - Ensure we collect exactly one value per “kept” key (normal_termination, electronic_energy, opt_coords, vibs, gibbs_energy).
          - If a key is missing, insert a default so that the final column has the same length
            as df.
        """
        df_out     = df.copy()
        coord_col  = self._last_coord_col(df_out)
        all_row_data: list[dict[str, object]] = []

        for i, row in df_out.iterrows():
            logger.info(f"[{prefix}] row {i} ({row['custom_name']})…")

            # Step 1: create a folder base_dir/prefix/TS(...)
            if self.save_calc_dirs and self.save_output or save_step:
                engine_base = self.base_dir / prefix
                engine_base.mkdir(parents=True, exist_ok=True)
                save_dir_name = row["custom_name"] + "_" + str(row["cid"])
                calc_dir = str(make_step_dir(engine_base, save_dir_name))
            else:
                calc_dir = None

            base_args = {
                "atoms":   row["atoms"],
                "coords":  [list(c) for c in row[coord_col]],
                "n_cores": self.n_cores,
                "calc_dir": calc_dir,
            }

            inputs = {**base_args, **build_inputs(row)}

            try:
                out = engine_fn(**inputs) or {}
            except Exception as e:
                if self.debug:
                    print(f"  → engine error: {e}")
                out = {"normal_termination": False}

            out.setdefault("normal_termination", False)
            if out["normal_termination"]:
                out.setdefault("electronic_energy", np.nan)


            allowed_keys = {"normal_termination", "electronic_energy", "opt_coords", "vibs"}
            if "vibs" in out and "gibbs_energy" in out:
                allowed_keys.add("gibbs_energy")
                
            row_data: dict[str, object] = {}
            for key in allowed_keys:
                if key in out:
                    col_name = f"{prefix}-{key}"
                    row_data[col_name] = out[key]

            all_row_data.append(row_data)

        all_cols = set().union(*(rd.keys() for rd in all_row_data))

        final_cols = sorted(all_cols)

        column_arrays: dict[str, list] = {col: [] for col in final_cols}
        for row_data in all_row_data:
            for col in final_cols:
                if col in row_data:
                    column_arrays[col].append(row_data[col])
                else:
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
        constraint: bool = False,
        save_step = False,
    ) -> pd.DataFrame:
        
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

            if constraint:
                atom = [x+1 for x in row["constraint_atoms"]]
                block = textwrap.dedent(f"""
                $constrain
                  force constant=10
                  distance: {atom[B]}, {atom[H]}, 2.07696
                  distance: {atom[N]}, {atom[H]}, 1.5127
                  distance: {atom[H]}, {atom[C]}, 1.29095
                  distance: {atom[B]}, {atom[C]}, 1.68461
                  distance: {atom[B]}, {atom[N]}, 3.06223
                  angle: {atom[N]}, {atom[H]}, {atom[C]}, 170.1342
                  angle: {atom[H]}, {atom[C]}, {atom[B]}, 87.4870
                $end
                """).strip()

                # Append constraint block if there was already some input,
                # or assign it fresh if none existed.
                if "detailed_input_str" in inp:
                    inp["detailed_input_str"] += "\n\n" + block
                else:
                    inp["detailed_input_str"] = block
            return inp

        return self._run_engine(df, self.xtb_fn, prefix, build_xtb, save_step)


    def orca(
        self,
        df: pd.DataFrame,
        name: str = "orca",
        options: dict | None = None,
        xtra_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
    ) -> pd.DataFrame:
        opts = options or {}
        keys = list(opts)
        if len(keys) < 1:
            raise ValueError("`options` must include at least one ORCA method key")

        # prefix = "orca-FUNC-BASIS[-OptTS or Freq or NoSym]" 
        if len(keys) == 1:
            prefix = f"{name}-{keys[0]}"
        else:
            func, basis = keys[0], keys[1]
            opt_flag     = next((k for k in keys[2:] if k in ("OptTS", "Freq", "NoSym")), None)
            prefix       = f"{name}-{func}-{basis}" + (f"-{opt_flag}" if opt_flag else "")

        def build_orca(row: Series) -> dict:
            inp = {
                "options":      opts,
                "xtra_inp_str": xtra_inp_str.strip(),
                "memory":       self.memory_gb,
                "n_cores":      self.n_cores
            }
            if "Freq" in opts:
                block = textwrap.dedent("""
                %geom
                  Calc_Hess true
                end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block
            return inp

        return self._run_engine(df, self.orca_fn, prefix, build_orca, save_step)