# frustactivation/stepper.py
from pathlib import Path
from .optimizers.uff import optimize_mol_uff
from .dirs import prepare_base_dir, make_step_dir
import pandas as pd
import numpy as np
from rdkit import Chem
import re
import textwrap

from tooltoad.xtb import xtb_calculate

# indices of atoms in the constraint_atoms
B = 0
N = 1
H = 4
C = 5

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
        **kwargs
    ):
        self.debug           = debug
        self.live            = live
        self.dump_each_step  = dump_each_step
        self.n_cores         = n_cores
        self.memory_gb       = memory_gb

        # call prepare_base_dir *with* your ligands list,
        # and *use* its return value as your base_dir
        self.base_dir = prepare_base_dir(output_base, ligands_smiles, job_id)

    @staticmethod
    def _last_coord_col(df: pd.DataFrame) -> str:
        """
        Return the name of the last column in `df` whose name contains 'coords'.
        This will catch columns like 'coords_embedded', 'xtb-gfn-coords', etc.
        """
        coord_cols = [c for c in df.columns if "coords" in c]
        if not coord_cols:
            raise ValueError("No column containing 'coords' found in DataFrame")
        return coord_cols[-1]

    @staticmethod
    def build_initial_df(embedded_dict: dict):
        """Create a tidy DataFrame from `embed_ts` output (with or without UFF).

        The helper detects whether each value-tuple contains an “energies”
        entry:

            (mol, cids, keep_idxs, smi)                     # optimize=False
            (mol, cids, keep_idxs, smi, energies)           # optimize=True

        If `energies` is present it adds an `energy_uff` column, otherwise that
        column is filled with `None`.

        Returns
        -------
        pd.DataFrame
            One row per conformer with columns

            ─ custom_name           TS name used as a key
            ─ ligand_name           parsed ligand fragment name
            ─ rpos                  reactive ring position (int)
            ─ atom_indices_to_keep  list[int]  (constraints)
            ─ conf_id_embedded      conformer ID (int)
            ─ smiles                ligand SMILES
            ─ atoms                 list[str]  (element symbols)
            ─ coords_embedded       list[tuple]  (Å)
            ─ energy_uff            float | None
        """
        rows = []
        pattern = re.compile(r"^TS\((?P<ligand>.+?)_rpos\((?P<rpos>\d+)\)\)$")

        for ts_name, value in embedded_dict.items():
            # ----- unpack, allowing 4-tuple or 5-tuple -----
            if len(value) == 4:               # no UFF energies
                mol, cids, keep_idxs, smi = value
                energies = []                 # empty list
            elif len(value) == 5:             # UFF energies included
                mol, cids, keep_idxs, smi, energies = value
            else:
                raise ValueError(
                    f"Unexpected tuple length ({len(value)}) for key {ts_name}"
                )

            # ----- parse ligand / rpos from key -----
            m = pattern.match(ts_name)
            if not m:
                raise ValueError(f"Bad TS name format: {ts_name}")
            ligand_name = m.group("ligand")
            rpos = int(m.group("rpos"))

            # ----- atom symbols -----
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]

            # map cid -> energy for fast lookup (may be empty)
            e_dict = {cid: e for e, cid in energies} if energies else {}

            # ----- one row per conformer -----
            for cid in list(cids):
                conf = mol.GetConformer(cid)
                coords = [
                    tuple(conf.GetAtomPosition(i))
                    for i in range(mol.GetNumAtoms())
                ]
                rows.append(
                    {
                        "custom_name": ts_name,
                        "ligand_name": ligand_name,
                        "rpos": rpos,
                        "constraint_atoms": keep_idxs,
                        "cid": cid,
                        "smiles": smi,
                        "atoms": atoms,
                        "coords_embedded": coords,
                        "energy_uff": e_dict.get(cid),  # None if not optimized
                    }
                )
        
        return pd.DataFrame(rows)


    def xtb(self, df: pd.DataFrame,
            name: str = "xtb",
            options: dict | None = None,
            detailed_inp_str: str = "",
            constraint: bool = False,
        ):

        options = options or {"gfn": 0}

        keys = list(options.keys())
        if len(keys) > 1:
            calc_name = f"{name}-{keys[0]}-{keys[1]}"
        else:
            calc_name = f"{name}-{keys[0]}"

        df_out = df.copy()
        last_coords_col = self._last_coord_col(df_out)

        # Prepare result‐lists
        converged_list = []
        EE_list        = []
        coords_list    = []     

        for index, row in df_out.iterrows():
            row_name = row["custom_name"]
            atoms = row['atoms']
            coords = row[last_coords_col]
            coords = [list(c) for c in coords]
            calc_dir = make_step_dir(self.base_dir, row_name)
            full_input_str = detailed_inp_str.strip()

            if constraint:
                atom = row["constraint_atoms"]
                constraint_block = textwrap.dedent(f"""
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

                full_input_str = f"{full_input_str}\n\n{constraint_block}"

            try:
                res = xtb_calculate(
                    atoms=atoms,
                    coords=coords,
                    options=options,
                    detailed_input_str=full_input_str,
                    n_cores=self.n_cores,
                    calc_dir=str(calc_dir),
                )
                ok = res.get("normal_termination", False)
                converged_list.append(ok)
                if ok:
                    EE = res.get("electronic_energy", np.nan)
                    optc = res.get("opt_coords", None)
                else:
                    EE, optc = np.nan, None
            except Exception:
                converged_list.append(False)
                EE, optc = np.nan, None

            EE_list.append(EE)
            coords_list.append(optc)

        df_out[f"{calc_name}-converged"] = converged_list
        df_out[f"{calc_name}-EE"]        = EE_list
        df_out[f"{calc_name}-coords"]    = coords_list     
            
        return df_out

    def orca(self, df):
        pass
