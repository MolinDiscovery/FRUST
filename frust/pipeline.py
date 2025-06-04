import pandas as pd
from pathlib import Path
from .config import Settings
from .io import dump_df
from .transformer_mols import transform_mols
from .transformer_ts import TSTransformer
from .embedder import generate_conformers
from .monitor import maybe_start_nuse
from typing import List

class BasePipeline:
    def __init__(self, settings: Settings):
        self.cfg = settings
        self.base_dir = self.cfg.dump_base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def maybe_dump(self, df: pd.DataFrame, step: str):
        if self.cfg.dump_each_step:
            dump_df(df, step, self.base_dir)

class TSPipeline(BasePipeline):
    def run(self, ligands: List[str]) -> pd.DataFrame:
        # 0) monitor
        maybe_start_nuse(self.cfg.live)

        # 1) transform → DataFrame
        raw = [transform_mols(smi, "", self.cfg.bonds_to_remove[0]) for smi in ligands]
        df0 = pd.DataFrame(raw).assign(step="transform")
        self.maybe_dump(df0, "step0_transform")

        # 2) TS UFF conformers
        ts_data = TSTransformer(
            ligand_smiles=ligands[0],
            ts_guess_struct=str(self.cfg.ts_guess_xyz),
            bonds_to_remove=self.cfg.bonds_to_remove,
            num_confs=self.cfg.n_ts_confs,
        )
        records = []
        for name, (mol, energies, idxs) in ts_data.items():
            for e, cid in energies:
                records.append({
                    "name": name, "conf_id": cid, "uff_E": e, "step": "ts_uff"
                })
        df1 = pd.DataFrame(records)
        self.maybe_dump(df1, "step1_ts_uff")

        # … 3) select lowest_UFF_E → xTB-FF → xTB-SP → cluster → DFT …
        # at each sub-step build a DataFrame and maybe_dump(…)

        # return the final DataFrame
        return df1