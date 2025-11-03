# frust/pipes/run_ts_per_rpos.py
from pathlib import Path
from frust.stepper import Stepper
from frust.embedder import embed_ts, embed_mols
from frust.transformers import transformer_mols
from rdkit.Chem.rdchem import Mol
import os
import pandas as pd

# ─── SHARED SETTINGS (inherit across steps) ─────────────────────────────
FUNCTIONAL = "wB97X-D3"
BASISSET = "6-31G**"
BASISSET_SOLV = "6-31+G**"  # for solvent SP

try:
    if "SLURM_JOB_ID" in os.environ:
        from nuse import start_monitoring
        start_monitoring(filter_cgroup=True)
except ImportError:
    pass

def _best_rows(df):
    last_energy = [c for c in df.columns if c.endswith("_energy")][-1]
    return (df.sort_values(["ligand_name", "rpos", last_energy])
              .groupby(["ligand_name", "rpos"]).head(1))


def run_init(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    work_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
):
    import re
    pattern = re.compile(
    r'^(?:(?P<prefix>(?:TS|INT)\d*|Mols)\()?'
    r'(?P<ligand>.+?)_rpos\('        
    r'(?P<rpos>\d+)\)\)?$'           
    )

    name = list(ts_struct.keys())[0]
    m = pattern.match(name)
    ts_type = m.group("prefix")
    
    embedded = embed_ts(ts_struct, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    ligand_smiles = list(ts_struct.values())[0][2]

    step = Stepper(
    ligand_smiles,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_calc_dirs=True,
    save_output_dir=save_output_dir,
    work_dir=work_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, options={"gfn": 2})
    df = step.xtb(df, options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    df = step.orca(df, name="DFT-pre-SP", options={
        FUNCTIONAL  : None,
        BASISSET    : None,
        "TightSCF"  : None,
        "SP"        : None,
        "NoSym"     : None,
    })

    last_energy = [c for c in df.columns if c.endswith("_energy")][-1]
    df = (df.sort_values(["ligand_name", "rpos", last_energy]
                        ).groupby(["ligand_name", "rpos"]).head(1))
    
    if output_parquet:
        df.to_parquet(output_parquet)
    
    return df


def run_hess(
    parquet_path: str,
    *,
    n_cores: int = 2,
    mem_gb: int = 32,
    debug: bool = False,
    out_dir: str | None = None,
    work_dir: str | None = None,
):
    """Compute a (numerical) Hessian to seed OptTS. Low RAM vs analytic Freq."""

    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df

    df = _best_rows(df)

    ligand_smiles = list(dict.fromkeys(df["smiles"].tolist()))
    step = Stepper(
        ligand_smiles,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_calc_dirs=True,
        save_output_dir=True,
        work_dir=work_dir,
    )

    df = step.orca(df, name="Hess", options={
        FUNCTIONAL: None,
        BASISSET: None,
        "TightSCF": None,
        "NumFreq": None,
        "NoSym": None,
    })

    stem = os.path.splitext(parquet_path)[0]
    out_parquet = stem + ".hess.parquet"
    df.to_parquet(out_parquet)

    return df


def run_OptTS():
    pass

def run_freq():
    pass

def run_solv_SP():
    pass