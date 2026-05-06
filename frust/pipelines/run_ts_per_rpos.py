# frust/pipes/run_ts_per_rpos.py
from pathlib import Path
from frust.stepper import Stepper
from frust.embedder import embed_ts, embed_mols
from frust.transformers import transformer_mols
from rdkit.Chem.rdchem import Mol
import os
import inspect
import pandas as pd
import numpy as np
from frust.schema import energy_columns, normalize_dataframe

# ─── SHARED SETTINGS (inherit across steps) ─────────────────────────────
FUNCTIONAL = "wB97X-D3" # "wB97X-D3"
BASISSET = "6-31G**" # "6-31G**"
BASISSET_SOLV = "6-31+G**"  # for solvent SP


def _resolve_theory(
    *,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
) -> tuple[str, str, str]:
    """Resolve workflow theory settings for the TS preset pipeline.

    Parameters
    ----------
    functional : str or None, optional
        ORCA functional override.
    basisset : str or None, optional
        ORCA gas-phase basis set override.
    basisset_solv : str or None, optional
        ORCA solvent single-point basis set override.

    Returns
    -------
    tuple[str, str, str]
        Functional, gas-phase basis set, and solvent basis set used for the
        current stage call.
    """
    return (
        functional or FUNCTIONAL,
        basisset or BASISSET,
        basisset_solv or BASISSET_SOLV,
    )

try:
    if "SLURM_JOB_ID" in os.environ:
        from nuse import start_monitoring
        start_monitoring(filter_cgroup=True)
except ImportError:
    pass

def _best_rows(df):
    df = normalize_dataframe(df)
    last_energy = energy_columns(df)[-1]
    group_cols = ["substrate_name", "structure_type", "molecule_role", "rpos"]
    return (df.sort_values(group_cols + [last_energy])
              .groupby(group_cols, dropna=False).head(1))


def run_init(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    save_dir: str | None = None,
    work_dir: str | None = None,
    save_output_dir: bool = True,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
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

    step = Stepper(
    step_type=ts_type,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=save_dir,
    save_output_dir=save_output_dir,
    work_dir=work_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, constraint=True, n_cores=2)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2}, n_cores=2)
    df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n, n_cores=2)

    current_functional, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
    )

    df = step.orca(df, name="DFT-pre-SP", options={
        current_functional: None,
        current_basisset  : None,
        "TightSCF"  : None,
        "SP"        : None,
        "NoSym"     : None,
    })

    df = step.orca(df, name="DFT-pre-Opt", options={
        current_functional: None,
        current_basisset  : None,
        "TightSCF" : None,
        "SlowConv" : None,
        "Opt"      : None,
        "NoSym"    : None,
    }, constraint=True, lowest=1)
    
    fn_name = inspect.currentframe().f_code.co_name
    parquet_name = fn_name.split("_")[1]
    df.to_parquet(f"{save_dir}/{parquet_name}.parquet")
    
    return df


def run_hess(
    parquet_path: str,
    *,
    n_cores: int = 2,
    mem_gb: int = 32,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
):
    
    df = normalize_dataframe(pd.read_parquet(f"{save_dir}/{parquet_path}"))
    if df.empty:
        return df

    df = _best_rows(df)

    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    current_functional, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
    )

    df = step.orca(df, name="Hess", options={
        current_functional: None,
        current_basisset: None,
        "TightSCF": None,
        "Freq": None,
        "NoSym": None,
    }, read_files=["input.hess"])

    stem = parquet_path.rsplit('.', 1)[0]
    out_parquet = stem + ".hess.parquet"
    df.to_parquet(f"{save_dir}/{out_parquet}")

    return df


def run_OptTS(
    parquet_path: str,
    *,
    n_cores: int = 8,
    mem_gb: int = 40,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
):
    df = normalize_dataframe(pd.read_parquet(f"{save_dir}/{parquet_path}"))

    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    # Read previously computed Hessian (*.hess from ORCA)
    current_functional, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
    )

    df = step.orca(df, name="OptTS", options={
        current_functional: None,
        current_basisset: None,
        "TightSCF": None,
        "SlowConv": None,
        "OptTS": None,
        "NoSym": None,
    }, use_last_hess=True)

    stem = parquet_path.rsplit('.', 1)[0]
    out_parquet = stem + ".optts.parquet"
    df.to_parquet(f"{save_dir}/{out_parquet}")

    return df


def run_freq(
    parquet_path: str,
    *,
    n_cores: int = 8,
    mem_gb: int = 40,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,        
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
):
    df = normalize_dataframe(pd.read_parquet(f"{save_dir}/{parquet_path}"))

    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    current_functional, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
    )

    df = step.orca(df, name="Freq", options={
        current_functional: None,
        current_basisset: None,
        "TightSCF": None,
        "SlowConv": None,
        "Freq": None,
        "NoSym": None,
    })

    stem = parquet_path.rsplit('.', 1)[0]
    out_parquet = stem + ".freq.parquet"
    df.to_parquet(f"{save_dir}/{out_parquet}")

    return df


def run_solv(
    parquet_path: str,
    *,
    n_cores: int = 8,
    mem_gb: int = 40,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,        
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
):
    df = normalize_dataframe(pd.read_parquet(f"{save_dir}/{parquet_path}"))

    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    current_functional, _, current_basisset_solv = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
    )

    df = step.orca(df, name="DFT-solv", options={
        current_functional: None,
        current_basisset_solv: None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")

    stem = parquet_path.rsplit('.', 1)[0]
    out_parquet = stem + ".solv.parquet"
    df.to_parquet(f"{save_dir}/{out_parquet}")

    return df


def run_cleanup(save_dir):
    print("[INFO]: Cleanup initiated...")

    """Deletes residual .parquet files"""

    parquet_files = list(Path(save_dir).glob("*.parquet"))
    if not parquet_files:
        print("No parquet files found.")
        return

    depths = {f: len(f.suffixes) for f in parquet_files}
    max_depth = max(depths.values())

    kept = []
    for f, d in depths.items():
        if d < max_depth:
            print(f"[INFO]: Removing residual parquet file {f}")
            try:
                f.unlink()
            except Exception as e:
                print(f"[WARN]: Could not remove {f}: {e}")
        else:
            kept.append(f)

    for f in kept:
        try:
            df = normalize_dataframe(pd.read_parquet(f))
        except Exception as e:
            print(f"[WARN]: Could not read {f}: {e}")
            continue

        hess_cols = []
        for col in df.columns:
            if col.endswith(".hess"):
                nonnull = next(
                    (v for v in df[col].tolist()
                     if v is not None and not (isinstance(v, float) and np.isnan(v))),
                    None
                )
                if nonnull is None or isinstance(nonnull, (bytes, bytearray, str)):
                    hess_cols.append(col)

        if hess_cols:
            print(f"[INFO]: Dropping .hess columns from {f.name}: {hess_cols}")
            df = df.drop(columns=hess_cols)
            try:
                df.to_parquet(f)
            except Exception as e:
                print(f"[WARN]: Failed writing cleaned Parquet {f}: {e}")
        else:
            print(f"[INFO]: No .hess columns in {f.name}")

    print("[INFO]: Cleanup done!")


# ––– Test stuff ────────────────────────────────────────────────────────–
# run_dir = "run_ts_per_rpos"                                              
# from frust.utils.mols import create_ts_per_rpos                          
# job_inputs = create_ts_per_rpos(["CN1C=CC=C1"], "../structures/ts1.xyz") 
# job_inputs = job_inputs[0]                                               
# ───────────────────────────────────────────────────────────────────────–

# Example local run
# _ = run_init(job_inputs, n_confs=1, save_dir=test_dir)
# _ = run_hess("init.parquet", save_dir=test_dir, n_cores=10)
# _ = run_OptTS("init.hess.parquet", save_dir=test_dir, n_cores=10)
# _ = run_freq("init.hess.optts.parquet", save_dir=test_dir, n_cores=10)
# _ = run_solv("init.hess.optts.freq.parquet", save_dir=test_dir, n_cores=10)
