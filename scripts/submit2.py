# scripts/submit2.py
import os
import inspect
import pandas as pd
import submitit
from itertools import islice
import importlib

# ─── CONFIG ─────────────────────────────────────────────────────────────
PIPELINE_NAME  = "run_ts_per_rpos"  # "run_ts_per_rpos", "run_ts_per_lig", "run_mols"
PRODUCTION     = True
USE_SLURM      = True
DEBUG          = False
BATCH_SIZE     = 1
CSV_PATH       = "../datasets/font_smiles.csv"
OUT_DIR        = "results_ts4_TMP_font_1"
LOG_DIR        = "logs/ts4_TMP_font_1"
SAVE_OUT_DIRS  = False
CPUS_PER_JOB   = 7
MEM_GB         = 31
TIMEOUT_MIN    = 14400
N_CONFS        = None if PRODUCTION else 1
DFT            = True
# ─── TS SPECIFIC ─────────────────────────────────────────────────────────
TS_XYZ         = "../structures/ts4_TMP.xyz"
# ─── MOL SPECIFIC ────────────────────────────────────────────────────────
SELECT_MOLS    = ["HH"] # "all", "uniques", "generics", or specific names

def batched(iterable, n):
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch
 
# 1) load the requested pipeline
pipes_mod   = importlib.import_module("frust.pipes")
pipeline_fn = getattr(pipes_mod, PIPELINE_NAME)
sig         = inspect.signature(pipeline_fn)

# 2) read & dedupe SMILES
df       = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))

# determine job inputs
if PIPELINE_NAME == "run_ts_per_rpos":
    from frust.pipes import create_ts_per_rpos
    job_inputs = create_ts_per_rpos(smi_list, TS_XYZ)
elif PIPELINE_NAME in {"run_ts_per_lig", "run_mols", "run_small_test"}:
    job_inputs = smi_list
else:
    raise ValueError(f"Unknown pipeline {PIPELINE_NAME!r}")

# 3) make sure output dirs exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 4) pick executor
executor = submitit.AutoExecutor(LOG_DIR) if USE_SLURM else submitit.LocalExecutor(LOG_DIR)
executor.update_parameters(
    slurm_partition="kemi1" if USE_SLURM else None,
    cpus_per_task=CPUS_PER_JOB,
    mem_gb=MEM_GB,
    timeout_min=TIMEOUT_MIN,
)

# 5) dispatch batches
futures = []
if PIPELINE_NAME == "run_ts_per_rpos":
    for ts_struct in job_inputs:
        tag = f"{PIPELINE_NAME}_{list(ts_struct.keys())[0]}"
        if USE_SLURM:
            executor.update_parameters(slurm_job_name=tag)
        all_kwargs = {
            "ligand_smiles_list": None,
            "ts_guess_xyz":       TS_XYZ,
            "ts_struct":          None,
            "n_confs":            N_CONFS,
            "n_cores":            CPUS_PER_JOB,
            "mem_gb":             MEM_GB,
            "debug":              DEBUG,
            "out_dir":            OUT_DIR,
            "output_parquet":     os.path.join(OUT_DIR, f"{tag}.parquet"),
            "save_output_dir":    SAVE_OUT_DIRS,
            "DFT":                DFT,
            "select_mols":        SELECT_MOLS,
        }
        merged = all_kwargs.copy()
        merged.update({"ts_struct": ts_struct})
        call_kwargs = {k: v for k, v in merged.items() if k in sig.parameters}
        fut = executor.submit(pipeline_fn, **call_kwargs)
        futures.append(fut)
else:
    for batch in batched(job_inputs, BATCH_SIZE):
        tag = f"{PIPELINE_NAME}_batch_{hash(tuple(map(str,batch))) & 0xffffffff:x}"
        if USE_SLURM:
            executor.update_parameters(slurm_job_name=tag)

        all_kwargs = {
            "ligand_smiles_list": None,
            "ts_guess_xyz":       TS_XYZ,
            "ts_struct":          None,
            "n_confs":            N_CONFS,
            "mem_gb":             MEM_GB,            
            "n_cores":            CPUS_PER_JOB,
            "debug":              DEBUG,
            "out_dir":            OUT_DIR,
            "output_parquet":     os.path.join(OUT_DIR, f"{tag}.parquet"),
            "save_output_dir":    SAVE_OUT_DIRS,
            "DFT":                DFT,
            "select_mols":        SELECT_MOLS,
        }
        specific = {"ligand_smiles_list": batch}
        merged = all_kwargs.copy()
        merged.update(specific)
        call_kwargs = {k: v for k, v in merged.items() if k in sig.parameters}
        fut = executor.submit(pipeline_fn, **call_kwargs)
        futures.append(fut)

# 6) report & optionally wait
if USE_SLURM:
    print("Submitted Slurm job IDs:", [f.job_id for f in futures])
else:
    print(f"Dispatched {len(futures)} local futures …")
    for i, f in enumerate(futures, 1):
        f.result()
        print(f"Completed {i}/{len(futures)}", end="\r")
    print("\nAll done.")
