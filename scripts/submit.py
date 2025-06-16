#!/usr/bin/env python3
import os
import inspect
import pandas as pd
import submitit
from itertools import islice
import importlib

# ─── CONFIG ─────────────────────────────────────────────────────────────
PIPELINE_NAME  = "run_ts"  # or "run_mols"
PRODUCTION     = False
USE_SLURM      = False
DEBUG          = False
BATCH_SIZE     = 8
CSV_PATH       = "../datasets/ir_borylation.csv" if PRODUCTION else "../datasets/ir_borylation_test.csv"
TS_XYZ         = "../structures/ts1_guess.xyz"
OUT_DIR        = "results_test"
LOG_DIR        = "logs/test"
SAVE_OUT_DIRS  = False
CPUS_PER_JOB   = 4
MEM_GB         = 8
TIMEOUT_MIN    = 7200
N_CONFS        = None if PRODUCTION else 1
DFT            = True
# ────────────────────────────────────────────────────────────────────────

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

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 3) pick executor
executor = submitit.AutoExecutor(LOG_DIR) if USE_SLURM else submitit.LocalExecutor(LOG_DIR)
executor.update_parameters(
    slurm_partition="kemi1" if USE_SLURM else None,
    cpus_per_task=CPUS_PER_JOB,
    mem_gb=MEM_GB,
    timeout_min=TIMEOUT_MIN,
)

# 4) dispatch batches
futures = []
for batch in batched(smi_list, BATCH_SIZE):
    tag = f"{PIPELINE_NAME}_batch_{hash(tuple(batch)) & 0xffffffff:x}"
    if USE_SLURM:
        executor.update(slurm_job_name=tag)

    # master dict of every possible argument
    all_kwargs = {
        "ligand_smiles_list": batch,
        "ts_guess_xyz":       TS_XYZ,
        "n_confs":            N_CONFS,
        "n_cores":            CPUS_PER_JOB,
        "debug":              DEBUG,
        "out_dir":            OUT_DIR,
        "output_parquet":     os.path.join(OUT_DIR, f"{tag}.parquet"),
        "save_output_dir":    SAVE_OUT_DIRS,
        "DFT":                DFT,
    }

    # filter to only those the pipeline actually declares
    call_kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}

    fut = executor.submit(pipeline_fn, **call_kwargs)
    futures.append(fut)

# 5) report & (optionally) wait
if USE_SLURM:
    print("Submitted Slurm job IDs:", [f.job_id for f in futures])
else:
    print(f"Dispatched {len(futures)} local futures …")
    for i, f in enumerate(futures, 1):
        f.result()
        print(f"Completed {i}/{len(futures)}", end="\r")
    print("\nAll done.")