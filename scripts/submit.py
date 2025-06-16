#!/usr/bin/env python3
import os
import inspect
import pandas as pd
import submitit
from itertools import islice
import importlib

# ─── CONFIG ─────────────────────────────────────────────────────────────
PIPELINE_NAME  = "run_ts" # "run_ts" or "run_mols"
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
# ────────────────────────────────────────────────────────────────────────

def batched(iterable, n):
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch

# load pipeline
pipes_mod   = importlib.import_module("frust.pipes")
pipeline_fn = getattr(pipes_mod, PIPELINE_NAME)
sig         = inspect.signature(pipeline_fn)
accepts_ts  = 'ts_guess_xyz' in sig.parameters

# read smiles
df       = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

executor = (
    submitit.AutoExecutor(LOG_DIR)
    if USE_SLURM
    else submitit.LocalExecutor(LOG_DIR)
)
executor.update_parameters(
    slurm_partition="kemi1" if USE_SLURM else None,
    cpus_per_task=CPUS_PER_JOB,
    mem_gb=MEM_GB,
    timeout_min=TIMEOUT_MIN,
)

futures = []
for batch in batched(smi_list, BATCH_SIZE):
    tag = f"{PIPELINE_NAME}_batch_{hash(tuple(batch)) & 0xffffffff:x}"
    if USE_SLURM:
        executor.update(slurm_job_name=tag)

    common_kwargs = dict(
        ligand_smiles_list=batch,
        n_confs=N_CONFS,
        n_cores=CPUS_PER_JOB,
        debug=DEBUG,
        out_dir=OUT_DIR,
        output_parquet=os.path.join(OUT_DIR, f"{tag}.parquet"),
        save_output_dir=SAVE_OUT_DIRS,
    )

    if accepts_ts:
        fut = executor.submit(pipeline_fn, ts_guess_xyz=TS_XYZ, **common_kwargs)
    else:
        fut = executor.submit(pipeline_fn, **common_kwargs)

    futures.append(fut)

if USE_SLURM:
    print("Submitted Slurm job IDs:", [f.job_id for f in futures])
else:
    print(f"Dispatched {len(futures)} local futures …")
    for i, f in enumerate(futures, 1):
        f.result()
        print(f"Completed {i}/{len(futures)}", end="\r")
    print("\nAll done.")