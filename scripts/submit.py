#!/usr/bin/env python3
import os
import pandas as pd
import submitit
from itertools import islice
from frust.pipes import run_ts1

# ─── CONFIG ─────────────────────────────────────────────────────────────
PRODUCTION      = True
USE_SLURM       = True
DEBUG           = False
BATCH_SIZE      = 4          # ← ligands per task / node
CSV_PATH        = "../datasets/ir_borylation.csv" if PRODUCTION else "../datasets/ir_borylation_test.csv"
# CSV_PATH        = "../datasets/font_smiles.csv"
TS_XYZ          = "../structures/ts1_guess.xyz"
OUT_DIR         = "results"
LOG_DIR         = "logs/ts"
SAVE_OUT_DIRS   = True
CPUS_PER_JOB    = 8
MEM_GB          = 16
TIMEOUT_MIN     = 7200       # one work week
N_CONFS         = None # None = use default rule
# ─────────────────────────────────────────────────────────────────────────

def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

# 1) Read and dedupe SMILES
df        = pd.read_csv(CSV_PATH)
smi_list  = list(dict.fromkeys(df["smiles"]))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 2) Pick executor
executor = submitit.AutoExecutor(LOG_DIR) if USE_SLURM else submitit.LocalExecutor(LOG_DIR)
executor.update_parameters(
    slurm_partition = "kemi1" if USE_SLURM else None,
    cpus_per_task   = CPUS_PER_JOB,
    mem_gb          = MEM_GB,
    timeout_min     = TIMEOUT_MIN,
)

# 3) Submit one future per batch
futures = []
for batch in batched(smi_list, BATCH_SIZE):
    out_name = f"batch_{hash(tuple(batch)) & 0xffffffff:x}.parquet"
    executor.update(slurm_job_name=out_name)
    fut = executor.submit(
        run_ts1,
        batch,                      # list[str]  ← run_ts1 already expects this
        ts_guess_xyz    = TS_XYZ,
        n_confs         = N_CONFS,
        n_cores         = CPUS_PER_JOB,
        debug           = DEBUG,
        out_dir         = OUT_DIR,
        output_parquet  = os.path.join(OUT_DIR, out_name),
        save_output_dir = SAVE_OUT_DIRS,
    )
    futures.append(fut)

# 4) Report / wait
if USE_SLURM:
    print("Submitted Slurm job IDs:", [f.job_id for f in futures])
else:
    print(f"Dispatched {len(futures)} local futures …")
    for i, f in enumerate(futures, 1):
        f.result()
        print(f"Completed {i}/{len(futures)}", end="\r")
    print("\nAll done.")