#!/usr/bin/env python3
import os
import pandas as pd
import submitit
from frust.pipes import run_ts1

# ─── CONFIG ─────────────────────────────────────────────────────────────
PRODUCTION      = False
USE_SLURM       = True
DEBUG           = False
CSV_PATH        = "../datasets/ir_borylation.csv" if PRODUCTION else "../datasets/ir_borylation_test.csv"
TS_XYZ          = "../structures/ts1_guess.xyz"
OUT_DIR         = "results"
LOG_DIR         = "logs/ts"
SAVE_OUT_DIRS   = True
CPUS_PER_JOB    = 8
MEM_GB          = 16
TIMEOUT_MIN     = 7200 # one work week
N_CONFS         = None if PRODUCTION else 2
# ─────────────────────────────────────────────────────────────────────────

# 1) Read and dedupe SMILES
df       = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 2) Pick executor
executor = (
    submitit.AutoExecutor(folder=LOG_DIR)
    if USE_SLURM
    else submitit.LocalExecutor(folder=LOG_DIR)
)
executor.update_parameters(
    slurm_partition = "kemi1" if USE_SLURM else None,
    cpus_per_task   = CPUS_PER_JOB,
    mem_gb          = MEM_GB,
    timeout_min     = TIMEOUT_MIN,
)

# 3) Submit one future per ligand
futures = []
for smi in smi_list:
    fut = executor.submit(
        run_ts1,
        [smi],
        ts_guess_xyz    = TS_XYZ,
        n_confs         = N_CONFS,
        n_cores         = CPUS_PER_JOB,
        debug           = DEBUG,
        out_dir         = OUT_DIR,
        output_parquet  = os.path.join(OUT_DIR, f"{smi}.parquet"),
        save_output_dir = SAVE_OUT_DIRS,
    )
    futures.append(fut)

# 4) Report
if USE_SLURM:
    job_ids = [f.job_id for f in futures]
    print("Submitted Slurm job IDs:", job_ids)
else:
    print(f"Dispatched {len(futures)} local futures.")
    # Optionally: futures[i].cancel() or futures[i].result()

# (Optional) block for local until done
if not USE_SLURM:
    for i, f in enumerate(futures, 1):
        f.result()  # wait
        print(f"Completed {i}/{len(futures)}", end="\r")
    print("\nAll done.")