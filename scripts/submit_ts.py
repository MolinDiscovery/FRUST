#!/usr/bin/env python3
import os
import pandas as pd
import submitit
from frust.pipes import run_ts1

# ─── CONFIGURE HERE ─────────────────────────────────────────────────────────────
USE_SLURM       = False            # True → submit to Slurm; False → run locally
DEBUG           = True
CSV_PATH        = "../datasets/ir_borylation_test.csv"
TS_XYZ          = "../structures/ts1_guess.xyz"
OUT_DIR         = "results"        # per-SMI .parquet will go here
LOG_DIR         = "logs/ts"        # submitit’s log folder
SAVE_OUT_DIRS   = False
CPUS_PER_JOB    = 8
MEM_GB          = 16
TIMEOUT_MIN     = 60
N_CONFS         = 2
# ────────────────────────────────────────────────────────────────────────────────

# read and dedupe SMILES
df = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# pick executor type
if USE_SLURM:
    executor = submitit.AutoExecutor(folder=LOG_DIR)
else:
    executor = submitit.LocalExecutor(folder=LOG_DIR)

executor.update_parameters(
    cpus_per_task=CPUS_PER_JOB,
    mem_gb=MEM_GB,
    timeout_min=TIMEOUT_MIN,
)

# dispatch one job per ligand
jobs = executor.map_array(
    lambda smi: run_ts1(
        [smi],
        ts_guess_xyz=TS_XYZ,
        n_confs=N_CONFS,
        n_cores=CPUS_PER_JOB,
        debug=DEBUG,
        output_parquet=os.path.join(OUT_DIR, f"{smi}.parquet"),
        save_output_dir=SAVE_OUT_DIRS
    ),
    smi_list
)


if USE_SLURM:
    print("Submitted jobs:", jobs.job_ids)
else:
    print(f"Local run submitted {len(jobs)} ligands.")