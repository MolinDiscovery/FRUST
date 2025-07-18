# scripts/submit.py
import os
import inspect
import pandas as pd
import submitit
from itertools import islice
import importlib

# ─── CONFIG ─────────────────────────────────────────────────────────────
PIPELINE_NAME  = "run_ts"  # "run_ts" or "run_mols"
PRODUCTION     = False
USE_SLURM      = True
DEBUG          = True
BATCH_SIZE     = 1
#CSV_PATH       = "../datasets/ir_borylation.csv" if PRODUCTION else "../datasets/ir_borylation_test.csv"
CSV_PATH       = "../datasets/1m.csv"
OUT_DIR        = "results_test_dist"
LOG_DIR        = "logs/test_dist"
SAVE_OUT_DIRS  = False
CPUS_PER_JOB   = 12
MEM_GB         = 30
TIMEOUT_MIN    = 7200
N_CONFS        = None if PRODUCTION else 1 # if this is set to None, the following rule goes. Let R me rotatable bonds. 50 confs when R < 7, 200 confs when R 7-12 bonds and 300 if more.
DFT            = True
# ─── TS SPECIFIC ─────────────────────────────────────────────────────────
TS_XYZ         = "../structures/ts2_guess_old.xyz"
# ─── MOL SPECIFIC ────────────────────────────────────────────────────────
SELECT_MOLS    = ["HBpin-mol", "HH"] # "all", "uniques", "generics", or specific names in a list ['dimer','ligand','catalyst','int2','mol2','HBpin-ligand','HBpin-mol']


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
        executor.update_parameters(slurm_job_name=tag)

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
        "select_mols":        SELECT_MOLS,
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