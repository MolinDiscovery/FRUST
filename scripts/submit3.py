# scripts/submit2.py
import os
import inspect
import pandas as pd
import submitit
from itertools import islice
import importlib

# ─── CONFIG ─────────────────────────────────────────────────────────────
PIPELINE_NAME = "run_ts_per_rpos"  # "run_ts_per_rpos", "run_ts_per_lig",
                                   # "run_mols"
PRODUCTION = True
USE_SLURM = True
DEBUG = False
BATCH_SIZE = 1
CSV_PATH = "../datasets/temps/temp_ts4.csv"
OUT_DIR = "results_ts4_dft_redo"
LOG_DIR = "logs/ts4_dft_redo"
SAVE_OUT_DIRS = True
CPUS_PER_JOB = 4
MEM_GB = 28
TIMEOUT_MIN = 7200
N_CONFS = None if PRODUCTION else 1
DFT = True
# ─── TS SPECIFIC ─────────────────────────────────────────────────────────
TS_XYZ = "../structures/ts4_TMP.xyz"
# ─── MOL SPECIFIC ────────────────────────────────────────────────────────
# "all", "uniques", "generics",
# or ['dimer','HH','ligand','catalyst','int2','mol2','HBpin-ligand','HBpin-mol']
SELECT_MOLS = "all"

# ─── SLURM ARRAY THROTTLING ─────────────────────────────────────────────
# Max number of array tasks to run concurrently (like --array %LIMIT)
ARRAY_LIMIT = 3


def batched(iterable, n):
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch


# 1) load the requested pipeline
pipes_mod = importlib.import_module("frust.pipes")
pipeline_fn = getattr(pipes_mod, PIPELINE_NAME)
sig = inspect.signature(pipeline_fn)


# 2) read & dedupe SMILES
df = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))


# determine job inputs
if PIPELINE_NAME in {"run_ts_per_rpos", "run_ts_per_rpos_UMA"}:
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
executor = (
    submitit.AutoExecutor(LOG_DIR) if USE_SLURM
    else submitit.LocalExecutor(LOG_DIR)
)
executor.update_parameters(
    slurm_partition="kemi1" if USE_SLURM else None,
    cpus_per_task=CPUS_PER_JOB,
    mem_gb=MEM_GB,
    timeout_min=TIMEOUT_MIN,
    slurm_array_parallelism=ARRAY_LIMIT if USE_SLURM else None,
)


def _runner(call_kwargs):
    return pipeline_fn(**call_kwargs)


# 5) build task list (each entry is kwargs for one task)
tasks = []
if PIPELINE_NAME in {"run_ts_per_rpos", "run_ts_per_rpos_UMA"}:
    for ts_struct in job_inputs:
        tag = f"{PIPELINE_NAME}_{list(ts_struct.keys())[0]}"
        all_kwargs = {
            "ligand_smiles_list": None,
            "ts_guess_xyz": TS_XYZ,
            "ts_struct": None,
            "n_confs": N_CONFS,
            "n_cores": CPUS_PER_JOB,
            "mem_gb": MEM_GB,
            "debug": DEBUG,
            "out_dir": OUT_DIR,
            "output_parquet": os.path.join(OUT_DIR, f"{tag}.parquet"),
            "save_output_dir": SAVE_OUT_DIRS,
            "DFT": DFT,
            "select_mols": SELECT_MOLS,
        }
        merged = all_kwargs.copy()
        merged.update({"ts_struct": ts_struct})
        call_kwargs = {
            k: v for k, v in merged.items() if k in sig.parameters
        }
        tasks.append(call_kwargs)
else:
    for batch in batched(job_inputs, BATCH_SIZE):
        # stable short tag from the batch contents
        h = hash(tuple(map(str, batch))) & 0xFFFFFFFF
        tag = f"{PIPELINE_NAME}_batch_{h:08x}"

        all_kwargs = {
            "ligand_smiles_list": None,
            "ts_guess_xyz": TS_XYZ,
            "ts_struct": None,
            "n_confs": N_CONFS,
            "mem_gb": MEM_GB,
            "n_cores": CPUS_PER_JOB,
            "debug": DEBUG,
            "out_dir": OUT_DIR,
            "output_parquet": os.path.join(OUT_DIR, f"{tag}.parquet"),
            "save_output_dir": SAVE_OUT_DIRS,
            "DFT": DFT,
            "select_mols": SELECT_MOLS,
        }
        specific = {"ligand_smiles_list": batch}
        merged = all_kwargs.copy()
        merged.update(specific)
        call_kwargs = {
            k: v for k, v in merged.items() if k in sig.parameters
        }
        tasks.append(call_kwargs)


# one array job name for the whole batch
if USE_SLURM:
    executor.update_parameters(slurm_job_name=PIPELINE_NAME)

# submit: one SLURM array with capped parallelism; local falls back to many
if USE_SLURM:
    futures = executor.map_array(_runner, tasks)
    print("Submitted Slurm job IDs:", [f.job_id for f in futures])
else:
    futures = [executor.submit(_runner, t) for t in tasks]
    print(f"Dispatched {len(futures)} local futures …")
    for i, f in enumerate(futures, 1):
        f.result()
        print(f"Completed {i}/{len(futures)}", end="\r")
    print("\nAll done.")