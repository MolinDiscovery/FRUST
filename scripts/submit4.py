# scripts/submit4.py
import os
import inspect
import pandas as pd
import submitit
from itertools import islice
import importlib

# ─── CONFIG ─────────────────────────────────────────────────────────────
PIPELINE_NAME = "run_ts_per_rpos"  # unchanged entrypoint name
PRODUCTION = True
USE_SLURM, PARTITION = True, "kemi1"
DEBUG = False
BATCH_SIZE = 1
CSV_PATH = "../datasets/temps/temp_ts4.csv"
OUT_DIR = "results_test"
WORK_DIR = None
LOG_DIR = "logs/test"
SAVE_OUT_DIRS = True
CPUS_PER_JOB = 6
MEM_GB = 40
TIMEOUT_MIN = 7200
N_CONFS = None if PRODUCTION else 1
DFT = True

# Split out Freq as Stage-2
SPLIT_FREQ = True
FREQ_CPUS = 2
FREQ_MEM_GB = 20
FREQ_TIMEOUT_MIN = 7200

# ─── TS SPECIFIC ────────────────────────────────────────────────────────
TS_XYZ = "../structures/ts4_TMP.xyz"

# ─── MOL SPECIFIC ───────────────────────────────────────────────────────
SELECT_MOLS = "all"

def batched(iterable, n):
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch

# 1) load the requested pipeline symbolically (for signature filtering)
pipes_mod = importlib.import_module("frust.pipes")
pipeline_fn = getattr(pipes_mod, PIPELINE_NAME)
sig = inspect.signature(pipeline_fn)

# 2) read & dedupe SMILES
df = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))

# determine job inputs
if PIPELINE_NAME in {
    "run_ts_per_rpos",
    "run_ts_per_rpos_UMA",
    "run_ts_per_rpos_UMA_short",
    "run_orca_smoke_test",
}:
    from frust.pipes import create_ts_per_rpos
    if PIPELINE_NAME == "run_orca_smoke_test":
        job_inputs = [{"rand mol": "mol object"}]
    else:
        job_inputs = create_ts_per_rpos(smi_list, TS_XYZ)
elif PIPELINE_NAME in {"run_ts_per_lig", "run_mols", "run_small_test"}:
    job_inputs = smi_list
else:
    raise ValueError(f"Unknown pipeline {PIPELINE_NAME!r}")

# 3) make sure output dirs exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
if WORK_DIR:
    os.makedirs(WORK_DIR, exist_ok=True)

# 4) pick executor
executor = (submitit.AutoExecutor(LOG_DIR) if USE_SLURM
            else submitit.LocalExecutor(LOG_DIR))
executor.update_parameters(
    slurm_partition=PARTITION if USE_SLURM else None,
    cpus_per_task=CPUS_PER_JOB,
    mem_gb=MEM_GB,
    timeout_min=TIMEOUT_MIN,
)

# 5) dispatch
futures = []

def _tag_for_ts(ts_struct: dict) -> str:
    key = list(ts_struct.keys())[0]
    return f"{PIPELINE_NAME}_{key}"

if PIPELINE_NAME in {
    "run_ts_per_rpos",
    "run_ts_per_rpos_UMA",
    "run_ts_per_rpos_UMA_short",
}:
    # stage identifiers in pipes
    stage1_name = "run_ts_per_rpos_geom_only" if SPLIT_FREQ else PIPELINE_NAME
    stage2_name = "run_ts_freq_only_from_parquet"

    stage1_fn = getattr(pipes_mod, stage1_name)
    stage1_sig = inspect.signature(stage1_fn)
    stage2_fn = getattr(pipes_mod, stage2_name)

    functional = "wB97X-D3"
    basisset = "6-31G**"

    for ts_struct in job_inputs:
        tag = _tag_for_ts(ts_struct)
        out_parquet = os.path.join(OUT_DIR, f"{tag}.parquet")

        # ── Stage 1 submission ───────────────────────────────────────────
        if USE_SLURM:
            executor.update_parameters(slurm_job_name=f"{tag}_geom")
        all_kwargs = {
            "ligand_smiles_list": None,
            "ts_guess_xyz": TS_XYZ,
            "ts_struct": None,
            "n_confs": N_CONFS,
            "n_cores": CPUS_PER_JOB,
            "mem_gb": MEM_GB,
            "debug": DEBUG,
            "out_dir": OUT_DIR,
            "work_dir": WORK_DIR,
            "output_parquet": out_parquet,
            "save_output_dir": SAVE_OUT_DIRS,
            "DFT": DFT,
            "select_mols": SELECT_MOLS,
        }
        merged = all_kwargs.copy()
        merged.update({"ts_struct": ts_struct})
        call_kwargs = {k: v for k, v in merged.items()
                       if k in stage1_sig.parameters}

        fut1 = executor.submit(stage1_fn, **call_kwargs)
        futures.append(fut1)

        if not SPLIT_FREQ:
            continue

        # ── Stage 2 submission (Freq-only, afterok) ─────────────────────
        dep = None
        if USE_SLURM:
            dep = {"dependency": f"afterok:{fut1.job_id}"}

        if USE_SLURM:
            executor.update_parameters(
                slurm_job_name=f"{tag}_freq",
                cpus_per_task=FREQ_CPUS,
                mem_gb=FREQ_MEM_GB,
                timeout_min=FREQ_TIMEOUT_MIN,
                slurm_additional_parameters=dep,
            )
        else:
            executor.update_parameters(
                cpus_per_task=FREQ_CPUS,
                mem_gb=FREQ_MEM_GB,
                timeout_min=FREQ_TIMEOUT_MIN,
            )

        freq_kwargs = dict(
            parquet_path=out_parquet,
            functional=functional,
            basisset=basisset,
            n_cores=FREQ_CPUS,
            mem_gb=FREQ_MEM_GB,
            debug=DEBUG,
            out_dir=OUT_DIR,
            work_dir=WORK_DIR,
        )

        fut2 = executor.submit(stage2_fn, **freq_kwargs)
        futures.append(fut2)

        # restore general resources for next loop iteration
        if USE_SLURM:
            executor.update_parameters(
                cpus_per_task=CPUS_PER_JOB,
                mem_gb=MEM_GB,
                timeout_min=TIMEOUT_MIN,
                slurm_additional_parameters=None,
            )

else:
    # non-TS pipelines retained as before
    for batch in batched(job_inputs, BATCH_SIZE):
        tag = (f"{PIPELINE_NAME}_batch_"
               f"{hash(tuple(map(str, batch))) & 0xffffffff:x}")
        if USE_SLURM:
            executor.update_parameters(slurm_job_name=tag)
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