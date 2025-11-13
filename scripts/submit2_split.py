# scripts/submit2.py
import os
import inspect
import pandas as pd
import submitit
from itertools import islice
import importlib

# ─── CONFIG ─────────────────────────────────────────────────────────────
PIPELINE_NAME_1         = "run_ts_per_rpos_geom_only"
PIPELINE_NAME_2         = "run_ts_freq_only_from_parquet"
PRODUCTION              = False
USE_SLURM, PARTITION    = True, "kemi1"
DEBUG                   = False
CSV_PATH                = "../datasets/1m.csv"
OUT_DIR                 = "results_test"
WORK_DIR                = "calcs"
LOG_DIR                 = "logs/test"
SAVE_OUT_DIRS           = True
CPUS_PER_JOB            = 20
MEM_GB                  = 40
TIMEOUT_MIN             = 7200
N_CONFS                 = None if PRODUCTION else 1
DFT                     = True

FREQ_CPUS = 10
FREQ_MEM_GB = 60
FREQ_TIMEOUT_MIN = 7200

# ─── TS SPECIFIC ─────────────────────────────────────────────────────────
TS_XYZ                  = "../structures/ts1.xyz"
# ─── MOL SPECIFIC ────────────────────────────────────────────────────────
SELECT_MOLS             = "all" # "all", "uniques", "generics", or ['dimer','HH','ligand','catalyst','int2','mol2','HBpin-ligand','HBpin-mol']
 
# 1) load the requested pipeline
pipes_mod   = importlib.import_module("frust.pipes")
stage1_fn   = getattr(pipes_mod, PIPELINE_NAME_1)
stage2_fn   = getattr(pipes_mod, PIPELINE_NAME_2)
stage1_sig  = inspect.signature(stage1_fn)

# 2) read & dedupe SMILES
df       = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))

# determine job inputs
if PIPELINE_NAME_1 in {
    "run_ts_per_rpos_geom_only",
    "run_orca_smoke_geom_only"
    }:
    from frust.pipes import create_ts_per_rpos
    if PIPELINE_NAME_1 == "run_orca_smoke_geom_only":
        job_inputs = [{"rand mol": "mol object"}]
    else:
        job_inputs = create_ts_per_rpos(smi_list, TS_XYZ)

# 3) make sure output dirs exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
if WORK_DIR:
    os.makedirs(WORK_DIR, exist_ok=True)

# 4) pick executor
executor = submitit.AutoExecutor(LOG_DIR)
executor.update_parameters(
    slurm_partition=PARTITION,
    cpus_per_task=CPUS_PER_JOB,
    mem_gb=MEM_GB,
    timeout_min=TIMEOUT_MIN,
)

# 5) dispatch batches
jobs = []
for ts_struct in job_inputs:
    tag = f"{PIPELINE_NAME_1}_{list(ts_struct.keys())[0]}"
    executor.update_parameters(slurm_job_name=tag, slurm_additional_parameters={})
    out_parquet = os.path.join(OUT_DIR, f"{tag}.parquet")

    all_kwargs = {
        "ligand_smiles_list": None,
        "ts_guess_xyz":       TS_XYZ,
        "ts_struct":          None,
        "n_confs":            N_CONFS,
        "n_cores":            CPUS_PER_JOB,
        "mem_gb":             MEM_GB,
        "debug":              DEBUG,
        "out_dir":            OUT_DIR,
        "output_parquet":     out_parquet,
        "save_output_dir":    SAVE_OUT_DIRS,
        "DFT":                DFT,
        "work_dir":           WORK_DIR,
    }
    merged = all_kwargs.copy()
    merged.update({"ts_struct": ts_struct})
    call_kwargs = {k: v for k, v in merged.items() if k in stage1_sig.parameters}
    job = executor.submit(stage1_fn, **call_kwargs)
    jobs.append(job)

    # ── Stage 2 submission (Freq-only, afterok) ─────────────────────
    dep = {"dependency": f"afterok:{job.job_id}"}

    executor.update_parameters(
        slurm_job_name=f"{tag}_freq",
        cpus_per_task=FREQ_CPUS,
        mem_gb=FREQ_MEM_GB,
        timeout_min=FREQ_TIMEOUT_MIN,
        slurm_additional_parameters=dep,
    )

    freq_kwargs = dict(
        parquet_path=out_parquet,
        n_cores=FREQ_CPUS,
        mem_gb=FREQ_MEM_GB,
        debug=DEBUG,
        out_dir=OUT_DIR,
        work_dir=WORK_DIR,
    )

    job = executor.submit(stage2_fn, **freq_kwargs)
    jobs.append(job)

    # ... reset parameters for the next job
    executor.update_parameters(
        cpus_per_task=CPUS_PER_JOB,
        mem_gb=MEM_GB,
        timeout_min=TIMEOUT_MIN,
        slurm_additional_parameters={},
    )    

# 6) report & optionally wait
print("Submitted Slurm job IDs:", [j.job_id for j in jobs])
