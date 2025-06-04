#!/usr/bin/env python
"""
Submit one Slurm job per SMILES string contained in CONFIG.json:

    python submit_pipeline.py CONFIG.json --partition gpu ...

The script:
  1. loads CONFIG.json
  2. writes N single-SMILES configs into a temp folder
  3. submits each via SubmitIt
"""
from __future__ import annotations
import argparse, json, tempfile, shutil
from pathlib import Path
import submitit

# (reuse the driver you built earlier)
from scripts.run_pipeline import main as run_pipeline   # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="master JSON config produced by PipelineBuilder")
    p.add_argument("--partition", default="cpu")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--cpus-per-task", default="8", type=int)
    args = p.parse_args()

    cfg_master = json.loads(Path(args.config).read_text())
    smiles_all = cfg_master["ligands_smiles"]

    tmp = Path(tempfile.mkdtemp(prefix="frust_cfgs_"))
    print("writing single-SMILES configs to", tmp)

    exec_folder = Path("slurm_logs")
    ex = submitit.AutoExecutor(folder=exec_folder)
    ex.update_parameters(
        slurm_partition=args.partition,
        slurm_time=args.time,
        cpus_per_task=args.cpus_per_task,
    )

    jobs = []
    with ex.batch():
        for smi in smiles_all:
            cfg = cfg_master.copy()
            cfg["ligands_smiles"] = [smi]          # one ligand only
            cfg_path = tmp / f"{smi.replace('/','_')}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            job = ex.submit(run_pipeline, str(cfg_path))
            jobs.append(job)
            print("submitted", smi, "→", job.job_id)

    print("✓", len(jobs), "jobs in batch")

if __name__ == "__main__":
    main()