#!/usr/bin/env python3

import os
import time
from datetime import datetime

import submitit
from frust.config import Settings
from frust.pipeline import run_pipeline  # your generic mol pipeline

def main():
    # Timestamped log folder
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f"submitit_logs/mols_{now}"
    os.makedirs(logdir, exist_ok=True)

    # Settings: live run with dumps at each step
    settings = Settings(debug=False, live=True, dump_each_step=True)

    executor = submitit.AutoExecutor(folder=logdir)
    executor.update_parameters(
        slurm_partition="kemi1",
        cpus_per_task=settings.cores,
        mem_gb=settings.memory_gb,
        slurm_time=3 * 24 * 60,  # three days
        gpus_per_node=0,
    )

    # Example list of ligands
    ligands = [
        "C1=CC=CO1",      # furan
        "CC1=CC=CO1",     # 2-methylfuran
        # …add more SMILES here…
    ]

    # Optional catalyst SMILES (if your run_pipeline needs it)
    catalyst = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"

    for smi in ligands:
        job_name = f"mol_{smi[:6]}"
        executor.update_parameters(slurm_job_name=job_name)

        # run_pipeline signature:
        #   run_pipeline(ligands, catalyst, cores, memory_gb, debug, show_resources, output_base_dir)
        job = executor.submit(
            run_pipeline,
            [smi],
            catalyst,
            settings.cores,
            settings.memory_gb,
            settings.debug,
            False,                       # show_resources
            str(settings.dump_base_dir), # where to write results
        )
        print(f"Submitted MOL job {job.job_id} for {smi}")
        time.sleep(1)

    print("All MOL jobs submitted. Retrieve results with job.result().")

if __name__ == "__main__":
    main()