#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=24
#SBATCH --error=/lustre/hpc/kemi/jmni/dev/FRUST/pipes/logs/irc/%j_0_log.err
#SBATCH --job-name='IRC_TS2(min-gode)'
#SBATCH --mem=60GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/lustre/hpc/kemi/jmni/dev/FRUST/pipes/logs/irc/%j_0_log.out
#SBATCH --partition=kemi1
#SBATCH --signal=USR2@90
#SBATCH --time=14400
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /lustre/hpc/kemi/jmni/dev/FRUST/pipes/logs/irc/%j_%t_log.out --error /lustre/hpc/kemi/jmni/dev/FRUST/pipes/logs/irc/%j_%t_log.err /groups/kemi/jmni/miniconda3/envs/FrustActivation/bin/python -u -m submitit.core._submit /lustre/hpc/kemi/jmni/dev/FRUST/pipes/logs/irc
