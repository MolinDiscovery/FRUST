#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.env"
checkout=/lustre/hpc/kemi/jmni/dev/FRUST-uma-task03
python=/groups/kemi/jmni/miniconda3/envs/UMA/bin/python
out_dir=${FRUST_UMA_TASK03_OUT:?Set the task 03 output directory before submission}
export PYTHONPATH="$checkout:$checkout/dev/uma-workflow/evidence/task03${PYTHONPATH:+:$PYTHONPATH}"
export FRUST_TASK03_REVISION="$(git -C "$checkout" rev-parse --short HEAD)"
mkdir -p "$out_dir"
"$python" "$checkout/dev/uma-workflow/evidence/task03/submit_lifecycle.py"
