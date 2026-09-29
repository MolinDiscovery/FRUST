#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.env"
checkout=/lustre/hpc/kemi/jmni/dev/FRUST-uma-task03
python=/groups/kemi/jmni/miniconda3/envs/UMA/bin/python
export PYTHONPATH="$checkout:$checkout/dev/uma-workflow/evidence/task04${PYTHONPATH:+:$PYTHONPATH}"
export FRUST_TASK04_REVISION="$(git -C "$checkout" rev-parse --short HEAD)"
"$python" "$checkout/dev/uma-workflow/evidence/task04/submit_references.py" "$@"
