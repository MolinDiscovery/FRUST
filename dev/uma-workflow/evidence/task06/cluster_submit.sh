#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.env"
checkout=/lustre/hpc/kemi/jmni/dev/FRUST
python=/groups/kemi/jmni/miniconda3/envs/UMA/bin/python
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
"$python" "$checkout/dev/uma-workflow/evidence/task06/submit_smoke.py" "$@"
