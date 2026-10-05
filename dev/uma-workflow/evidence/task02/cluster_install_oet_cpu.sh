#!/usr/bin/env bash
set -euo pipefail

runtime=/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu
source_repo=/lustre/hpc/kemi/jmni/software/orca-external-tools-src
python=/groups/kemi/jmni/miniconda3/envs/UMA/bin/python

printf 'host=%s\njob=%s\n' "$(hostname)" "${SLURM_JOB_ID:-}"
git -C "$source_repo" rev-parse --short HEAD
"$python" -m venv "$runtime"
"$runtime/bin/python" -m pip install --no-cache-dir --extra-index-url \
  https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
"$runtime/bin/python" -m pip install --no-cache-dir -e "$source_repo"
"$runtime/bin/python" -m pip install --no-cache-dir 'fairchem-core==2.23.0'
"$runtime/bin/python" - <<'PY'
import importlib.metadata as metadata
import oet
import torch
from fairchem.core.calculate.pretrained_mlip import available_models

for name in ("oet", "fairchem-core", "torch", "ase", "huggingface-hub"):
    print(f"{name}={metadata.version(name)}")
print(f"oet_source={oet.__file__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"uma-s-1p2p1_available={'uma-s-1p2p1' in available_models}")
PY
