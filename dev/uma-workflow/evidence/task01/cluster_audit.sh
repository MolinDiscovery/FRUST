#!/usr/bin/env bash
set -u

source "$HOME/.env"

printf 'host=%s\njob=%s\nnode_list=%s\n' "$(hostname)" "${SLURM_JOB_ID:-}" "${SLURM_JOB_NODELIST:-}"
printf 'frust_checkout=%s\n' '/lustre/hpc/kemi/jmni/dev/FRUST'
git -C /lustre/hpc/kemi/jmni/dev/FRUST rev-parse --short HEAD 2>&1
printf 'orca_exe=%s\n' "$ORCA_EXE"
"$ORCA_EXE" --version 2>&1 | grep -m 1 'Program Version' || true
printf 'xtb_exe=%s\n' "$XTB_EXE"
"$XTB_EXE" --version 2>&1 | grep -m 1 'xtb version' || true
printf 'gxtb_exe=%s\n' "$GXTB_EXE"
"$GXTB_EXE" --version 2>&1 | grep -m 1 'xtb version' || true
printf 'oet_tools=%s\n' "$OET_TOOLS"
"$OET_TOOLS/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import oet
import sys

print('oet_python=' + sys.version.split()[0])
print('oet_source=' + str(oet.__file__))
for name in ('oet', 'fairchem-core', 'torch', 'ase', 'huggingface-hub'):
    try:
        print(name + '=' + metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name + '=not installed')
PY
/groups/kemi/jmni/miniconda3/envs/UMA/bin/python - <<'PY'
import importlib.metadata as metadata
import frust
import sys

print('uma_python=' + sys.version.split()[0])
print('frust_source=' + str(frust.__file__))
for name in ('frust', 'fairchem-core', 'torch', 'ase', 'huggingface-hub'):
    try:
        print('UMA_env_' + name + '=' + metadata.version(name))
    except metadata.PackageNotFoundError:
        print('UMA_env_' + name + '=not installed')
PY
"$OET_TOOLS/bin/oet_uma" -h 2>&1 | grep -E 'Options: uma-|Default: uma-' | head -n 5 || true
