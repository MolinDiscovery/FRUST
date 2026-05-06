# HPC Setup For Custom OET, UMA, And g-xTB

This page describes how to set up the custom ORCA-External-Tools installation
needed by FRUST on an HPC cluster.

The setup has three separate pieces:

```text
FRUST                 Python workflow code
Tooltoad              FRUST's QM backend layer
orca-external-tools   OET fork used by ORCA for UMA and ORCA-driven g-xTB
g-xTB                 Special xtb binary that supports --gxtb
```

## Recommended Directory Layout

Use paths that live on a shared filesystem visible from the compute nodes. Do
not install this only on a login-node-local disk.

Example:

```bash
export PROJECT_ROOT=$HOME/FrustActivationProject
export FRUST_ROOT=$PROJECT_ROOT/FRUST
export TOOLTOAD_ROOT=$PROJECT_ROOT/tool-toad
export OET_SRC=$PROJECT_ROOT/orca-external-tools
export OET_TOOLS=$HOME/.local/orca-external-tools
export GXTB_ROOT=$HOME/.local/g-xtb
export GXTB_EXE=$GXTB_ROOT/bin/xtb
```

`OET_TOOLS` is the active OET installation root. FRUST expects:

```text
$OET_TOOLS/bin/oet_server
$OET_TOOLS/bin/oet_client
$OET_TOOLS/bin/oet_uma
$OET_TOOLS/bin/oet_gxtb
```

`GXTB_EXE` must point to the special g-xTB `xtb` executable, not the normal
xTB executable.

## Clone The Repositories

FRUST and Tooltoad:

```bash
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git clone https://github.com/MolinDiscovery/FRUST.git
git clone https://github.com/MolinDiscovery/tool-toad.git
```

Custom OET fork:

```bash
git clone https://github.com/MolinDiscovery/orca-external-tools.git
cd "$OET_SRC"
git remote add upstream https://github.com/faccts/orca-external-tools.git 2>/dev/null || true
git switch main
```

## Python Environment

Use the Python environment you normally use for FRUST. On the cluster this is
often a conda environment.

Example:

```bash
conda create -n frust python=3.12
conda activate frust
```

Install FRUST and Tooltoad from the local checkouts:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e "$TOOLTOAD_ROOT"
python -m pip install -e "$FRUST_ROOT"
```

Install any remaining project dependencies according to the cluster setup you
normally use. The important point is that the environment running FRUST imports
this local Tooltoad checkout.

## Install Custom OET

The OET fork should be installed with its own virtual environment. This keeps
the OET command-line wrappers stable and avoids mixing OET's UMA dependencies
with the main FRUST environment.

```bash
cd "$OET_SRC"

python install.py \
  --venv-dir "$OET_TOOLS/.venv" \
  --script-dir "$OET_TOOLS/bin" \
  -e uma
```

This creates:

```text
$OET_TOOLS/.venv/
$OET_TOOLS/bin/oet_server
$OET_TOOLS/bin/oet_client
$OET_TOOLS/bin/oet_uma
$OET_TOOLS/bin/oet_gxtb
```

If you later update the OET fork, reinstall into the same active path:

```bash
cd "$OET_SRC"
git pull --ff-only origin main

python install.py \
  --venv-dir "$OET_TOOLS/.venv" \
  --script-dir "$OET_TOOLS/bin" \
  -e uma
```

Do not point `OET_TOOLS` at the source checkout. Point it at the installation
root containing `bin/`.

## Install g-xTB

Install or unpack the g-xTB distribution somewhere on the shared filesystem.
The final executable should be:

```text
$GXTB_EXE
```

Example layout:

```text
$HOME/.local/g-xtb/bin/xtb
```

Make it executable:

```bash
chmod +x "$GXTB_EXE"
```

Check that this is the g-xTB-capable binary:

```bash
"$GXTB_EXE" --help | grep -- --gxtb
"$GXTB_EXE" --help | grep -- --grad
```

Do not put `GXTB_ROOT/bin` first on `PATH` unless you intentionally want the
cluster shell command `xtb` to mean g-xTB. FRUST and Tooltoad use `GXTB_EXE`
directly, so the normal `XTB_EXE` can stay pointed at the normal xTB install.

## Environment Variables

Set these variables before running FRUST jobs:

```bash
export OET_TOOLS=$HOME/.local/orca-external-tools
export GXTB_ROOT=$HOME/.local/g-xtb
export GXTB_EXE=$GXTB_ROOT/bin/xtb
```

`UMA_TOOLS` is still accepted for older setups:

```bash
export UMA_TOOLS=$OET_TOOLS
```

If both `OET_TOOLS` and `UMA_TOOLS` are set, FRUST uses `OET_TOOLS`.

You can put the exports in a cluster setup script, for example:

```bash
# $HOME/env/frust-hpc.sh
export PROJECT_ROOT=$HOME/FrustActivationProject
export FRUST_ROOT=$PROJECT_ROOT/FRUST
export TOOLTOAD_ROOT=$PROJECT_ROOT/tool-toad
export OET_TOOLS=$HOME/.local/orca-external-tools
export UMA_TOOLS=$OET_TOOLS
export GXTB_ROOT=$HOME/.local/g-xtb
export GXTB_EXE=$GXTB_ROOT/bin/xtb
```

Then source it from notebooks, shell sessions, and Slurm scripts:

```bash
source "$HOME/env/frust-hpc.sh"
```

## Smoke Tests

Run these checks on the cluster after installation.

Check OET scripts:

```bash
test -x "$OET_TOOLS/bin/oet_server"
test -x "$OET_TOOLS/bin/oet_client"
test -x "$OET_TOOLS/bin/oet_uma"
test -x "$OET_TOOLS/bin/oet_gxtb"

"$OET_TOOLS/bin/oet_server" --help
"$OET_TOOLS/bin/oet_client" --help
"$OET_TOOLS/bin/oet_gxtb" --help
```

Check g-xTB:

```bash
"$GXTB_EXE" --help | grep -- --gxtb
"$GXTB_EXE" --help | grep -- --grad
```

Check FRUST sees the paths:

```bash
python - <<'PY'
from frust.config import get_oet_tools
from frust.utils.gxtb import get_gxtb_exe

print("OET_TOOLS:", get_oet_tools())
print("GXTB_EXE:", get_gxtb_exe())
PY
```

Check UMA server startup without running a full ORCA job:

```bash
python - <<'PY'
from frust.utils.uma import uma_server

with uma_server(server_cores=1, memory_per_thread_mib=500) as server:
    print("UMA server:", server.bind)
PY
```

This should start `oet_server uma`, wait for `/healthz`, print the localhost
bind address, and then shut the server down.

## Slurm Job Template

Use the same setup inside the allocation that runs FRUST. This matters because
FRUST starts the UMA server inside the current process; on Slurm that means the
server starts on the allocated compute node.

Example:

```bash
#!/bin/bash
#SBATCH --job-name=frust
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00

set -euo pipefail

module purge
# module load miniconda
# module load orca

source "$HOME/env/frust-hpc.sh"
conda activate frust

cd "$FRUST_ROOT"

python run_my_frust_job.py
```

Do not start a long-lived shared UMA server manually on the login node. FRUST
starts a short-lived server bound to `127.0.0.1:<free_port>` for the current
ORCA-backed UMA step and shuts it down after ORCA finishes.

## UMA Usage From FRUST

Default server mode:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
)
```

Explicit model:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p1",
)
```

GPU request:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
    uma_device="cuda",
)
```

Control server resources:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
    n_cores=8,
    uma_server_cores=8,
    uma_memory_per_thread_mib=500,
)
```

Keep UMA server logs:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
    uma_keep_logs="always",
    uma_log_dir="UMA-logs",
)
```

The default is `uma_keep_logs="on_failure"`, which avoids bloating successful
runs with server logs.

## g-xTB Usage From FRUST

Direct g-xTB through Tooltoad:

```python
df = step.gxtb(
    df,
    name="gxtb-opt",
    options={"opt": None},
    save_step=True,
)
```

ORCA-driven g-xTB optimization through OET:

```python
df = step.orca(
    df,
    name="gxtb-Opt",
    options={"Opt": None},
    gxtb=True,
    save_step=True,
)
```

ORCA-driven g-xTB transition-state optimization:

```python
df = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None},
    gxtb=True,
    save_step=True,
)
```

If the TS search needs a better initial Hessian:

```python
df = step.gxtb(
    df,
    name="gxtb-hess",
    options={"hess": None},
    save_step=True,
)

df = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None},
    gxtb=True,
    use_last_hess=True,
    save_step=True,
)
```

Use `NumFreq`, not `Freq`, with `gxtb=True`:

```python
df = step.orca(
    df,
    name="gxtb-OptTS-NumFreq",
    options={"OptTS": None, "NumFreq": None},
    gxtb=True,
    save_step=True,
)
```

## Updating An Existing HPC Installation

Update the code:

```bash
cd "$FRUST_ROOT"
git pull --ff-only origin main

cd "$TOOLTOAD_ROOT"
git pull --ff-only origin main

cd "$OET_SRC"
git pull --ff-only origin main
```

Reinstall FRUST and Tooltoad into the active FRUST environment:

```bash
conda activate frust
python -m pip install -e "$TOOLTOAD_ROOT"
python -m pip install -e "$FRUST_ROOT"
```

Reinstall OET into the active OET path:

```bash
cd "$OET_SRC"

python install.py \
  --venv-dir "$OET_TOOLS/.venv" \
  --script-dir "$OET_TOOLS/bin" \
  -e uma
```

Run the smoke tests again after updating.

## Troubleshooting

If FRUST cannot find OET:

```text
RuntimeError: Set OET_TOOLS, or legacy UMA_TOOLS, to the orca-external-tools root.
```

check:

```bash
echo "$OET_TOOLS"
ls -l "$OET_TOOLS/bin/oet_server"
```

If FRUST cannot find g-xTB:

```text
RuntimeError: Set GXTB_EXE to the g-xTB v2 xtb executable.
```

check:

```bash
echo "$GXTB_EXE"
ls -l "$GXTB_EXE"
"$GXTB_EXE" --help | grep -- --gxtb
```

If UMA starts on the wrong node, make sure the FRUST Python process itself is
running inside the Slurm allocation. The server is launched by FRUST using
`oet_server uma --bind 127.0.0.1:<free_port>`, so it runs wherever the FRUST
process runs.

If a UMA server fails, rerun with preserved logs:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
    uma_keep_logs="always",
    uma_log_dir="UMA-logs",
)
```

Inspect the latest `UMA-logs/oet_uma_server_*.log`.

