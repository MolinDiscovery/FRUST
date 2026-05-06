# External Tool Setup

This page is the advanced part of installation. It assumes FRUST is already
installed as a Python package.

The goal is simple: install the external programs you need, put their paths in
`~/.env`, and run smoke tests before starting a real workflow.

## What Goes Where

FRUST and Tooltoad read stable tool paths from `.env` files in this order:

1. path set by `TOOLTOAD_DOTENV_PATH`;
2. `.env` in the current working directory;
3. `~/.env`.

For normal use, put your personal machine paths in `~/.env`.

Use `${HOME}`, not bare `$HOME`, inside `.env` files. `python-dotenv` expands
the brace form reliably.

## Minimal `.env` Template

Start with the variables for the tools you actually use:

```bash
ORCA_EXE=/path/to/orca
OPEN_MPI_DIR=${HOME}/openmpi
XTBPATH=/path/to/xtb/share/xtb
XTB_EXE=/path/to/xtb
OET_TOOLS=/path/to/orca-external-tools
HF_HOME=${HOME}/.cache/huggingface
GXTB_ROOT=/path/to/g-xtb
GXTB_EXE=/path/to/g-xtb/bin/xtb
export DYLD_LIBRARY_PATH="/path/to/orca:${OPEN_MPI_DIR}/lib:${DYLD_LIBRARY_PATH}"
export PATH="/path/to/orca:${PATH}"
```

Shell `export ...` commands still work for temporary sessions, CI, and
submitted jobs, but `~/.env` should be the normal place for stable local paths.

## Standard xTB And ORCA

For standard xTB stages, install xTB and set:

```bash
XTB_EXE=/path/to/xtb
XTBPATH=/path/to/xtb/share/xtb
```

For ORCA stages, install ORCA and Open MPI, then set:

```bash
ORCA_EXE=/path/to/orca
OPEN_MPI_DIR=/path/to/openmpi
```

On macOS, ORCA often also needs ORCA and Open MPI on `PATH` and the dynamic
library path:

```bash
export DYLD_LIBRARY_PATH="/path/to/orca:${OPEN_MPI_DIR}/lib:${DYLD_LIBRARY_PATH}"
export PATH="/path/to/orca:${PATH}"
```

## ORCA-External-Tools

ORCA-External-Tools is needed for UMA and for ORCA-driven g-xTB. It is separate
from FRUST and separate from ORCA.

!!! info "Why FRUST uses a custom OET fork"

    FRUST uses the MolinDiscovery OET fork because the g-xTB route was changed
    to work with the newer g-xTB 2.0 architecture. Upstream OET's documented
    g-xTB path is based on the older preliminary g-xTB setup, where ORCA's
    wrapper route used numerical gradients. For FRUST we want ORCA to drive
    optimizations while g-xTB 2.0 supplies the external energies and gradients.

    The custom `oet_gxtb` wrapper does that bridge: ORCA calls `oet_gxtb`,
    `oet_gxtb` calls the g-xTB 2.0 `xtb` binary from `GXTB_EXE`, and the
    resulting energy/gradient information is passed back to ORCA. Direct
    `Stepper.gxtb(...)` uses the same g-xTB 2.0 executable through Tooltoad.

Choose stable install paths:

```bash
PROJECT_ROOT=${HOME}/FrustActivationProject
OET_SRC=${PROJECT_ROOT}/orca-external-tools
OET_TOOLS=${HOME}/.local/orca-external-tools
```

Clone the OET source:

```bash
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git clone https://github.com/MolinDiscovery/orca-external-tools.git
```

Install the OET command wrappers into their own virtual environment:

```bash
python install.py \
  --venv-dir "$OET_TOOLS/.venv" \
  --script-dir "$OET_TOOLS/bin" \
  -e uma
```

This should create:

```text
$OET_TOOLS/bin/oet_server
$OET_TOOLS/bin/oet_client
$OET_TOOLS/bin/oet_uma
$OET_TOOLS/bin/oet_gxtb
```

Put the installed root in `~/.env`:

```bash
OET_TOOLS=${HOME}/.local/orca-external-tools
```

Do not point `OET_TOOLS` at the source checkout. Point it at the install root
that contains `bin/`.

## g-xTB

Install or unpack the [g-xTB 2.0.0](https://github.com/grimme-lab/g-xtb/releases/tag/v2.0.0) distribution somewhere stable. The final executable
should be the special g-xTB-capable `xtb`, not the normal xTB executable.

Example layout:

```text
${HOME}/.local/g-xtb/bin/xtb
```

Make it executable:

```bash
chmod +x "$GXTB_EXE"
```

Put the paths in `~/.env`:

```bash
GXTB_ROOT=${HOME}/.local/g-xtb
GXTB_EXE=${GXTB_ROOT}/bin/xtb
```

FRUST does not automatically reuse `XTB_EXE` for g-xTB because normal xTB
installs usually do not support `--gxtb`.

## Smoke Tests

Run these checks in the same environment where FRUST will run.

Check that FRUST sees OET and g-xTB:

```bash
python - <<'PY'
from frust.config import get_oet_tools
from frust.utils.gxtb import get_gxtb_exe

print("OET_TOOLS:", get_oet_tools())
print("GXTB_EXE:", get_gxtb_exe())
PY
```

For shell checks, load your `~/.env` into the current shell first:

```bash
set -a
source ~/.env
set +a
```

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

## Submitit Notes

FRUST cluster workflows are meant to be launched through `submitit`, using
`frust.cluster`. You normally should not write a custom Slurm script for each
FRUST workflow.

The important setup rule is that the submitted Python job must see the same
tool paths as your login shell:

- put stable paths in a readable `~/.env`;
- install external tools on a filesystem visible from compute nodes;
- use `ClusterConfig.extra_slurm_parameters` for scheduler options such as
  account or qos.

FRUST starts UMA servers inside the submitted Python process. With the Slurm
submitit backend, that means the UMA server starts on the allocated compute
node. Do not start a long-lived shared UMA server on the login node.

See [Cluster Submission](../cluster/submission.md) for the submitit interface.

## Updating External Tools

Update FRUST and Tooltoad in the Python environment where you run workflows:

```bash
cd "$FRUST_ROOT"
git pull --ff-only origin main

cd "$TOOLTOAD_ROOT"
git pull --ff-only origin main

conda activate frust
python -m pip install -e "$TOOLTOAD_ROOT"
python -m pip install -e "$FRUST_ROOT"
```

Update OET by pulling the source and reinstalling into the same `OET_TOOLS`
install root:

```bash
cd "$OET_SRC"
git pull --ff-only origin main

python install.py \
  --venv-dir "$OET_TOOLS/.venv" \
  --script-dir "$OET_TOOLS/bin" \
  -e uma
```

Run the smoke tests again after updating.

## Troubleshooting

If FRUST cannot find OET:

```text
RuntimeError: Set OET_TOOLS to the orca-external-tools root.
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

If UMA starts on the wrong node, make sure the FRUST Python process was launched
through the Slurm submitit backend rather than from the login node.