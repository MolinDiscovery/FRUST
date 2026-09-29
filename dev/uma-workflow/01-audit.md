# 01 — Audit the UMA and OET baseline

## Goal

Establish exactly what is installed locally and on the cluster before changing
the OET fork or FRUST. This task is read-only apart from small disposable
checks and this plan's Completion record.

## Work

1. Record the FRUST, OET fork, ORCA, xTB, `fairchem-core`, Torch, and available
   UMA model/checkpoint revisions on the Mac and in the intended compute-node
   environment. Distinguish installed versions from the latest available
   upstream versions; do not upgrade automatically.
2. Compare the current OET fork's g-xTB v2 gradient route with upstream OET.
   State whether the fork is still required for FRUST's g-xTB use, using a
   minimal executable check where source inspection is inconclusive.
3. Inspect the current FRUST UMA call path, OET server/client arguments, and
   ORCA input generation. Identify where a solvent option and a job-scoped
   server should attach.
4. Run one minimal existing UMA energy/gradient calculation in the UMA
   environment to establish a working baseline. Keep the input and output.
   Do not run a large screen.

## Acceptance

- A compact version matrix names the executable or package source, revision,
  host, and UMA checkpoint used by the baseline check.
- The current g-xTB fork requirement and UMA solvent/server gaps are stated
  with evidence.
- The baseline calculation reports a usable energy and gradient, or the exact
  installation blocker is recorded for task 02.
- No FRUST or OET production code is changed in this task.

## Completion record

Completed 2026-09-29. The local and cluster FRUST checkouts were both at
`839777d`. The cluster checks ran within Slurm allocations on `node066` in
`kemi1` (jobs `65658476` and `65658535`). All installed OET packages below are
from a separate OET virtual environment; the `UMA` conda environment is not
the runtime used when OET evaluates an ORCA external call.

| Component | Mac | `node066.cluster` | Current upstream reference |
| --- | --- | --- | --- |
| FRUST | Local checkout `839777d`, package 0.1.0 | `/lustre/hpc/kemi/jmni/dev/FRUST` at `839777d`, package 0.1.0 | Local development repository |
| ORCA | 6.1.0, `/Users/jacobmolinnielsen/Library/orca_6_1_0/orca` | 6.1.0, `/groups/kemi/jmni/software/orca_6_1_0_generic/orca` | Version parity is sufficient for this audit |
| OET fork | Source `c13336b`, `/Users/jacobmolinnielsen/Developer/FrustActivationProject/orca-external-tools`; installed at `/Users/jacobmolinnielsen/Library/orca-external-tools` | Source `c13336b`, `/lustre/hpc/kemi/jmni/software/orca-external-tools-src`; installed at `/lustre/hpc/kemi/jmni/software/orca-external-tools` | [Official latest release: 2.0.0](https://github.com/faccts/orca-external-tools/releases); main is still developing |
| OET runtime | Python 3.12.5; `fairchem-core` 2.19.0; Torch 2.8.0; ASE 3.28.0; `huggingface-hub` 1.13.0 | Python 3.12.8; `fairchem-core` 2.20.0; Torch 2.8.0; ASE 3.28.0; `huggingface-hub` 1.15.0 | [`fairchem-core` 2.23.0](https://pypi.org/project/fairchem-core/) (2026-09-24) |
| `UMA` conda environment | Python 3.12.8; `fairchem-core` 2.3.0; Torch 2.6.0; ASE 3.26.0 | Python 3.12.8; `fairchem-core` 2.3.0; Torch 2.6.0; ASE 3.26.0 | Used for FRUST Python and tests, not the OET calculator process |
| Model used in both baseline runs | `omol@uma-s-1p1`, cached checkpoint blob `07068e9c76702ca173d13155095f2117c1b327ec228557e64cd2709c777b824a` | Same task, model and checkpoint blob | [FAIR Chemistry lists `uma-s-1p2p1` as its latest small model](https://github.com/facebookresearch/fairchem) |
| Standard xTB | 6.7.1 (`edcfbbe`) | 6.7.1 (`edcfbbe`) | [Latest stable xTB: 6.7.1](https://github.com/grimme-lab/xtb/releases) |
| g-xTB executable | Modified xTB 6.7.1 (`28d6122`), `/Users/jacobmolinnielsen/Library/g-xtb/xtb-6.7.1-gxtb-210426-macos-arm64/bin/xtb` | Modified xTB 6.7.1 (`26dd68d`), `/lustre/hpc/kemi/jmni/software/g-xtb/bin/xtb` | [Latest g-xTB release: 2.0.1](https://github.com/grimme-lab/g-xtb/releases) |

The Mac OET editable-install metadata reports `2.0.1.dev1+g2f604db51`, while
the imported source checkout is `c13336b`. The cluster metadata reports
`2.0.1.dev8+gc13336b4b`. The source commit, rather than the stale Mac package
metadata, identifies the code being imported. The installed OET CLIs advertise
`uma-s-1p2` and `uma-s-1p1`, defaulting to `uma-s-1p1`; neither advertises
`uma-s-1p2p1` yet. Do not silently switch models when continuing this plan.

### Fork and integration findings

- The [current upstream g-xTB calculator](https://github.com/faccts/orca-external-tools/blob/main/src/oet/calculator/gxtb.py) still invokes the old standalone `gxtb` route with `.gxtb`, `.eeq`, and `.basisq` files and `-grad`. The local fork's `src/oet/calculator/gxtb.py` invokes modified `xtb` with `--gxtb` and `--grad` and parses its namespaced gradient. [g-xTB 2.0 moved to the xTB implementation and removed external parameter files](https://github.com/grimme-lab/g-xtb/releases). The fork is still required for FRUST's g-xTB v2 route. This source-level interface mismatch is conclusive; no additional g-xTB calculation was needed.
- Both the [upstream UMA calculator](https://github.com/faccts/orca-external-tools/blob/main/src/oet/calculator/uma.py) and this fork lack an xTB solvent-difference option. `frust/utils/uma.py` exposes only task, model, device, cache and offline settings; `uma_ext_args()` passes those to OET and `uma_orca_block()` writes them to `%method Ext_Params`. Task 02 should attach the explicit correction option there and implement the matched energy **and** gradient terms in OET.
- `frust/stepper.py` currently enters `uma_server()` separately inside each `Stepper.orca(..., uma=...)` call. That server binds to `127.0.0.1` and is closed on return. A multistage submitted job therefore creates a fresh server for each UMA stage. Task 03 should own a server around the job's stage group in `frust/workflows/core.py` and pass its bind to stage calls, including numerical-frequency calls. Actual client/server placement and reuse remain to be verified in task 03.

### Reproducible baseline

The three-atom [water input](evidence/task01/water.xyz) and [OET external-call request](evidence/task01/water.ext) asked for one OMol energy and gradient with `uma-s-1p1` on CPU. Both calls exited successfully, used a cached model offline, and returned nine finite gradient components:

| Run | OET command | Energy (Eh) | Representative gradient, O z (Eh/Bohr) | Output |
| --- | --- | ---: | ---: | --- |
| Mac | `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/jacobmolinnielsen/Library/orca-external-tools/bin/oet_uma -t omol -m uma-s-1p1 -d cpu -o True water.ext` | -76.42751493608 | +0.07668146491 | [stdout](evidence/task01/baseline.stdout), [gradient](evidence/task01/water.engrad) |
| `node066` | Same arguments through `/lustre/hpc/kemi/jmni/software/orca-external-tools/bin/oet_uma` in Slurm job `65658535` | -76.42751493306 | +0.07668145001 | [script](evidence/task01/cluster_baseline.sh), [captured output](evidence/task01/cluster_baseline.txt) |

The [compute-node version audit](evidence/task01/cluster_audit.txt) and its
[script](evidence/task01/cluster_audit.sh) preserve the cluster paths and
versions. Focused FRUST tests ran with
`conda run -n UMA python -m pytest tests/test_uma_oet.py -q`:
**11 passed, 3 deselected**. These direct OET calls establish the existing
calculator's energy and gradient baseline. They do not yet verify the full
FRUST → ORCA → OET client/server path; that belongs to tasks 03 and 06.

Before profile calculations, choose and pin the UMA model and compatible OET
runtime in task 02. `uma-s-1p2p1` is the current upstream small-model target,
but its OET compatibility and checkpoint availability have not been tested.
No FRUST or OET production code was changed in task 01.
