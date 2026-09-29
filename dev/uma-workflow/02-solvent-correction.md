# 02 — Add the UMA solvent correction in OET

## Goal

Make the local OET UMA backend return a composite energy and gradient for
GFN2-xTB ALPB(chloroform) correction, while leaving gas-phase UMA available.
The work belongs in the local OET development repository; FRUST passes the
chosen setting through its ORCA input.

## Intended calculation

For each ORCA-requested geometry, charge, and multiplicity:

```text
E_total = E_UMA + E_xTB,ALPB(chloroform) - E_xTB,gas
∇E_total = ∇E_UMA + ∇E_xTB,ALPB(chloroform) - ∇E_xTB,gas
```

Use the same GFN2-xTB settings and atom ordering for the gas and solvated xTB
calls. Convert all energies and gradients to OET's ORCA-facing units before
combining them. The OET option should make the correction visible in the saved
ORCA `%method` / `Ext_Params` block. Keep the exact option spelling consistent
between OET and FRUST; choose it during implementation.

## Work

1. Add the solvent option and composite calculation to the OET UMA backend.
   Keep xTB subprocesses on the same compute node as the server and use unique
   scratch names for simultaneous ORCA evaluations.
2. Surface the UMA model, xTB method, solvent model, and solvent in a concise
   OET output record and FRUST result provenance. Do not flood the ORCA output
   with every optimization displacement.
3. Check a gas-phase UMA call, a solvated single point, and a solvated gradient
   against finite differences of the composite energy. Then run a tiny ORCA
   optimization using the option in the generated input.
4. Record the local OET commit and the installed OET revision. Handle the
   cluster install through versioned repository updates, not ad hoc edits to
   mounted FRUST source.

## Acceptance

- The solvent-on result matches the equation above; solvent-off remains UMA.
- The analytical composite gradient agrees with finite differences to a
  tolerance justified for the chosen geometry and step size.
- The ORCA input visibly identifies the correction, and a saved output or
  metadata record identifies what ran.
- Failure of either xTB calculation fails the external evaluation clearly;
  it does not silently return an uncorrected UMA result.

## Completion record

Completed 2026-09-29. The new OET option is `--xtb-alpb chloroform`, with
`--xtb-exe PATH` when an explicit normal xTB binary is needed. FRUST exposes
these as `uma_xtb_alpb="chloroform"` and `uma_xtb_exe=...`. For example:

```python
result = step.orca(
    water,
    name="uma_alpb_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_xtb_alpb="chloroform",
    uma_inference_settings="batch",
)
```

The [saved ORCA input](evidence/task02/orca-smoke/input.inp) includes
`--xtb-alpb chloroform` and `--inference-settings batch` in `%method
Ext_Params`. OET writes one concise, per-request
[`input_EXT.uma.json`](evidence/task02/orca-smoke/input_EXT.uma.json) record
with model, inference mode, xTB method, solvent, and energy components. FRUST
also stores these choices in `df.attrs["frust_steps"][stage]`, demonstrated in
the [saved provenance summary](evidence/task02/orca-smoke-summary.json).
Gas-phase UMA remains available by omitting `uma_xtb_alpb`.

| Item | Pinned result |
| --- | --- |
| OET fork | [Commit `1b4fcda`](https://github.com/MolinDiscovery/orca-external-tools/commit/1b4fcda), branch `feature/uma-xtb-alpb`; correction implementation entered at `6eb7970` and the later commit repaired the standalone test executable path |
| Installed Mac OET | `2.0.1.dev10+g1b4fcda92`, editable source from the local OET checkout; `fairchem-core` 2.23.0, Torch 2.13.0 |
| Installed cluster OET | `2.0.1.dev10+g1b4fcda92`, editable source from the GitHub-updated cluster checkout; separate CPU runtime `/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu`, `fairchem-core` 2.23.0, Torch `2.13.0+cpu`; [node066 version record](evidence/task02/cluster_runtime_versions.txt) |
| Target model | `omol@uma-s-1p2p1`, checkpoint snapshot `f611b917d9c68566bbbeccbb0aa0f7cad1696cb2`, checkpoint blob SHA-256 `b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce` on both hosts |
| Correction | GFN2-xTB 6.7.1, ALPB(chloroform) minus gas; both xTB calls use the same coordinates, charge, multiplicity and atom order, with private scratch directories |

The cluster's existing OET virtualenv packages were left intact; its editable
source checkout now points to `1b4fcda`. The new CPU runtime was installed on
`node066` with the [versioned install script](evidence/task02/cluster_install_oet_cpu.sh).
The gated 2.2 GB checkpoint was unavailable to the cluster's Hugging Face
login (401), so the already cached Mac checkpoint and its small reference files
were copied into the cluster OET cache. The transferred checkpoint's SHA-256
matched the blob name. Future cluster UMA jobs must set `OET_TOOLS` to the new
CPU runtime path until the workflow's job environment is configured in task 03.

### Numerical checks

The [water geometry](evidence/task02/uma-latest-alpb/water.xyz) used in these
small functional checks has O at `(0, 0, 0)` Å and H at
`(±0.758602, 0, 0.504284)` Å. Independent [xTB gas](evidence/task02/xtb-gas/gas.energy)
and [ALPB](evidence/task02/xtb-alpb/alpb.energy) calls and the
[OET component record](evidence/task02/uma-latest-alpb/water.uma.json) gave:

| Component | Energy (Eh), Mac |
| --- | ---: |
| UMA `uma-s-1p2p1`, gas | -76.42749337447343 |
| GFN2-xTB, ALPB(chloroform) | -5.068696341452 |
| GFN2-xTB, gas | -5.065772968305 |
| Composite | **-76.43041674762043** |

The composite energy equals `UMA + ALPB - gas`; its analytical gradient was
compared against central differences of **the whole composite**, not just the
xTB term. The [reproduction script](evidence/task02/check_composite.py) varied
all nine Cartesian coordinates by ±0.001 Å. The [maximum absolute gradient
error](evidence/task02/finite_difference.json) was
`4.07 × 10^-6 Eh/Bohr`, below the `1 × 10^-5 Eh/Bohr` tolerance appropriate
for this step and the model's finite numerical precision.

The [node066 corrected request](evidence/task02/cluster_latest_baseline.sh)
ran offline in Slurm job `65665353` and returned
`-76.43041676121588 Eh`, about `1.36 × 10^-8 Eh` from the Mac result;
[output and gradients](evidence/task02/cluster_latest_baseline.txt) are saved.
The [local FRUST → ORCA → OET water optimization](evidence/task02/run_tiny_orca.py)
converged in five cycles, terminated normally, and returned
`-76.437169050600 Eh`; its [ORCA output excerpt](evidence/task02/orca-smoke/orca_excerpt.txt)
and input are saved. This is a functional integration check, not a screening
benchmark.

OET tests: `7 passed` for the new correction cases and existing UMA standalone
energy/gradient cases in the installed OET environment. FRUST focused tests:
`13 passed, 3 deselected` with
`conda run -n UMA python -m pytest tests/test_uma_oet.py -q`. Both xTB subprocess
failure modes raise a clear error in the new tests; no uncorrected UMA value is
returned. On the Mac, SciPy in the project `UMA` conda environment was repaired
from 1.15.0 to 1.17.1 after its old binary prevented Stepper from importing.

The original FRUST `uma="omol"` default remains `uma-s-1p1` for compatibility.
The new workflow should explicitly select the pinned `uma-s-1p2p1` model in
task 05. Cluster server lifetime and client/server host placement are task 03.
