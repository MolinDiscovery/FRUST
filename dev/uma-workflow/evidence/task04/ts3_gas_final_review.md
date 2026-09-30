# Final gas UMA TS3 candidate: accepted review

Job `65685852` on `node066` completed normally in 11 min 57 sec with FRUST
revision `ed58fe1`, OET runtime `oet-uma-2p23-cpu`, and
`omol@uma-s-1p2p1`. Its source was the ALPB TS3 optimized geometry, displaced
by `+0.8` along each of the two shallow peripheral imaginary modes, followed
by **gas-phase** UMA `OptTS` and `NumFreq` in one job-local UMA server lifetime.
The saved result is
`/lustre/hpc/kemi/jmni/results/uma-task04-ts3-gas-final-20260930/workflow/ts3_gas/final.parquet`.

The optimization and final numerical frequency calculation terminated
normally. Exactly one imaginary mode remains, mode 0 at **−94.46 cm⁻¹**; the
next frequency is +26.02 cm⁻¹. That satisfies the numerical first-order
saddle criterion.

| Geometry | Bcat–H (Å) | Bpin–H (Å) | Bcat–Csub (Å) | Bpin–Csub (Å) |
| --- | ---: | ---: | ---: | ---: |
| ALPB TS3 seed | 1.391 | 1.245 | 1.635 | 2.082 |
| New gas result labelled TS3 | 1.229 | 1.590 | 1.879 | 1.624 |
| Existing gas TS4 | 1.236 | 1.536 | 1.847 | 1.639 |

The new gas geometry lies much closer to the separately optimized gas TS4
reactive core than to the TS3 seed. TS3 and TS4 are closely related, so these
distances alone do not distinguish their chemical assignments. The signed
changes in five reactive distances across ±0.25 of mode 0 have a cosine of
**−0.9978** against the gas
TS4 imaginary-mode distance-change vector. Mode sign is arbitrary, so the
magnitudes and directions of the underlying motions closely agree. Visual
inspection of the `ft.plot_vibs` animation and its two displaced endpoint
structures likewise shows Bpin/substrate/catalyst-core movement.
The local result directory contains `ts3_gas_reactive_mode.html`,
`ts3_gas_mode_endpoints_compact.html`, and `ts3_vs_ts4_modes.html` for direct
visual review. The py3Dmol viewers were also rotated interactively to inspect
the three-dimensional arrangement and mode from more than one direction.

**Decision: accept as the UMA gas TS3 profile reference.** The user reviewed
the notebook and found the TS3 structure and reactive motion chemically right.
The numerical saddle and mode checks support that assessment. Similarity to
TS4 remains an observation, not evidence of a wrong assignment. A matched
ωB97 frequency for this exact TMP/thiophene source remains unavailable, so
the broader ωB97 TS3 distribution is context only.

The user-executed review notebook is
`dev/uma-workflow/evidence/task04/ts3_gas_review.ipynb`; its original executed
state was backed up as `ts3_gas_review_executed_backup.ipynb` alongside the
local result in `fruits/results-2026/uma-task04-ts3-gas-final-20260930/`.
