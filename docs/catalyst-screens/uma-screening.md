# UMA Method Choices For Catalyst Screens

Start with one substrate, one catalyst, and one TS type. A `full` screen
calculates the TS and the ligand, dimer, HBpin, and H₂ references required for
its barrier.

```csv
role,smiles,compound_name,rpos
substrate,CN1C=CC=C1,n_methyl_pyrrole,2
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,
```

Save this as `screen.csv`. These choices use the same TS1 chemistry;
calculator methods and candidate cutoffs change independently.

| Choice | Screening and selection | Final barrier |
| --- | --- | --- |
| UMA → ωB97 | UMA SP, then constrained UMA Opt; select one | ωB97 TS and references |
| UMA final, one candidate | UMA SP, then constrained UMA Opt; refine one | UMA TS and references |
| g-xTB → UMA SP → ωB97 | Constrained g-xTB Opt, then UMA SP reranking; no UMA Opt | ωB97 TS and references |
| g-xTB → UMA SP → UMA | Constrained g-xTB Opt, solvent UMA SP selection, constrained gas UMA preoptimization | Gas UMA refinement and frequencies, terminal solvent UMA SP |
| UMA final, several candidates | UMA SP, then constrained UMA Opt; refine several distinct geometries | UMA barrier for each acceptable TS candidate |

`[C]` in the stage paths means the reactive core is constrained. The constraints
come from the selected TS guess profile. Final `OptTS` searches release them.

## g-xTB Screening With Independent UMA Validation

For g-xTB geometries selected by UMA and then validated by UMA, use one native
workflow:

```python
import frust as ft

paired = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    n_confs=200,
    prune_initial=False,
    screening="gxtb-default",
    top_n=20,
    ranking="uma-alpb-chloroform",
    ranking_top_n=1,
    method="uma-gas-opt-alpb-chloroform",
    level="full",
)
```

The selected g-xTB geometry is the input to constrained gas UMA preoptimization.
Hessian, OptTS and final numerical frequencies release the TS constraints;
reference targets receive unconstrained minimum optimization and frequencies.
The terminal SP adds ALPB(chloroform). `method` can independently select DFT
validation instead. See [screening, selection and validation](../workflows/workflow-methods.md#screening-selection-and-validation)
for the stage path and separate result tiers.

## UMA Screening With ωB97 Final Validation

```python
import frust as ft

common = dict(
    csv_path="screen.csv",
    ts_types=["TS1"],
    spec_profile="omol-uma-s-1p2p1/gas",
    spec_match="exact",
    scope="barriers",
    dimer_reference="dimer",
    n_confs=8,
    top_n=4,
)

wb97 = ft.workflows.catalyst_screen(
    **common,
    screening="uma-gas",
    method="wb97xd3-631g",
    level="full",
    include_dft_rank_sp=False,
)

wb97.show_stages()[["branch", "stage", "engine", "solvent"]]
```

```text
TS guesses → RMSD prune → GFN-FF Opt [C] → UMA SP → filter → UMA Opt [C]
           → ωB97 preopt [C] → Hessian → OptTS → Freq → chloroform SP
```

The references receive ωB97 calculations. The reported `full` barrier uses
ωB97 TS and reference energies with the ωB97 thermochemistry recipe.
`uma_opt-EE` stays a screening result. `wb97.targets()` and
`wb97.show_stages()` inspect the plan without starting embedding or
calculators; `wb97.run(...)` and `wb97.submit(...)` use the same stage graph.
See [Running Screens](running.md) for execution details.

## UMA As The Final Method

Change both the screening and final method to the same UMA environment:

```python
uma = ft.workflows.catalyst_screen(
    **common,
    screening="uma-alpb-chloroform",
    method="uma-alpb-chloroform",
    level="full",
    include_dft_rank_sp=False,
    ts_refine_n=1,
)
```

```text
TS guesses → RMSD prune → GFN-FF Opt [C] → UMA SP → filter → UMA Opt [C]
           → distinct-candidate prune → select one → release constraints
           → UMA Hessian → OptTS → NumFreq → inspect imaginary mode
```

References use UMA Opt and NumFreq in the **same** gas or ALPB environment.
The final geometry is `uma_ts_opt-oc` for a TS or `uma_min_opt-oc` for a
minimum. `uma_freq-EE`, `uma_freq-GE`, and `uma_freq-vibs` carry the final
electronic energy, Gibbs energy, and modes. This is an UMA result; no DFT
calculation is implied.

To keep several distinct TS candidates through refinement, increase the final
cutoff:

```python
several = ft.workflows.catalyst_screen(
    **common,
    screening="uma-alpb-chloroform",
    method="uma-alpb-chloroform",
    level="full",
    include_dft_rank_sp=False,
    ts_refine_n=3,
)
```

`top_n=4` limits the constrained UMA optimization input here.
`ts_refine_n=3` then allows up to three geometrically distinct UMA optimized
TSs into Hessian, OptTS, and NumFreq. A candidate is usable only if its TS
mode and all required reference minima pass quality checks.

!!! info "Gas and ALPB are separate final methods"

    Use `screening="uma-gas", method="uma-gas"` for a gas UMA barrier.
    Use `screening="uma-alpb-chloroform", method="uma-alpb-chloroform"` for
    the corrected potential. Keep their outputs and reference stores separate.

## Gas UMA Geometry With A Terminal Chloroform Single Point

```python
validation = ft.workflows.catalyst_screen(
    **common,
    screening="uma-gas",
    method="uma-gas-opt-alpb-chloroform",
    level="full",
    ts_refine_n=3,
)
validation.show_stages()
```

| Structure | Final stages | Analysis energy | Thermal correction |
| --- | --- | --- | --- |
| TS candidates | Gas Hessian → gas OptTS → gas NumFreq → ALPB(chloroform) SP | `uma_solv_sp-EE` | `uma_freq-GE - uma_freq-EE` |
| Reference minima | Gas Opt → gas NumFreq → ALPB(chloroform) SP | `uma_solv_sp-EE` | `uma_freq-GE - uma_freq-EE` |

This preset uses pinned **UMA-S 1.2.1 OMol** throughout. Its gas optimization
and frequency stages follow the gas-geometry/solvent-SP protocol used by
`r2scan-3c`, with UMA and ALPB replacing DFT and SMD. The terminal single point
evaluates each optimized gas geometry without further optimization:

```text
G = E[UMA + ALPB(chloroform)] + (G[UMA gas frequencies] - E[UMA gas frequencies])
  = uma_solv_sp-EE + (uma_freq-GE - uma_freq-EE)
```

For example, solvent electronic energy −100.020 Hartree and gas-frequency
energies G = −99.950, E = −100.000 Hartree give G = −99.970 Hartree.
The same assembly applies to the substrate (`ligand`), each requested catalyst
dimer alternative, HBpin, and H₂ references. Barrier analysis then subtracts
the required reference free energies with their stoichiometric coefficients.
Optimized coordinates remain `uma_ts_opt-oc` or `uma_min_opt-oc`; gas
`uma_freq-vibs` supplies frequency and imaginary-mode characterization.

!!! info "Keep each protocol in its own run directory"

    Use gas screening with this mixed preset. `uma-gas` continues to use gas
    frequency Gibbs energies; `uma-alpb-chloroform` continues to optimize and
    characterize on the ALPB potential. Full UMA currently supports
    `scope="barriers"`. Inspect imaginary modes before accepting TS barriers.

To compare solvent UMA SP energies on **g-xTB geometries**, keep the existing
screening-only reranking path in a separate run:

```python
comparison = ft.workflows.catalyst_screen(
    **common,
    screening="gxtb-default",
    ranking="uma-alpb-chloroform",
    level="uma_ranked",
    uma_rank_top_n=3,
)
```

This comparison uses `uma_rank_sp-EE` on `xtb_opt-oc` and reports electronic
barriers (ΔE). It does not supply the gas UMA thermal correction needed for ΔG.

## Rerank g-xTB Geometries With UMA SP

Use a broad g-xTB cutoff before a smaller UMA SP cutoff:

```python
hybrid = ft.workflows.catalyst_screen(
    **{**common, "top_n": 8},
    screening="gxtb-default",
    ranking="uma-gas",
    method="wb97xd3-631g",
    level="full",
    include_dft_rank_sp=False,
    uma_rank_top_n=2,
)
```

```text
TS guesses → RMSD prune → GFN-FF Opt [C] → g-xTB SP → broad filter
           → g-xTB Opt [C] → UMA SP on g-xTB geometries → rerank
           → ωB97 preopt [C] → Hessian → OptTS → Freq → chloroform SP
```

Here `top_n=8` advances up to eight g-xTB candidates to UMA SP;
`uma_rank_top_n=2` advances up to two of those to ωB97 refinement. There is
**no UMA optimization** in this path. `xtb_opt-oc` is the geometry evaluated
by `uma_rank_sp-EE`; `dft_ts_opt-oc` is the final TS geometry. The final
barrier uses ωB97 TS and reference calculations.

## Read The Results

Run one chosen workflow in a fresh directory and inspect its portable result:

```python
run = uma.run(out_dir="runs/uma-alpb-ts1", n_cores=4, mem_gb=20)

run.available_analysis_levels()
# ('low_cost', 'full')

run.candidate_barriers()[[
    "ts_cid", "selected", "delta_e_kcal_mol", "delta_g_kcal_mol",
    "quality_status", "quality_issues",
]]
```

`run.barriers()` shows the selected barrier per TS target;
`run.candidate_barriers()` shows **every retained full UMA TS candidate**.
One N-methylpyrrole TS1 check produced these method-specific results:

| Full result | Final energy and geometry | ΔE‡ | ΔG‡ | Quality |
| --- | --- | ---: | ---: | --- |
| Gas UMA | `uma_freq-EE`; `uma_ts_opt-oc` | 25.15 | 30.66 | `invalid`: ligand minimum has one imaginary mode |
| UMA + ALPB(chloroform) | `uma_freq-EE`; `uma_ts_opt-oc` | 19.70 | 26.04 | `ready` after TS mode review |
| ωB97 + SMD(chloroform), seeded from the selected gas UMA TS | `dft_solv_sp-EE`; `dft_ts_opt-oc` | 26.08 | 32.21 | `ready` after TS mode review |

Energies are in kcal/mol. The gas UMA and ωB97 rows are a paired candidate
comparison, but the gas UMA barrier remains unusable because of its ligand
reference. The ALPB row is a separate run. These values show how to read the
output; they do not establish general method accuracy.

| Saved tier | Meaning | Example stage columns |
| --- | --- | --- |
| `low_cost` | Constrained screening: UMA for choices 1, 2, and 4; g-xTB for choice 3 | `uma_sp-EE`, `uma_opt-EE`, `uma_opt-oc` **or** `xtb_sp-EE`, `xtb_opt-EE`, `xtb_opt-oc` |
| `uma_ranked` | UMA SP ranking of g-xTB optimized geometries in choice 3 | `uma_rank_sp-EE`, with geometry from `xtb_opt-oc` |
| `dft_ranked` | Optional DFT single-point ranking when `include_dft_rank_sp=True` in a compatible ωB97 path | `dft_rank_sp-EE` |
| `full` | Final TS and references at the chosen method | UMA: `uma_ts_opt-oc`, `uma_freq-EE`, `uma_freq-GE`; ωB97: `dft_ts_opt-oc`, `dft_freq-GE`, `dft_solv_sp-EE` |

`*-EE` values are electronic energies in hartree; `*-oc` values are Cartesian
geometries. A numerical ΔG‡ requires frequency thermochemistry for both the
TS and its references. An electronic-only tier cannot supply one.

| `quality_status` | Meaning |
| --- | --- |
| `ready` | Required calculations, frequencies, reference minima, and TS mode review pass. |
| `review` | A numerical result exists, but a TS mode or other flagged result needs scientific inspection. |
| `invalid` | A TS or reference fails a required quality check; any displayed numerical barrier is diagnostic only. |
| `incomplete` | A required calculation, reference, or thermal quantity is missing. |

Inspect the state table when a barrier is `incomplete` or `invalid`:

```python
run.states()[[
    "state_id", "frequency_gibbs_energy_hartree",
    "n_imag", "quality_status", "quality_issues",
]]
```

A missing `frequency_gibbs_energy_hartree` for a required state prevents a
thermal ΔG‡. A minimum with `n_imag=1` makes its dependent barrier `invalid`.

The gas UMA ligand in the table had an imaginary frequency at −66.53 cm⁻¹.
Its numerical barrier should not be used for prediction.

Inspect and review a TS mode before accepting its barrier:

```python
queue = run.review_queue()
queue[["result_id", "state_id", "imaginary_frequencies_cm1"]]

result_id = queue.iloc[0]["result_id"]
run.plot_vibration(result_id, mode=0)

# Record this only after confirming the intended transfer motion:
run.set_review(result_id, "approved", note="Mode follows N-H to substrate C-H transfer.")
```

See [Vibrations](../visualization/vibrations.md) for visual inspection.
`ready` is not implied merely by one imaginary frequency; the mode must
follow the intended chemistry. Reference minima require **zero** imaginary
modes.

## Compare A Selected UMA Candidate With ωB97

This optional follow-on starts from a completed full UMA run and recalculates
the selected candidate and required references with ωB97:

```python
parent = ft.screen.open_run("runs/uma-alpb-ts1")
comparison_wf = ft.workflows.wb97_comparison(parent, candidates="selected")
comparison = comparison_wf.run(out_dir="runs/uma-alpb-ts1-wb97")

paired = comparison.method_comparison()
paired[[
    "parent_uma_result_id", "match_status",
    "uma_delta_g_kcal_mol", "uma_quality_status",
    "wb97_delta_g_kcal_mol", "wb97_quality_status",
]]
```

`match_status="matched"` confirms candidate identity, atom composition, and
barrier reference definition agree. Each method's `quality_status` still
decides whether its own barrier is usable. The ωB97 result does not borrow UMA
energies or references. A pair can be `matched` while
`uma_quality_status="invalid"` and `wb97_quality_status="ready"`.
UMA ALPB and ωB97 SMD use different solvent treatments, so a numerical
difference between them does not isolate the electronic method alone.

## Choose The Guess Profile Separately

`screening` selects the screening energy and gradient; `method` selects the
final energy and thermochemistry; `spec_profile` sets initial TS geometry and
reactive-core constraints. The examples use the reviewed UMA **gas** guess
profile even when the method uses ALPB(chloroform).

!!! note "No reviewed UMA ALPB geometry profile yet"

    `spec_profile="omol-uma-s-1p2p1/gas"` is a gas geometry profile. It is
    an explicit choice, not an ALPB-specific guess. Use
    `spec_match="exact"` to prevent an unintended fallback. The reviewed
    `wb97xd3-631g/gas` profile is another option for ωB97 final paths.

The UMA ALPB potential adds a GFN2-xTB ALPB(chloroform) minus gas correction
to both UMA energy and gradient for SP, optimization, Hessian displacements,
and frequency calculations. See [UMA With FRUST](../external-tools/uma.md)
for the formula and lower-level controls.
