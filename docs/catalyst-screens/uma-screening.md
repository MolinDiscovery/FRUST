# UMA Screening And ωB97 Validation

Start with one substrate, one catalyst, and one TS type. After inspecting the
first result, increase the conformer count or add more TS types.

```csv
role,smiles,compound_name,rpos
substrate,CN1C=CC=C1,n_methyl_pyrrole,2
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,
```

Save those rows as `screen.csv`, then construct an ALPB(chloroform) UMA screen:

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    ts_types=["TS1"],
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="low_cost",
    spec_profile="wb97xd3-631g/gas",
    spec_match="exact",
    scope="barriers",
    dimer_reference="dimer",
    n_confs=2,
    top_n=1,
)

wf.show_stages()[["branch", "stage", "engine", "solvent"]]
```

The TS screening path is:

```text
TS guess → initial MOI/RMSD pruning → xtb_preopt (GFN-FF)
         → uma_sp → uma_sp_filter → uma_opt
```

| Stage | Action | Saved result |
| --- | --- | --- |
| `xtb_preopt` | GFN-FF preoptimization; reactive-core constraints apply to TSs and INT3 | `xtb_preopt-EE`, `xtb_preopt-oc` |
| `uma_sp` | Evaluate candidate geometries with UMA-S 1.2.1 (OMol) plus GFN2-xTB ALPB(chloroform) correction | `uma_sp-EE` |
| `uma_sp_filter` | Keep the lowest `top_n` UMA single-point rows per structure | Filtering metadata in `ft.show_steps(df)` |
| `uma_opt` | Optimize retained geometries with the same corrected UMA potential; TSs and INT3 remain constrained | `uma_opt-EE`, `uma_opt-oc` |

`*-EE` is an electronic energy in hartree; `*-oc` contains optimized Cartesian
coordinates. For example, one TS1 row ended with `uma_sp-EE =
-915.0120606762` and `uma_opt-EE = -915.0312486223` Eh. The final `uma_opt-oc`
array is the geometry passed to the next stage. Molecule references run without
reactive-core constraints.

To use gas-phase UMA, change only the screening choice:

```python
gas_wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    ts_types=["TS1"],
    screening="uma-gas",
    method="wb97xd3-631g",
    level="low_cost",
    spec_profile="wb97xd3-631g/gas",
    spec_match="exact",
    scope="barriers",
    dimer_reference="dimer",
    n_confs=2,
    top_n=1,
)
```

The gas workflow has the same stage names and output columns; its UMA calls
omit the xTB ALPB correction. Keep gas and ALPB runs in separate directories.

## Choose The Saved Level

| `level` | After `uma_opt` | Analysis energy | Saved tiers |
| --- | --- | --- | --- |
| `low_cost` | Stop | `uma_opt-EE` | `low_cost` only; electronic ΔE, no ΔG |
| `dft_ranked` | ωB97 ranking single point and filter | `dft_rank_sp-EE` | `low_cost`, `dft_ranked`; electronic ΔE, no ΔG |
| `full`, default | ωB97 optimization, TS optimization, frequencies, final SMD(chloroform) single point | `dft_solv_sp-EE`; gas-frequency thermal correction supplies G | `low_cost`, `full` |
| `full`, `include_dft_rank_sp=True` | ωB97 ranking single point before full validation | Final `dft_solv_sp-EE` | `low_cost`, `dft_ranked`, `full` |

`dft_ranked` always runs the ranking single point. A `full` UMA run skips it by
default; pass `include_dft_rank_sp=True` when you want that extra selection and
its independently saved result tier:

```python
full = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    ts_types=["TS1"],
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="full",
    include_dft_rank_sp=False,  # default for UMA
    spec_profile="wb97xd3-631g/gas",
    spec_match="exact",
    scope="barriers",
    dimer_reference="dimer",
    n_confs=2,
    top_n=1,
)

with_ranking = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    ts_types=["TS1"],
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="full",
    include_dft_rank_sp=True,
    spec_profile="wb97xd3-631g/gas",
    spec_match="exact",
    scope="barriers",
    dimer_reference="dimer",
    n_confs=2,
    top_n=1,
)
```

For the default full path, TS1 continues through `dft_preopt`, `dft_hessian`,
`dft_ts_opt`, `dft_freq`, and `dft_solv_sp`. Minimum references use `dft_opt`,
`dft_freq`, and `dft_solv_sp`. The ranking-enabled path inserts
`dft_rank_sp` and `dft_rank_filter` before refinement. The saved `low_cost`
winner is an actual UMA checkpoint; FRUST does not reconstruct it from the
later ωB97 winner.

## Choose The Guess Profile Separately

The `screening` choice sets the UMA energy and gradient. `spec_profile` sets
the initial TS geometry and numerical constraints. The examples above use the
reviewed **ωB97 gas** profile because ωB97 performs final validation. To try
the reviewed **UMA gas** profile, set:

```python
uma_guess = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    ts_types=["TS1"],
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="full",
    spec_profile="omol-uma-s-1p2p1/gas",
    spec_match="exact",
    scope="barriers",
    dimer_reference="dimer",
    n_confs=2,
    top_n=1,
)
```

!!! note "No reviewed UMA ALPB geometry profile yet"

    The second example still screens with ALPB(chloroform), but its initial
    geometry and constraints come from **gas-phase** UMA. `spec_match="exact"`
    prevents an unintended profile fallback. The run records the chosen
    `guess_profile` independently of the screening potential. Neither this
    profile nor the ωB97 gas profile is an ALPB geometry reference.

## Run, Inspect, And Review

Run a small calculation in a fresh output directory:

```python
run = full.run(out_dir="runs/uma-ts1-full", n_cores=10, mem_gb=20)

run.available_analysis_levels()
# ('low_cost', 'full')

run.barriers(level="low_cost")[["ts_type", "delta_e_kcal_mol", "quality_status"]]
run.barriers(level="full")[[
    "ts_type", "delta_e_kcal_mol", "delta_g_kcal_mol",
    "delta_g_corrected_kcal_mol", "quality_status",
]]
```

An analyzed N-methylpyrrole result with valid reference minima can look like
this:

| TS1 result | Barrier (kcal/mol) | Quality |
| --- | ---: | --- |
| UMA + ALPB `low_cost` ΔE‡ | 19.77 | `ready` |
| ωB97 + SMD `full` ΔE‡ | 19.65 | `review` |
| ωB97 + SMD `full` ΔG‡ | 25.43 | `review` |
| Corrected ωB97 ΔG‡ (−1.89 kcal/mol correction) | 23.54 | `review` |

A `review` barrier has a numerical value, but its TS mode still needs inspection
before the barrier is accepted. These values illustrate one substrate/catalyst
case; they do not establish method agreement in general. UMA uses an xTB ALPB
correction while ωB97 uses SMD, so the solvent treatments also differ.

```python
queue = run.review_queue()
queue[["result_id", "state_id", "n_imag", "vibration_flags"]]

# After inspecting the TS geometry and animated imaginary mode:
result_id = queue.iloc[0]["result_id"]
run.plot_vibration(result_id)
```

Use [Vibrations](../visualization/vibrations.md) to inspect the motion before
recording a review decision. The common result layout, portable analysis, and
cluster submission steps are described in
[End-To-End Calculation And Analysis](end-to-end.md). The lower-level
`Stepper.orca(...)` controls and solvent formula are in
[UMA With FRUST](../external-tools/uma.md).
