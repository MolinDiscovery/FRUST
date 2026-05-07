# TS Guess Generation

FRUST transition-state workflows start from two pieces of information:

- a ligand or substrate table, usually with a `smiles` column;
- a transition-state template geometry such as `structures/ts1.xyz` or
  `structures/ts2.xyz`.

The template tells FRUST what kind of TS-like structure to build. The ligand
table tells FRUST which substrates and reactive positions should be expanded.

```mermaid
flowchart TD
    A["Ligand table<br/>smiles, names, optional rpos"]
    B["TS template<br/>ts_guess_xyz"]
    C["Reactive-position mapping"]
    D["Template type detection<br/>TS1, TS2, TS3, TS4, INT3"]
    E["Generated TS structures"]
    F["Conformer embedding"]
    G["Initial FRUST DataFrame<br/>atoms, coords_embedded, rpos, cid"]
    H["Stepper or pipeline stages"]

    A --> C
    B --> D
    C --> E
    D --> E
    E --> F --> G --> H
```

!!! warning "Template geometry is chemical input"

    FRUST can generate and screen structures from a TS template, but it cannot
    prove that the template represents the intended reaction. Inspect the final
    imaginary mode before using a barrier.

## Choosing The TS Entry Point

Use `run_ts_per_lig(...)` when one TS template should be applied once per
ligand:

```python
from frust.pipes import run_ts_per_lig

df = run_ts_per_lig(
    ligands,
    ts_guess_xyz="structures/ts1.xyz",
    n_confs=2,
    DFT=False,
)
```

Use `run_ts_per_rpos(...)` when the same ligand can react at multiple positions:

```python
from frust.pipes import run_ts_per_rpos

df = run_ts_per_rpos(
    ligands,
    ts_guess_xyz="structures/ts2.xyz",
    n_confs=2,
    DFT=False,
)
```

!!! tip "Use fewer conformers for wiring checks"

    For a new template or CSV, start with `n_confs=1`, `DFT=False`, and a tiny
    input table. Once the structure generation and reactive-position mapping
    look correct, increase conformer coverage and launch the expensive stages.

## What The Initial DataFrame Contains

After embedding, FRUST builds a dataframe where each row is a generated
structure or conformer. The columns that matter first are:

| Column | Meaning |
| --- | --- |
| `substrate_name` | ligand or substrate identity |
| `structure_type` | TS or intermediate type, for example `TS1` or `INT3` |
| `rpos` | reactive position used for this generated structure |
| `cid` | conformer id |
| `atoms` | element symbols |
| `coords_embedded` | embedded starting coordinates |

!!! example "Check that reactive positions were generated"

    ```python
    df[["substrate_name", "structure_type", "rpos", "cid"]].head()
    df.groupby(["substrate_name", "rpos"], dropna=False).size()
    ```

## What To Inspect Before Running DFT

- Confirm each ligand generated the expected number of reactive-position rows.
- Confirm `rpos` points to the intended atom in the substrate.
- Inspect a few embedded structures before spending ORCA time.
- Check whether the lowest conformer after a cheap optimization is chemically
  sensible.

For the post-run checks, continue with
[Inspecting Results](inspecting-results.md).
