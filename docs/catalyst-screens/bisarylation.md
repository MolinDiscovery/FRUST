# Reviewing The Bisarylation Branch

The second-arylation branch needs its own structures and balanced energy
accounting. For example, H₂ leaves between INT4 and INT5, and HBpin enters the
exchange segment. Comparing those isolated structures without the missing
fragments would compare different atom inventories.

| Saddle | Forward reference | Reverse reference |
| --- | --- | --- |
| TS5, second C–H activation | Monoaryl catalyst + incoming substrate | INT4 adduct |
| TS6, H₂ elimination | INT4 | INT5 + H₂ |
| TS7, two-saddle bridge entry | INT5 + HBpin | INT6 |
| TS8, two-saddle bridge exit | INT6 | Monoaryl catalyst + arylBpin |
| Direct TS7-only or TS8-only exchange | INT5 + HBpin | Monoaryl catalyst + arylBpin |

FRUST cancels common spectators before evaluating a local barrier. Thus the
TS6 forward barrier needs its saddle and INT4, while the TS7 entry barrier also
needs HBpin. A missing dimer reference does not hide a qualified local barrier.
Dimer-referenced saddle elevations remain separate columns with their own quality.

!!! warning "Geometry references and mechanism assignment are separate"

    Bisaryl geometry references currently remain inactive candidates. Native
    construction is available through explicitly reviewed specification mappings;
    the factory does not activate those mappings in the registry. Software support
    for three route shapes does not establish that every shape occurs chemically.

## Construct And Calculate

Start from an expanded system table, as for an ordinary catalyst screen:

| system_name | substrate_smiles | catalyst_smiles | rpos |
| --- | --- | --- | ---: |
| pyrrole_TMP | `CN1C=CC=C1` | `CC1(C)CCCC(C)(C)N1c1ccccc1B` | 2 |

By default the initially bound aryl uses the same substrate and site. Add
`bound_substrate_smiles` and `bound_rpos` when they differ.

The following assumes `systems` is that table and `reviewed_specs` is an explicit
mapping of reviewed TS5–TS8 and INT4–INT6 geometry specifications supplied during
profile maintenance. Inactive registry references cannot substitute for that
mapping. See [geometry-profile maintenance](../development/ts-spec-profiles.md).

```python
import frust as ft

wf = ft.workflows.bisarylation_screen(
    systems=systems,
    method="uma-gas",
    screening="uma-gas",
    spec_profile="uma-gas",
    candidate_specs=reviewed_specs,
    n_confs=8,
    validation_top_n=3,
)
wf.plan()
run = wf.run(out_dir="runs/bisaryl", n_cores=4, mem_gb=16)
```

`plan()` inspects targets without embedding structures or running calculators.
Local execution and cluster submission use the same native target definitions.
All three retained candidates receive requested full characterization. The
same contract applies to capable DFT final plans with an explicitly selected
guess profile. Optional ranking uses the ordinary shared method-plan policy.

Racemic dimer pairing is currently unsupported by this composed workflow; supplied
catalyst configurations and the shared stereoisomer-expansion policy are supported.

## Review Results Before Assigning A Route

Inspect modes and minimum identities first:

```python
run.review_queue()
run.states()[["state_id", "cid", "result_id", "quality_status"]]
```

Use `set_review` for a TS mode or `set_identity_review` for a minimum after
inspecting its chemistry. Approval never overrides failed convergence or
vibration checks. INT4, INT5 and INT6 require explicit identity review.

Then inspect the candidates for a particular route hypothesis:

```python
pool = run.pathway_candidates(
    system_name="pyrrole_TMP", rpos=2, variant="two_saddle"
)
pool[["role", "cid", "result_id", "quality_status", "selected"]]
```

The helper lists every applicable role and candidate, including failed or missing
choices. It does not pick a mechanism. Select the chemically qualified rows using
the actual paths, endpoint minima and source bindings. A resolved decision needs
saved evidence and exact result IDs for each required role:

```python
# reviewed_pool contains one explicitly reviewed result for each required role.
chosen = reviewed_pool.set_index("role")["result_id"].to_dict()
run.set_pathway_review(
    system_name="pyrrole_TMP",
    rpos=2,
    variant="two_saddle",
    result_ids=chosen,
    evidence_files=["reviewed_paths_and_minima.json"],
    note="Both exchange saddles and their adjacent minima were checked.",
)
```

| Variant | Meaning | Exchange states omitted from analysis |
| --- | --- | --- |
| `two_saddle` | TS7 → INT6 → TS8 with reviewed local connections | None |
| `ts7_only` | One directly connecting exchange saddle assigned TS7 | INT6 and TS8 |
| `ts8_only` | One directly connecting exchange saddle assigned TS8 | TS7 and INT6 |
| `unresolved` | No supported complete assignment yet | None implied absent |

Raw calculated candidates remain retained after a route decision. Unused branches
are omitted from that route's profile and barrier tables. A planning hypothesis
does not approve newly calculated rows. Approved prior hypotheses can be reused via
`run.pathway_reviews().attrs["frust_pathway_reviews"]`, but the next run still needs
result-bound review.

Evidence files are copied into the portable run. Changed results, revoked identity
approval, a different method or changed/missing proof files invalidate the decision.
Use `variant="unresolved"` with empty bindings to withdraw a decision explicitly.

## Keep Neighboring Conformers Explicit

TS7 and TS8 can reach chemically identical bridge minima in different conformers.
Bind each saddle's actual neighbor when their result IDs differ:

```python
run.set_pathway_review(
    system_name="pyrrole_TMP", rpos=2, variant="two_saddle",
    result_ids=chosen,
    minimum_result_ids={"TS8": {"INT6": bridge_from_TS8}},
    evidence_files=["reviewed_paths_and_minima.json"],
    note="TS7 and TS8 reach distinct bridge conformers; interconversion is unresolved.",
)
```

The TS8 local barrier now uses that specific minimum. The ordered ledger still uses
its explicitly selected INT6 row. `reference_result_ids`, `product_result_ids` and
`uses_alternate_minimum` expose the distinction in the barrier table. This does not
establish an interconversion barrier or a continuous kinetic path.

## Inspect Local Barriers And The Balanced Ledger

```python
run.barriers()[[
    "ts_type", "reference_state", "delta_g_kcal_mol",
    "reverse_delta_g_kcal_mol", "quality_status",
]]
run.candidate_barriers()
run.profile(quantity="electronic", include_invalid=True)
run.pathway_reviews()
```

`candidate_barriers()` retains every TS result. An alternative exchange saddle
does not inherit the selected candidate's path approval. A recorded dimer choice
can be changed in memory to another available calculated topology through
`profile(dimer_reference="dimer_eight_membered")`.

| Quality | Interpretation |
| --- | --- |
| `ready` | Required numerical, identity and applicable review checks pass |
| `review` | Scientific review or a current pathway decision is still needed |
| `invalid` | Numerical, identity, composition or protocol checks fail |
| `incomplete` | A required result or energy is missing |

Forward, reverse and dimer-elevation quality are separate. A reverse minimum may
be missing even when the forward local barrier is available. Unresolved exchange
energies are withheld. A balanced electronic ledger alone does not establish
selectivity, solvent effects or concentration-dependent rates.
