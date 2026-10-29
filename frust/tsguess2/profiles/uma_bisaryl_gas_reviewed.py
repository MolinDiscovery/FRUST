"""Active revision 2 of the reviewed native TMP bisaryl UMA gas geometries.

Reference identity, native regeneration and transferred local connections were
reviewed before publication. The original candidate revision remains unchanged.
These construction references do not assign a mechanism to another system.
Direct single-saddle chemistry and conformer-interconversion kinetics remain
unvalidated; target results still require their own chemical and pathway review.
"""

from dataclasses import replace

from frust.tsguess2.profiles.uma_bisaryl_gas import (
    GEOMETRIES as CANDIDATE_GEOMETRIES,
    GEOMETRY_KEY,
)


_PUBLICATION_NOTES = (
    " Active revision 2 publishes the unchanged native geometry after reviewed "
    "final-profile regeneration and independent transferred TS-family connections. "
    "INT6 regeneration required eight starts, three characterized candidates and "
    "identity selection; lower-energy products and the original small-ensemble "
    "failure remain excluded. Validated exchange examples have two local saddles "
    "and distinct bridge conformers. Direct TS7-only/TS8-only chemistry, bridge "
    "interconversion and target-system selectivity are not established."
)

GEOMETRIES = {
    state: replace(
        geometry,
        revision=2,
        status="active",
        reference=replace(
            geometry.reference,
            notes="Historical candidate record: " + geometry.reference.notes
            + _PUBLICATION_NOTES,
        ),
    )
    for state, geometry in CANDIDATE_GEOMETRIES.items()
}
