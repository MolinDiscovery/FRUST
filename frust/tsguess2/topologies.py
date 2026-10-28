"""Method-independent construction topologies for ``tsguess2``."""

from __future__ import annotations

from frust.tsguess2.models import ConstraintDef, CoreTopology


BISARYL_CH_TOPOLOGY = CoreTopology(
    state="TS5",
    builder_key="bisaryl_adduct",
    core_smarts="[#5]~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]1~[#7]",
    constraints=(
        ConstraintDef("incomingC_transferH", "distance", ("incoming_C", "N_transfer_H")),
        ConstraintDef("catN_transferH", "distance", ("cat_N", "N_transfer_H")),
        ConstraintDef("catB_incomingC", "distance", ("cat_B", "incoming_C")),
        ConstraintDef("catB_boundC", "distance", ("cat_B", "bound_C")),
        ConstraintDef("catB_retainedH", "distance", ("cat_B", "B_transfer_H")),
        ConstraintDef("catB_catN", "distance", ("cat_B", "cat_N")),
    ),
)


BISARYL_H2_TOPOLOGY = CoreTopology(
    state="TS6",
    builder_key="bisaryl_h2",
    core_smarts="[#5]~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]1~[#7]",
    constraints=(
        ConstraintDef("catN_NtransferH", "distance", ("cat_N", "N_transfer_H")),
        ConstraintDef("catB_BtransferH", "distance", ("cat_B", "B_transfer_H")),
        ConstraintDef("eliminating_HH", "distance", ("N_transfer_H", "B_transfer_H")),
        ConstraintDef("catB_catN", "distance", ("cat_B", "cat_N")),
        ConstraintDef("catB_boundC", "distance", ("cat_B", "bound_C")),
        ConstraintDef("catB_incomingC", "distance", ("cat_B", "incoming_C")),
    ),
)


BISARYL_EXCHANGE_TOPOLOGIES = {
    state: CoreTopology(
        state=state,
        builder_key="bisaryl_exchange",
        core_smarts="[#5]~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]1~[#7]",
        constraints=(
            ConstraintDef("catB_transferH", "distance", ("cat_B", "transfer_H")),
            ConstraintDef("pinB_transferH", "distance", ("pin_B", "transfer_H")),
            ConstraintDef("catB_exchangingC", "distance", ("cat_B", "exchanging_C")),
            ConstraintDef("pinB_exchangingC", "distance", ("pin_B", "exchanging_C")),
            ConstraintDef("catB_retainedC", "distance", ("cat_B", "retained_C")),
            ConstraintDef("catB_pinB", "distance", ("cat_B", "pin_B")),
            ConstraintDef("catB_catN", "distance", ("cat_B", "cat_N")),
        ),
    ) for state in ("TS7", "INT6", "TS8")
}


_HBPIN_BRIDGED_CORE_SMARTS = (
    "[#5;!$([#5](~[#8])~[#8])]~[#1]~"
    "[#5;$([#5](~[#8])~[#8])]~[#6]"
)


CORE_TOPOLOGIES: dict[str, CoreTopology] = {
    "TS1": CoreTopology(
        state="TS1",
        builder_key="ts1_ts2",
        core_smarts="[#1]~[#7]~[*]~[*]~[#5]~[#6]",
        constraints=(
            ConstraintDef("catN_transferH", "distance", ("cat_N", "transfer_H")),
            ConstraintDef(
                "transferH_substrateC",
                "distance",
                ("transfer_H", "substrate_C"),
            ),
            ConstraintDef("catB_substrateC", "distance", ("cat_B", "substrate_C")),
            ConstraintDef("catB_catN", "distance", ("cat_B", "cat_N")),
        ),
    ),
    "TS2": CoreTopology(
        state="TS2",
        builder_key="ts1_ts2",
        core_smarts="[#1]~[#7]~[*]~[*]~[#5]~[#6]",
        constraints=(
            ConstraintDef(
                "catB_BtransferH",
                "distance",
                ("cat_B", "B_transfer_H"),
            ),
            ConstraintDef(
                "catN_NtransferH",
                "distance",
                ("cat_N", "N_transfer_H"),
            ),
            ConstraintDef("catB_catN", "distance", ("cat_B", "cat_N")),
            ConstraintDef(
                "catB_BtransferH_catN",
                "angle",
                ("cat_B", "B_transfer_H", "cat_N"),
            ),
        ),
    ),
    "TS3": CoreTopology(
        state="TS3",
        builder_key="ts3_ts4",
        core_smarts=_HBPIN_BRIDGED_CORE_SMARTS,
        constraints=(
            ConstraintDef("transferH_catB", "distance", ("transfer_H", "cat_B")),
            ConstraintDef("transferH_pinB", "distance", ("transfer_H", "pin_B")),
            ConstraintDef(
                "transferH_substrateC",
                "distance",
                ("transfer_H", "substrate_C"),
            ),
            ConstraintDef("catB_substrateC", "distance", ("cat_B", "substrate_C")),
            ConstraintDef("pinB_substrateC", "distance", ("pin_B", "substrate_C")),
            ConstraintDef("pinB_catB", "distance", ("pin_B", "cat_B")),
            ConstraintDef(
                "catB_transferH_pinB",
                "angle",
                ("cat_B", "transfer_H", "pin_B"),
            ),
            ConstraintDef(
                "catB_substrateC_pinB",
                "angle",
                ("cat_B", "substrate_C", "pin_B"),
            ),
        ),
    ),
    "TS4": CoreTopology(
        state="TS4",
        builder_key="ts3_ts4",
        core_smarts=_HBPIN_BRIDGED_CORE_SMARTS,
        constraints=(
            ConstraintDef("catB_pinB", "distance", ("cat_B", "pin_B")),
            ConstraintDef("pinB_transferH", "distance", ("pin_B", "transfer_H")),
            ConstraintDef(
                "substrateC_transferH",
                "distance",
                ("substrate_C", "transfer_H"),
            ),
            ConstraintDef("catB_transferH", "distance", ("cat_B", "transfer_H")),
            ConstraintDef("catB_substrateC", "distance", ("cat_B", "substrate_C")),
            ConstraintDef("pinB_substrateC", "distance", ("pin_B", "substrate_C")),
            ConstraintDef(
                "catB_transferH_pinB",
                "angle",
                ("cat_B", "transfer_H", "pin_B"),
            ),
            ConstraintDef(
                "catB_substrateC_pinB",
                "angle",
                ("cat_B", "substrate_C", "pin_B"),
            ),
        ),
    ),
    "INT3": CoreTopology(
        state="INT3",
        builder_key="ts3_ts4",
        core_smarts=_HBPIN_BRIDGED_CORE_SMARTS,
        constraints=(
            ConstraintDef("catB_transferH", "distance", ("cat_B", "transfer_H")),
            ConstraintDef("catB_substrateC", "distance", ("cat_B", "substrate_C")),
            ConstraintDef("pinB_transferH", "distance", ("pin_B", "transfer_H")),
            ConstraintDef("pinB_substrateC", "distance", ("pin_B", "substrate_C")),
            ConstraintDef(
                "catB_transferH_pinB",
                "angle",
                ("cat_B", "transfer_H", "pin_B"),
            ),
            ConstraintDef(
                "catB_substrateC_pinB",
                "angle",
                ("cat_B", "substrate_C", "pin_B"),
            ),
        ),
    ),
}
