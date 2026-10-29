"""Effective molecular roles shared by construction, planning and review."""

from __future__ import annotations

import pandas as pd


def effective_bound_role(system, rpos):
    """Resolve a bound molecule and site without changing SMILES atom order."""
    from frust.tsguess.matching import parse_rpos_value

    incoming = str(system["substrate_smiles"])
    bound = system.get("bound_substrate_smiles")
    bound = incoming if bound is None or pd.isna(bound) else str(bound)
    site = system.get("bound_rpos")
    if site is None or pd.isna(site):
        if bound != incoming:
            raise ValueError("bound_rpos is required for a different bound substrate")
        site = rpos
    sites = parse_rpos_value(site, bound)
    if len(sites) != 1:
        raise ValueError("bound_rpos must name exactly one reactive site")
    return bound, int(sites[0])
