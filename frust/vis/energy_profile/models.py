from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnergyState:
    """Normalized energy-profile state.

    Parameters
    ----------
    label
        State label shown on the x-axis or point annotation.
    energy
        Relative energy for the state.
    placement
        Optional placement token such as ``"t"``, ``"bottom"``, or
        ``"ttrr"``.
    """

    label: Any
    energy: float
    placement: Any = None


@dataclass(frozen=True)
class ParsedProfile:
    """Parsed profile entries and side-reaction metadata."""

    entries: list[tuple[Any, float, Any]]
    segment_ids: list[int]
    side_anchor_label: str | None
    side_connector_rise_frac: float | None
    side_legend_label: str | None
    main_to_product_anchor_idx: int | None
    main_to_product_drop_frac: float | None
    no_product: bool


@dataclass(frozen=True)
class ProfileStyle:
    """Resolved Matplotlib style for a profile."""

    main_color: Any
    side_color: Any
    marker: str
    alpha: float
    linewidth: float
