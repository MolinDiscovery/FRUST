"""Shared candidate policy resolution, independent of calculator choice."""

from dataclasses import dataclass
from numbers import Integral


def _positive_count(value, name):
    """Validate a candidate count without silently truncating it."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _resolve_count(
    *, validation_top_n=None, ranking_top_n=None, uma_rank_top_n=None, ts_refine_n=None
):
    """Resolve explicit aliases and reject conflicting candidate instructions."""
    values = {
        name: _positive_count(value, name)
        for name, value in {
            "validation_top_n": validation_top_n,
            "ranking_top_n": ranking_top_n,
            "uma_rank_top_n": uma_rank_top_n,
        }.items()
        if value is not None
    }
    if len(set(values.values())) > 1:
        raise ValueError("validation_top_n, ranking_top_n and uma_rank_top_n disagree")
    if ts_refine_n is not None:
        _positive_count(ts_refine_n, "ts_refine_n")
        if validation_top_n is not None and ts_refine_n != validation_top_n:
            raise ValueError("validation_top_n and ts_refine_n disagree")
    return next(iter(values.values()), None)


@dataclass(frozen=True)
class CandidatePolicy:
    """Resolved selection boundary stored as compact workflow provenance."""

    count: int
    energy_stage: str
    boundary: str
    diversity: bool = False

    def to_dict(self):
        """Return durable, JSON-compatible scientific selection settings."""
        return dict(
            count=self.count,
            energy_stage=self.energy_stage,
            boundary=self.boundary,
            finite_only=True,
            diversity=self.diversity,
        )
