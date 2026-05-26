"""Diagnostics for generated TS core geometries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def core_metrics(
    coords: Sequence[Sequence[float]],
    roles: Mapping[str, int],
    constraint_spec: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Measure generated role distances and angles.

    Parameters
    ----------
    coords : sequence
        Cartesian coordinates indexed by atom index.
    roles : mapping
        Mapping from role names to atom indices.
    constraint_spec : sequence of mapping
        Row-level constraint entries.

    Returns
    -------
    list of dict
        Measured values and deviations from the reference constraint values.
    """
    metrics: list[dict[str, object]] = []
    for entry in constraint_spec:
        kind = str(entry["kind"])
        role_names = [str(role) for role in entry["roles"]]
        value = float(entry["value"])
        if kind == "distance":
            measured = _distance(coords[roles[role_names[0]]], coords[roles[role_names[1]]])
        elif kind == "angle":
            measured = _angle(
                coords[roles[role_names[0]]],
                coords[roles[role_names[1]]],
                coords[roles[role_names[2]]],
            )
        else:
            continue
        metrics.append(
            {
                "kind": kind,
                "roles": role_names,
                "reference": value,
                "measured": measured,
                "delta": measured - value,
            }
        )
    return metrics


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(tuple(a), tuple(b))


def _angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ba = [float(x) - float(y) for x, y in zip(a, b)]
    bc = [float(x) - float(y) for x, y in zip(c, b)]
    dot = sum(x * y for x, y in zip(ba, bc))
    nba = math.sqrt(sum(x * x for x in ba))
    nbc = math.sqrt(sum(x * x for x in bc))
    cosang = max(-1.0, min(1.0, dot / (nba * nbc)))
    return math.degrees(math.acos(cosang))
