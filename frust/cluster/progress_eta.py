"""Estimate the time to the last calculation worker finishing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from heapq import heapify, heapreplace
import json
from pathlib import Path
import re

from frust.cluster.naming import sanitize_tag


_LOOKBACK = 2 * 60 * 60
_MIN_TIMINGS = 10
_DRAWS = 256


@dataclass(frozen=True)
class _Target:
    tag: str
    status: str | None
    start: float | None
    end: float | None


@dataclass(frozen=True)
class _Batch:
    attempt: str
    job_id: str | None
    targets: tuple[_Target, ...]
    start: float | None
    end: float | None
    finished: bool


@dataclass(frozen=True)
class _Estimate:
    seconds: float | None = None
    reason: str | None = None


def _timestamp(value: str | None) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp.timestamp() if stamp.tzinfo is not None else None


@lru_cache(maxsize=4)
def _load_molecular_sizes(path: Path, modified_ns: int, length: int) -> tuple[dict, dict]:
    from rdkit import Chem

    @lru_cache(maxsize=None)
    def atom_count(smiles: str | None) -> int | None:
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
        return Chem.AddHs(molecule).GetNumAtoms() if molecule is not None else None

    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        return {}, {}
    components = {
        sanitize_tag(row["compound_name"]): atom_count(row.get("smiles"))
        for row in manifest.get("components") or [] if row.get("compound_name")
    }
    systems = {}
    for row in manifest.get("systems") or []:
        substrate = components.get(sanitize_tag(row.get("substrate_name") or ""))
        catalyst = components.get(sanitize_tag(row.get("catalyst_name") or ""))
        substrate = substrate or atom_count(row.get("substrate_smiles"))
        catalyst = catalyst or atom_count(row.get("catalyst_smiles"))
        name = row.get("system_name")
        if substrate and catalyst and isinstance(name, str) and name:
            systems[sanitize_tag(name)] = substrate + catalyst
    return components, systems


def _molecular_sizes(path: Path) -> tuple[dict, dict]:
    try:
        info = path.stat()
        return _load_molecular_sizes(path, info.st_mtime_ns, info.st_size)
    except (OSError, ValueError):
        return {}, {}


def _target_features(tag: str, components: dict, systems: dict) -> tuple[str, int | None]:
    kind, _, suffix = tag.partition("__")
    if kind.startswith("dimer"):
        atoms = components.get(suffix.split("__pair_", 1)[0])
        return kind, 2 * atoms if atoms else None
    if kind == "ligand":
        return kind, components.get(suffix)
    return kind, systems.get(re.sub(r"__r-?\d+$", "", suffix))


def _worker_slots(history: list[_Batch], active: list[_Batch], now: float) -> int:
    lower = max(now - _LOOKBACK, min(batch.start for batch in history if batch.start is not None))
    if lower >= now:
        return len(active)
    observed = sum(max(0.0, min(batch.end, now) - max(batch.start, lower)) for batch in history
                   if batch.start is not None and batch.end is not None)
    observed += sum(now - max(batch.start, lower) for batch in active)
    return max(len(active), round(observed / (now - lower)))


def _schedule(active: list[float], queued: list[float], slots: int) -> float:
    workers = active + [0.0] * (slots - len(active))
    heapify(workers)
    for duration in queued:
        heapreplace(workers, workers[0] + duration)
    return max(workers)


def _estimate_worker_time(
    history: list[_Batch],
    remaining: list[_Batch],
    manifest: Path,
    now: float,
    *,
    references: bool,
) -> _Estimate:
    active = [batch for batch in remaining if batch.start is not None]
    if not active:
        return _Estimate(reason="waiting for active worker timings")
    observed = [target for batch in history for target in batch.targets]
    dimers = references and sum(t.tag.startswith("dimer") for t in observed) > len(observed) / 2
    training = [t for t in observed if t.status == "success" and t.start is not None
                and t.end is not None and t.start < t.end <= now
                and (not dimers or t.tag.startswith("dimer"))]
    if len(training) < _MIN_TIMINGS:
        return _Estimate(reason="waiting for more completed target timings")
    components, systems = _molecular_sizes(manifest)
    timed = [(target, *_target_features(target.tag, components, systems)) for target in training]
    matched = [(target, kind, size) for target, kind, size in timed if size is not None]
    if not matched:
        return _Estimate(reason="comparable molecular sizes are unavailable")
    if len(matched) < _MIN_TIMINGS:
        return _Estimate(reason="waiting for more completed timings with molecular sizes")
    if max(target.end for target, _, _ in matched) < now - _LOOKBACK:
        return _Estimate(reason="waiting for recent target completions")
    sizes = [size for _, _, size in matched]
    low, high = min(sizes), max(sizes)
    kinds = sorted({kind for _, kind, _ in matched})
    targets = [t for batch in remaining for t in batch.targets]
    remaining_features = [_target_features(t.tag, components, systems) for t in targets]
    comparable = sum(kind in kinds and size is not None and low <= size <= high
                     for kind, size in remaining_features)
    if comparable < 0.9 * len(targets):
        return _Estimate(reason="waiting for timings for comparable targets")

    import numpy as np

    center = float(np.median(sizes))

    def feature_row(kind, size):
        size = center if size is None else min(high, max(low, size))
        return [1, (size - center) / 20, *(kind == label for label in kinds[1:])]

    elapsed = np.array([target.end - target.start for target, _, _ in matched])
    train = np.array([feature_row(kind, size) for _, kind, size in matched])
    fit, *_ = np.linalg.lstsq(train, np.log(elapsed), rcond=None)
    residuals = np.exp(np.log(elapsed) - train @ fit)
    typical = np.exp(np.array([feature_row(*entry) for entry in remaining_features]) @ fit)
    rng = np.random.default_rng(22)
    draws = typical * rng.choice(residuals, size=(_DRAWS, len(targets)))
    for column, target in enumerate(targets):
        if target.start is None:
            continue
        age = now - target.start
        residual = typical[column] * residuals - age
        residual = residual[residual > 0]
        if not len(residual):
            return _Estimate(reason="an active target has outlasted comparable timings")
        draws[:, column] = rng.choice(residual, size=_DRAWS)

    overheads = [max(0.0, batch.end - batch.start - sum(t.end - t.start for t in batch.targets))
                 for batch in history if batch.start is not None and batch.end is not None and batch.end <= now
                 and all(t.start is not None and t.end is not None for t in batch.targets)]
    overhead = float(np.median(overheads)) if overheads else 0.0
    live_costs, queued_costs = [], []
    column = 0
    for batch in remaining:
        stop = column + len(batch.targets)
        cost = draws[:, column:stop].sum(axis=1)
        if batch.start is None:
            queued_costs.append(cost + overhead)
        else:
            live_costs.append(cost)
        column = stop

    slots = _worker_slots(history, active, now)
    live_draws = np.asarray(live_costs).T
    queued_draws = np.asarray(queued_costs).T if queued_costs else [[]] * _DRAWS
    seconds = float(np.median([_schedule(list(live), list(queued), slots)
                              for live, queued in zip(live_draws, queued_draws)]))
    return _Estimate(seconds=seconds)
