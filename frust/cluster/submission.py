"""Internal, calculation-free planning and durable submission bookkeeping."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4

from frust.cluster.config import Resources, SubmissionRecord


@dataclass(frozen=True)
class PlannedGroup:
    name: str
    resources: Resources
    output_name: str
    batches: tuple[tuple[str, ...], ...]
    parallelism: int | None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer (not a boolean)")
    return value


def _plan_submission(
    tags: Sequence[str],
    groups: Sequence[tuple[str, Resources, str]],
    *,
    execution: str,
    array: bool = False,
    array_parallelism: int | Mapping[str, int] | None = None,
    targets_per_task: int = 1,
) -> tuple[PlannedGroup, ...]:
    """Validate scheduler options and group lightweight target identities."""
    if not isinstance(array, bool):
        raise ValueError("array must be a boolean")
    _positive_integer(targets_per_task, "targets_per_task")
    names = [name for name, _, _ in groups]
    if len(names) != len(set(names)):
        raise ValueError("Stage-group names must be unique")
    if len(tags) != len(set(tags)):
        raise ValueError("Selected targets have duplicate output identities")
    for tag in tags:
        if not tag or tag in {".", ".."} or Path(tag).name != tag or "\\" in tag:
            raise ValueError(f"Target tag must be a single output directory name: {tag!r}")
    limits = dict.fromkeys(names)
    if not array:
        if array_parallelism is not None or targets_per_task != 1:
            raise ValueError("array_parallelism and target batching require array=True")
    else:
        if execution != "single_job" and targets_per_task != 1:
            raise ValueError("Staged arrays require targets_per_task=1")
        if isinstance(array_parallelism, Mapping):
            if execution == "single_job":
                raise ValueError("single_job requires a scalar array_parallelism")
            if set(array_parallelism) != set(names):
                raise ValueError(f"array_parallelism must specify exactly these groups: {names}")
            limits = {
                name: _positive_integer(array_parallelism[name], f"array_parallelism[{name!r}]")
                for name in names
            }
        else:
            limit = _positive_integer(array_parallelism, "array_parallelism")
            limits = dict.fromkeys(names, limit)
    batches = tuple(tuple(tags[i:i + targets_per_task]) for i in range(0, len(tags), targets_per_task))
    return tuple(
        PlannedGroup(name, resources, output, batches, limits[name])
        for name, resources, output in groups
    )


def _validate_array_scheduler_options(cluster):
    extra = cluster.extra_slurm_parameters or {}
    conflicts = {str(key).lstrip("-").replace("_", "-") for key in extra}
    conflicts &= {"array", "dependency", "array-parallelism", "map-count"}
    if conflicts:
        raise ValueError(f"FRUST manages array and dependency options; remove: {sorted(conflicts)}")


def _validate_array_size(cluster, size):
    if cluster.max_array_size is not None:
        _positive_integer(cluster.max_array_size, "cluster.max_array_size")
        if size > cluster.max_array_size:
            raise ValueError(
                f"Array has {size} elements, exceeding cluster.max_array_size={cluster.max_array_size}. "
                "Select fewer targets with targets=; FRUST does not split arrays automatically."
            )


def _submit_array_jobs(
    executor, cluster, fn, arguments, *, parallelism, on_submitted,
    on_array_submitted=None,
):
    """Submit identical workers with Slurm or bounded local concurrency.

    Local submission waits for a free slot before launching the next worker;
    callers can therefore block until earlier workers finish. Failure results
    do not stop submission of independent targets.
    """
    if not arguments:
        return []
    if cluster.backend == "slurm":
        executor.update_parameters(slurm_array_parallelism=parallelism)
        try:
            jobs = executor.map_array(fn, *zip(*arguments))
        except Exception as error:
            raise RuntimeError(
                "Slurm array submission failed. Check the scheduler for accepted jobs before retrying; "
                "check MaxArraySize and submission limits, or select fewer targets with targets=. "
                "FRUST does not split arrays or fall back to individual submission."
            ) from error
        if len(jobs) != len(arguments):
            raise RuntimeError("Submitit returned an unexpected number of array jobs")
        if on_array_submitted is not None:
            on_array_submitted(jobs)
        else:
            for index, job in enumerate(jobs):
                on_submitted(index, job)
        return jobs
    if cluster.backend != "local":
        raise ValueError("cluster.backend must be either 'slurm' or 'local'")
    jobs = []
    active = []
    for index, args in enumerate(arguments):
        while len(active) >= parallelism:
            active = [job for job in active if not job.done()]
            if len(active) >= parallelism:
                time.sleep(0.05)
        job = executor.submit(fn, *args)
        on_submitted(index, job)
        jobs.append(job)
        active.append(job)
    return jobs


class SubmissionLedger:
    """Persist each accepted job before attempting another submission."""

    def __init__(self, root, groups, *, mode, backend, array):
        self.attempt_id = uuid4().hex
        self.path = Path(root) / ".frust" / "submissions" / f"{self.attempt_id}.json"
        self.mode = mode
        self.backend = backend
        self.array = array
        self.groups = groups
        self.collection_job_id = None
        self.records = [
            SubmissionRecord(
                target=tag, group=group.name, batch_index=index, batch_targets=batch,
                save_dir=str(Path(root) / tag),
                output_path=str(Path(root) / tag / group.output_name),
                attempt_id=self.attempt_id,
            )
            for group in groups
            for index, batch in enumerate(group.batches)
            for tag in batch
        ]
        self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "attempt_id": self.attempt_id,
            "mode": self.mode, "backend": self.backend, "array": self.array,
            "groups": [asdict(group) for group in self.groups],
            "records": [asdict(record) for record in self.records],
            "collection_job_id": self.collection_job_id,
        }
        _atomic_write_submission_json(self.path, payload)

    def submitted(self, group, index, job):
        self.submitted_many(group, {index: job})

    def submitted_many(self, group, jobs):
        updates = {}
        for index, job in jobs.items():
            job_id = job.job_id
            match = re.fullmatch(r"(\d+)_(\d+)", str(job_id)) if self.backend == "slurm" else None
            updates[index] = dict(
                status="submitted", job_id=job_id,
                array_job_id=match[1] if match else None,
                array_index=int(match[2]) if match else None,
            )
        self.records = [
            replace(record, **updates[record.batch_index])
            if record.group == group and record.batch_index in updates else record
            for record in self.records
        ]
        self._write()

    def failed(self, group, indices, error):
        self.records = [
            replace(record, status="submission_failed", error=f"{type(error).__name__}: {error}")
            if record.group == group and record.batch_index in indices and record.status == "planned"
            else record
            for record in self.records
        ]
        self._write()

    def submit(self, group, index, executor, fn, *args):
        try:
            job = executor.submit(fn, *args)
        except Exception as error:
            self.failed(group, [index], error)
            raise
        self.submitted(group, index, job)
        return job

    def collected(self, job):
        self.collection_job_id = job.job_id
        self._write()


def _atomic_write_submission_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
