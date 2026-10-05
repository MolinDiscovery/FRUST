"""Internal, calculation-free planning and durable submission bookkeeping."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import hashlib
import pickle
import shutil
from contextlib import contextmanager
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
        self.created_ns = time.time_ns()
        self.fingerprints = {}
        self.previous_attempts = {}
        self.archives = {}
        self.log_dir = None
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
            "created_ns": self.created_ns, "fingerprints": self.fingerprints,
            "previous_attempts": self.previous_attempts, "archives": self.archives,
            "log_dir": self.log_dir,
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


def _submission_history(root):
    history = []
    for path in (Path(root) / '.frust/submissions').glob('*.json'):
        payload = json.loads(path.read_text())
        payload['_path'] = str(path)
        history.append(payload)
    return sorted(history, key=lambda item: item.get('created_ns', Path(item['_path']).stat().st_mtime_ns))


def _latest_targets(root, attempt_id=None):
    latest = {}
    for attempt in _submission_history(root):
        if attempt_id is not None and attempt['attempt_id'] != attempt_id:
            continue
        for record in attempt['records']:
            latest[record['target']] = (attempt, record)
    return latest


@contextmanager
def _mutation_lock(root, *, wait=False):
    path = Path(root) / '.frust/mutation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + (60 if wait else 0)
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Another submission/collection owns {path}; inspect its owner before removing a stale lock') from error
            time.sleep(0.1)
    try:
        _atomic_write_submission_json(path / 'owner.json', {'pid': os.getpid(), 'started_ns': time.time_ns()})
        yield
    finally:
        shutil.rmtree(path)


def _completion_path(root, attempt_id, group, index):
    return Path(root) / '.frust/completions' / attempt_id / f'{group}_{index}.json'


def _mark_completed(root, attempt_id, group, index):
    if attempt_id is not None:
        _atomic_write_submission_json(_completion_path(root, attempt_id, group, index), {'finished_ns': time.time_ns()})


def _job_terminal(attempt, job_id):
    if job_id is None or not attempt.get('log_dir'):
        return False
    from frust.cluster.executor import _load_submitit
    submitit = _load_submitit()
    job_type = submitit.SlurmJob if attempt['backend'] == 'slurm' else submitit.LocalJob
    job = job_type(folder=attempt['log_dir'], job_id=str(job_id))
    if job.paths.result_pickle.exists():
        return True
    if attempt['backend'] != 'slurm':
        return False
    try:
        return job.state in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE'}
    except Exception:
        return False


def _attempt_terminal(root, attempt):
    seen = set()
    for record in attempt['records']:
        key = (record['group'], record['batch_index'])
        if key in seen:
            continue
        seen.add(key)
        completed = _completion_path(root, attempt['attempt_id'], *key).exists()
        batch = Path(root) / '.frust/batches' / attempt['attempt_id'] / f"{record['batch_index']}.json"
        completed |= batch.exists() and bool(json.loads(batch.read_text()).get('finished_at'))
        if not completed and not _job_terminal(attempt, record.get('job_id')):
            return False
    collector = attempt.get('collection_job_id')
    return collector is None or _completion_path(root, attempt['attempt_id'], 'collect', 0).exists() or _job_terminal(attempt, collector)


def _target_fingerprints(workflow, targets):
    config = {key: value for key, value in vars(workflow).items()
              if key not in {'_target_cache', 'dataframe', 'csv_path', 'smiles'} and not callable(value)}
    config['method'] = workflow.method.fingerprint()
    identity = (type(workflow).__module__, type(workflow).__qualname__, config)
    return {target.tag: hashlib.sha256(pickle.dumps((identity, target.payload, target.metadata), protocol=5)).hexdigest()
            for target in targets}


@contextmanager
def _submission_guard(root, plan, workflow, targets, cluster, *, mode, array, retry):
    if retry and (mode != 'single_job'):
        raise NotImplementedError('Explicit retries currently require execution="single_job"')
    fingerprints = _target_fingerprints(workflow, targets) if mode == 'single_job' else {}
    with _mutation_lock(root):
        latest = _latest_targets(root)
        outcomes = _target_outcomes(root)
        predecessors = {}
        checked = set()
        for target in targets:
            previous = latest.get(target.tag)
            directory = Path(root) / target.tag
            if previous is None:
                if retry:
                    raise ValueError(f'No validated submission history for retry target {target.tag}; use a new out_dir')
                if array and directory.exists() and any(directory.iterdir()):
                    raise ValueError(f'Existing artifacts for {target.tag}; use a new out_dir')
                continue
            attempt, record = previous
            # Existing staged screen restarts are validated by their manifest.
            # Array/retry attempts remain protected until staged support lands.
            if (mode != 'single_job' and attempt['mode'] != 'single_job'
                    and not array and not retry and not attempt.get('array')
                    and not attempt.get('previous_attempts')):
                continue
            if attempt['attempt_id'] not in checked and not _attempt_terminal(root, attempt):
                raise RuntimeError(f'Previous attempt {attempt["attempt_id"]} is active or unverified; overlapping writes are blocked')
            checked.add(attempt['attempt_id'])
            if not retry:
                raise ValueError(f'{target.tag} has submission history; select failed targets with retry=True or use a new out_dir')
            if attempt.get('fingerprints', {}).get(target.tag) != fingerprints[target.tag]:
                raise ValueError(f'Incompatible or legacy chemistry fingerprint for {target.tag}; use a new out_dir')
            import pandas as pd
            final = directory / 'final.parquet'
            if final.exists():
                try:
                    frame = pd.read_parquet(final)
                except Exception:
                    frame = None
                if frame is not None:
                    nt = [name for name in frame if str(name).endswith('-NT')]
                    belongs = frame.attrs.get('frust_submission', {}).get('attempt_id') == attempt['attempt_id']
                    target_failed = outcomes[target.tag].get('status') in {'failed', 'interrupted', 'unattempted'}
                    if belongs and not target_failed and (not nt or frame[nt].fillna(False).astype(bool).all().all()):
                        raise ValueError(f'Target {target.tag} already succeeded; exclude it from retry targets')
            predecessors[target.tag] = attempt['attempt_id']
        ledger = SubmissionLedger(root, plan, mode=mode, backend=cluster.backend, array=array)
        ledger.fingerprints = fingerprints
        ledger.previous_attempts = predecessors
        ledger.log_dir = str(Path(cluster.log_dir).resolve())
        ledger._write()
        for tag, predecessor in predecessors.items():
            source = Path(root) / tag
            if source.exists():
                destination = Path(root) / '.frust/history' / ledger.attempt_id / tag
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                ledger.archives[tag] = str(destination)
        ledger._write()
        yield ledger


def _target_outcomes(root, attempt_id=None, *, workers_finished=False):
    outcomes = {}
    for tag, (attempt, record) in _latest_targets(root, attempt_id).items():
        batch = Path(root) / '.frust/batches' / attempt['attempt_id'] / f"{record['batch_index']}.json"
        status = {}
        if batch.exists():
            payload = json.loads(batch.read_text())
            status = dict(next((item for item in payload['targets'] if item['target'] == tag), {}))
            if status.get('status') == 'unattempted':
                status['error'] = payload.get('error')
            if status.get('status') == 'running' and (
                workers_finished or payload.get('finished_at') or _job_terminal(attempt, record.get('job_id'))
            ):
                status.update(status='interrupted', error='Worker terminated without a final target outcome; cause is unknown')
        outcomes[tag] = dict(status, target=tag, attempt_id=attempt['attempt_id'], job_id=record.get('job_id'),
                             output_path=record['output_path'], tracked=bool(attempt.get('fingerprints', {}).get(tag)))
    return outcomes


def _ensure_collection_ready(root, targets):
    latest = _latest_targets(root)
    checked = set()
    for target in targets:
        if target.tag not in latest:
            continue
        attempt, _ = latest[target.tag]
        if attempt['attempt_id'] not in checked and not _attempt_terminal(root, attempt):
            raise RuntimeError('Collection requires completed worker and collection jobs; an attempt is active or unverified')
        checked.add(attempt['attempt_id'])
