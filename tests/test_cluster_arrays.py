from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.cluster import ClusterConfig, JobSubmissionResult, Resources
from frust.cluster.submission import (
    SubmissionLedger, _plan_submission, _submit_array_jobs,
)


RESOURCES = Resources(cpus=2, mem_gb=3, timeout_min=5)
GROUPS = [("single_job", RESOURCES, "final.parquet")]


def test_plan_batches_are_ordered_and_calculation_free():
    plan = _plan_submission(
        ["A", "B", "C", "D", "E"], GROUPS, execution="single_job",
        array=True, array_parallelism=2, targets_per_task=2,
    )
    assert plan[0].batches == (("A", "B"), ("C", "D"), ("E",))
    assert plan[0].resources == RESOURCES
    assert plan[0].parallelism == 2
    assert _plan_submission([], GROUPS, execution="single_job")[0].batches == ()


@pytest.mark.parametrize("options", [
    {"array": 1}, {"targets_per_task": True}, {"targets_per_task": 0},
    {"targets_per_task": 1.5}, {"array_parallelism": 2}, {"targets_per_task": 2},
    {"array": True}, {"array": True, "array_parallelism": True},
    {"array": True, "array_parallelism": 0},
    {"array": True, "array_parallelism": 1.5},
    {"array": True, "array_parallelism": {"single_job": 1}},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        _plan_submission(["A"], GROUPS, execution="single_job", **options)


@pytest.mark.parametrize("tags", [["A", "A"], ["../A"], ["."], [""], ["A\\B"]])
def test_duplicate_or_unsafe_output_identities(tags):
    with pytest.raises(ValueError):
        _plan_submission(tags, GROUPS, execution="single_job")


def test_staged_limits_and_batching():
    groups = [("init", RESOURCES, "init.parquet"), ("freq", RESOURCES, "init_freq.parquet")]
    plan = _plan_submission(["A", "B"], groups, execution="fully_staged", array=True,
                            array_parallelism={"init": 2, "freq": 1})
    assert [group.parallelism for group in plan] == [2, 1]
    assert plan[0].batches == plan[1].batches == (("A",), ("B",))
    for limits in ({"init": 1}, {"init": 1, "freq": 1, "extra": 1}, {"init": True, "freq": 1}):
        with pytest.raises(ValueError):
            _plan_submission(["A"], groups, execution="fully_staged", array=True,
                             array_parallelism=limits)
    with pytest.raises(ValueError, match="targets_per_task=1"):
        _plan_submission(["A"], groups, execution="fully_staged", array=True,
                         array_parallelism=2, targets_per_task=2)


def test_ledger_preserves_accepted_jobs_and_later_failure(tmp_path):
    plan = _plan_submission(["A", "B", "C"], GROUPS, execution="single_job")
    ledger = SubmissionLedger(tmp_path, plan, mode="single_job", backend="slurm", array=False)
    ledger.submitted("single_job", 0, SimpleNamespace(job_id="124"))
    ledger.failed("single_job", [1], RuntimeError("scheduler rejected submission"))
    payload = json.loads(ledger.path.read_text())
    assert payload["schema_version"] == 1
    assert [r["status"] for r in payload["records"]] == ["submitted", "submission_failed", "planned"]
    assert payload["records"][0]["job_id"] == "124"
    assert payload["records"][0]["array_job_id"] is None
    assert "scheduler rejected" in payload["records"][1]["error"]
    assert not list(ledger.path.parent.glob("*.tmp"))


def test_whole_array_mapping_is_persisted_in_one_write(tmp_path):
    plan = _plan_submission(["A", "B", "C"], GROUPS, execution="single_job", array=True, array_parallelism=2)
    ledger = SubmissionLedger(tmp_path, plan, mode="single_job", backend="slurm", array=True)
    with patch.object(ledger, "_write", wraps=ledger._write) as write:
        ledger.submitted_many("single_job", {i: SimpleNamespace(job_id=f"124_{i}") for i in range(3)})
    assert write.call_count == 1
    assert [r.job_id for r in ledger.records] == ["124_0", "124_1", "124_2"]


def test_batched_and_staged_mapping_and_result_defaults(tmp_path):
    groups = [("init", RESOURCES, "init.parquet"), ("freq", RESOURCES, "init_freq.parquet")]
    plan = _plan_submission(["A", "B"], groups, execution="fully_staged", array=True, array_parallelism=2)
    ledger = SubmissionLedger(tmp_path, plan, mode="fully_staged", backend="slurm", array=True)
    ledger.submitted("init", 1, SimpleNamespace(job_id="124_1"))
    record = ledger.records[1]
    assert (record.target, record.group, record.array_job_id, record.array_index) == ("B", "init", "124", 1)
    assert ledger.records[3].status == "planned"
    batch_plan = _plan_submission(["A", "B"], GROUPS, execution="single_job", array=True,
                                 array_parallelism=1, targets_per_task=2)
    batch_ledger = SubmissionLedger(tmp_path, batch_plan, mode="single_job", backend="local", array=True)
    batch_ledger.submitted("single_job", 0, SimpleNamespace(job_id="124_0"))
    assert [r.job_id for r in batch_ledger.records] == ["124_0", "124_0"]
    assert all(r.array_job_id is None for r in batch_ledger.records)
    assert batch_ledger.attempt_id != ledger.attempt_id
    old = JobSubmissionResult([1], ["A"], ["A"], "mols", "local")
    assert asdict(old)["records"] == []
    assert old.array_job_ids == [] and old.submission_path is None


def _workflow():
    return ft.workflows.raw_mols(dataframe=pd.DataFrame({
        "compound_name": ["A", "B", "C"], "smiles": ["CCO", "CCN", "CCC"],
    }))


def test_validation_before_output_or_executor_creation(tmp_path):
    wf = _workflow()
    with patch("frust.workflows.core.create_executor") as create, \
         patch.object(wf, "_prepare_initial_df") as prepare:
        with pytest.raises(ValueError, match="duplicate"):
            wf.submit(out_dir=tmp_path / "bad", cluster=ClusterConfig(), targets=[0, 0])
    assert not (tmp_path / "bad").exists()
    create.assert_not_called()
    prepare.assert_not_called()


class ArrayExecutor:
    def __init__(self, ids):
        self.ids = ids
        self.parameters = []
        self.calls = []

    def update_parameters(self, **kwargs):
        self.parameters.append(kwargs)

    def map_array(self, fn, *columns):
        self.calls.append((fn, list(zip(*columns))))
        return [SimpleNamespace(job_id=job_id) for job_id in self.ids]

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        return SimpleNamespace(job_id="999")


def test_workflow_array_resources_mapping_and_collection(tmp_path):
    worker = ArrayExecutor(["123_0", "123_1", "123_2"])
    collector = ArrayExecutor([])
    wf = _workflow()
    with patch("frust.workflows.core.create_executor", side_effect=[worker, collector]), \
         patch.object(wf, "_prepare_initial_df") as prepare:
        result = wf.submit(
            out_dir=tmp_path, cluster=ClusterConfig(), execution="single_job",
            array=True, array_parallelism=2, stage_resources={"single_job": RESOURCES},
        )
    prepare.assert_not_called()
    assert len(worker.calls) == 1
    fn, arguments = worker.calls[0]
    assert fn.__name__ == "_run_target_submitted_job"
    assert [args[1].tag for args in arguments] == result.tags
    assert all(args[3].n_cores == 2 and args[3].mem_gb == pytest.approx(2.4) for args in arguments)
    assert worker.parameters[-1] == {"slurm_array_parallelism": 2}
    assert worker.parameters[0]["mem_gb"] == 3
    assert result.array_job_ids == ["123"]
    assert [r.array_index for r in result.records] == [0, 1, 2]
    assert collector.parameters[-1]["slurm_additional_parameters"]["dependency"] == "afterany:123"
    assert all("slurm_array_parallelism" not in p for p in collector.parameters)
    assert collector.parameters[-1]["mem_gb"] == 4
    assert json.loads(open(result.submission_path).read())["collection_job_id"] == "999"


def test_single_target_real_id_and_empty_selection(tmp_path):
    worker = ArrayExecutor(["123"])
    with patch("frust.workflows.core.create_executor", return_value=worker):
        result = _workflow().submit(
            out_dir=tmp_path / "one", cluster=ClusterConfig(), execution="single_job",
            array=True, array_parallelism=2, targets=[0], collect=False,
        )
    assert result.job_ids == ["123"] and result.array_job_ids == []
    assert result.records[0].array_index is None
    with patch("frust.workflows.core.create_executor") as create:
        empty = _workflow().submit(
            out_dir=tmp_path / "empty", cluster=ClusterConfig(), execution="single_job",
            array=True, array_parallelism=2, targets=[],
        )
    create.assert_not_called()
    assert not empty.job_ids and empty.collection_job_id is None


@pytest.mark.parametrize("extra", [{"array": "0-9"}, {"dependency": "afterok:1"}, {"--array": "0-9"}])
def test_conflicting_scheduler_options_rejected(tmp_path, extra):
    with patch("frust.workflows.core.create_executor") as create:
        with pytest.raises(ValueError, match="FRUST manages"):
            _workflow().submit(out_dir=tmp_path / "bad", cluster=ClusterConfig(extra_slurm_parameters=extra),
                               array=True, array_parallelism=2)
    create.assert_not_called()
    assert not (tmp_path / "bad").exists()


def test_array_limit_and_future_features_rejected_before_submission(tmp_path):
    with patch("frust.workflows.core.create_executor") as create:
        with pytest.raises(ValueError, match="Select fewer targets"):
            _workflow().submit(out_dir=tmp_path / "size", cluster=ClusterConfig(max_array_size=2),
                               array=True, array_parallelism=2)
        with pytest.raises(NotImplementedError, match="Staged arrays"):
            _workflow().submit(out_dir=tmp_path / "staged", cluster=ClusterConfig(),
                               execution="fully_staged", array=True, array_parallelism=2)
    create.assert_not_called()


def test_array_submission_failure_keeps_report_and_cause(tmp_path):
    worker = ArrayExecutor([])
    with patch("frust.workflows.core.create_executor", return_value=worker), \
         patch.object(worker, "map_array", side_effect=ValueError("invalid job array specification")):
        with pytest.raises(RuntimeError, match="MaxArraySize") as failure:
            _workflow().submit(out_dir=tmp_path, cluster=ClusterConfig(), array=True, array_parallelism=2)
    assert isinstance(failure.value.__cause__, ValueError)
    report = json.loads(next((tmp_path / ".frust/submissions").glob("*.json")).read_text())
    assert all(r["status"] == "submission_failed" for r in report["records"])


def test_local_partial_submission_preserves_ids(tmp_path):
    executor = ArrayExecutor([])
    plan = _plan_submission(["A", "B", "C"], GROUPS, execution="single_job", array=True, array_parallelism=3)
    ledger = SubmissionLedger(tmp_path, plan, mode="single_job", backend="local", array=True)
    with patch.object(executor, "submit", side_effect=[SimpleNamespace(job_id="11"), RuntimeError("failed")]):
        with pytest.raises(RuntimeError):
            _submit_array_jobs(executor, ClusterConfig(backend="local"), sum, [(1,), (2,), (3,)],
                               parallelism=3, on_submitted=lambda i, j: ledger.submitted("single_job", i, j))
    assert ledger.records[0].job_id == "11"
    assert ledger.records[1].status == "planned"


def test_local_concurrency_and_collection_after_failure(tmp_path):
    submitit = pytest.importorskip("submitit")

    def worker(workflow, target, save_dir, options, submitted_at, attempt_id=None, batch_index=0):
        from pathlib import Path
        import json
        import time
        import pandas as pd

        directory = Path(save_dir)
        start = time.time()
        time.sleep(0.5)
        try:
            if target.tag == "B":
                raise RuntimeError("injected failure")
            df = pd.DataFrame({"target": [target.tag], "calc-NT": [True]})
            df.attrs["frust_submission"] = {"attempt_id": attempt_id, "target": target.tag}
            df.to_parquet(directory / "final.parquet")
        finally:
            (directory / "interval.json").write_text(json.dumps([start, time.time()]))

    with patch("frust.workflows.core._run_target_submitted_job", worker):
        result = _workflow().submit(
            out_dir=tmp_path / "run", cluster=ClusterConfig(backend="local", log_dir=tmp_path / "logs"),
            array=True, array_parallelism=2, target_retention="all",
        )
    job = submitit.LocalJob(folder=tmp_path / "logs", job_id=result.collection_job_id)
    deadline = time.monotonic() + 40
    while not job.paths.result_pickle.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert job.paths.result_pickle.exists(), "Collector did not finish"
    job.result()
    events = []
    for directory in result.save_dirs:
        start, end = json.loads((Path(directory) / "interval.json").read_text())
        events.extend([(start, 1), (end, -1)])
    active = maximum = 0
    for _, change in sorted(events):
        active += change
        maximum = max(maximum, active)
    assert maximum <= 2
    assert len(events) == 6
    merged = pd.read_parquet(result.collection_output)
    assert sorted(merged["target"].tolist()) == ["A", "C"]
    report = json.loads(open(result.collection_report).read())
    assert report["n_missing"] == 1 and report["n_collected"] == 2
    assert result.array_job_ids == []
