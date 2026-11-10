"""Hardware-independent allocation and validation deduplication policies."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import frust as ft
from frust.cluster import ClusterConfig, Resources
from frust.utils.uma import resolve_uma_server_cores
from frust.workflows.core import ExecutionOptions, StageDef, _apply_calculator
from frust.workflows.methods import CalculatorSpec
from test_modular_workflow_validation import workflow
from test_workflows import FakeExecutor


@pytest.mark.parametrize("cores,memory,expected", [
    (10, 20, 5), (4, 8, 2), (8, 14.4, 4), (6, 20.8, 6),
    (8, 24, 7), (8, 32, 8), (1, 100, 1), (8, 1, 1), (8, 20.4, 6),
])
def test_memory_scaled_server_budget(cores, memory, expected):
    assert resolve_uma_server_cores(cores, memory) == expected
    assert resolve_uma_server_cores(cores, memory, 3) == 3


@pytest.mark.parametrize("engine,options,uma,expected", [
    ("xtb", {"opt": None}, False, (1, 8)),
    ("gxtb", {"opt": None}, False, (1, 8)),
    ("orca", {"SP": None}, True, (1, 4)),
    ("orca", {"Opt": None}, True, (8, 1)),
    ("orca", {"OptTS": None}, True, (8, 1)),
    ("orca", {"NumFreq": None}, True, (6, 1)),
    ("orca", {"SP": None}, False, (8, 1)),
    ("orca", {"Freq": None}, False, (8, 1)),
])
def test_stage_dispatch_uses_scientific_operation(engine, options, uma, expected):
    spec = CalculatorSpec(engine=engine, options=options,
                          kwargs={"uma": "omol"} if uma else {})
    calls = []
    step = SimpleNamespace(**{name: lambda df, **kwargs: calls.append(kwargs) or df
                              for name in ("xtb", "gxtb", "orca")})
    _apply_calculator(step, pd.DataFrame(), StageDef("custom_name", "custom stage"), spec,
                      options=ExecutionOptions(n_cores=8, mem_gb=20.8))
    assert (calls[0]["n_cores"], calls[0]["n_workers"]) == expected
    if uma:
        assert calls[0]["uma_server_cores"] == 6
    assert spec.kwargs == ({"uma": "omol"} if uma else {})


def test_stage_overrides_and_shared_server_pool():
    calls = []
    step = SimpleNamespace(orca=lambda df, **kwargs: calls.append(kwargs) or df)
    runtime = ExecutionOptions(n_cores=8, mem_gb=5, uma_server_cores=6)
    for keywords in ({"SP": None}, {"Opt": None}, {"NumFreq": None}):
        _apply_calculator(step, pd.DataFrame(),
                          StageDef("stage", "stage", n_cores=2, n_workers=3),
                          CalculatorSpec("orca", keywords, kwargs={"uma": "omol"}),
                          options=runtime)
    assert [call["n_workers"] for call in calls] == [3, 3, 1]
    assert {call["uma_server_cores"] for call in calls} == {6}


def test_composed_local_run_forwards_server_override(tmp_path):
    from test_modular_workflow_validation import FakeStepper, run_mocked

    run_mocked(workflow(), tmp_path, uma_server_cores=3)
    uma_calls = [kwargs for _, name, _, _, kwargs in FakeStepper.calls if name.startswith("uma_")]
    assert uma_calls and all(kwargs["uma_server_cores"] == 3 for kwargs in uma_calls)


@pytest.mark.parametrize("method", ["uma-gas", "wb97xd3-631g"])
def test_explicit_validation_selection_without_reranking_deduplicates(method):
    wf = workflow(ranking=None, method=method, ranking_top_n=None, validation_top_n=3)
    for child in wf.children().values():
        assert "validation_dedup" in {stage.id for stage in child._stage_defs()}


@pytest.mark.parametrize("method", ["uma-gas-opt-alpb-chloroform", "wb97xd3-631g"])
def test_deduplication_precedes_frequencies_in_every_validation_branch(method):
    wf = workflow(method=method, validation_top_n=3, ranking_top_n=None, scope="full_cycle")
    fingerprint = wf.method.fingerprint()
    for child in wf.children().values():
        stages = child._stage_defs()
        dedup = next(stage for stage in stages if stage.id == "validation_dedup")
        position = stages.index(dedup)
        assert stages[position + 1].operation == "frequencies"
        opt = stages[position - 1]
        assert opt.operation in {"ts_search", "minimum_optimization"}
        assert dedup.prune_options["coords_col"] == opt.id + "-oc"
        assert dedup.prune_options["energy_col"] == opt.id + "-EE"
        groups = child._stage_groups("dft_staged")
        group = next(group for group in groups if dedup in group)
        assert group[-1].operation == "frequencies"
        assert child._group_name(group) == stages[position + 1].id
    assert wf.method.fingerprint() == fingerprint


@pytest.mark.parametrize("method", ["uma-gas-opt-alpb-chloroform", "wb97xd3-631g"])
def test_real_deduplication_dispatch_preserves_lowest_candidate_and_failures(tmp_path, method):
    from frust.workflows.core import _run_stage_calculation

    child = workflow(method=method, validation_top_n=3, ranking_top_n=None).children()["transition_states"]
    target = child.targets()[0]
    stage = next(stage for stage in child._stage_defs() if stage.id == "validation_dedup")
    coord_col, energy_col = (stage.prune_options[key] for key in ("coords_col", "energy_col"))
    coords = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    df = pd.DataFrame({
        "substrate_name": ["water"] * 4,
        "cid": [0, 1, 2, 3],
        "atoms": [["O", "H", "H"]] * 4,
        coord_col: [coords, coords, coords, None],
        energy_col: [-100., -100.00004, -99.9, np.nan],
    })
    result = _run_stage_calculation(child, target, stage, df, save_dir=None,
                                    options=ExecutionOptions(work_dir=str(tmp_path)))
    assert result.cid.tolist() == [1, 2, 3]
    assert list(result.columns) == list(df.columns)
    assert ft.show_steps(result).loc["validation_dedup", "dropped_rows"] == 1


@pytest.mark.parametrize("execution", ["single_job", "dft_staged", "fully_staged"])
@pytest.mark.parametrize("full", [False, True])
def test_submission_resources_and_override_forwarding(tmp_path, execution, full):
    wf = ft.workflows.raw_mols(dataframe=pd.DataFrame({"smiles": ["O"]}), dft=full)
    fake = FakeExecutor()
    cluster = ClusterConfig(backend="slurm", partition="generic", log_dir=str(tmp_path / "logs"))
    with patch("frust.workflows.core.create_executor", return_value=fake):
        wf.submit(out_dir=tmp_path / "run", cluster=cluster, execution=execution,
                  collect=False, uma_server_cores=2)
    for group, parameters, (_, args, _) in zip(wf._stage_groups(execution), fake.parameters, fake.submissions):
        validation = full and any(stage.phase == "validation" for stage in group)
        assert (parameters["cpus_per_task"], parameters["mem_gb"]) == (
            (6, 26) if validation else (4, 20)
        )
        options = args[3] if execution == "single_job" else args[6]
        assert options.uma_server_cores == 2
        assert options.mem_gb == pytest.approx((26 if validation else 20) * 0.8)
    fake = FakeExecutor()
    with patch("frust.workflows.core.create_executor", return_value=fake):
        wf.submit(out_dir=tmp_path / "override", cluster=cluster, execution="single_job",
                  collect=False, stage_resources={"single_job": Resources(cpus=3, mem_gb=12, timeout_min=60)})
    assert fake.parameters[0]["cpus_per_task"] == 3
    assert fake.parameters[0]["mem_gb"] == 12
