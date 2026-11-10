"""Row concurrency, result ordering, and backend environment isolation."""

import os
from pathlib import Path
import threading
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.stepper import Stepper


def rows():
    return pd.DataFrame({
        "substrate_name": ["water"] * 4,
        "cid": [0, 1, 2, 3],
        "atoms": [["O", "H", "H"]] * 4,
        "coords_embedded": [
            [[float(i), 0, 0], [float(i) + 1, 0, 0], [float(i), 1, 0]]
            if i != 1 else None for i in range(4)
        ],
    }, index=[8, 2, 8, -1])


@pytest.mark.parametrize("engine", ["xtb", "gxtb", "orca"])
def test_parallel_rows_preserve_serial_results_and_restore_environment(tmp_path, monkeypatch, engine):
    import tooltoad.gxtb  # Load calculator configuration before snapshotting the stage environment.
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "OMP_STACKSIZE", "OMP_MAX_ACTIVE_LEVELS", "XTBPATH")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setenv("XTBPATH", "original")
    before = {key: os.environ.get(key) for key in keys}
    outputs = []
    for workers in (1, 3):
        barrier = threading.Barrier(3)
        release_first = threading.Event()
        seen = []
        completed = []

        def backend(atoms, coords, n_cores, scr, options, **kwargs):
            cid = int(coords[0][0])
            assert n_cores == 1
            for key in keys[:3]:
                assert os.environ[key] == "1"
            seen.append(scr)
            if workers > 1:
                Path(scr, "row.txt").write_text(str(cid))
                barrier.wait(timeout=5)
                assert Path(scr, "row.txt").read_text() == str(cid)
                if cid == 0:
                    assert release_first.wait(timeout=5)
            options["row_mutation"] = cid
            os.environ["XTBPATH"] = "backend-mutation"
            if cid == 2:
                raise RuntimeError("row two failed")
            completed.append(cid)
            if cid == 3:
                release_first.set()
            return {"normal_termination": True, "electronic_energy": -float(cid),
                    "opt_coords": coords, "input.hess": f"hessian-{cid}"}

        step = Stepper(n_cores=4, n_workers=workers, save_output_dir=False, work_dir=str(tmp_path))
        setattr(step, f"{engine}_fn", backend)
        options = {"SP": None} if engine == "orca" else {"gfn": 2}
        out = getattr(step, engine)(rows(), name="stage", options=options, n_cores=1)
        outputs.append(out)
        assert {key: os.environ.get(key) for key in keys} == before
        assert options == ({"SP": None} if engine == "orca" else {"gfn": 2})
        assert step.n_cores == 4 and step.n_workers == workers
        assert out.index.tolist() == [8, 2, 8, -1]
        assert out["stage-NT"].tolist() == [True, False, False, True]
        assert out.loc[-1, "stage-input.hess"] == "hessian-3"
        assert "row two failed" in out.iloc[2]["stage-error"]
        assert out.attrs["frust_steps"]["stage"]["calculator"]["resources"]["n_workers"] == workers
        timing = out.attrs["frust_steps"]["stage"]["timing"]
        assert (timing["processed_rows"], timing["skipped_rows"]) == (3, 1)
        assert ft.show_steps(out).loc["stage", "n_workers"] == workers
        if workers > 1:
            assert completed == [3, 0]
            assert len(set(seen)) == 3
            assert all(not Path(directory).exists() for directory in seen)
    pd.testing.assert_frame_equal(outputs[0], outputs[1])


def test_worker_override_and_stage_exception_restore_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    step = Stepper(n_workers=3, save_output_dir=False, work_dir=str(tmp_path))

    def broken_inputs(row):
        assert os.environ["OMP_NUM_THREADS"] == "1"
        raise ValueError("invalid row input")

    with pytest.raises(ValueError, match="invalid row input"):
        step._run_engine(rows(), step.xtb_fn, "bad", broken_inputs, False, None,
                         n_workers=2, n_cores=1)
    assert os.environ["OMP_NUM_THREADS"] == "9"
    assert "MKL_NUM_THREADS" not in os.environ
    assert list(tmp_path.iterdir()) == []
    calls = []
    step.xtb_fn = lambda **kwargs: calls.append(threading.get_ident()) or {"normal_termination": True}
    out = step.xtb(rows(), n_workers=1)
    assert set(calls) == {threading.get_ident()}
    assert step.n_workers == 3
    assert next(iter(out.attrs["frust_steps"].values()))["calculator"]["resources"]["n_workers"] == 1


def test_parallel_orca_keeps_hessians_logs_and_duplicate_row_directories(tmp_path):
    df = rows().iloc[[0, 3]].copy()
    df["cid"] = 0
    df["seed-input.hess"] = ["first seed", "last seed"]
    barrier = threading.Barrier(2)
    seen = []
    step = Stepper(n_workers=2, n_cores=1, work_dir=str(tmp_path), output_base=tmp_path / "saved")

    def backend(coords, calc_dir, data2file, read_files, memory, **kwargs):
        assert "orca.out" in read_files
        assert memory == 10
        Path(calc_dir, "input.hess").write_text(data2file["private_input.hess"])
        seen.append(calc_dir)
        barrier.wait(timeout=5)
        return {"normal_termination": True, "electronic_energy": -1,
                "opt_coords": coords, "orca.out": "optimization log"}

    step.orca_fn = backend
    with patch("frust.utils.optimization._optimization_record",
               side_effect=lambda row, *args: {"label": float(row.coords_embedded[0][0])}):
        out = step.orca(df, name="opt", options={"Opt": None}, save_step=True, use_last_hess=True)
    assert len(set(seen)) == 2
    assert sorted(Path(path, "input.hess").read_text() for path in seen) == ["first seed", "last seed"]
    assert out.attrs["frust_optimization_checks"] == [{"label": 0}, {"label": 3}]
    assert "opt-orca.out" not in out


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_worker_budget(value, tmp_path):
    with pytest.raises(ValueError, match="n_workers"):
        Stepper(n_workers=value, work_dir=str(tmp_path))


def test_empty_and_single_row_do_not_create_executor(tmp_path):
    step = Stepper(debug=True, n_workers=4, save_output_dir=False, work_dir=str(tmp_path))
    with patch("frust.stepper.ThreadPoolExecutor", side_effect=AssertionError("unexpected pool")):
        assert step.xtb(rows().iloc[:0]).empty
        assert len(step.xtb(rows().iloc[:1])) == 1


def test_tooltoad_orca_runtime_is_resolved_before_parallel_path_mutations(tmp_path, monkeypatch):
    import tooltoad.orca as backend

    monkeypatch.setenv("XTBPATH", "original-parameters")
    barrier = threading.Barrier(2)
    commands = []
    resolutions = []

    def resolve(orca_cmd=None, set_env=None):
        resolutions.append(set_env)
        return "orca", set_env or f"env XTBPATH={os.environ['XTBPATH']}"

    def stream(command, cwd):
        commands.append(command)
        barrier.wait(timeout=5)
        yield "calculation failed\n"

    step = Stepper(n_cores=1, n_workers=2, save_output_dir=False, work_dir=str(tmp_path))
    step.orca_fn = backend.orca_calculate
    with patch.object(backend, "_resolve_orca_runtime", side_effect=resolve), patch.object(backend, "stream", stream):
        out = step.orca(rows().iloc[[0, 3]], name="orca_sp", options={"SP": None})
    assert not out["orca_sp-NT"].any()
    assert resolutions == [None, "env XTBPATH=original-parameters", "env XTBPATH=original-parameters"]
    assert all("XTBPATH=original-parameters" in command for command in commands)
    assert os.environ["XTBPATH"] == "original-parameters"
    assert list(tmp_path.iterdir()) == []
