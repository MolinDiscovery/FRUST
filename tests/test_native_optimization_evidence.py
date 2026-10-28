"""A normal ORCA exit must not hide a capped final geometry optimization."""
from copy import deepcopy

import pandas as pd
import pytest

from frust.stepper import Stepper
from frust.utils.dataframes import merge_dataframe_attrs
from frust.utils.optimization import _optimization_check, _optimization_record


@pytest.mark.parametrize("converged", [True, False])
@pytest.mark.parametrize("explicit_log", [True, False])
def test_stepper_captures_native_log_without_widening_results(converged, explicit_log):
    calls = []
    log = "THE OPTIMIZATION HAS CONVERGED" if converged else "Maximum optimization cycles reached\nORCA TERMINATED NORMALLY"

    def engine(atoms, coords, read_files, **kwargs):
        calls.append(read_files)
        return dict(normal_termination=True, electronic_energy=-1., opt_coords=coords, **{"orca.out": log})

    frame = pd.DataFrame(dict(atoms=[["H", "H"]], coords_embedded=[[[0., 0., 0.], [0., 0., .74]]], cid=[0], structure_id=["HH"]))
    step = Stepper(save_output_dir=False)
    step.orca_fn = engine
    out = step._run_engine(frame, engine, "final", lambda row: dict(options={"Opt": None}, read_files=["orca.out"] if explicit_log else []), False, None)
    assert calls == [["orca.out"]]
    assert bool(out.iloc[0]["final-NT"])
    assert ("final-orca.out" in out) == explicit_log
    check = _optimization_check(out.attrs["frust_optimization_checks"], out.iloc[0], "final")
    assert check["status"] == ("converged" if converged else "not_converged")


def test_merge_retains_every_geometry_and_rejects_stale_or_conflicting_evidence():
    frames = []
    for cid in range(3):
        row = pd.Series(dict(atoms=["H"], cid=cid, structure_id="x", **{"opt-oc": [[cid, 0., 0.]]}))
        record = _optimization_record(row, "opt", row["opt-oc"], "THE OPTIMIZATION HAS CONVERGED")
        frame = pd.DataFrame([row]);frame.attrs["frust_optimization_checks"] = [record]
        frames.append(frame)
    attrs = merge_dataframe_attrs(frames)
    assert len(attrs["frust_optimization_checks"]) == 3
    assert _optimization_check(attrs["frust_optimization_checks"], frames[2].iloc[0], "opt")["status"] == "converged"
    changed = frames[2].iloc[0].copy();changed["opt-oc"] = [[9., 0., 0.]]
    assert _optimization_check(attrs["frust_optimization_checks"], changed, "opt")["status"] == "stale"
    conflict = deepcopy(attrs["frust_optimization_checks"][-1]);conflict["status"] = "not_converged"
    assert _optimization_check([*attrs["frust_optimization_checks"], conflict], frames[2].iloc[0], "opt")["status"] == "conflicting"


@pytest.mark.parametrize("options", [{"Freq": None, "ExtOpt": None}, {"TightOpt": None}, {"SP": None}])
def test_nonoptimization_actions_do_not_invent_convergence_checks(options):
    from frust.utils.optimization import _orca_optimization_requested

    assert not _orca_optimization_requested(dict(options=options))
    assert not _orca_optimization_requested(dict(options={"Opt": None}, xtra_inp_str="%geom Scan B 0 1 = 1, 2, 5 end end"))
