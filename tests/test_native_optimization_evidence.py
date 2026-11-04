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


@pytest.mark.parametrize('family',['uma','dft'])
def test_reference_cache_rejects_caps_and_rebinds_exact_native_evidence(tmp_path,monkeypatch,family):
    from dataclasses import replace
    import frust as ft
    from frust.results import attach_result_contract
    from frust.screen.references import ReferenceLibrary

    if family=='uma':
        from test_uma_full_analysis import _run_full_uma,_components
        run=_run_full_uma(tmp_path/'source',monkeypatch)
        frame=pd.read_parquet(run.path/'calculations/references/computed.parquet')
        frame=frame.loc[frame.state_id.eq('ligand')].iloc[[0]].copy()
        wf=ft.workflows.catalyst_screen(dataframe=_components(),ts_types=['TS1'],
                                      method='uma-gas',screening='uma-gas',dimer_reference='dimer')
        target=next(t for t in wf.children()['references'].targets() if t.state_id=='ligand')
        method=wf.method;stage='uma_min_opt'
    else:
        from test_screen_end_to_end import _row,_components
        wf=ft.workflows.catalyst_screen(dataframe=_components(),ts_types=['TS1'],method='r2scan-3c-solv',dimer_reference='dimer')
        target=next(t for t in wf.children()['references'].targets() if t.state_id=='dimer')
        method=wf.method;stage='dft_opt';frame=pd.DataFrame([_row('dimer',-40.,{'H':2})])
        attach_result_contract(frame,'minimum',dft=True,include_terminal_solv_sp=False,
                               thermochemistry=method.thermochemistry)
    record=_optimization_record(frame.iloc[0],stage,frame.iloc[0][stage+'-oc'],'ORCA TERMINATED NORMALLY')
    frame.attrs['frust_optimization_checks']=[record];library=ReferenceLibrary(tmp_path/'library').initialize()
    with pytest.raises(ValueError,match='native optimization is not_converged'):
        library.publish(frame,target,method)
    record['status']='converged';published=library.publish(frame,target,method)
    alias=replace(target,target_id='renamed:'+target.target_id,tag='renamed')
    rebound=published.materialize(alias)
    assert _optimization_check(rebound.attrs['frust_optimization_checks'],rebound.iloc[0],stage)['status']=='converged'
    assert published.dataframe().attrs['frust_optimization_checks'][0]['structure_id']==record['structure_id']
    record['output_sha256']='a'*64
    changed=library.publish(frame,target,method)
    assert changed.reference_id!=published.reference_id
