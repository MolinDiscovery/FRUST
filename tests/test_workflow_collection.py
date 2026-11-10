"""Collection of compatible Parquet schemas from different workflow targets."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import frust as ft
from frust.workflows.core import _stream_parquet_files


def _saved_workflow(root, frames, method="uma-gas-opt-alpb-chloroform"):
    workflow = ft.workflows.raw_mols(
        dataframe=pd.DataFrame({"compound_name": list(frames), "smiles": "CC"}),
        method=method,
    )
    for tag, frame in frames.items():
        directory = root / tag
        directory.mkdir()
        frame.to_parquet(directory / "final.parquet", index=False)
    return workflow


@pytest.mark.parametrize(
    "method,stage",
    [("uma-gas-opt-alpb-chloroform", "uma_freq"), ("wb97xd3-631g", "dft_freq")],
)
def test_collect_aligns_roles_and_preserves_values_and_provenance(tmp_path, method, stage):
    frames = {
        "TS4": pd.DataFrame({
            "cid": [10, 11],
            "constraint_roles": [{"cat_B": 0, "pin_B": 4, "substrate_C": 8}, None],
            f"{stage}-EE": [-100.25, -100.5],
            f"{stage}-NT": True,
            f"{stage}-oc": [[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]],
        }),
        "TS1": pd.DataFrame({
            "cid": [12],
            "constraint_roles": [{"cat_B": 2, "cat_N": 6, "substrate_C": 9}],
            f"{stage}-EE": [-101.25],
            f"{stage}-NT": True,
            f"{stage}-oc": [[[6.0, 7.0, 8.0]]],
        }),
    }
    for tag, frame in frames.items():
        frame.attrs["frust_steps"] = {tag: {"engine": "orca", "columns": [f"{stage}-EE"]}}
    workflow = _saved_workflow(tmp_path, frames, method)
    originals = {tag: (tmp_path / tag / "final.parquet").read_bytes() for tag in frames}

    merged = workflow.collect(tmp_path, require_normal_termination=True)

    rows = pq.read_table(tmp_path / "merged.parquet").to_pylist()
    assert [row["cid"] for row in rows] == [10, 11, 12]
    assert [row["constraint_roles"] for row in rows] == [
        {"cat_B": 0, "pin_B": 4, "substrate_C": 8, "cat_N": None},
        None,
        {"cat_B": 2, "pin_B": None, "substrate_C": 9, "cat_N": 6},
    ]
    assert [row[f"{stage}-EE"] for row in rows] == [-100.25, -100.5, -101.25]
    assert [row[f"{stage}-oc"] for row in rows] == [
        [[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0]],
    ]
    assert set(merged.attrs["frust_steps"]) == {"TS4", "TS1"}
    assert len(merged.attrs["frust_merge"]["input_files"]) == 2
    report = json.loads((tmp_path / "collection_report.json").read_text())
    assert (report["n_collected"], report["n_rows"], report["n_failures"]) == (2, 3, 0)
    for tag, content in originals.items():
        assert (tmp_path / tag / "final.parquet").read_bytes() == content


@pytest.mark.parametrize("container", [pa.list_, pa.large_list])
def test_stream_aligns_structs_inside_lists_and_preserves_nulls(tmp_path, container):
    first_type = container(pa.struct([("cat_B", pa.int64()), ("pin_B", pa.int64())]))
    second_type = container(pa.struct([("cat_N", pa.int64()), ("cat_B", pa.int64())]))
    first = pa.table({"nested": pa.array([[{"cat_B": 1, "pin_B": 2}, None]], first_type)})
    second = pa.table({"nested": pa.array([None, [None, {"cat_N": 3, "cat_B": 4}]], second_type)})
    files = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    pq.write_table(first, files[0])
    pq.write_table(second, files[1], row_group_size=1)

    output = tmp_path / "merged.parquet"
    _stream_parquet_files(files, output, attrs={}, expected_rows=3)

    assert pq.read_table(output)["nested"].to_pylist() == [
        [{"cat_B": 1, "pin_B": 2, "cat_N": None}, None],
        None,
        [None, {"cat_B": 4, "pin_B": None, "cat_N": 3}],
    ]


def test_collect_handles_missing_null_columns_and_numeric_promotion(tmp_path):
    frames = {
        "first": pd.DataFrame({"cid": [0], "energy_uff": [None], "correction": [None]}),
        "second": pd.DataFrame({"cid": [1], "correction": [0.125]}),
    }
    workflow = _saved_workflow(tmp_path, frames)

    workflow.collect(tmp_path)

    merged = pq.read_table(tmp_path / "merged.parquet")
    assert merged.schema.field("energy_uff").type == pa.null()
    assert merged.to_pylist() == [
        {"cid": 0, "energy_uff": None, "correction": None},
        {"cid": 1, "energy_uff": None, "correction": 0.125},
    ]


def test_incompatible_schemas_preserve_existing_collection(tmp_path):
    workflow = _saved_workflow(tmp_path, {
        "first": pd.DataFrame({"constraint_roles": [{"cat_B": 1}]}),
        "second": pd.DataFrame({"constraint_roles": ["invalid role mapping"]}),
    })
    output = tmp_path / "merged.parquet"
    pd.DataFrame({"previous_result": [1]}).to_parquet(output)
    original = output.read_bytes()

    with pytest.raises(pa.ArrowTypeError):
        workflow.collect(tmp_path)

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".merged.parquet.*.tmp")) == []


def test_stream_verification_failure_preserves_existing_collection(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "merged.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(source)
    pd.DataFrame({"previous_result": [2]}).to_parquet(output)
    original = output.read_bytes()

    with pytest.raises(ValueError, match="Collected row count mismatch"):
        _stream_parquet_files([source], output, attrs={}, expected_rows=2)

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".merged.parquet.*.tmp")) == []
