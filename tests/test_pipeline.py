import pytest
import pandas as pd
from frust.config import Settings
from frust.pipeline import TSPipeline

@pytest.fixture
def tmp_settings(tmp_path):
    return Settings(
        debug=True,
        live=False,
        dump_each_step=True,
        dump_base_dir=tmp_path
    )

def test_pipeline_minimal_run(tmp_settings):
    pipeline = TSPipeline(tmp_settings)
    # run on furan (C1=CC=CO1)
    df = pipeline.run(["C1=CC=CO1"])
    assert isinstance(df, pd.DataFrame)
    # must have a 'step' column
    assert "step" in df.columns
    # at least one row
    assert len(df) > 0
    # after step1, should see 'step1_ts_uff'
    assert "ts_uff" in df["step"].unique()

def test_pipeline_dumps_intermediate_csv(tmp_settings):
    pipeline = TSPipeline(tmp_settings)
    df = pipeline.run(["C1=CC=CO1"])
    # since dump_each_step=True, check that files exist
    import os
    dumped = os.listdir(str(tmp_settings.dump_base_dir))
    # expect at least one CSV file named 'step0_transform.csv'
    assert any(name.startswith("step0_transform") and name.endswith(".csv") for name in dumped)