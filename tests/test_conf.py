import pytest
from frust.config import Settings
from pathlib import Path

def test_defaults():
    cfg = Settings()
    assert cfg.debug is False
    assert cfg.live is False
    assert cfg.dump_each_step is False
    assert cfg.cores == 8
    assert cfg.memory_gb == 20.0
    assert isinstance(cfg.dump_base_dir, Path)

def test_override_and_types():
    cfg = Settings(
        debug=True,
        live=True,
        dump_each_step=True,
        cores=4,
        memory_gb=10.5,
        dump_base_dir="my_results"
    )
    assert cfg.debug
    assert cfg.live
    assert cfg.dump_each_step
    assert cfg.cores == 4
    assert cfg.memory_gb == 10.5
    assert str(cfg.dump_base_dir) == "my_results"

@pytest.mark.parametrize("bad_val", ["not_an_int", -1, 0])
def test_invalid_cores(bad_val):
    with pytest.raises(Exception):
        Settings(cores=bad_val)