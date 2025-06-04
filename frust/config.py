from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    # run‐time switches
    debug: bool = False
    live: bool = False
    dump_each_step: bool = False

    # resources
    cores: int = 8
    memory_gb: float = 20.0

    # results location (can be overridden by prepare_base_dir)
    base_dir: Path = Path(".")