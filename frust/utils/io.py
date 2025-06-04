import pandas as pd
from pathlib import Path

def dump_df(df: pd.DataFrame, step: str, base_dir: Path) -> Path:
    """
    If dump_each_step is True, writes DataFrame to `base_dir/{step}.csv`.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{step}.csv"
    df.to_csv(path, index=False)
    return path