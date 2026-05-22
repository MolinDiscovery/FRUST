# frust/utils/dirs.py
from pathlib import Path

def prepare_base_dir(path: Path | str | None, job_id: int | None = None) -> Path:
    """Prepare and create a base directory for pipeline results.
    
    A base directory is created at the specified path (or the current working
    directory if ``path`` is None). Inside this location, a uniquely named
    subdirectory is generated using a prefix and an optional job identifier.
    
    Parameters
    ----------
    path : Path or str or None
        Base path where the directory should be created. If None, the current
        working directory is used.
    job_id : int or None, optional
        Optional job identifier to include in the generated directory name.
    
    Returns
    -------
    Path
        Path object pointing to the created base results directory.
    
    Examples
    --------
    >>> prepare_base_dir("/tmp", job_id=123)
    PosixPath('/tmp/FRUST_results_123_<timestamp>')
    
    >>> prepare_base_dir(None, job_id=456)
    PosixPath('./FRUST_results_456_<timestamp>')
    """

    if path:
        base_dir = Path(path)
        base_dir.mkdir(exist_ok=True, parents=True)
    else:
        base_dir = Path(".")

    # Import lazily so importing Stepper does not load molecule/visualization helpers.
    from .mols import generate_id

    # Always create a unique, timestamped run directory when output is requested.
    results_preface = "FRUST_results"
    results_dir_name = generate_id(results_preface, job_id)
    base_results_dir = base_dir / results_dir_name
    base_results_dir.mkdir(exist_ok=True, parents=True)

    return base_results_dir


def make_step_dir(base: Path, step: str) -> Path:
    """
    Make (or reuse) a sub-directory ``base/step`` and return it.
    """
    d = base / step
    d.mkdir(parents=True, exist_ok=True)
    return d
