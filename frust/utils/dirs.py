# frust/utils/dirs.py
from pathlib import Path
from .mols import get_molecule_name, generate_id

def prepare_base_dir(path: Path | str | None, ligands_smiles: list[str], job_id: int | None = None) -> Path:
    """Prepare and create a base directory for pipeline results.

    Creates a base directory structure for storing pipeline results. If multiple ligands
    are provided, creates a generic timestamped directory. For single ligands, uses the
    molecule name in the directory name.

    Args:
        path: Base path where the directory should be created. If None, uses current directory.
        ligands_smiles: List of SMILES strings for the ligands being processed.
        job_id: Optional job identifier to include in the directory name.

    Returns:
        Path object pointing to the created base results directory.

    Examples:
        >>> prepare_base_dir("/tmp", ["CCO"], job_id=123)
        PosixPath('/tmp/ethanol_123_20240527_143022')
        
        >>> prepare_base_dir(None, ["CCO", "CCC"], job_id=456)
        PosixPath('./pipeline_ts_results_456_20240527_143022')
    """

    if path:
        base_dir = Path(path)
        base_dir.mkdir(exist_ok=True, parents=True)
    else:
        base_dir = Path(".")

    # Only create a unique, timestamped sub‐folder when there's more than one ligand
    if len(ligands_smiles) > 1:
        results_preface = "ts_results"
        results_dir_name = generate_id(results_preface, job_id)
        base_results_dir = base_dir / results_dir_name
    else:
        results_preface = get_molecule_name(ligands_smiles[0])
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