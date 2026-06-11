"""Cluster stages for screen-based TS guess workflows."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd

from frust.schema import energy_columns, infer_group_columns, normalize_dataframe
from frust.screen import create_ts_guesses
from frust.stepper import Stepper

FUNCTIONAL = "wB97X-D3"
BASISSET = "6-31G**"
BASISSET_SOLV = "6-31+G**"

try:
    if "SLURM_JOB_ID" in os.environ:
        from nuse import start_monitoring

        start_monitoring(filter_cgroup=True)
except ImportError:
    pass


def _resolve_theory(
    *,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    composite_method: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Resolve workflow theory settings.

    Parameters
    ----------
    functional : str or None, optional
        ORCA functional override.
    basisset : str or None, optional
        ORCA gas-phase basis set override.
    basisset_solv : str or None, optional
        ORCA solvent basis set override.
    composite_method : str or None, optional
        Complete ORCA composite-method keyword, such as ``"r2SCAN-3c"``. When
        provided, no separate basis set keywords are emitted.

    Returns
    -------
    tuple[str, str or None, str or None]
        Method keyword, gas-phase basis set, and solvent basis set.
    """
    if composite_method is not None:
        conflicting = [
            name
            for name, value in (
                ("functional", functional),
                ("basisset", basisset),
                ("basisset_solv", basisset_solv),
            )
            if value is not None
        ]
        if conflicting:
            joined = ", ".join(f"`{name}`" for name in conflicting)
            raise ValueError(
                "`composite_method` cannot be combined with "
                f"{joined}; ORCA composite methods already include their basis/corrections."
            )
        return composite_method, None, None

    return (
        functional or FUNCTIONAL,
        basisset or BASISSET,
        basisset_solv or BASISSET_SOLV,
    )


def _orca_options(method: str, basis: str | None, *keywords: str) -> dict[str, None]:
    """Build ORCA simple-input options for a method and optional basis."""
    options: dict[str, None] = {method: None}
    if basis:
        options[basis] = None
    options.update({keyword: None for keyword in keywords})
    return options


def _best_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the best row per available structure identity group.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with one or more energy columns.

    Returns
    -------
    pandas.DataFrame
        Lowest-energy row per inferred structure group.
    """
    df = normalize_dataframe(df)
    if df.empty:
        return df
    last_energy = energy_columns(df)[-1]
    group_cols = infer_group_columns(df)
    if not group_cols:
        return df.sort_values(last_energy).head(1)
    return df.sort_values(group_cols + [last_energy]).groupby(group_cols, dropna=False).head(1)


def _single_screen_target(screen_target: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Validate and normalize one screen-chain target.

    Parameters
    ----------
    screen_target : pandas.DataFrame
        One-row expanded screen dataframe containing a fixed ``ts_type``.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        Normalized target dataframe and uppercase TS type.
    """
    if not isinstance(screen_target, pd.DataFrame):
        raise TypeError("`screen_target` must be a pandas DataFrame")
    if len(screen_target) != 1:
        raise ValueError("`screen_target` must contain exactly one TS/system/rpos row")
    if "ts_type" not in screen_target.columns:
        raise ValueError("`screen_target` must contain a 'ts_type' column")

    target = screen_target.copy()
    ts_type = str(target["ts_type"].iloc[0]).upper()
    target.loc[target.index[0], "ts_type"] = ts_type
    return target, ts_type


def run_init(
    screen_target: pd.DataFrame,
    *,
    ts_backend: str = "tsguess2",
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    save_dir: str | None = None,
    work_dir: str | None = None,
    save_output_dir: bool = True,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    composite_method: str | None = None,
) -> pd.DataFrame:
    """Generate one screen TS target and run the initialization filter chain.

    Parameters
    ----------
    screen_target : pandas.DataFrame
        One-row expanded screen dataframe with fixed ``ts_type`` and ``rpos``.
    ts_backend : {"tsguess2", "tsguess"}, optional
        TS guess backend used by :func:`frust.screen.create_ts_guesses`.
    n_confs : int or None, optional
        Number of TS guess conformers to generate. ``None`` selects the
        backend's rotatable-bond heuristic.
    n_cores : int, optional
        CPU cores used for embedding and calculator stages.
    mem_gb : int, optional
        Memory in GB forwarded to :class:`frust.stepper.Stepper`.
    debug : bool, optional
        If ``True``, calculator debug mode is enabled.
    top_n : int, optional
        Number of xTB-optimized conformers retained before DFT filtering.
    save_dir : str or None, optional
        Directory for intermediate and output parquet files.
    work_dir : str or None, optional
        Calculator scratch directory.
    save_output_dir : bool, optional
        Whether to preserve calculator output directories.
    functional : str or None, optional
        ORCA functional override.
    basisset : str or None, optional
        ORCA gas-phase basis set override.
    basisset_solv : str or None, optional
        ORCA solvent basis set override.
    composite_method : str or None, optional
        Complete ORCA composite-method keyword, such as ``"r2SCAN-3c"``. When
        provided, no separate basis set keywords are emitted.

    Returns
    -------
    pandas.DataFrame
        Initialization-stage dataframe written to ``init.parquet``.
    """
    target, ts_type = _single_screen_target(screen_target)
    save_path = Path(save_dir or ".")
    save_path.mkdir(parents=True, exist_ok=True)

    guesses = create_ts_guesses(
        target,
        ts_types=[ts_type],
        n_confs=n_confs,
        n_cores=n_cores,
        backend=ts_backend,
    )
    df = guesses[ts_type]
    if df.empty:
        raise ValueError(f"No TS guesses generated for screen target {ts_type!r}")
    df.to_parquet(save_path / "ts_guess.parquet")

    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_path,
        save_output_dir=save_output_dir,
        work_dir=work_dir,
    )

    df = step.xtb(
        df,
        name="xtb_preopt",
        options={"gfnff": None, "opt": None},
        constraint=True,
        n_cores=2,
    )
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2}, n_cores=2)
    df = step.xtb(
        df,
        name="xtb_opt",
        options={"gfn": 2, "opt": None},
        constraint=True,
        lowest=top_n,
        n_cores=2,
    )

    current_method, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
        composite_method=composite_method,
    )

    df = step.orca(
        df,
        name="DFT-pre-SP",
        options=_orca_options(current_method, current_basisset, "TightSCF", "SP", "NoSym"),
    )

    df = step.orca(
        df,
        name="DFT-pre-Opt",
        options=_orca_options(
            current_method,
            current_basisset,
            "TightSCF",
            "SlowConv",
            "Opt",
            "NoSym",
        ),
        constraint=True,
        lowest=1,
    )

    fn_name = inspect.currentframe().f_code.co_name
    parquet_name = fn_name.split("_")[1]
    df.to_parquet(save_path / f"{parquet_name}.parquet")
    return df


def run_hess(
    parquet_path: str,
    *,
    n_cores: int = 2,
    mem_gb: int = 32,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    composite_method: str | None = None,
) -> pd.DataFrame:
    """Run the Hessian stage for the best initialized screen TS rows."""
    df = normalize_dataframe(pd.read_parquet(Path(save_dir or ".") / parquet_path))
    if df.empty:
        return df

    df = _best_rows(df)
    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    current_method, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
        composite_method=composite_method,
    )

    df = step.orca(
        df,
        name="Hess",
        options=_orca_options(current_method, current_basisset, "TightSCF", "Freq", "NoSym"),
        read_files=["input.hess"],
    )

    stem = parquet_path.rsplit(".", 1)[0]
    out_parquet = stem + ".hess.parquet"
    df.to_parquet(Path(save_dir or ".") / out_parquet)
    return df


def run_OptTS(
    parquet_path: str,
    *,
    n_cores: int = 8,
    mem_gb: int = 40,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    composite_method: str | None = None,
) -> pd.DataFrame:
    """Run the ORCA OptTS stage using the previous Hessian."""
    df = normalize_dataframe(pd.read_parquet(Path(save_dir or ".") / parquet_path))
    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    current_method, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
        composite_method=composite_method,
    )

    df = step.orca(
        df,
        name="OptTS",
        options=_orca_options(
            current_method,
            current_basisset,
            "TightSCF",
            "SlowConv",
            "OptTS",
            "NoSym",
        ),
        use_last_hess=True,
    )

    stem = parquet_path.rsplit(".", 1)[0]
    out_parquet = stem + ".optts.parquet"
    df.to_parquet(Path(save_dir or ".") / out_parquet)
    return df


def run_freq(
    parquet_path: str,
    *,
    n_cores: int = 8,
    mem_gb: int = 40,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    composite_method: str | None = None,
) -> pd.DataFrame:
    """Run the final frequency stage."""
    df = normalize_dataframe(pd.read_parquet(Path(save_dir or ".") / parquet_path))
    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    current_method, current_basisset, _ = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
        composite_method=composite_method,
    )

    df = step.orca(
        df,
        name="Freq",
        options=_orca_options(
            current_method,
            current_basisset,
            "TightSCF",
            "SlowConv",
            "Freq",
            "NoSym",
        ),
    )

    stem = parquet_path.rsplit(".", 1)[0]
    out_parquet = stem + ".freq.parquet"
    df.to_parquet(Path(save_dir or ".") / out_parquet)
    return df


def run_solv(
    parquet_path: str,
    *,
    n_cores: int = 8,
    mem_gb: int = 40,
    debug: bool = False,
    save_dir: str | None = None,
    work_dir: str | None = None,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    composite_method: str | None = None,
) -> pd.DataFrame:
    """Run the solvent single-point stage."""
    df = normalize_dataframe(pd.read_parquet(Path(save_dir or ".") / parquet_path))
    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=save_dir,
        save_output_dir=True,
        work_dir=work_dir,
    )

    current_method, _, current_basisset_solv = _resolve_theory(
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
        composite_method=composite_method,
    )

    df = step.orca(
        df,
        name="DFT-solv",
        options=_orca_options(current_method, current_basisset_solv, "TightSCF", "SP", "NoSym"),
        xtra_inp_str='%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend',
    )

    stem = parquet_path.rsplit(".", 1)[0]
    out_parquet = stem + ".solv.parquet"
    df.to_parquet(Path(save_dir or ".") / out_parquet)
    return df


def run_cleanup(save_dir: str) -> None:
    """Remove intermediate parquet files and bulky Hessian byte columns."""
    print("[INFO]: Cleanup initiated...")

    parquet_files = list(Path(save_dir).glob("*.parquet"))
    if not parquet_files:
        print("No parquet files found.")
        return

    depths = {f: len(f.suffixes) for f in parquet_files}
    max_depth = max(depths.values())

    kept = []
    for file_path, depth in depths.items():
        if depth < max_depth:
            print(f"[INFO]: Removing residual parquet file {file_path}")
            try:
                file_path.unlink()
            except Exception as exc:
                print(f"[WARN]: Could not remove {file_path}: {exc}")
        else:
            kept.append(file_path)

    for file_path in kept:
        try:
            df = normalize_dataframe(pd.read_parquet(file_path))
        except Exception as exc:
            print(f"[WARN]: Could not read {file_path}: {exc}")
            continue

        hess_cols = []
        for col in df.columns:
            if col.endswith(".hess"):
                nonnull = next(
                    (
                        value
                        for value in df[col].tolist()
                        if value is not None
                        and not (isinstance(value, float) and np.isnan(value))
                    ),
                    None,
                )
                if nonnull is None or isinstance(nonnull, (bytes, bytearray, str)):
                    hess_cols.append(col)

        if hess_cols:
            print(f"[INFO]: Dropping .hess columns from {file_path.name}: {hess_cols}")
            df = df.drop(columns=hess_cols)
            try:
                df.to_parquet(file_path)
            except Exception as exc:
                print(f"[WARN]: Failed writing cleaned Parquet {file_path}: {exc}")
        else:
            print(f"[INFO]: No .hess columns in {file_path.name}")

    print("[INFO]: Cleanup done!")
