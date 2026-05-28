# frust/pipes.py
from pathlib import Path
from frust import screen
from frust.stepper import Stepper
from frust.embedder import embed_ts, embed_mols
from frust.transformers import transformer_mols
from frust.utils.io import read_ts_type_from_xyz
from frust.utils.mols import create_ts_per_rpos, create_mol_per_rpos
from frust.schema import energy_columns
from frust.utils.dataframes import lowest_energy_rows
import pandas as pd

from rdkit.Chem.rdchem import Mol

import os
try:
    if "SLURM_JOB_ID" in os.environ:
        from nuse import start_monitoring
        start_monitoring(filter_cgroup=True)
except ImportError:
    pass


def _normalize_smiles_input(
    ligand_smiles_df: pd.DataFrame | list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize molecule pipeline input to a DataFrame and unique SMILES list."""
    if isinstance(ligand_smiles_df, pd.DataFrame):
        df = ligand_smiles_df.copy()
        if "smiles" not in df.columns:
            raise ValueError("ligand_smiles_df must contain a 'smiles' column")
    else:
        df = pd.DataFrame({"smiles": list(ligand_smiles_df)})

    if df["smiles"].isna().any():
        raise ValueError("ligand_smiles_df['smiles'] contains missing values")

    unique_smiles = list(dict.fromkeys(df["smiles"].tolist()))
    return df, unique_smiles


def _screen_input_to_systems(screen_input: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Return expanded screen systems from a CSV, component table, or systems table.

    Parameters
    ----------
    screen_input : str, pathlib.Path, or pandas.DataFrame
        Screen CSV/component table with ``role`` and ``smiles`` columns, or an
        already-expanded systems dataframe with screen system columns.

    Returns
    -------
    pandas.DataFrame
        Expanded substrate-catalyst systems ready for TS guess generation.
    """
    system_columns = {"system_name", "substrate_smiles", "catalyst_smiles", "rpos"}
    if isinstance(screen_input, pd.DataFrame) and system_columns.issubset(screen_input.columns):
        return screen_input.copy()
    return screen.expand(screen.read(screen_input))


def run_screen_ts_per_rpos(
    screen_input: str | Path | pd.DataFrame,
    *,
    ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    work_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
) -> pd.DataFrame:
    """Run the screen-based TS workflow for substrate/catalyst systems.

    Parameters
    ----------
    screen_input : str, pathlib.Path, or pandas.DataFrame
        Screen CSV/component table accepted by :func:`frust.screen.read`, or an
        expanded systems dataframe from :func:`frust.screen.expand`.
    ts_types : tuple or list of str, optional
        Transition-state types to generate. Defaults to TS1-TS4.
    n_confs : int or None, optional
        Number of conformers per TS guess. ``None`` uses the TS guess module's
        rotatable-bond heuristic.
    n_cores : int, optional
        Number of CPU cores used by embedding and calculator stages.
    mem_gb : int, optional
        Memory in GB forwarded to :class:`frust.stepper.Stepper`.
    debug : bool, optional
        Forwarded to :class:`frust.stepper.Stepper`.
    top_n : int, optional
        Number of low-energy xTB rows retained before DFT filtering.
    out_dir : str or None, optional
        Base output directory for calculation artifacts.
    work_dir : str or None, optional
        Calculator scratch directory.
    output_parquet : str or None, optional
        If provided, write the resulting dataframe to this Parquet file.
    save_output_dir : bool, optional
        Whether to keep the output directory structure created by the stepper.
    DFT : bool, optional
        If ``True``, continue from the xTB/DFT prescreen into DFT refinement.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing screened or DFT-refined TS candidates.
    """
    systems = _screen_input_to_systems(screen_input)
    ts_guesses = screen.create_ts_guesses(
        systems,
        ts_types=ts_types,
        n_confs=n_confs,
        n_cores=n_cores,
    )
    df = pd.concat(ts_guesses.values(), ignore_index=True)
    if df.empty:
        raise ValueError("No screen TS guesses were generated")

    step = Stepper(
        step_type=None,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_calc_dirs=True,
        save_output_dir=save_output_dir,
        work_dir=work_dir,
    )

    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2})
    df = step.xtb(
        df,
        name="xtb_opt",
        options={"gfn": 2, "opt": None},
        constraint=True,
        lowest=top_n,
    )

    functional = "wB97X-D3"
    basisset = "6-31G**"
    basisset_solv = "6-31+G**"
    freq = "Freq"

    df = step.orca(
        df,
        name="DFT-pre-SP",
        options={
            functional: None,
            basisset: None,
            "TightSCF": None,
            "SP": None,
            "NoSym": None,
        },
    )

    if not DFT:
        df = lowest_energy_rows(df)
        if output_parquet:
            df.to_parquet(output_parquet)
        return df

    df = step.orca(
        df,
        name="DFT-pre-Opt",
        options={
            functional: None,
            basisset: None,
            "TightSCF": None,
            "SlowConv": None,
            "Opt": None,
            "NoSym": None,
        },
        constraint=True,
        lowest=1,
    )

    df = step.orca(
        df,
        name="DFT",
        options={
            functional: None,
            basisset: None,
            "TightSCF": None,
            "SlowConv": None,
            "OptTS": None,
            freq: None,
            "NoSym": None,
        },
        lowest=1,
    )

    df = step.orca(
        df,
        name="DFT-SP",
        options={
            functional: None,
            basisset_solv: None,
            "TightSCF": None,
            "SP": None,
            "NoSym": None,
        },
        xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""",
    )

    if output_parquet:
        df.to_parquet(output_parquet)
    return df


def run_ts_per_rpos(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    work_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
):
    import re
    pattern = re.compile(
    r'^(?:(?P<prefix>(?:TS|INT)\d*|Mols)\()?'
    r'(?P<ligand>.+?)_rpos\('        
    r'(?P<rpos>\d+)\)\)?$'           
    )

    name = list(ts_struct.keys())[0]
    m = pattern.match(name)
    ts_type = m.group("prefix")
    
    embedded = embed_ts(ts_struct, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    step = Stepper(
    step_type=ts_type,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_calc_dirs=True,
    save_output_dir=save_output_dir,
    work_dir=work_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2})
    df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    functional      = "wB97X-D3" # wB97X-D3, wB97M-V
    basisset        = "6-31G**" # 6-31G**, def2-TZVPD
    basisset_solv   = "6-31+G**" # 6-31+G**, def2-TZVPD
    freq            = "Freq" # NumFreq, Freq

    df = step.orca(df, name="DFT-pre-SP", options={
        functional  : None,
        basisset    : None,
        "TightSCF"  : None,
        "SP"        : None,
        "NoSym"     : None,
    })

    if not DFT:
        last_energy = energy_columns(df)[-1]
        df = (df.sort_values(["substrate_name", "structure_type", "molecule_role", "rpos", last_energy]
                            ).groupby(["substrate_name", "structure_type", "molecule_role", "rpos"], dropna=False).head(1))
        
        if output_parquet:
            df.to_parquet(output_parquet)            
        return df

    # ↓↓↓↓↓↓↓↓ This code only executes if DFT is True ↓↓↓↓↓↓↓↓
    df = step.orca(df, name="DFT-pre-Opt", options={
        functional : None,
        basisset   : None,
        "TightSCF" : None,
        "SlowConv" : None,
        "Opt"      : None,
        "NoSym"    : None,
    }, constraint=True, lowest=1)

    if ts_type.upper() == "INT3":
        opt = "Opt"
    else:
        opt = "OptTS"

    df = step.orca(df, name="DFT", options={
        functional : None,
        basisset   : None,
        "TightSCF" : None,
        "SlowConv" : None,
        opt        : None,
        freq       : None,
        "NoSym"    : None,
    }, lowest=1)

    df = step.orca(df, name="DFT-SP", options={
        functional      : None,
        basisset_solv   : None,
        "TightSCF"      : None,
        "SP"            : None,
        "NoSym"         : None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")
    
    if output_parquet:
        df.to_parquet(output_parquet)

    return df

def run_ts_per_rpos_UMA(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
):
    import re
    pattern = re.compile(
    r'^(?:(?P<prefix>(?:TS|INT)\d*|Mols)\()?'
    r'(?P<ligand>.+?)_rpos\('        
    r'(?P<rpos>\d+)\)\)?$'           
    )

    # Get type...
    name = list(ts_struct.keys())[0]
    m = pattern.match(name)
    ts_type = m.group("prefix")
    
    embedded = embed_ts(ts_struct, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    step = Stepper(
    step_type=ts_type,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2})
    df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    last_energy = energy_columns(df)[-1]
    df3_filt = (
        df.sort_values(["substrate_name", "structure_type", "molecule_role", "rpos", last_energy])
           .groupby(["substrate_name", "structure_type", "molecule_role", "rpos"], dropna=False)
           .head(1)
    )
    
    if not DFT:
        if output_parquet:
            df3_filt.to_parquet(output_parquet)            
        return df3_filt

    # ↓↓↓↓↓↓↓↓ This code only executes if DFT is True ↓↓↓↓↓↓↓↓

    df = step.orca(df, name="DFT-pre-SP", options={
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    })

    df = step.orca(df, name="DFT-pre-Opt", options={
        "wB97X-D3" : None,
        "6-31G**"  : None,
        "TightSCF" : None,
        "SlowConv" : None,
        "Opt"      : None,
        "NoSym"    : None,
    }, constraint=True, lowest=1)

    if ts_type.upper() == "INT3":
        opt = "Opt"
    else:
        opt = "OptTS"

    df = step.orca(df, name="UMA", options={"ExtOpt": None, "OptTS": None, "NumFreq": None}, xtra_inp_str="""%geom
  Calc_Hess  true
  NumHess    true
  Recalc_Hess 5
  MaxIter    300
end""", lowest=1)

    df = step.orca(df, name="DFT-SP", options={
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP"      : None,
        "NoSym"   : None,
    })

    df = step.orca(df, name="DFT-SP-solvent", options={
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP"      : None,
        "NoSym"   : None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")
    
    if output_parquet:
        df.to_parquet(output_parquet)
    return df

def run_ts_per_rpos_UMA_short(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    work_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
):
    import re
    pattern = re.compile(
    r'^(?:(?P<prefix>(?:TS|INT)\d*|Mols)\()?'
    r'(?P<ligand>.+?)_rpos\('        
    r'(?P<rpos>\d+)\)\)?$'           
    )

    # Get type...
    name = list(ts_struct.keys())[0]
    m = pattern.match(name)
    ts_type = m.group("prefix")
    
    embedded = embed_ts(ts_struct, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    step = Stepper(
    step_type=ts_type,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    work_dir=work_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, constraint=True, n_cores=2)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2}, lowest=20, n_cores=2)
    df = step.orca(df, name="uma_opt", options={"ExtOpt": None, "Opt": None}, constraint=True, lowest=10, uma="omol@uma-s-1p1")
    df = step.orca(df, name="uma_tsopt", options={"ExtOpt": None, "OptTS": None, "NumFreq": None}, lowest=1, uma="omol@uma-s-1p1")
    
    if output_parquet:
        df.to_parquet(output_parquet)
    return df


def run_ts_per_lig(
    ligand_smiles_df: pd.DataFrame,
    ts_guess_xyz: str,
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
):
    """Run the TS workflow for each ligand in a ligand table.

    The function expands each ligand into TS structures using the
    transition-state guess geometry, generates conformers, runs the XTB
    pre-screening steps, and optionally performs ORCA DFT refinement.

    Parameters
    ----------
    ligand_smiles_df : pandas.DataFrame
        Input table containing at least a ``smiles`` column with ligand
        SMILES strings.
    ts_guess_xyz : str
        Path to the XYZ file containing the transition-state guess geometry.
        The TS type is inferred from this file.
    n_confs : int or None, optional
        Number of conformers to generate per TS structure. If ``None``, the
        embedder default is used.
    n_cores : int, optional
        Number of CPU cores to use for downstream calculations.
    mem_gb : int, optional
        Memory limit in gigabytes passed to :class:`frust.stepper.Stepper`.
    debug : bool, optional
        If ``True``, disables conformer optimization and keeps the workflow
        lighter for debugging.
    top_n : int, optional
        Number of lowest-energy structures retained after the XTB screening
        stage.
    out_dir : str or None, optional
        Base output directory for calculation artifacts.
    output_parquet : str or None, optional
        If provided, write the resulting dataframe to this Parquet file.
    save_output_dir : bool, optional
        Whether to keep the output directory structure created by the stepper.
    DFT : bool, optional
        If ``True``, continue from the XTB screen into the DFT refinement
        stages. If ``False``, return after the pre-screening workflow.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the screened or DFT-refined TS candidates.

    """
    ts_structs = create_ts_per_rpos(
        ligand_smiles_df,
        ts_guess_xyz,
        return_format="dict",
    )
    
    ts_type = read_ts_type_from_xyz(ts_guess_xyz)

    embedded = embed_ts(ts_structs, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    step = Stepper(
    step_type=ts_type,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    )
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2})
    df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    functional      = "wB97X-D3" # wB97X-D3, wB97M-V
    basisset        = "6-31G**" # 6-31G**, def2-TZVPD
    basisset_solv   = "6-31+G**" # 6-31+G**, def2-TZVPD
    freq            = "Freq" # NumFreq, Freq

    df = step.orca(df, name="DFT-pre-SP", options={
        functional  : None,
        basisset    : None,
        "TightSCF"  : None,
        "SP"        : None,
        "NoSym"     : None,
    })

    if not DFT:
        if output_parquet:
            df.to_parquet(output_parquet)
        return df

    # ↓↓↓↓↓↓↓↓ This code only executes if DFT is True ↓↓↓↓↓↓↓↓

    df = step.orca(df, name="DFT-pre-Opt", options={
        functional : None,
        basisset   : None,
        "TightSCF" : None,
        "SlowConv" : None,
        "Opt"      : None,
        "NoSym"    : None,
    }, constraint=True, lowest=1)
    
    if ts_type.upper() == "INT3":
        opt = "Opt"
    else:
        opt = "OptTS"

    df = step.orca(df, name="DFT", options={
        functional : None,
        basisset   : None,
        "TightSCF" : None,
        "SlowConv" : None,
        opt        : None,
        freq       : None,
        "NoSym"    : None,
    }, lowest=1)

    df = step.orca(df, name="DFT-SP", options={
        functional      : None,
        basisset_solv   : None,
        "TightSCF"      : None,
        "SP"            : None,
        "NoSym"         : None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")
    
    if output_parquet:
        df.to_parquet(output_parquet)
    return df


# ──────────────────────  catalytic-cycle molecules  ──────────────────────
def run_mols(
    ligand_smiles_df: pd.DataFrame | list[str],
    *,
    n_confs: int | None = 5,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
    select_mols: str | list[str] = "all",  # "all", "uniques", "generics", or specific names
):
    """Run the standard molecular catalytic-cycle workflow for ligand SMILES.

    This is the molecule analogue of the transition-state pipelines: FRUST
    expands each ligand into the molecular states produced by
    :func:`frust.transformers.transformer_mols`, embeds conformers, runs the
    default xTB preoptimization/single-point/optimization cascade, and
    optionally continues through the DFT ORCA branch.

    Parameters
    ----------
    ligand_smiles_df : pandas.DataFrame or list[str]
        Ligand input. DataFrame inputs must contain a ``smiles`` column. A
        plain list is interpreted as a list of ligand SMILES and is converted
        to a one-column DataFrame internally. Duplicate SMILES are only
        expanded once.
    n_confs : int or None, optional
        Number of conformers to embed for each generated molecular state. Pass
        ``None`` to let the embedder use its own default behavior.
    n_cores : int, optional
        Number of CPU cores used for conformer embedding and as the default
        resource setting for the :class:`frust.stepper.Stepper`.
    mem_gb : int, optional
        Memory in GB passed to the :class:`frust.stepper.Stepper`.
    debug : bool, optional
        Enable debug behavior in the underlying :class:`frust.stepper.Stepper`.
    top_n : int, optional
        Number of lowest-energy conformers retained after the xTB optimization
        stage.
    out_dir : str or None, optional
        Base directory for calculation output if output directories are saved.
    output_parquet : str or None, optional
        If provided, write the final dataframe to this parquet path before
        returning it.
    save_output_dir : bool, optional
        Whether the :class:`frust.stepper.Stepper` should save a FRUST output
        directory.
    DFT : bool, optional
        If ``False``, stop after the xTB cascade and ``DFT-pre-SP``. If
        ``True``, also run the ORCA ``DFT-Opt`` and solvent ``DFT-SP`` stages.
    select_mols : str or list[str], optional
        Molecular states to generate before embedding. Use ``"all"`` for the
        full set. Use ``"uniques"`` for the ligand and rpos-dependent states
        such as ``int2_rpos(...)``, ``mol2_rpos(...)``, and
        ``HBpin-ligand_rpos(...)``. Use ``"generics"`` for states shared
        across all ligands/rpos values: ``dimer``, ``HH``, ``catalyst``, and
        ``HBpin-mol``. Use a string or list to select explicit families from
        ``"dimer"``, ``"HH"``, ``"ligand"``, ``"catalyst"``, ``"int2"``,
        ``"mol2"``, ``"HBpin-ligand"``, and ``"HBpin-mol"``.

    Returns
    -------
    pandas.DataFrame
        Result dataframe containing embedded structures and all calculation
        columns produced by the requested xTB/ORCA stages.

    Examples
    --------
    Run the full default molecular-state expansion for two conformers per
    state:

    >>> import frust as ft
    >>> df = ft.pipes.run_mols(["COc1ccccc1"], n_confs=2)

    Run only ligand and catalyst reference states:

    >>> df = ft.pipes.run_mols(
    ...     ["COc1ccccc1"],
    ...     select_mols=["ligand", "catalyst"],
    ... )

    Run only reactive-position variants for the ``int2`` family:

    >>> df = ft.pipes.run_mols(["COc1ccccc1"], select_mols="int2")
    """
    ligand_smiles_df, ligand_smiles_list = _normalize_smiles_input(ligand_smiles_df)

    # 1) build generic-cycle molecules (with optional selection)
    mols = create_mol_per_rpos(
        ligand_smiles_df,
        return_format="dict",
        select_mols=select_mols,
    )

    # 2) embed
    embedded = embed_mols(mols, n_confs=n_confs, n_cores=n_cores)

    # 3) xTB cascade
    step = Stepper(
        step_type="MOLS",
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_output_dir=save_output_dir,
        save_calc_dirs=False,
    )
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, n_cores=2)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2}, n_cores=2)
    df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None}, lowest=top_n, n_cores=2)

    functional      = "wB97X-D3" # wB97X-D3, wB97M-V
    basisset        = "6-31G**" # 6-31G**, def2-TZVPD
    basisset_solv   = "6-31+G**" # 6-31+G**, def2-TZVPD
    freq            = "Freq" # Freq, NumFreq

    df = step.orca(df, name="DFT-pre-SP", options={
        functional  : None,
        basisset    : None,
        "TightSCF"  : None,
        "SP"        : None,
        "NoSym"     : None,
    })

    # 4) if no DFT requested, save/return
    if not DFT:
        df = lowest_energy_rows(df)
        if output_parquet:
            df.to_parquet(output_parquet)
        return df

    # ↓↓↓↓↓↓↓↓ DFT branch ↓↓↓↓↓↓↓↓

    df = step.orca(df, "DFT-Opt", options={
        functional  : None,
        basisset    : None,
        "TightSCF"  : None,
        "SlowConv"  : None,
        "Opt"       : None,
        freq        : None,
        "NoSym"     : None,
    }, lowest=1)

    df = step.orca(df, name="DFT-SP", options={
        functional      : None,
        basisset_solv   : None,
        "TightSCF"      : None,
        "SP"            : None,
        "NoSym"         : None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")

    if output_parquet:
        df.to_parquet(output_parquet)
    return df


def run_mols_custom(
    ligand_smiles_df: pd.DataFrame | list[str],
    *,
    n_confs: int | None = 5,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
    select_mols: str | list[str] = "all",  # "all", "uniques", "generics", or specific names
):
    """Run the standard molecular catalytic-cycle workflow for ligand SMILES.

    This is the molecule analogue of the transition-state pipelines: FRUST
    expands each ligand into the molecular states produced by
    :func:`frust.transformers.transformer_mols`, embeds conformers, runs the
    default xTB preoptimization/single-point/optimization cascade, and
    optionally continues through the DFT ORCA branch.

    Parameters
    ----------
    ligand_smiles_df : pandas.DataFrame or list[str]
        Ligand input. DataFrame inputs must contain a ``smiles`` column. A
        plain list is interpreted as a list of ligand SMILES and is converted
        to a one-column DataFrame internally. Duplicate SMILES are only
        expanded once.
    n_confs : int or None, optional
        Number of conformers to embed for each generated molecular state. Pass
        ``None`` to let the embedder use its own default behavior.
    n_cores : int, optional
        Number of CPU cores used for conformer embedding and as the default
        resource setting for the :class:`frust.stepper.Stepper`.
    mem_gb : int, optional
        Memory in GB passed to the :class:`frust.stepper.Stepper`.
    debug : bool, optional
        Enable debug behavior in the underlying :class:`frust.stepper.Stepper`.
    top_n : int, optional
        Number of lowest-energy conformers retained after the xTB optimization
        stage.
    out_dir : str or None, optional
        Base directory for calculation output if output directories are saved.
    output_parquet : str or None, optional
        If provided, write the final dataframe to this parquet path before
        returning it.
    save_output_dir : bool, optional
        Whether the :class:`frust.stepper.Stepper` should save a FRUST output
        directory.
    DFT : bool, optional
        If ``False``, stop after the xTB cascade and ``DFT-pre-SP``. If
        ``True``, also run the ORCA ``DFT-Opt`` and solvent ``DFT-SP`` stages.
    select_mols : str or list[str], optional
        Molecular states to generate before embedding. Use ``"all"`` for the
        full set. Use ``"uniques"`` for the ligand and rpos-dependent states
        such as ``int2_rpos(...)``, ``mol2_rpos(...)``, and
        ``HBpin-ligand_rpos(...)``. Use ``"generics"`` for states shared
        across all ligands/rpos values: ``dimer``, ``HH``, ``catalyst``, and
        ``HBpin-mol``. Use a string or list to select explicit families from
        ``"dimer"``, ``"HH"``, ``"ligand"``, ``"catalyst"``, ``"int2"``,
        ``"mol2"``, ``"HBpin-ligand"``, and ``"HBpin-mol"``.

    Returns
    -------
    pandas.DataFrame
        Result dataframe containing embedded structures and all calculation
        columns produced by the requested xTB/ORCA stages.

    Examples
    --------
    Run the full default molecular-state expansion for two conformers per
    state:

    >>> import frust as ft
    >>> df = ft.pipes.run_mols(["COc1ccccc1"], n_confs=2)

    Run only ligand and catalyst reference states:

    >>> df = ft.pipes.run_mols(
    ...     ["COc1ccccc1"],
    ...     select_mols=["ligand", "catalyst"],
    ... )

    Run only reactive-position variants for the ``int2`` family:

    >>> df = ft.pipes.run_mols(["COc1ccccc1"], select_mols="int2")
    """
    ligand_smiles_df, ligand_smiles_list = _normalize_smiles_input(ligand_smiles_df)

    # 1) build generic-cycle molecules (with optional selection)
    mols = create_mol_per_rpos(
        ligand_smiles_df,
        return_format="dict",
        select_mols=select_mols,
    )

    # 2) embed
    embedded = embed_mols(mols, n_confs=n_confs, n_cores=n_cores)

    # 3) xTB cascade
    step = Stepper(
        step_type="MOLS",
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_output_dir=save_output_dir,
        save_calc_dirs=False,
    )
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, n_cores=2)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2}, n_cores=2)
    #df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None}, lowest=top_n, n_cores=2)
    df = step.gxtb(df, name="gxtb_opt", options={"opt": None}, n_cores=2)

    functional      = "wB97X-D3" # wB97X-D3, wB97M-V
    basisset        = "6-31G**" # 6-31G**, def2-TZVPD
    basisset_solv   = "6-31+G**" # 6-31+G**, def2-TZVPD
    freq            = "Freq" # Freq, NumFreq

    df = step.orca(df, name="DFT-pre-SP", options={
        functional  : None,
        basisset_solv : None,
        "TightSCF"  : None,
        "SP"        : None,
        "NoSym"     : None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")

    # 4) if no DFT requested, save/return
    if not DFT:
        df = lowest_energy_rows(df)
        if output_parquet:
            df.to_parquet(output_parquet)
        return df

    # ↓↓↓↓↓↓↓↓ DFT branch ↓↓↓↓↓↓↓↓

    df = step.orca(df, "DFT-Opt", options={
        functional  : None,
        basisset    : None,
        "TightSCF"  : None,
        "SlowConv"  : None,
        "Opt"       : None,
        freq        : None,
        "NoSym"     : None,
    }, lowest=1)

    df = step.orca(df, name="DFT-SP", options={
        functional      : None,
        basisset_solv   : None,
        "TightSCF"      : None,
        "SP"            : None,
        "NoSym"         : None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")

    if output_parquet:
        df.to_parquet(output_parquet)
    return df


def run_mols_UMA(
    ligand_smiles_df: pd.DataFrame | list[str],
    *,
    n_confs: int = 5,
    n_cores: int = 4,
    mem_gb: int = 20,    
    debug: bool = False,
    top_n: int = 5,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
    select_mols: str | list[str] = "all",  # "all", "uniques", "generics", or specific names
):
    ligand_smiles_df, ligand_smiles_list = _normalize_smiles_input(ligand_smiles_df)

    # 1) build generic-cycle molecules (with optional selection)
    mols = create_mol_per_rpos(
        ligand_smiles_df,
        return_format="dict",
        select_mols=select_mols,
    )

    # 2) embed
    embedded = embed_mols(mols, n_confs=n_confs, n_cores=n_cores)

    # 3) cascade
    step = Stepper(
        step_type="MOLS",
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_output_dir=save_output_dir,
    )
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, n_cores=1)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2}, n_cores=1)
    df = step.orca(df, name="uma_opt", options={"ExtOpt": None, "Opt": None}, uma="omol", lowest=10)
    #df = step.orca(df, options={"ExtOpt": None, "NumFreq": None}, uma="omol@uma-s-1p1", lowest=1)
    
    if output_parquet:
            df.to_parquet(output_parquet)

    return df


def run_test(
    ligand_smiles_list: list[str],
    *,
    n_confs: int = 5,
    n_cores: int = 4,
    debug: bool = False,
    top_n: int = 5,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
    select_mols: str | list[str] = "all",  # "all", "uniques", "generics", or specific names
):
    # 1) build generic-cycle molecules (with optional selection)
    mols = {}
    for smi in ligand_smiles_list:
        if select_mols == "all":
            tmp = transformer_mols(ligand_smiles=smi)
        elif select_mols == "uniques":
            tmp = transformer_mols(ligand_smiles=smi, only_uniques=True)
        elif select_mols == "generics":
            tmp = transformer_mols(ligand_smiles=smi, only_generics=True)
        else:
            tmp = transformer_mols(ligand_smiles=smi, select=select_mols)

        mols.update(tmp)

    # 2) embed
    embedded = embed_mols(mols, n_confs=n_confs, n_cores=n_cores)

    # 3) xTB cascade
    step = Stepper(
        step_type="MOLS",
        n_cores=n_cores,
        debug=debug,
        output_base=out_dir,
        save_output_dir=save_output_dir
    )
    df0 = step.build_initial_df(embedded)

    # ↓↓↓↓↓↓↓↓ DFT branch ↓↓↓↓↓↓↓↓

    # a) TS-like Hess-calc & frequency for each ligand
    detailed_inp = """%geom\nCalc_Hess true\nend"""
    orca_opts = {
        "wB97X-D3": None,
        "6-31G**": None,
        "TightSCF": None,
        "SlowConv": None,
        "Opt":     None,
        "Freq":    None,
        "NoSym":   None,
    }
    print(df0)
    df5 = step.orca(df0, name="DFT-Opt", options=orca_opts, xtra_inp_str=detailed_inp)

    # b) single-point with solvent model
    detailed_inp = """%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend"""
    orca_opts = {
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP":       None,
        "NoSym":    None,
    }
    df6 = step.orca(df5, name="DFT-SP", options=orca_opts, xtra_inp_str=detailed_inp)

    if output_parquet:
        df6.to_parquet(output_parquet)
    return df6


def run_small_test(
    ligand_smiles_list: list[str],
    ts_guess_xyz: str,
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,    
):
    from rdkit import Chem
    smi = ligand_smiles_list[0]
    m = Chem.MolFromSmiles(smi)
    mol_dict = {"mol": m}
    mols_dict_embedded = embed_mols(mol_dict, n_confs=n_confs)

    step = Stepper(
        step_type="MOLS",
        n_cores=n_cores,
        memory_gb=2,
        debug=False,
        output_base=out_dir,
        save_output_dir=save_output_dir,
    )
    df0 = step.build_initial_df(mols_dict_embedded)

    df1 = step.xtb(df0, name="xtb_sp", options={"gfn": 2})
    df2 = step.orca(df0, name="orca_test", options={"HF": None, "STO-3G": None})


def run_orca_smoke_test(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    work_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,        
):
    from tooltoad.chemutils import xyz2mol

    f = Path("../structures/misc/HH.xyz")
    mols = {}
    with open(f, "r") as file:
        xyz_block = file.read()
        mol = xyz2mol(xyz_block)
        mols[f.stem] = (mol, [0])

    step = Stepper(step_type="MOLS", save_output_dir=False)
    df = step.build_initial_df(mols)

    name = df["custom_name"].iloc[0]

    step = Stepper(step_type=None,
                    debug=debug,
                    save_output_dir=save_output_dir,
                    output_base=out_dir,
                    n_cores=n_cores,
                    memory_gb=mem_gb,
                    work_dir=work_dir,
                    save_calc_dirs=True)
    
    df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None})
    df = step.orca(df, name="DFT-SP", options={
        "wB97X-D3":     None,
        "6-31G**":      None,
        "TightSCF":     None,
        "SP":          None,
        "NoSym":        None,
        "Freq":         None,
    })

    if output_parquet:
            df.to_parquet(output_parquet)            
    
    return df


def run_ts_for_rpos(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    mem_gb: int = 20,
    debug: bool = False,
    top_n: int = 10,
    out_dir: str | None = None,
    work_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = False,
):
    import re
    pattern = re.compile(
    r'^(?:(?P<prefix>(?:TS|INT)\d*|Mols)\()?'
    r'(?P<ligand>.+?)_rpos\('        
    r'(?P<rpos>\d+)\)\)?$'           
    )

    name = list(ts_struct.keys())[0]
    m = pattern.match(name)
    ts_type = m.group("prefix")
    
    embedded = embed_ts(ts_struct, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    step = Stepper(
    step_type=ts_type,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_calc_dirs=True,
    save_output_dir=save_output_dir,
    work_dir=work_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, name="xtb_sp", options={"gfn": 2})
    df = step.xtb(df, name="xtb_opt", options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    functional      = "wB97X-D3" # wB97X-D3, wB97M-V
    basisset        = "6-31G**" # 6-31G**, def2-TZVPD
    basisset_solv   = "6-31+G**" # 6-31+G**, def2-TZVPD
    freq            = "Freq" # NumFreq, Freq

    df = step.orca(df, name="DFT-pre-SP", options={
        functional  : None,
        basisset    : None,
        "TightSCF"  : None,
        "SP"        : None,
        "NoSym"     : None,
    })

    if not DFT:
        last_energy = energy_columns(df)[-1]
        df = (df.sort_values(["substrate_name", "structure_type", "molecule_role", "rpos", last_energy]
                            ).groupby(["substrate_name", "structure_type", "molecule_role", "rpos"], dropna=False).head(1))
        
        if output_parquet:
            df.to_parquet(output_parquet)            
        return df

    # ↓↓↓↓↓↓↓↓ This code only executes if DFT is True ↓↓↓↓↓↓↓↓
    df = step.orca(df, name="DFT-pre-Opt", options={
        functional : None,
        basisset   : None,
        "TightSCF" : None,
        "SlowConv" : None,
        "Opt"      : None,
        "NoSym"    : None,
    }, constraint=True, lowest=1)

    if ts_type.upper() == "INT3":
        opt = "Opt"
    else:
        opt = "OptTS"

    df = step.orca(df, name="DFT", options={
        functional : None,
        basisset   : None,
        "TightSCF" : None,
        "SlowConv" : None,
        opt        : None,
        freq       : None,
        "NoSym"    : None,
    }, lowest=1)

    df = step.orca(df, name="DFT-SP", options={
        functional      : None,
        basisset_solv   : None,
        "TightSCF"      : None,
        "SP"            : None,
        "NoSym"         : None,
    }, xtra_inp_str="""%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend""")
    
    if output_parquet:
        df.to_parquet(output_parquet)

    return df


if __name__ == '__main__':
    FRUST_path = str(Path(__file__).resolve().parent.parent)
    print(f"Running in main. FRUST path: {FRUST_path}")
    # run_ts_per_lig(
    #     ["CN1C=CC=C1"],
    #     ts_guess_xyz=f"{FRUST_path}/structures/ts2.xyz",
    #     n_confs=1,
    #     debug=False,
    #     out_dir="noob",
    #     save_output_dir=False,
    #     #output_parquet="TS3_test.parguet",
    #     DFT=True,
    #     top_n=1
    # )

    # ts_mols = create_ts_per_rpos(["CN1C=CC=C1"], ts_guess_xyz=f"{FRUST_path}/structures/int3.xyz")
    # for ts_rpos in ts_mols:
    #     run_ts_per_rpos(ts_rpos, save_output_dir=False, n_confs=1)

    # run_mols(
    #     ["CN1C=CC=C1", "CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C"],
    #     debug=False,
    #     save_output_dir=False,
    #     output_parquet="HH.parquet",
    #     DFT=True,
    #     select_mols=["HH"]
    # )

    # ts_mols = create_ts_per_rpos(["CN1C=CC=C1"], ts_guess_xyz=f"{FRUST_path}/structures/ts1.xyz")
    # for ts_rpos in ts_mols:
    #     run_ts_per_rpos_UMA_short(ts_rpos, out_dir="noob", save_output_dir=True, n_confs=2)    
