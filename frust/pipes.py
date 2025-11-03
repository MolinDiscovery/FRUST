# frust/pipes.py
from pathlib import Path
from frust.stepper import Stepper
from frust.embedder import embed_ts, embed_mols
from frust.transformers import transformer_mols
from frust.utils.io import read_ts_type_from_xyz

from rdkit.Chem.rdchem import Mol

import os
try:
    if "SLURM_JOB_ID" in os.environ:
        from nuse import start_monitoring
        start_monitoring(filter_cgroup=True)
except ImportError:
    pass


def create_ts_per_rpos(
    ligand_smiles_list: list[str],
    ts_guess_xyz: str,        
    ) -> list[dict[str, Mol]]:
    """
    Generate transition state (TS) structures for each ligand SMILES using a TS guess XYZ template.

    Args:
        ligand_smiles_list (List[str]): List of ligand SMILES strings for which to create TS structures.
        ts_guess_xyz (str): Path to an XYZ file containing the TS guess geometry. The TS type
            (e.g., 'TS1', 'TS2', 'TS3', 'TS4') is inferred from the comment line.

    Returns:
        List[Dict[str, rdkit.Chem.Mol]]: A list of dictionaries, each mapping a TS identifier
            (e.g., reaction position key) to an RDKit Mol object representing the generated TS.
    """

    ts_type = read_ts_type_from_xyz(ts_guess_xyz)

    if ts_type == 'TS1':
        from frust.transformers import transformer_ts1
        transformer_ts = transformer_ts1
    elif ts_type == 'TS2':
        from frust.transformers import transformer_ts2
        transformer_ts = transformer_ts2
    elif ts_type == 'TS3':
        from frust.transformers import transformer_ts3
        transformer_ts = transformer_ts3
    elif ts_type == 'TS4':
        from frust.transformers import transformer_ts4
        transformer_ts = transformer_ts4
    elif ts_type == 'INT3':
        from frust.transformers import transformer_int3
        transformer_ts = transformer_int3        
    else:
        raise ValueError(f"Unrecognized TS type: {ts_type}")

    ts_structs = {}
    for smi in ligand_smiles_list:
        ts_mols = transformer_ts(smi, ts_guess_xyz)
        ts_structs.update(ts_mols)

    ts_structs_list = []
    for k, i in ts_structs.items():
        ts_structs_list.append({k:i})   

    return ts_structs_list


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

    ligand_smiles = list(ts_struct.values())[0][2]

    step = Stepper(
    ligand_smiles,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_calc_dirs=True,
    save_output_dir=save_output_dir,
    work_dir=work_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, options={"gfn": 2})
    df = step.xtb(df, options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

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
        last_energy = [c for c in df.columns if c.endswith("_energy")][-1]
        df = (df.sort_values(["ligand_name", "rpos", last_energy]
                            ).groupby(["ligand_name", "rpos"]).head(1))
        
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

    ligand_smiles = list(ts_struct.values())[0][2]

    step = Stepper(
    ligand_smiles,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, options={"gfn": 2})
    df = step.xtb(df, options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    last_energy = [c for c in df.columns if c.endswith("_energy")][-1]
    df3_filt = (
        df.sort_values(["ligand_name", "rpos", last_energy])
           .groupby(["ligand_name", "rpos"])
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

    ligand_smiles = list(ts_struct.values())[0][2]

    step = Stepper(
    ligand_smiles,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    work_dir=work_dir,
    )
    
    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None}, constraint=True, n_cores=2)
    df = step.xtb(df, options={"gfn": 2}, lowest=20, n_cores=2)
    df = step.orca(df, options={"ExtOpt": None, "Opt": None}, constraint=True, lowest=10, uma="omol@uma-s-1p1")
    df = step.orca(df, options={"ExtOpt": None, "OptTS": None, "NumFreq": None}, lowest=1, uma="omol@uma-s-1p1")
    
    if output_parquet:
        df.to_parquet(output_parquet)
    return df


def run_ts_per_lig(
    ligand_smiles_list: list[str],
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
    
    ts_type = read_ts_type_from_xyz(ts_guess_xyz)
    print(ts_type)

    if ts_type == 'TS1':
        from frust.transformers import transformer_ts1
        transformer_ts = transformer_ts1
    elif ts_type == 'TS2':
        from frust.transformers import transformer_ts2
        transformer_ts = transformer_ts2
    elif ts_type == 'TS3':
        from frust.transformers import transformer_ts3
        transformer_ts = transformer_ts3
    elif ts_type == 'TS4':
        from frust.transformers import transformer_ts4
        transformer_ts = transformer_ts4
    elif ts_type == 'INT3':
        from frust.transformers import transformer_int3
        transformer_ts = transformer_int3
    else:
        raise ValueError(f"Unrecognized TS type: {ts_type}")

    ts_structs = {}
    for smi in ligand_smiles_list:
        ts_mols = transformer_ts(smi, ts_guess_xyz)
        ts_structs.update(ts_mols)

    embedded = embed_ts(ts_structs, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    step = Stepper(
    ligand_smiles_list,
    n_cores=n_cores,
    memory_gb=mem_gb,
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    )
    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None}, constraint=True)
    df = step.xtb(df, options={"gfn": 2})
    df = step.xtb(df, options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

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
    }, lowest=1)

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
    ligand_smiles_list: list[str],
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
        ligand_smiles_list,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_output_dir=save_output_dir,
        save_calc_dirs=False,
    )
    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None})
    df = step.xtb(df, options={"gfn": 2})
    df = step.xtb(df, options={"gfn": 2, "opt": None}, lowest=top_n)

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
    }, lowest=1)

    # 4) if no DFT requested, save/return
    if not DFT:

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

    df = step.orca(df, options={
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
    ligand_smiles_list: list[str],
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

    # 3) cascade
    step = Stepper(
        ligand_smiles_list,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_output_dir=save_output_dir,
    )
    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None}, n_cores=1)
    df = step.xtb(df, options={"gfn": 2}, n_cores=1)
    df = step.orca(df, options={"ExtOpt": None, "Opt": None}, uma="omol", lowest=10)
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
        ligand_smiles_list,
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
    df5 = step.orca(df0, options=orca_opts, xtra_inp_str=detailed_inp)

    # b) single-point with solvent model
    detailed_inp = """%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend"""
    orca_opts = {
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP":       None,
        "NoSym":    None,
    }
    df6 = step.orca(df5, options=orca_opts, xtra_inp_str=detailed_inp)

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
        [smi],
        n_cores=n_cores,
        memory_gb=2,
        debug=False,
        output_base=out_dir,
        save_output_dir=save_output_dir,
    )
    df0 = step.build_initial_df(mols_dict_embedded)

    df1 = step.xtb(df0, options={"gfn": 2})
    df2 = step.orca(df0, options={"HF": None, "STO-3G": None})


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

    step = Stepper(list(mols.keys()), step_type="mol", save_output_dir=False)
    df = step.build_initial_df(mols)

    name = df["custom_name"].iloc[0]

    step = Stepper([name],
                    step_type="none",
                    debug=debug,
                    save_output_dir=save_output_dir,
                    output_base=out_dir,
                    n_cores=n_cores,
                    memory_gb=mem_gb,
                    work_dir=work_dir,
                    save_calc_dirs=True)
    
    df = step.xtb(df, options={"gfn": 2, "opt": None})
    df = step.orca(df, options={
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


# ─── NEW: two-stage helpers for splitting Freq ──────────────────────────
import re
import os
import pandas as pd
from frust.stepper import Stepper  # adjust import if different

def run_ts_per_rpos_geom_only(
    ts_struct: dict[str, tuple["Mol", list, str]],
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
    """Stage-1: run everything except the costly frequency calculation.

    Args:
        ts_struct: One TS structure dict as produced by create_ts_per_rpos.
        n_confs: Number of conformers or None.
        n_cores: CPU cores for xTB/ORCA.
        mem_gb: Memory in GB.
        debug: If True, lighter/shorter steps.
        top_n: Keep N lowest xTB conformers before DFT.
        out_dir: Output directory base.
        work_dir: Working directory (scratch).
        output_parquet: Path to write stage-1 manifest parquet.
        save_output_dir: Whether to save calc dirs.
        DFT: Whether to do DFT optimization stages.
        select_mols: Which molecule subsets to include.
    """

    pattern = re.compile(
        r'^(?:(?P<prefix>(?:TS|INT)\d*|Mols)\()?'
        r'(?P<ligand>.+?)_rpos\('
        r'(?P<rpos>\d+)\)\)?$'
    )

    name = list(ts_struct.keys())[0]
    m = pattern.match(name)
    ts_type = m.group("prefix") if m else None

    embedded = embed_ts(
        ts_struct, ts_type=ts_type, n_confs=n_confs, optimize=not debug
    )

    ligand_smiles = list(ts_struct.values())[0][2]

    step = Stepper(
        ligand_smiles,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_calc_dirs=True,
        save_output_dir=save_output_dir,
        work_dir=work_dir,
    )

    df = step.build_initial_df(embedded)
    df = step.xtb(df, options={"gfnff": None, "opt": None}, constraint=True, n_cores=2)
    df = step.xtb(df, options={"gfn": 2}, n_cores=2)
    df = step.xtb(df, options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n, n_cores=2)

    functional = "wB97X-D3"
    basisset = "6-31G**"
    basisset_solv = "6-31+G**"

    df = step.orca(df, name="DFT-pre-SP", options={
        functional: None,
        basisset: None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    })

    if not DFT:
        last_energy = [c for c in df.columns if c.endswith("_energy")][-1]
        df = (df.sort_values(["ligand_name", "rpos", last_energy])
                .groupby(["ligand_name", "rpos"]).head(1))
        if output_parquet:
            df.to_parquet(output_parquet)
        return df

    # DFT geometry optimization (no Freq here)
    df = step.orca(df, name="DFT-pre-Opt", options={
        functional: None,
        basisset: None,
        "TightSCF": None,
        "SlowConv": None,
        "Opt": None,
        "NoSym": None,
    }, constraint=True, lowest=1)

    opt_key = "Opt" if (ts_type or "").upper() == "INT3" else "szq"

    df = step.orca(df, name="DFT", options={
        functional: None,
        basisset: None,
        "TightSCF": None,
        "SlowConv": None,
        opt_key: None,
        "NoSym": None,
    }, lowest=1)

    # Keep your SP-in-solvent as part of stage-1 if desired
    df = step.orca(df, name="DFT-SP", options={
        functional: None,
        basisset_solv: None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    }, xtra_inp_str=(
        "%CPCM\nSMD TRUE\nSMDSOLVENT \"chloroform\"\nend"
    ))

    if output_parquet:
        df.to_parquet(output_parquet)

    # Optional sentinel
    if output_parquet:
        done = os.path.splitext(output_parquet)[0] + ".geom.done"
        try:
            open(done, "a").close()
        except OSError:
            pass

    return df


def run_ts_freq_only_from_parquet(
    parquet_path: str,
    *,
    n_cores: int = 5,
    mem_gb: int = 35,
    debug: bool = False,
    out_dir: str | None = None,
    work_dir: str | None = None,
):
    """Stage-2: run frequency only from a stage-1 parquet.

    Args:
        parquet_path: Path to stage-1 parquet with the chosen structure(s).
        n_cores: CPU cores for ORCA.
        mem_gb: Memory in GB.
        debug: If True, lighter/shorter steps.
        out_dir: Output directory base.
        work_dir: Working directory (scratch).
    """
    df = pd.read_parquet(parquet_path)

    # Idempotency: skip if already has Freq artifacts or sentinel exists.
    stem = os.path.splitext(parquet_path)[0]
    sentinel = stem + ".freq.done"
    if os.path.exists(sentinel):
        return df

    last_energy = [c for c in df.columns if c.endswith("_energy")][-1]
    df = (df.sort_values(["ligand_name", "rpos", last_energy])
            .groupby(["ligand_name", "rpos"]).head(1))

    ligand_smiles = list(dict.fromkeys(df["smiles"].tolist()))
    step = Stepper(
        ligand_smiles,
        n_cores=n_cores,
        memory_gb=mem_gb,
        debug=debug,
        output_base=out_dir,
        save_calc_dirs=True,
        save_output_dir=True,
        work_dir=work_dir,
    )

    df = step.orca(df, name="DFT-Freq", options={
        # "wB97X-D3": None,
        # "6-31G**": None,
        "PBE": None,
        "def2-SVP": None,
        "TightSCF": None,
        "Freq": None,
        "NoSym": None,
    }, lowest=1)

    # Write back next to the original parquet (overwrite/merge as you prefer)
    out_parquet = stem + ".with_freq.parquet"
    df.to_parquet(out_parquet)

    try:
        open(sentinel, "a").close()
    except OSError:
        pass

    return df


def run_orca_smoke_geom_only(
    ts_struct: dict[str, tuple[Mol, list, str]],
    *,
    n_confs: int | None = None,
    n_cores: int = 2,
    mem_gb: int = 8,
    debug: bool = False,
    top_n: int = 1,
    out_dir: str | None = None,
    work_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
    DFT: bool = True,
):
    """
    Stage-1 (SMOKE): build a tiny DF and run a cheap SP (no Freq).
    Produces a parquet hand-off + `.geom.done` sentinel.
    """
    from tooltoad.chemutils import xyz2mol

    f = Path("../structures/misc/HH.xyz")
    mols: dict[str, tuple[Mol, list[int]]] = {}

    with open(f, "r") as file:
        xyz_block = file.read()
        mol = xyz2mol(xyz_block)
        mols[f.stem] = (mol, [0])

    # Initial DF
    step = Stepper(
        list(mols.keys()),
        step_type="mol",
        save_output_dir=False,
        debug=debug,
    )
    df = step.build_initial_df(mols)

    # Use the one name we just created
    name = df["custom_name"].iloc[0]

    # Real stepper for actual compute
    step = Stepper(
        [name],
        step_type="none",
        debug=debug,
        save_output_dir=save_output_dir,
        output_base=out_dir,
        n_cores=n_cores,
        memory_gb=mem_gb,
        work_dir=work_dir,
        save_calc_dirs=True,
    )

    # Light compute to exercise IO without big memory use
    df = step.xtb(df, options={"gfn": 2, "opt": None}, save_step=True)
    df = step.orca(df, options={
        "wB97X-D3": None,
        "6-31G**":  None,
        "TightSCF": None,
        "SP":       None,
        "NoSym":    None,
    }, save_step=True)

    if output_parquet:
        df.to_parquet(output_parquet)
        stem = os.path.splitext(output_parquet)[0]
        try:
            open(stem + ".geom.done", "a").close()
        except OSError:
            pass

    return df


def run_orca_smoke_freq_only_from_parquet(
    parquet_path: str,
    *,
    n_cores: int = 1,
    mem_gb: int = 12,
    debug: bool = False,
    out_dir: str | None = None,
    work_dir: str | None = None,
    # accept extra kwargs for drop-in compatibility
    **kwargs,
):
    """
    Stage-2 (SMOKE): load Stage-1 parquet and run a cheap Freq-only step.
    Writes `<stem>.with_freq.parquet` + `.freq.done` sentinel.
    """
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df

    stem = os.path.splitext(parquet_path)[0]
    sentinel = stem + ".freq.done"
    if os.path.exists(sentinel):
        return df

    # Keep a single row; these smoke tests are just plumbing checks
    df = df.head(1)
    name = df["custom_name"].iloc[0]

    step = Stepper(
        [name],
        step_type="none",
        debug=debug,
        save_output_dir=True,
        output_base=out_dir,
        n_cores=n_cores,
        memory_gb=mem_gb,
        work_dir=work_dir,
        save_calc_dirs=True,
    )

    df = step.orca(df, options={
        "wB97X-D3": None,
        "6-31G**":  None,
        "TightSCF": None,
        "Freq":     None,
        "NoSym":    None,
    })

    out_parquet = stem + ".with_freq.parquet"
    df.to_parquet(out_parquet)

    try:
        open(sentinel, "a").close()
    except OSError:
        pass

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


