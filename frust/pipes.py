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
    ):
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
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    )
    df0 = step.build_initial_df(embedded)
    df1 = step.xtb(df0, options={"gfnff": None, "opt": None}, constraint=True)
    df2 = step.xtb(df1, options={"gfn": 2})
    df3 = step.xtb(df2, options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    last_energy = [c for c in df3.columns if c.endswith("_energy")][-1]
    df3_filt = (
        df3.sort_values(["ligand_name", "rpos", last_energy])
           .groupby(["ligand_name", "rpos"])
           .head(1)
    )
    
    if not DFT:
        if output_parquet:
            df3_filt.to_parquet(output_parquet)            
        return df3_filt

    # ↓↓↓↓↓↓↓↓ This code only executes if DFT is True ↓↓↓↓↓↓↓↓
    options = {
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    }

    df4 = step.orca(df3, name="DFT-pre-SP", options=options, save_step=True)

    detailed_inp = """%geom\nCalc_Hess true\nend"""
    options = {
        "wB97X-D3" : None,
        "6-31G**"  : None,
        "TightSCF" : None,
        "SlowConv" : None,
        "OptTS"    : None,
        "Freq"     : None,
        "NoSym"    : None,
    }

    df5 = step.orca(df4, name="DFT", options=options, xtra_inp_str=detailed_inp, save_step=True, lowest=1)

    detailed_inp = """%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend"""
    options = {
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP"      : None,
        "NoSym"   : None,
    }

    df6 = step.orca(df5, name="DFT-SP", options=options, xtra_inp_str=detailed_inp, save_step=True)
    
    if output_parquet:
        df6.to_parquet(output_parquet)
    return df6


def run_ts_per_lig(
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

    embedded = embed_ts(ts_structs, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    step = Stepper(
    ligand_smiles_list,
    n_cores=n_cores,
    debug=debug,
    output_base=out_dir,
    save_output_dir=save_output_dir,
    )
    df0 = step.build_initial_df(embedded)
    df1 = step.xtb(df0, options={"gfnff": None, "opt": None}, constraint=True)
    df2 = step.xtb(df1, options={"gfn": 2})
    df3 = step.xtb(df2, options={"gfn": 2, "opt": None}, constraint=True, lowest=top_n)

    last_energy = [c for c in df3.columns if c.endswith("_energy")][-1]
    df3_filt = (
        df3.sort_values(["ligand_name", "rpos", last_energy])
           .groupby(["ligand_name", "rpos"])
           .head(1)
    )
    
    if not DFT:
        if output_parquet:
            df3_filt.to_parquet(output_parquet)            
        return df3_filt

    # ↓↓↓↓↓↓↓↓ This code only executes if DFT is True ↓↓↓↓↓↓↓↓
    options = {
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    }
    
    df4 = step.orca(df3, name="DFT-pre-SP", options=options, save_step=False)
    
    if ts_type == "INT3":
        opt = "Opt"
    else:
        opt = "OptTS"

    detailed_inp = """%geom\nCalc_Hess true\nend"""
    options = {
        "wB97X-D3" : None,
        "6-31G**"  : None,
        "TightSCF" : None,
        "SlowConv" : None,
        opt        : None,
        "Freq"     : None,
        "NoSym"    : None,
    }

    df5 = step.orca(df4, name="DFT", options=options, xtra_inp_str=detailed_inp, save_step=True, lowest=1)

    detailed_inp = """%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend"""
    options = {
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP"      : None,
        "NoSym"   : None,
    }

    df6 = step.orca(df5, name="DFT-SP", options=options, xtra_inp_str=detailed_inp, save_step=True)
    
    if output_parquet:
        df6.to_parquet(output_parquet)
    return df6


# ──────────────────────  catalytic-cycle molecules  ──────────────────────
def run_mols(
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
    df1 = step.xtb(df0, options={"gfnff": None, "opt": None})
    df2 = step.xtb(df1, options={"gfn": 2})
    df3 = step.xtb(df2, options={"gfn": 2, "opt": None}, lowest=top_n)

    df3_fin = (
        df3
        .sort_values(["ligand_name", "xtb-gfn-opt-electronic_energy"])
        .groupby("ligand_name")
        .head(1)
    )

    # 4) if no DFT requested, save/return
    if not DFT:
        if output_parquet:
            df3_fin.to_parquet(output_parquet)
        return df3_fin

    # ↓↓↓↓↓↓↓↓ DFT branch ↓↓↓↓↓↓↓↓
    options = {
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    }

    df4 = step.orca(df3, name="DFT-pre-SP", options=options, save_step=True)


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
    df5 = step.orca(df4, options=orca_opts, xtra_inp_str=detailed_inp, lowest=1)

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
    *,
    out_dir: str | None = None,
    save_output_dir: bool = True,      
):
    from rdkit import Chem
    smi = ligand_smiles_list[0]
    m = Chem.MolFromSmiles(smi)
    mol_dict = {"mol": m}
    mols_dict_embedded = embed_mols(mol_dict, n_confs=1)

    step = Stepper(
        [smi],
        n_cores=1,
        memory_gb=2,
        debug=False,
        output_base=out_dir,
        save_output_dir=save_output_dir,
    )
    df0 = step.build_initial_df(mols_dict_embedded)

    df1 = step.xtb(df0, options={"gfn": 2})
    df2 = step.orca(df0, options={"HF": None, "STO-3G": None})


if __name__ == '__main__':
    FRUST_path = str(Path(__file__).resolve().parent.parent)
    print(f"Running in main. FRUST path: {FRUST_path}")
    run_ts_per_lig(
        ["CN1C=CC=C1"],
        ts_guess_xyz=f"{FRUST_path}/structures/ts2.xyz",
        n_confs=1,
        debug=False,
        out_dir="noob",
        save_output_dir=False,
        #output_parquet="TS3_test.parguet",
        DFT=True,
        top_n=1
    )

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
