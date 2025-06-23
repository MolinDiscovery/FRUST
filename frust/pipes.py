# frust/pipes.py
from pathlib import Path
from frust.stepper import Stepper
from frust.embedder import embed_ts, embed_mols
from frust.transformers import transformer_mols
from frust.utils.io import read_ts_type_from_xyz

import os
try:
    if "SLURM_JOB_ID" in os.environ:
        from nuse import start_monitoring
        start_monitoring(filter_cgroup=True)
except ImportError:
    pass

# ─────────────────────────  TS xTB  ──────────────────────────
def run_ts(
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
    
    df3_filt.to_parquet("test.parquet")

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

    df5 = step.orca(df4, name="DFT", options=options, xtra_inp_str=detailed_inp, save_step=True, lowest=1, distribute=True)

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
    df5 = step.orca(df4, options=orca_opts, xtra_inp_str=detailed_inp, lowest=1, distribute=True)

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
    df5 = step.orca(df0, options=orca_opts, xtra_inp_str=detailed_inp, lowest=1, distribute=True)

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

if __name__ == '__main__':
    FRUST_path = str(Path(__file__).resolve().parent.parent)
    print(f"Running in main. FRUST path: {FRUST_path}")
    run_ts(
        ["CN1C=CC=C1"],
        ts_guess_xyz=f"{FRUST_path}/structures/ts2_guess.xyz",
        n_confs=3,
        debug=True,
        save_output_dir=False,
        #output_parquet="TS3_test.parguet",
        DFT=True,
        top_n=1
    )
    # run_mols(
    #     ["CN1C=CC=C1", "CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C"],
    #     debug=False,
    #     save_output_dir=False,
    #     output_parquet="HH.parquet",
    #     DFT=True,
    #     select_mols=["HH"]
    # )