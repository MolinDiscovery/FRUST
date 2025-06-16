# frust/pipes.py
from pathlib import Path
from frust.stepper import Stepper
from frust.embedder import embed_ts, embed_mols
from frust.transformers import transformer_mols
from frust.utils.io import read_ts_type_from_xyz


# ─────────────────────────  TS xTB  ──────────────────────────
def run_ts(
    ligand_smiles_list: list[str],
    ts_guess_xyz: str,
    *,
    n_confs: int | None = None,
    n_cores: int = 4,
    debug: bool = False,
    top_n: int = 5,
    out_dir: str | None = None,
    output_parquet: str | None = None,
    save_output_dir: bool = True,
):
    
    ts_type = read_ts_type_from_xyz(ts_guess_xyz)

    if ts_type == 'TS1':
        from frust.transformers import transformer_ts1
        transformer_ts = transformer_ts1
    elif ts_type == 'TS2':
        from frust.transformers import transformer_ts2
        transformer_ts = transformer_ts2

    # 1) build TS guesses
    ts_structs = {}
    for smi in ligand_smiles_list:
        ts_structs.update({
            name: (mol, idxs, smi)
            for name, (mol, idxs) in transformer_ts(
                ligand_smiles=smi, ts_guess_struct=ts_guess_xyz
            ).items()
        })

    # 2) embed
    embedded = embed_ts(ts_structs, ts_type=ts_type, n_confs=n_confs, optimize=not debug)

    # 3) xTB cascade
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

    # ---------- identical filtering to notebook ----------
    df2_filt = (
        df2.sort_values(["ligand_name", "rpos", "xtb-gfn-electronic_energy"])
           .groupby(["ligand_name", "rpos"])
           .head(top_n)
    )

    df3 = step.xtb(df2_filt, options={"gfn": 2, "opt": None}, constraint=True)

    df3_fin = (
        df3.sort_values(["ligand_name", "rpos", "xtb-gfn-opt-electronic_energy"])
           .groupby(["ligand_name", "rpos"])
           .head(1)
    )

    if output_parquet:
        df3_fin.to_parquet(output_parquet)
    return df3_fin


# ──────────────────────  TS DFT  ──────────────────────



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
    save_output_dir: bool = True
):
    mols = {}
    for smi in ligand_smiles_list:
        tmp = transformer_mols(ligand_smiles=smi, only_generics=True)
        ligand_key = list(tmp.keys())[1]
        mols[ligand_key] = tmp[ligand_key]

    embedded = embed_mols(mols, n_confs=n_confs, n_cores=n_cores)

    step = Stepper(
        ligand_smiles_list,
        n_cores=n_cores,
        debug=debug,
        output_base=out_dir,
        save_output_dir=save_output_dir
    )
    df0 = step.build_initial_df(embedded)

    df1 = step.xtb(
        df0,
        options={"gfnff": None, "opt": None},
    )
    df2 = step.xtb(
        df1,
        options={"gfn": 2},
    )

    df2_filt = (
        df2
        .sort_values(["ligand_name", "xtb-gfn-electronic_energy"])
        .groupby("ligand_name")
        .head(top_n)
    )

    df3 = step.xtb(
        df2_filt,
        options={"gfn": 2, "opt": None},
    )

    df3_fin = (
        df3
        .sort_values(["ligand_name", "xtb-gfn-opt-electronic_energy"])
        .groupby("ligand_name")
        .head(1)
    )

    if output_parquet:
        df3_fin.to_parquet(output_parquet)
    return df3_fin


if __name__ == '__main__':
    FRUST_path = str(Path(__file__).resolve().parent.parent)
    print(f"FRUST path: {FRUST_path}")
    run_ts(["CN1C=CC=C1"], ts_guess_xyz=f"{FRUST_path}/structures/ts2_guess.xyz", n_confs=1, debug=False, save_output_dir=False)
    #run_mols(["CN1C=CC=C1"], debug=False, save_output_dir=False)