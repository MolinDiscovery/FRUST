from frust.pipes import create_ts_per_rpos, run_ts_per_rpos_UMA_short, run_mols_UMA

# ts_mols = create_ts_per_rpos(["CN1C=CC=C1"], ts_guess_xyz=f"../structures/ts1.xyz")
# for ts_rpos in ts_mols:
#     run_ts_per_rpos_UMA_short(ts_rpos, out_dir="noob", save_output_dir=True, n_confs=1)

df = run_mols_UMA(
    ["CN1C=CC=C1"],
    output_parquet="results.parquet",
    n_confs=1,
    n_cores=10,
    select_mols=["ligand"]
)