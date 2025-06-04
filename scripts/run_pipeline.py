# frust/run_pipeline.py
import json, argparse
from frust import stepper, embedder
from frust.transformers import transformer_ts

def main(cfg_path: str):
    cfg = json.loads(open(cfg_path).read())

    # ─── initial TS generation ──────────────────────────────────────────────
    ts   = transformer_ts(**cfg["transformer"])
    embd = embedder.embed_ts(ts, **cfg["embedder"])
    df   = stepper.Stepper.build_initial_df(embd)

    stp  = stepper.Stepper(
             ligands_smiles = cfg["ligands_smiles"],
             output_base    = cfg["output_base"],
             job_id         = cfg["job_id"],
             n_cores        = cfg["resources"]["cores"],
             memory_gb      = cfg["resources"]["mem_gb"],
             debug          = cfg["debug"],
           )

    # ─── iterate over recipe ────────────────────────────────────────────────
    for s in cfg["steps"]:
        engine = s["engine"].lower()
        if engine == "xtb":
            df = stp.xtb(df,
                         name        = s["name"],
                         options     = s["options"],
                         detailed_inp_str = s.get("inp", ""),
                         constraint  = s.get("constraint", False))
        elif engine == "orca":
            df = stp.orca(df,
                          name        = s["name"],
                          options     = s["options"],
                          xtra_inp_str = s.get("inp", ""),
                          constraint  = s.get("constraint", False))
        else:
            raise ValueError(f"Unknown engine {engine}")

    out = stp.base_dir / "final.parquet"
    df.to_parquet(out)
    print("✓ pipeline finished →", out)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("config", help="JSON file produced by PipelineBuilder")
    main(a.parse_args().config)