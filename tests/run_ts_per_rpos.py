import os
from pathlib import Path
import shutil

# --- make paths absolute relative to this file ---
HERE = Path(__file__).resolve().parent
CSV_PATH = str((HERE / "../datasets/1m.csv").resolve())
TS_XYZ   = str((HERE / "../structures/ts1.xyz").resolve())
WORK_DIR = str((HERE / "noob").resolve())  # keep your name, but absolute
os.makedirs(WORK_DIR, exist_ok=True)

# --- ORCA env bootstrap (macOS / Apple Silicon safe defaults) ---
def bootstrap_orca_env(
    orca_dir=Path("/Users/jacobmolinnielsen/Library/orca_6_1_0")
):
    orca_dir = Path(orca_dir).expanduser().resolve()
    if orca_dir.exists():
        os.environ.setdefault("ORCA_DIR", str(orca_dir))
        # Put ORCA binaries first on PATH
        os.environ["PATH"] = f"{orca_dir}:{os.environ.get('PATH','')}"
        # Prefer ORCA's own libs; avoid arm64 Homebrew libs for x86_64 ORCA
        dyld_parts = [str(orca_dir)]
        # If you also have Intel Homebrew, add its GCC libs (optional)
        intel_gcc = Path("/usr/local/opt/gcc")
        if intel_gcc.exists():
            # pick highest version dir under /usr/local/opt/gcc/lib/gcc/*
            gcc_lib_root = intel_gcc / "lib" / "gcc"
            if gcc_lib_root.exists():
                # append all subdirs (14, 13, etc.) that exist
                for p in sorted(gcc_lib_root.iterdir()):
                    if p.is_dir():
                        dyld_parts.append(str(p))
        # DO NOT insert /opt/homebrew (arm64) here.
        current = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = ":".join(
            [*dyld_parts, current] if current else dyld_parts
        )
        # Optional: help ORCA not oversubscribe
        os.environ.setdefault("OMP_NUM_THREADS", "10")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        # Clear variables that sometimes confuse downstream tools
        os.environ.pop("XTBPATH", None)
        os.environ.pop("XTBHOME", None)

    # quick sanity: ensure 'orca' resolves
    if not shutil.which("orca"):
        # fall back to ORCA in the dir if present
        cand = orca_dir / "orca"
        if cand.exists():
            os.environ["PATH"] = f"{orca_dir}:{os.environ.get('PATH','')}"
        else:
            print("WARNING: 'orca' not found on PATH and no orca in ORCA_DIR.")

bootstrap_orca_env()

import pandas as pd
from frust.pipes import create_ts_per_rpos, run_ts_per_rpos
import os

CSV_PATH = "../datasets/1m.csv"
TS_XYZ         = "../structures/ts1.xyz"
df       = pd.read_csv(CSV_PATH)
smi_list = list(dict.fromkeys(df["smiles"]))
job_inputs = create_ts_per_rpos(smi_list, TS_XYZ)

n = 0
out_dir = f"test{n}"

job = job_inputs[n]
tag = list(job.keys())[0]

df = run_ts_per_rpos(
    job,
    n_confs=1,
    n_cores=10,
    mem_gb=30,
    debug=False,
    DFT=True,
    out_dir=out_dir,
    output_parquet=os.path.join(out_dir, f"{tag}.parquet"),
    save_output_dir=True,
    work_dir="noob",
)
