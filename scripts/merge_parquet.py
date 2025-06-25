#!/usr/bin/env python3
"""
merge_parquet.py: Merge multiple Parquet files with identical schemas into a single file.
Usage:
    python merge_parquet.py -i <input_folder> -o <output_file>
"""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Parquet files with the same schema into one file"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "results"),
        help="Directory containing .parquet files to merge"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="merged.parquet",
        help="Output Parquet file path"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print(f"Error: input directory '{input_path}' does not exist.")
        return

    parquet_files = sorted(input_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in '{input_path}'.")
        return

    print(f"Found {len(parquet_files)} files. Reading and concatenating...")
    dfs = [pd.read_parquet(str(fp)) for fp in parquet_files]
    merged = pd.concat(dfs, ignore_index=True)

    merged.to_parquet(args.output)
    print(f"Merged {len(parquet_files)} files into '{args.output}'.")

if __name__ == "__main__":
    main()
