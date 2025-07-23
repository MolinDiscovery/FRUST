import pandas as pd

import pandas as pd

def summarize_ts_vibrations(
    df: pd.DataFrame,
    col: str = "DFT-wB97X-D3-6-31G**-OptTS-vibs",
    max_rows: int = 5
):
    true_ts_count = 0
    non_ts_count = 0
    rows = []

    for idx, row in df.iterrows():
        ligand = row.get("ligand_name", "")
        rpos   = row.get("rpos", "")
        vibs   = row[col]

        freqs = [entry.get('frequency') for entry in vibs]
        neg_freqs = [f for f in freqs if f < 0]
        pos_freqs = [f for f in freqs if f >= 0]

        is_true_ts = len(neg_freqs) == 1
        status = "✅ True TS" if is_true_ts else f"❌ Not TS ({len(neg_freqs)} neg)"

        if is_true_ts:
            true_ts_count += 1
        else:
            non_ts_count += 1

        if neg_freqs:
            neg_str = ", ".join(f"{f:.2f}" for f in neg_freqs[:3])
            if len(neg_freqs) > 3:
                neg_str += " ..."
        else:
            neg_str = "No negatives"
        neg_str += " |"

        if pos_freqs:
            pos_str = ", ".join(f"{f:.1f}" for f in pos_freqs[:3])
            if len(pos_freqs) > 3:
                pos_str += " ..."
        else:
            pos_str = "No positives"
        pos_str += " |"

        rows.append({
            "Structure": idx,
            "Ligand": ligand,
            "RPOS": rpos,
            "Status": status,
            "Neg. freqs": neg_str,
            "Pos. freqs": pos_str
        })

    result_df = pd.DataFrame(rows)

    print(result_df.head(max_rows).to_string(index=False))
    if len(result_df) > max_rows:
        print(f"\n... and {len(result_df) - max_rows} more rows.")

    print("\nSummary:")
    print(f"  ✅ True TSs : {true_ts_count}")
    print(f"  ❌ Non-TSs  : {non_ts_count}")